#!/usr/bin/env python3
"""B1000 MH/CEM-MH: paired seeded resets, repeatability gate, cross-population audit.

Python only; submit from a server-local Bash script. All oracle calls occur AFTER
closed-loop collection. No model training or CEM algorithm is replaced.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from cem_mh_diag_core import (NativeRecorder, array_hash, case_manifest, dump,
                             fixed_rollout, physics, plain, repeat_comparison,
                             restore_prefix, score_metrics)

ROOT = Path(__file__).resolve().parent
LABELS = ("mh", "cemmh")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clone(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    return copy.deepcopy(x)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ids(x):
    a = np.asarray(plain(x))
    return a.reshape(a.shape[0], -1)[:, -1].astype(np.int64)


def run_closed(cfg, dataset, process, model, seed, selected, iterations, trace_enabled):
    import hydra
    import stable_worldmodel as swm
    from eval import img_transform
    from omegaconf import OmegaConf
    from pusht_exact_replay import capture_live_reset_contexts

    seed_all(seed)
    model.eval().requires_grad_(False)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    world = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    taps, starts, contexts, idmap, traces, counts = [], [], [], {}, {}, {}
    env_spec = None

    class Policy(swm.policy.WorldModelPolicy):
        def set_env(self, env):
            nonlocal env_spec
            super().set_env(env)
            envs = getattr(env, "envs", None)
            if envs is None:
                envs = getattr(getattr(env, "unwrapped", None), "envs", None)
            if envs is None or len(envs) != 100:
                raise RuntimeError("Need 100 accessible native PushT envs, not a reduced subset")
            for i, e in enumerate(envs):
                taps.append(NativeRecorder(e.unwrapped, seed * 1000 + i))
            spec = envs[0].unwrapped.spec
            env_spec = {"id": spec.id, "kwargs": copy.deepcopy(spec.kwargs)}

        def get_action(self, info_dict, **kwargs):
            if not contexts:
                contexts.extend(capture_live_reset_contexts(self.env))
                if any(c["variation_count"] == 0 for c in contexts):
                    raise RuntimeError("No live variations captured; refusing exact-context claim")
                initial_ids = ids(info_dict["id"])
                if len(initial_ids) != 100 or len(set(initial_ids)) != 100:
                    raise RuntimeError("First policy call must contain 100 unique IDs")
                idmap.update({int(v): i for i, v in enumerate(initial_ids)})
                starts.extend(t.snapshot() for t in taps)
            return super().get_action(info_dict, **kwargs)

    original_cost = model.get_cost
    original_solver = solver

    def tapped_cost(info, candidates):
        # Official solver batch_size=1. Snapshot inputs BEFORE get_cost mutates them.
        if candidates.shape[:2] != (1, 100):
            raise RuntimeError(f"Unexpected official CEM shape: {candidates.shape}")
        env_id = int(ids(info["id"][:, 0])[0])
        i = idmap[env_id]
        number = counts.get(i, 0)
        counts[i] = number + 1
        solve_no, iteration = divmod(number, 10)
        entry = None
        if i in selected:
            key = (i, solve_no)
            if iteration == 0:
                base = {k: clone(v[:, 0]) for k, v in info.items()
                        if torch.is_tensor(v) or isinstance(v, np.ndarray)}
                if base["pixels"].shape[1] != 1:
                    raise RuntimeError("Diagnostic requires official world.history_size=1")
                traces[key] = {"eval_index": i, "solve_no": solve_no, "info": base,
                               "snapshot": taps[i].snapshot(), "populations": []}
            if iteration in iterations:
                entry = {"iteration": iteration, "candidates": clone(candidates[0])}
        cost = original_cost(info, candidates)
        if entry is not None:
            entry["source_cost"] = clone(cost[0])
            entry["source_elite"] = clone(torch.topk(cost[0], 10, largest=False).indices)
            traces[i, solve_no]["populations"].append(entry)
        return cost

    def tapped_solve(info_dict, *args, **kwargs):
        current_ids = ids(info_dict["id"])
        result = original_solver(info_dict, *args, **kwargs)
        for j, v in enumerate(current_ids):
            i = idmap[int(v)]
            if i in selected:
                n = counts[i]
                if n % 10:
                    raise RuntimeError("Official CEM did not make 10 cost calls per solve")
                traces[i, n // 10 - 1]["returned_actions"] = clone(result["actions"][j])
        return result

    class SolverTap:
        def __getattr__(self, name):
            return getattr(original_solver, name)

        def __call__(self, info_dict, *args, **kwargs):
            return tapped_solve(info_dict, *args, **kwargs)

        def solve(self, info_dict, *args, **kwargs):
            return tapped_solve(info_dict, *args, **kwargs)

    if trace_enabled:
        model.get_cost = tapped_cost
        solver = SolverTap()
    try:
        trans = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
        policy = Policy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                        process=process, transform=trans)
        world.set_policy(policy)
        metrics = world.evaluate_from_dataset(
            dataset, start_steps=list(cfg.diag_start), episodes_idx=list(cfg.diag_episodes),
            goal_offset_steps=25, eval_budget=50,
            callables=OmegaConf.to_container(cfg.eval.callables, resolve=True))
        successes = np.asarray(metrics["episode_successes"], dtype=bool).tolist()
        if len(successes) != 100 or len(starts) != 100:
            raise RuntimeError("Incomplete closed-loop capture")
        if trace_enabled and not selected.issubset({i for i, _ in traces}):
            raise RuntimeError("Some selected cases have no recorded solve")
        for (i, _), tr in traces.items():
            if "returned_actions" not in tr or len(tr["populations"]) != len(iterations):
                raise RuntimeError("Incomplete official solver trace")
            segment = taps[i].segments[tr["snapshot"]["reset_count"] - 1]
            tr["live_steps"] = copy.deepcopy(segment["steps"])
        result = {"successes": successes, "metrics": plain(metrics), "contexts": contexts,
                  "initial_physics": [s["physics"] for s in starts],
                  "initial_recipes": [s["recipe"] for s in starts],
                  "initial_images": [array_hash(s["image"]) for s in starts],
                  "native_reset_counts": [t.reset_count for t in taps],
                  "native_steps": [sum([seg["steps"] for seg in t.segments], []) for t in taps],
                  "env_spec": env_spec}
        return result, list(traces.values())
    finally:
        model.get_cost = original_cost
        for t in taps:
            t.restore()
        try:
            world.envs.close()
        except Exception:
            try:
                world.close()
            except Exception:
                pass


def check_initial(reference, actual):
    from pusht_exact_replay import assert_reset_contexts_match
    assert_reset_contexts_match(reference["contexts"], actual["contexts"])
    if reference["initial_images"] != actual["initial_images"]:
        raise RuntimeError("Initial renders differ across paired runs")
    if not np.allclose(reference["initial_physics"], actual["initial_physics"], rtol=0, atol=1e-6):
        raise RuntimeError("Initial full physics differs across paired runs")


def decode(candidates, scaler):
    a = np.asarray(plain(candidates), dtype=np.float32)
    if a.ndim != 3 or a.shape[1:] != (5, 10) or len(scaler.mean_) != 2:
        raise RuntimeError(f"Unexpected PushT packed actions {a.shape}")
    return scaler.inverse_transform(a.reshape(-1, 2)).reshape(len(a), 25, 2).astype(np.float32)


@torch.inference_mode()
def model_scores(model, info, candidates, device):
    c = torch.as_tensor(candidates, dtype=torch.float32, device=device)[None]
    expanded = {}
    for k, v in info.items():
        if torch.is_tensor(v):
            expanded[k] = v[:, None].expand(1, c.shape[1], *v.shape[1:]).clone()
        elif isinstance(v, np.ndarray):
            expanded[k] = np.repeat(v[:, None], c.shape[1], axis=1)
    costs = model.get_cost(expanded, c)[0]
    elite = torch.topk(costs, min(10, len(costs)), largest=False).indices
    return costs.detach().float().cpu().numpy(), elite.detach().cpu().numpy()


@torch.inference_mode()
def encode_images(model, transform, images, device):
    chunks = []
    for offset in range(0, len(images), 32):
        p = torch.stack([transform(im) for im in images[offset:offset+32]]).to(device)[:, None]
        chunks.append(model.encode({"pixels": p})["emb"][:, -1].cpu())
    return torch.cat(chunks)


def shared_frame(models):
    for name in ("encoder", "projector"):
        a, b = [getattr(models[l], name).state_dict() for l in LABELS]
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise RuntimeError(f"Frozen visual frame differs: {name}")
    if any(getattr(m, "coordinate_adapter", None) is not None for m in models.values()):
        raise RuntimeError("Coordinate adapters require an explicit frame check")


def validate_trace_execution(raw, tr, transform, scaler):
    snap = tr["snapshot"]
    restore_prefix(raw, snap)
    # Verify native renderer and official model-input preprocessing agree.
    rendered = transform(snap["image"])
    if not torch.allclose(rendered, tr["info"]["pixels"][0, -1], rtol=0, atol=1e-5):
        raise RuntimeError("Native render vs official policy pixels mismatch")
    # Validate decoded *actual returned plan* against native executed actions.
    executed = tr["live_steps"][len(snap["prefix"]):][:25]
    actions = decode(np.asarray(tr["returned_actions"])[None], scaler)[0]
    if not executed:
        raise RuntimeError("No live execution after traced solve")
    for j, step in enumerate(executed):
        if not np.allclose(actions[j], step["action"], rtol=0, atol=1e-5):
            raise RuntimeError("Action decoding differs from actual native execution")
        _, _, term, trunc, _ = raw.step(actions[j])
        if (bool(term) != step["term"] or bool(trunc) != step["trunc"]
                or not np.allclose(physics(raw), step["physics"], rtol=0, atol=1e-6)):
            raise RuntimeError("Returned-plan replay does not reproduce live physics")


def analyze(cfg, models, process, out, manifest, device):
    import gymnasium as gym
    from eval import img_transform
    transform = img_transform(cfg)
    shared_frame(models)
    groups = {r["eval_index"]: r["historical_group"] for r in manifest}
    results = []
    populations_dir = out / "populations"
    populations_dir.mkdir(exist_ok=False)
    for source in LABELS:
        # Only load artifacts created locally by this driver, never external .pt files.
        saved = torch.load(out / f"{source}_traces.pt", map_location="cpu", weights_only=False)
        env = gym.make(saved["env_spec"]["id"], **saved["env_spec"]["kwargs"])
        raw = env.unwrapped
        try:
            for tr in saved["traces"]:
                snap, info = tr["snapshot"], tr["info"]
                validate_trace_execution(raw, tr, transform, process["action"])
                with torch.inference_mode():
                    goal = models["mh"].encode({"pixels": info["goal"].to(device)})["emb"][:, -1].cpu()
                for pop in tr["populations"]:
                    key = f"{source}_case{tr['eval_index']}_solve{tr['solve_no']}_iter{pop['iteration']}"
                    print("ANALYZE", key, flush=True)
                    c = np.asarray(pop["candidates"])
                    scores, elites = {}, {}
                    for label in LABELS:
                        scores[label], elites[label] = model_scores(models[label], info, c, device)
                    if (not np.allclose(scores[source], np.asarray(pop["source_cost"].float()), rtol=1e-5, atol=1e-5)
                            or not np.array_equal(elites[source], np.asarray(pop["source_elite"]))):
                        raise RuntimeError(f"Source score/elite reconstruction failed: {key}")
                    replays = []
                    for acts in decode(c, process["action"]):
                        restore_prefix(raw, snap)
                        replays.append(fixed_rollout(raw, acts))
                    real = encode_images(models["mh"], transform, [r["image"] for r in replays], device)
                    oracle = ((real - goal) ** 2).sum(-1).numpy()
                    ever = np.asarray([r["ever_success"] for r in replays])
                    # Oracle tie handling is explicit. It only supplies a diagnostic mean.
                    oi = torch.topk(torch.from_numpy(oracle), 10, largest=False).indices.numpy()
                    elites["encoder"] = oi
                    # Reproduce official mean arithmetic on device, not NumPy float64.
                    ct = torch.as_tensor(c, device=device)
                    means = torch.stack([ct[torch.as_tensor(elites[l], device=device)].mean(0)
                                         for l in (*LABELS, "encoder")]).cpu().numpy()
                    if pop["iteration"] == 9:
                        source_mean = means[LABELS.index(source)]
                        if not np.allclose(source_mean, np.asarray(tr["returned_actions"]), rtol=0, atol=1e-6):
                            raise RuntimeError("Final source elite mean differs from official returned plan")
                    mean_replays = []
                    for acts in decode(means, process["action"]):
                        restore_prefix(raw, snap)
                        r = fixed_rollout(raw, acts)
                        r.pop("image")
                        mean_replays.append(r)
                    metrics = {}
                    for label in LABELS:
                        m, _ = score_metrics(scores[label], oracle, ever)
                        # Replace any CPU tie-dependent top-k counts with actual device top-k.
                        ei = elites[label]
                        m["elite_success_count"] = int(ever[ei].sum())
                        m["success_retention"] = float(ever[ei].sum()/ever.sum()) if ever.any() else None
                        m["oracle_elite_overlap"] = len(set(ei) & set(oi))/10
                        distance = np.asarray([r["official_endpoint_distance"] for r in replays])
                        m["distance_selection_regret"] = float(distance[m["selected_index"]] - distance.min())
                        metrics[label] = m
                    final = {"key": key, "source": source, "eval_index": tr["eval_index"],
                             "solve_no": tr["solve_no"], "iteration": pop["iteration"],
                             "historical_group": groups[tr["eval_index"]], "metrics": metrics,
                             "mean_replays": dict(zip((*LABELS, "encoder"), mean_replays)),
                             "fixed_steps": 25, "prefix_steps": len(snap["prefix"])}
                    results.append(final)
                    dump(populations_dir / f"{key}.json", final)
                    np.savez_compressed(populations_dir / f"{key}.npz", candidates=c,
                        mh_cost=scores["mh"], cemmh_cost=scores["cemmh"], encoder_cost=oracle,
                        mh_elite=elites["mh"], cemmh_elite=elites["cemmh"], encoder_elite=oi,
                        ever_success=ever, endpoint_success=[r["endpoint_success"] for r in replays],
                        first_success_step=[r["first_success_step"] or -1 for r in replays],
                        endpoint_states=np.stack([r["state"] for r in replays]),
                        official_endpoint_distance=[r["official_endpoint_distance"] for r in replays],
                        elite_means=means)
        finally:
            env.close()
    dump(out / "mechanism.json", {"status": "complete", "populations": results,
         "note": "Each row uses one source history/population scored by BOTH models. Fixed endpoint costs are distinct from ever-success. Means are diagnostic, not new closed-loop runs."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Prior evaluation's cem/ directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    p.add_argument("--stage", choices=("capture", "analyze", "all"), default="all")
    p.add_argument("--controls", type=int, default=2, help="Historical common-fail and common-success controls per seed")
    p.add_argument("--iterations", type=int, nargs="+", default=[0, 5, 9])
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.controls < 0 or len(set(args.iterations)) != len(args.iterations) or any(i not in range(10) for i in args.iterations):
        p.error("Invalid control count or iterations")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from omegaconf import OmegaConf
    import stable_worldmodel as swm
    from eval import get_dataset
    from eval_lowbudget_failure_autopsy import _build_process, _prepare_eval_rows
    from scripts.summarize_cem_mh_eval import load_result
    if importlib.metadata.version("stable-worldmodel") != "0.0.6":
        raise RuntimeError("This diagnostic targets recorded stable-worldmodel==0.0.6; do not silently upgrade")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the recorded official evaluation protocol")
    source = {l: load_result(Path(args.input_dir)/f"{l}_seed{args.seed}_b1000.json", args.seed, 1000) for l in LABELS}
    manifest = case_manifest(source["mh"], source["cemmh"], args.seed, args.controls)
    cfg = OmegaConf.load(ROOT / "config/eval/pusht.yaml")
    del cfg["defaults"]
    cfg.solver = OmegaConf.load(ROOT / "config/eval/solver/cem.yaml")
    cfg.seed, cfg.eval.num_eval, cfg.world.max_episode_steps = args.seed, 100, 100
    cfg.solver.num_samples, cfg.solver.n_steps, cfg.solver.topk = 100, 10, 10
    cfg.solver.batch_size, cfg.solver.var_scale, cfg.solver.device = 1, 1.0, args.device
    cfg.solver._target_ = "stable_worldmodel.solver.CEMSolver"
    cfg.world.history_size, cfg.world.frame_skip = 1, 1
    cfg.plan_config.horizon, cfg.plan_config.receding_horizon, cfg.plan_config.action_block = 5, 5, 5
    cfg.eval.dataset_name, cfg.eval.img_size = "pusht_expert_train", 224
    cfg.eval.goal_offset_steps, cfg.eval.eval_budget = 25, 50
    selection = source["mh"]["selection"]
    cfg.diag_start, cfg.diag_episodes = selection["start_steps"], selection["episodes_idx"]
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    _, _, ep, st = _prepare_eval_rows(cfg, dataset)
    if ep.tolist() != selection["episodes_idx"] or st.tolist() != selection["start_steps"]:
        raise RuntimeError("Current dataset selection differs from source evaluation")
    process = _build_process(cfg, dataset)
    models, hashes = {}, {}
    for l in LABELS:
        checkpoint = Path(swm.data.utils.get_cache_dir()) / (source[l]["policy"] + "_object.ckpt")
        hashes[l] = sha(checkpoint)
        models[l] = swm.policy.AutoCostModel(source[l]["policy"]).to(args.device).eval().requires_grad_(False)
        models[l].interpolate_pos_encoding = True
    shared_frame(models)
    if hashes["mh"] == hashes["cemmh"]:
        raise RuntimeError("Both labels resolve to the same checkpoint")
    out = Path(args.output_dir).resolve()
    if args.stage in ("capture", "all"):
        out.mkdir(parents=True, exist_ok=False)
        dump(out / "manifest.json", manifest)
        dump(out / "protocol.json", {"seed": args.seed, "checkpoint_sha256": hashes,
             "policies": {l: source[l]["policy"] for l in LABELS},
             "historical_successes": {l: source[l]["successes"] for l in LABELS}, "config": OmegaConf.to_container(cfg, resolve=True),
             "iterations": args.iterations, "controls": args.controls,
             "source_sha256": {l: sha(Path(args.input_dir)/f"{l}_seed{args.seed}_b1000.json") for l in LABELS},
             "cohort": "NEW explicitly seeded reset cohort; historical labels select cases only, not claimed historical exact replay"})
        (out/"installed_cem.py").write_text(inspect.getsource(swm.solver.CEMSolver))
        results = {}
        reference = None
        try:
            for label in LABELS:
                for repeat in range(2):
                    print(f"CAPTURE seed={args.seed} {label} repeat={repeat}", flush=True)
                    r, traces = run_closed(cfg, dataset, process, models[label], args.seed,
                                           {m["eval_index"] for m in manifest}, set(args.iterations), repeat == 0)
                    dump(out / f"{label}_repeat{repeat}.json", r)
                    if reference is None:
                        reference = r
                    check_initial(reference, r)
                    results[label, repeat] = r
                    if repeat == 0:
                        torch.save({"env_spec": r["env_spec"], "traces": traces}, out / f"{label}_traces.pt")
                        # Fail before more expensive runs if replay is not faithful.
                        import gymnasium as gym
                        from eval import img_transform
                        check_env = gym.make(r["env_spec"]["id"], **r["env_spec"]["kwargs"])
                        try:
                            for tr in traces:
                                validate_trace_execution(check_env.unwrapped, tr, img_transform(cfg), process["action"])
                        finally:
                            check_env.close()
                        dump(out / f"{label}_execution_check.json", {"passed": True, "traces": len(traces)})
            comparisons = {l: repeat_comparison(results[l, 0], results[l, 1]) for l in LABELS}
            for l in LABELS:
                if results[l, 0]["native_reset_counts"] != results[l, 1]["native_reset_counts"]:
                    comparisons[l]["passed"] = False
                    comparisons[l]["reset_count_mismatch"] = True
            gate = {"passed": all(x["passed"] for x in comparisons.values()), "models": comparisons}
            dump(out / "repeatability.json", gate)
            if not gate["passed"]:
                raise RuntimeError("Repeated model execution changed; mechanism analysis blocked")
        except Exception as exc:
            dump(out / "BLOCKED.json", {"error": str(exc), "stage": "capture"})
            raise
    if args.stage in ("analyze", "all"):
        gate = json.loads((out/"repeatability.json").read_text())
        protocol = json.loads((out/"protocol.json").read_text())
        if not gate["passed"] or protocol["checkpoint_sha256"] != hashes or protocol["seed"] != args.seed:
            raise RuntimeError("Repeatability/checkpoint/seed gate failed")
        manifest = json.loads((out/"manifest.json").read_text())
        try:
            analyze(cfg, models, process, out, manifest, args.device)
        except Exception as exc:
            dump(out / "BLOCKED.json", {"error": str(exc), "stage": "analyze"})
            raise
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    main()
