#!/usr/bin/env python3
"""Official B=9000 paired evaluation and frozen-model geometry diagnosis.

Each policy is executed once. Failure labels, solve observations, executed
actions and CEM populations come from those same runs. Physical candidate
replays are counterfactual state-reset experiments; mean replay error against
the recorded execution is reported, never substituted for actual execution.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig, OmegaConf

from eval import get_dataset, img_transform
from eval_lowbudget_failure_autopsy import (
    TraceCEMSolver, _build_process, _jsonable, _load_start_goal_states,
    _physical_cost, _prepare_eval_rows, _spearman,
)
from eval_b3000_paired_failure_analysis import (
    _extract_variations, _matched_controls, _normalized_to_raw, _slice_info,
)
from eval_b3000_critical_jepa_mechanism import (
    _find_trace_for_env, _goal_embedding_from_solver, _predict_processed_context,
    _record_map, _reset_state, _run_closed_loop_recording,
)
from eval_b3000_hardstate_encoder_geometry import (
    _component_costs, _partial_spearman, _selection_metrics,
)
from eval_pusht_action_ranking import _encode_images
from pusht_official_protocol import (
    finite_summary, matched_factor_accuracy, paired_outcomes, validate_protocol,
)


def write_json(path, data):
    def clean(x):
        x = _jsonable(x)
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_model(policy, device):
    model = swm.policy.AutoCostModel(str(policy)).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def model_fingerprint(model, visual_only=False):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if visual_only and not name.startswith(("encoder.", "projector.")):
            continue
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def check_cem_parity(cfg):
    """Check tracing against the installed official solver before expensive work."""
    version = importlib.metadata.version("stable-worldmodel")
    if version != "0.0.6":
        raise RuntimeError(f"Frozen evaluator requires stable-worldmodel==0.0.6, got {version}")

    class ToyCost:
        def get_cost(self, info, candidates):
            target = info["state"][..., 0].to(candidates.device)
            return ((candidates - target[..., None]) ** 2).sum(dim=(-2, -1))

    args = dict(model=ToyCost(), batch_size=1, num_samples=30, var_scale=1.0,
                n_steps=3, topk=3, device=str(cfg.solver.device), seed=42)
    official = hydra.utils.get_class("stable_worldmodel.solver.CEMSolver")(**args)
    traced = TraceCEMSolver(**args)
    plan = swm.PlanConfig(**cfg.plan_config)
    for solver in (official, traced):
        solver.configure(action_space=Box(-1.0, 1.0, shape=(3, 2), dtype=np.float32),
                         n_envs=3, config=plan)
    info = {"id": np.arange(3)[:, None], "state": torch.arange(3).float()[:, None, None]}
    records = []
    for warm in (None, torch.zeros(3, 2, 10)):
        a, b = official.solve(info, init_action=warm), traced.solve(info, init_action=warm)
        action_error = float(torch.max(torch.abs(a["actions"] - b["actions"])))
        cost_error = float(np.max(np.abs(np.asarray(a["costs"]) - b["costs"])))
        if action_error != 0.0 or cost_error > 1e-6:
            raise RuntimeError(f"Official/traced CEM mismatch: actions={action_error}, costs={cost_error}")
        records.append(dict(warm_start=warm is not None,
                            max_abs_action_error=action_error, max_abs_cost_error=cost_error))
    return {"stable_worldmodel_version": version, "passed": True, "checks": records}


def replay(env, start, goal, candidates_raw, variations, seed):
    states, images, contacts, hits = [], [], [], []
    for actions in candidates_raw:
        raw = _reset_state(env, start, goal, variations, seed)
        contact, hit = 0, False
        for action in actions:  # Official actions are not clipped to dataset support.
            obs, _, term, _, info = raw.step(action)
            hit |= bool(term)
            contact += int(info.get("n_contacts", 0) > 0)
        states.append(np.asarray(obs["state"], dtype=np.float64))
        images.append(np.asarray(raw.render()))
        contacts.append(contact)
        hits.append(hit)
    return np.stack(states), images, np.asarray(contacts), np.asarray(hits)


def score_metrics(score, components, k):
    out = {}
    factors = {"pusher": components["pusher_cost"], "block": components["block_cost"],
               "theta": components["theta_cost"]}
    targets = {**factors, "object": components["object_task_cost"],
               "physical": components["official_cost"]}
    for name, target in targets.items():
        iqr = float(np.diff(np.percentile(target, [25, 75]))[0])
        out[f"{name}_iqr"] = iqr
        metrics = _selection_metrics(score, target, k)
        # Do not give a constant-outcome population an arbitrary rank or elite overlap.
        informative = iqr > 1e-8
        out[f"{name}_informative"] = informative
        for key, value in metrics.items():
            out[f"{name}_{key}"] = value if informative else float("nan")
    for name, target in factors.items():
        others = np.stack([v for key, v in factors.items() if key != name], axis=1)
        out[f"{name}_partial_rho"] = (_partial_spearman(score, target, others)
                                        if out[f"{name}_informative"] else float("nan"))
        acc, pairs = matched_factor_accuracy(score, target, others)
        out[f"{name}_matched_accuracy"] = acc
        out[f"{name}_matched_pairs"] = pairs
    return out


def factor_head_costs(model, z, zg):
    heads = getattr(model, "factor_heads", None)
    if heads is None:
        return None
    a, b = heads(z), heads(zg[None])
    pusher = ((a["pusher_xy"] - b["pusher_xy"]) * 256 / 20).square().sum(-1)
    block = ((a["block_xy"] - b["block_xy"]) * 256 / 20).square().sum(-1)
    th_a = torch.atan2(a["theta_unit"][:, 0], a["theta_unit"][:, 1])
    th_b = torch.atan2(b["theta_unit"][:, 0], b["theta_unit"][:, 1])
    delta = th_a - th_b
    theta = (torch.atan2(delta.sin(), delta.cos()) / (np.pi / 9)).square()
    return (pusher + block + theta).detach().cpu().numpy()


def save_recording(path, run):
    data = {k: run[k] for k in ("label", "policy", "metrics", "success", "elapsed")}
    data.update(trace=run["solver"].trace, history=run["recorder"].history,
                solve_step_by_env=run["recorder"].solve_step_by_env,
                model_sha256=model_fingerprint(run["model"]))
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def load_recording(path, device):
    # These are this script's own local recordings, never third-party uploads.
    data = torch.load(path, map_location="cpu", weights_only=False)
    data["solver"] = SimpleNamespace(trace=data.pop("trace"))
    data["recorder"] = SimpleNamespace(history=data.pop("history"),
                                      solve_step_by_env=data.pop("solve_step_by_env"))
    data["model"] = load_model(data["policy"], device)
    if model_fingerprint(data["model"]) != data["model_sha256"]:
        raise RuntimeError(f"Checkpoint weights changed since recording: {data['policy']}")
    return data


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    dc = cfg.get("diagnostic", {})
    mode = str(dc.get("mode", "formal"))
    stage = str(dc.get("stage", "all"))
    if stage not in {"all", "replay"}:
        raise ValueError("diagnostic.stage must be all or replay")
    iterations = list(map(int, dc.get("replay_iterations", [0, 3, 9, 19, 29])))
    spec = {k: cfg.solver[k] for k in ("num_samples", "n_steps", "topk", "batch_size", "var_scale")}
    spec.update({k: cfg.world[k] for k in ("history_size", "frame_skip", "env_name")})
    spec.update(OmegaConf.to_container(cfg.plan_config, resolve=True))
    spec.update({k: cfg.eval[k] for k in ("num_eval", "eval_budget", "goal_offset_steps", "img_size", "dataset_name")})
    spec.update(seed=int(cfg.seed), replay_iterations=iterations)
    validate_protocol(spec, mode)
    if str(cfg.solver._target_) != "stable_worldmodel.solver.CEMSolver":
        raise ValueError("Use the frozen solver=cem config")
    out = Path(str(dc.get("output_dir", "outputs/pusht_official_diagnostic"))).resolve()
    if stage == "all" and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new output directory, or stage=replay: {out}")
    out.mkdir(parents=True, exist_ok=True)
    recordings, populations = out / "recordings", out / "populations"
    recordings.mkdir(exist_ok=True)
    populations.mkdir(exist_ok=True)
    device = torch.device(str(cfg.solver.device))
    batch = int(dc.get("model_batch_size", 64))
    if batch <= 0:
        raise ValueError("model_batch_size must be positive")
    policies = {
        "lewm": str(dc.get("lewm_policy", "lewm_epoch_10")),
        "ald_tf": str(dc.get("ald_policy", "pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10")),
    }
    factor_policy = str(dc.get("factor_policy", "")).strip()
    parity = check_cem_parity(cfg)
    write_json(out / "cem_parity.json", parity)
    cfg.world.max_episode_steps = max(2 * int(cfg.eval.eval_budget), int(cfg.eval.goal_offset_steps) + 1)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    _, rows, episodes, starts = _prepare_eval_rows(cfg, dataset)
    start_states, goals = _load_start_goal_states(dataset, episodes, starts, cfg.eval.goal_offset_steps)
    process = _build_process(cfg, dataset)
    transform = img_transform(cfg)
    identity = dict(protocol=spec, policies=policies, dataset_rows=rows.tolist(),
                    episodes=episodes.tolist(), start_steps=starts.tolist())
    if stage == "replay":
        saved = json.loads((out / "run_identity.json").read_text())
        if saved != identity:
            raise ValueError("Replay protocol/policies/starts differ from the recorded run")
        if (out / "summary.json").exists():
            raise FileExistsError("This diagnostic is already complete; use a new output directory")
    else:
        write_json(out / "run_identity.json", identity)
        (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
        write_json(out / "provenance.json", {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            "packages": {x: importlib.metadata.version(x) for x in ("torch", "stable-worldmodel", "stable-pretraining")},
        })
    models, runs = {}, {}
    # Fail early on a missing optional checkpoint, before the expensive benchmark.
    if factor_policy:
        models["factor"] = load_model(factor_policy, device)
    for label, policy in policies.items():
        path = recordings / f"{label}.pt"
        if stage == "replay":
            runs[label] = load_recording(path, device)
        else:
            runs[label] = _run_closed_loop_recording(
                cfg, dataset, process, policy, label, episodes, starts, list(range(len(rows))))
            save_recording(path, runs[label])
        models[label] = runs[label]["model"]
        write_json(out / f"{label}_closed_loop.json", {k: runs[label][k]
                   for k in ("policy", "metrics", "success", "elapsed")})

    labels, counts, paired = paired_outcomes(runs["lewm"]["success"], runs["ald_tf"]["success"])
    critical = [i for i, label in enumerate(labels) if label != "both_success"]
    n_controls = int(dc.get("max_success_controls", 3))
    if n_controls < 0:
        raise ValueError("max_success_controls must be nonnegative")
    controls = _matched_controls(start_states, goals, labels, critical, n_controls)
    if not critical:  # Even a 100/100 pair has informative success controls.
        difficulty = [_physical_cost(s[None], g)[0][0] for s, g in zip(start_states, goals)]
        controls = np.argsort(difficulty)[::-1][:max(n_controls, 0)].tolist()
    selected = sorted(set(critical + controls))
    manifest = [dict(eval_index=i, dataset_row=int(rows[i]), episode_idx=int(episodes[i]),
                     start_step=int(starts[i]), case_type=labels[i],
                     lewm_success=bool(runs["lewm"]["success"][i]),
                     ald_tf_success=bool(runs["ald_tf"]["success"][i]),
                     selected_for_diagnosis=i in selected, is_matched_control=i in controls)
                for i in range(len(rows))]
    write_csv(out / "paired_manifest.csv", manifest)
    write_json(out / "paired_summary.json", dict(**paired, paired_counts=counts,
               selected_eval_indices=selected, mode=mode, official_benchmark=mode == "formal"))
    print(json.dumps(dict(**paired, paired_counts=counts, selected=selected), indent=2), flush=True)

    metric_rows, mean_rows, skipped = [], [], []
    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    t0 = time.time()
    try:
        for env_i in selected:
            goal = goals[env_i]
            for source_label, source in runs.items():
                for solve in (0, 1):
                    tr, li = _find_trace_for_env(source["solver"].trace, env_i, solve)
                    if li is None:
                        skipped.append(dict(eval_index=env_i, source=source_label, solve=solve,
                                            reason="No solve for this environment in recorded execution"))
                        continue
                    info = _slice_info(tr["solver_info"], li)
                    rec = _record_map(source["recorder"], env_i)
                    step = source["recorder"].solve_step_by_env[(solve, env_i)]
                    start = np.asarray(rec[step]["state"], dtype=np.float64)
                    variations = _extract_variations(info)
                    px = torch.as_tensor(info["pixels"])[0, -1:]
                    zgs = {name: _goal_embedding_from_solver(m, info, device) for name, m in models.items()}
                    norm_mean = tr["mean_after"][li, -1][None]
                    raw_mean = _normalized_to_raw(norm_mean, process["action"], int(cfg.plan_config.action_block))
                    mean_state, _, _, _ = replay(env, start, goal, raw_mean, variations, 42)
                    chain = dict(eval_index=env_i, case_type=labels[env_i], source=source_label,
                                 solve=solve, world_step=step,
                                 initial_physical_cost=float(_physical_cost(start[None], goal)[0][0]),
                                 mean_replay_physical_cost=float(_physical_cost(mean_state, goal)[0][0]))
                    # A state vector does not restore every hidden simulator variable.
                    next_step = step + len(raw_mean[0])
                    active_boundary = source["recorder"].solve_step_by_env.get((solve + 1, env_i))
                    if active_boundary == next_step and all(s in rec for s in range(step, next_step + 1)):
                        actual_actions = np.stack([rec[s]["action"] for s in range(step, next_step)])
                        error = float(np.max(np.abs(actual_actions - raw_mean[0])))
                        chain["executed_action_max_abs_error"] = error
                        if error > 1e-4:
                            raise RuntimeError(f"Executed actions differ from CEM mean: {chain}")
                        actual = np.asarray(rec[next_step]["state"])
                        mismatch = _component_costs(mean_state, actual)
                        chain.update(actual_next_physical_cost=float(_physical_cost(actual[None], goal)[0][0]),
                                     mean_replay_vs_actual_pusher_px=float(mismatch["pusher_xy_error_px"][0]),
                                     mean_replay_vs_actual_block_px=float(mismatch["block_xy_error_px"][0]),
                                     mean_replay_vs_actual_theta_deg=float(mismatch["theta_error_deg"][0]),
                                     actual_endpoint_available=True)
                    else:
                        chain["actual_endpoint_available"] = False
                        chain["audit_unavailable_reason"] = "No next active solve boundary; a full plan may not have executed"
                    mean_rows.append(chain)
                    write_csv(out / "mean_execution_audit.csv", mean_rows)
                    for it in iterations:
                        print(f"replay env={env_i} {labels[env_i]} source={source_label} solve={solve} iter={it}", flush=True)
                        candidates = tr["candidates"][li, it]
                        raw = _normalized_to_raw(candidates, process["action"], int(cfg.plan_config.action_block))
                        final, images, contacts, hits = replay(env, start, goal, raw, variations, 42)
                        components = _component_costs(final, goal)
                        meta = dict(eval_index=env_i, case_type=labels[env_i], source=source_label,
                                    solve=solve, cem_iteration=it, num_samples=len(candidates),
                                    contact_fraction=float(np.mean(contacts > 0)),
                                    mean_contact_steps=float(contacts.mean()),
                                    block_motion_mean_px=float(np.linalg.norm(final[:, 2:4] - start[2:4], axis=1).mean()),
                                    population_ever_success_fraction=float(hits.mean()),
                                    population_endpoint_success_fraction=float(_physical_cost(final, goal)[3].mean()))
                        arrays = dict(candidates_normalized=candidates, final_state=final, goal_state=goal,
                                      start_state=start, contact_steps=contacts, ever_success=hits,
                                      final_mean_normalized=norm_mean[0], mean_replay_state=mean_state[0],
                                      native_cost=tr["predicted_costs"][li, it], **components)
                        for model_label, model in models.items():
                            zr = _encode_images(model, transform, images, device, batch)
                            zg = zgs[model_label]
                            zp = _predict_processed_context(model, px, np.empty((0, 2)), candidates,
                                                            process["action"], int(cfg.plan_config.action_block), device, batch)
                            enc = (zr - zg[None]).square().sum(-1).cpu().numpy()
                            pred = (zp - zg[None]).square().sum(-1).cpu().numpy()
                            endpoint_mse = (zp - zr).square().mean(-1).cpu().numpy()
                            encoded_spread = float((zr - zr.mean(0, keepdim=True)).square().mean())
                            if model_label == source_label:
                                native = arrays["native_cost"]
                                arrays["native_recomputed_max_abs"] = np.max(np.abs(native - pred))
                                if not np.allclose(native, pred, rtol=2e-4, atol=2e-4):
                                    raise RuntimeError(f"Native CEM cost audit failed: {meta}, max_abs={arrays['native_recomputed_max_abs']}")
                            common = dict(**meta, model=model_label,
                                          endpoint_mse_mean=float(endpoint_mse.mean()),
                                          endpoint_nmse_population=(float(endpoint_mse.mean()) / encoded_spread
                                                                    if encoded_spread > 1e-12 else float("nan")),
                                          pred_encoder_rho=_spearman(pred, enc))
                            for kind, cost in (("encoder_raw", enc), ("predictor_raw", pred)):
                                metric_rows.append(dict(**common, score=kind, **score_metrics(cost, components, int(cfg.solver.topk))))
                            with torch.inference_mode():
                                head = factor_head_costs(model, zr, zg)
                            if head is not None:
                                metric_rows.append(dict(**common, score="encoder_factor_head_diagnostic_only",
                                                       **score_metrics(head, components, int(cfg.solver.topk))))
                                arrays[f"{model_label}_head_cost"] = head
                            arrays.update({f"{model_label}_encoder_cost": enc,
                                           f"{model_label}_predictor_cost": pred,
                                           f"{model_label}_endpoint_mse": endpoint_mse})
                        np.savez_compressed(populations / f"eval{env_i:03d}_{source_label}_solve{solve}_iter{it:02d}.npz", **arrays)
                        write_csv(out / "population_metrics.csv", metric_rows)
    finally:
        env.close()
    write_csv(out / "population_metrics.csv", metric_rows)
    write_csv(out / "mean_execution_audit.csv", mean_rows)
    write_json(out / "skipped_solves.json", skipped)
    # Aggregate within episode first; 5 CEM iterations are not 5 independent cases.
    case_rows = []
    keys = ("eval_index", "case_type", "source", "solve", "model", "score")
    groups = dict.fromkeys(tuple(row[k] for k in keys) for row in metric_rows)
    numeric = [k for k, v in metric_rows[0].items() if isinstance(v, (int, float, np.number)) and k not in keys] if metric_rows else []
    for group in groups:
        rr = [r for r in metric_rows if tuple(r[k] for k in keys) == group]
        case_rows.append({**dict(zip(keys, group)), "population_count": len(rr),
                          **{k: finite_summary(r[k] for r in rr)["mean"] for k in numeric}})
    write_csv(out / "case_metrics.csv", case_rows)
    summary = dict(status="complete", mode=mode, official_benchmark=mode == "formal", protocol=spec,
                   paired=paired, paired_counts=counts, selected_eval_indices=selected,
                   policies={**policies, **({"factor": factor_policy} if factor_policy else {})},
                   model_sha256={label: model_fingerprint(model) for label, model in models.items()},
                   visual_encoder_sha256={label: model_fingerprint(model, visual_only=True) for label, model in models.items()},
                   factor_policy_role="Optional same-population offline scoring; no Factor closed-loop success measured",
                   num_populations=len(list(populations.glob("*.npz"))), replay_seconds=time.time() - t0,
                   cem_modified=False, real_context_frames=1, oracle_used_for_planning=False,
                   physical_cost="(pusher/20)^2 + (block/20)^2 + (wrapped theta/(pi/9))^2; diagnostic surrogate, not binary success",
                   matched_pair_definition="Match both other factor costs; observational association, not causal isolation",
                   replay_limitation="Candidates use observation-state reset, not a full simulator snapshot. Read mean_execution_audit.csv for mismatch against actual execution.",
                   interpretation="Compare encoder_raw vs physical, predictor_raw vs encoder_raw, endpoint MSE and per-factor metrics within each episode/source/solve. Low-spread factors are uninformative.")
    write_json(out / "summary.json", summary)
    print(f"=== DONE ===\nResults: {out}", flush=True)


if __name__ == "__main__":
    run()
