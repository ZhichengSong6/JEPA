#!/usr/bin/env python3
"""Critical-case JEPA mechanism diagnostic for B=3000 PushT.

Step 2 after paired failure/rescue analysis.

No training. No CEM modification. The official planner is rerun only to recover
the exact closed-loop trajectories and CEM populations. Formal mode requires
the paired outcome partition to match the previous Step-1 manifest exactly.

For every selected critical case, source CEM trajectory, solve, and CEM
iteration, this diagnostic measures on the SAME candidate population:

1) Encoder ceiling:
       C_enc(U) = || E(o_H(U)) - E(o_goal) ||^2
   versus diagnosis-only physical task cost.

2) Endpoint fidelity:
       MSE( z_pred_H(U), E(o_H(U)) )
   for LeWM and ALD+TF.

3) Causal-prefix fidelity:
   Re-score every candidate with C in {1,2,3} real observed coarse frames.
   Solve 1 uses observations/actions recorded directly from the ACTUAL official
   closed-loop run. Solve 0 uses dataset pre-history only when it exists;
   unavailable prefixes are reported, never fabricated.

4) Mean-plan causal chain:
   Compare the official policy's actually returned 25 raw actions directly
   against the solve-0 final CEM mean, then use the directly recorded solve-1
   state/image for physical progress and endpoint-fidelity analysis.

The physical simulator is diagnosis-only and never changes planning.

Outputs:
  closed_loop_audit.json
  critical_manifest.csv
  population_mechanism_metrics.csv
  mean_plan_causal_chain.csv
  candidate_mechanism_metrics.npz
  mechanism_summary.json
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from eval import get_dataset, img_transform
from eval_lowbudget_failure_autopsy import (
    _angle_error_rad,
    _build_process,
    _jsonable,
    _load_start_goal_states,
    _physical_cost,
    _prepare_eval_rows,
    _spearman,
)
from eval_b3000_paired_failure_analysis import (
    CrossTraceCEMSolver,
    _case_type,
    _extract_variations,
    _normalized_to_raw,
    _rank_percentile,
    _slice_info,
    _topk_recall,
)
from eval_pusht_horizon_directional import _encode, _transform_batch


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _numeric_summary(values):
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "count": 0, "mean": None, "median": None,
            "p10": None, "p90": None,
        }
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


class RecordingWorldModelPolicy(swm.policy.WorldModelPolicy):
    """Official 0.0.6 policy plus passive closed-loop observation/action logging.

    Planning behavior is unchanged: get_action delegates to super() exactly.
    We only snapshot raw observations before the call and raw actions returned
    by the official policy for selected environment indices.
    """

    def __init__(self, *args, record_indices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_indices = sorted(set(map(int, record_indices or [])))
        self.history = {i: [] for i in self.record_indices}
        self.solve_step_by_env = {}
        self._world_step = 0

    @staticmethod
    def _latest(v, env_i):
        x = np.asarray(v)
        y = x[int(env_i)]
        # World history_size=1 gives an explicit time dimension.
        if y.ndim >= 2 and y.shape[0] == 1:
            y = y[-1]
        return np.asarray(y).copy()

    def get_action(self, info_dict: dict, **kwargs):
        snaps = {}
        for i in self.record_indices:
            if i >= len(info_dict["state"]):
                continue
            snaps[i] = {
                "world_step": int(self._world_step),
                "state": self._latest(info_dict["state"], i),
                "pixels": self._latest(info_dict["pixels"], i),
            }

        before = len(self.solver.trace)
        action = super().get_action(info_dict, **kwargs)
        after = len(self.solver.trace)

        a = np.asarray(action)
        for i, snap in snaps.items():
            snap["action"] = np.asarray(a[i]).copy()
            self.history[i].append(snap)

        if after > before:
            for tr in self.solver.trace[before:after]:
                solve_idx = int(tr["solve_index"])
                for env_i in np.asarray(
                    tr["global_env_indices"], dtype=np.int64
                ):
                    if int(env_i) in self.history:
                        self.solve_step_by_env[(solve_idx, int(env_i))] = int(
                            self._world_step
                        )

        self._world_step += 1
        return action


def _close_world(world):
    try:
        world.envs.close()
    except Exception:
        try:
            world.close()
        except Exception:
            pass


def _run_closed_loop_recording(
    cfg,
    dataset,
    process,
    policy_name,
    label,
    eval_episodes,
    eval_start,
    record_indices,
):
    device = torch.device(str(cfg.solver.device))
    model = swm.policy.AutoCostModel(str(policy_name)).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    solver = CrossTraceCEMSolver(
        model=model,
        batch_size=int(cfg.solver.batch_size),
        num_samples=int(cfg.solver.num_samples),
        var_scale=float(cfg.solver.var_scale),
        n_steps=int(cfg.solver.n_steps),
        topk=int(cfg.solver.topk),
        device=str(cfg.solver.device),
        seed=int(cfg.seed),
        state_scaler=process.get("state"),
    )
    world = swm.World(**cfg.world, image_shape=(224, 224))
    plan_config = swm.PlanConfig(**cfg.plan_config)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    policy = RecordingWorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
        record_indices=record_indices,
    )
    world.set_policy(policy)

    print(f"===== CLOSED LOOP [{label}] =====")
    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start.tolist(),
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get("callables"), resolve=True
        ),
    )
    elapsed = time.time() - t0
    success = np.asarray(metrics["episode_successes"], dtype=bool)
    print(
        f"[{label}] success={float(metrics['success_rate']):.1f}% "
        f"failures={int((~success).sum())} solver_calls={len(solver.trace)} "
        f"time={elapsed:.1f}s"
    )
    _close_world(world)
    return {
        "label": label,
        "policy": str(policy_name),
        "model": model,
        "solver": solver,
        "recorder": policy,
        "metrics": metrics,
        "success": success,
        "elapsed": elapsed,
    }


def _record_map(recorder, env_i):
    return {
        int(r["world_step"]): r
        for r in recorder.history.get(int(env_i), [])
    }


def _build_recorded_solve1_history(
    recorder,
    env_i,
    solve_idx,
    exact_current_px,
    transform,
    action_block,
    max_context,
):
    key = (int(solve_idx), int(env_i))
    if key not in recorder.solve_step_by_env:
        raise RuntimeError(
            f"Missing recorded solve step for solve={solve_idx}, env={env_i}"
        )
    step = int(recorder.solve_step_by_env[key])
    rec = _record_map(recorder, env_i)
    if step not in rec:
        raise RuntimeError(
            f"Missing current closed-loop record at step={step}, env={env_i}"
        )

    histories = {
        1: {
            "px": exact_current_px[-1:].clone(),
            "past_raw": np.empty((0, 2), dtype=np.float32),
        }
    }
    for ctx in range(2, int(max_context) + 1):
        need = (ctx - 1) * int(action_block)
        frame_steps = [
            step - j * int(action_block)
            for j in range(ctx - 1, 0, -1)
        ]
        if any(s not in rec for s in frame_steps):
            histories[ctx] = None
            continue
        prior_images = [rec[s]["pixels"] for s in frame_steps]
        prior_px = _transform_batch(transform, prior_images)
        px = torch.cat(
            [prior_px, exact_current_px[-1:].cpu()], dim=0
        )
        action_steps = list(range(step - need, step))
        if any(s not in rec for s in action_steps):
            histories[ctx] = None
            continue
        past_raw = np.asarray(
            [rec[s]["action"] for s in action_steps],
            dtype=np.float32,
        )
        histories[ctx] = {"px": px, "past_raw": past_raw}
    return histories


def _read_reference_manifest(path: Path):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "eval_index": int(r["eval_index"]),
                "case_type": r["case_type"],
                "selected_for_cross_eval":
                    str(r.get("selected_for_cross_eval", "")).lower()
                    in {"1", "true", "yes"},
                "is_matched_control":
                    str(r.get("is_matched_control", "")).lower()
                    in {"1", "true", "yes"},
            })
    if not rows:
        raise RuntimeError(f"Empty reference manifest: {path}")
    return rows


def _find_trace_for_env(trace_list, env_i, solve_idx):
    if solve_idx >= len(trace_list):
        return None, None
    tr = trace_list[solve_idx]
    gidx = np.asarray(tr["global_env_indices"], dtype=np.int64)
    loc = np.where(gidx == int(env_i))[0]
    if len(loc) == 0:
        return tr, None
    return tr, int(loc[0])


def _reset_state(env, state, goal, variations, seed):
    names, vals = variations
    options = {}
    if names:
        options["variation"] = names
        options["variation_values"] = vals
    env.reset(seed=int(seed), options=options if options else None)
    raw = env.unwrapped
    raw._set_goal_state(np.asarray(goal, dtype=np.float64))
    raw._set_state(np.asarray(state, dtype=np.float64))
    return raw


def _render_state(env, state, goal, variations, seed):
    raw = _reset_state(env, state, goal, variations, seed)
    return np.asarray(raw.render())


def _replay_candidates_with_images(
    env,
    start_state,
    goal_state,
    raw_candidates,
    variations,
    seed_base,
):
    n = len(raw_candidates)
    phys = np.empty(n, dtype=np.float64)
    ever = np.zeros(n, dtype=bool)
    endpoint = np.zeros(n, dtype=bool)
    final_states = np.empty((n, len(start_state)), dtype=np.float64)
    final_images = []
    for ci, acts in enumerate(raw_candidates):
        raw = _reset_state(
            env, start_state, goal_state, variations, seed_base + ci
        )
        hit = False
        obs = None
        for a in acts:
            obs, _, term, _, _ = raw.step(a)
            hit = hit or bool(term)
        fs = np.asarray(obs["state"], dtype=np.float64)
        final_states[ci] = fs
        final_images.append(np.asarray(raw.render()))
        pc, _, _, suc = _physical_cost(fs[None], goal_state)
        phys[ci] = float(pc[0])
        ever[ci] = hit
        endpoint[ci] = bool(suc[0])
    return phys, ever, endpoint, final_states, final_images


def _replay_mean_with_coarse_history(
    env,
    start_state,
    goal_state,
    raw_actions,
    variations,
    seed,
    action_block,
):
    """Replay exact executed mean and capture states/images every action_block."""
    raw = _reset_state(env, start_state, goal_state, variations, seed)
    coarse_states = [np.asarray(start_state, dtype=np.float64).copy()]
    coarse_images = [np.asarray(raw.render())]
    obs = None
    ever = False
    for t, a in enumerate(np.asarray(raw_actions, dtype=np.float32), start=1):
        obs, _, term, _, _ = raw.step(a)
        ever = ever or bool(term)
        if t % int(action_block) == 0:
            coarse_states.append(
                np.asarray(obs["state"], dtype=np.float64).copy()
            )
            coarse_images.append(np.asarray(raw.render()))
    if obs is None:
        raise RuntimeError("Mean plan replay received no actions.")
    return {
        "states": coarse_states,
        "images": coarse_images,
        "final_state": np.asarray(obs["state"], dtype=np.float64),
        "final_image": np.asarray(raw.render()),
        "ever_success": bool(ever),
    }


def _normalize_past_raw(raw_actions, scaler, action_block):
    raw = np.asarray(raw_actions, dtype=np.float32)
    if len(raw) == 0:
        return np.empty((0, int(action_block) * 2), dtype=np.float32)
    if len(raw) % int(action_block) != 0:
        raise ValueError(
            f"Past raw length {len(raw)} not divisible by action_block={action_block}"
        )
    x = scaler.transform(raw.reshape(-1, 2)).astype(np.float32)
    return x.reshape(-1, int(action_block) * 2)


@torch.inference_mode()
def _goal_embedding_from_solver(model, info_one, device):
    goal = info_one["goal"]
    if not torch.is_tensor(goal):
        goal = torch.as_tensor(goal)
    # solver info before candidate expansion: (B, history, C, H, W)
    goal = goal.to(device=device, dtype=torch.float32)
    out = model.encode({"pixels": goal})
    return out["emb"][0, -1].detach()


@torch.inference_mode()
def _predict_processed_context(
    model,
    context_px,          # (C, channels, H, W), already transformed
    past_raw_actions,    # ((C-1)*action_block, 2)
    future_norm,         # (N, horizon, packed_dim), CEM-normalized
    action_scaler,
    action_block,
    device,
    batch_size,
):
    ctx = int(context_px.shape[0])
    past_norm = _normalize_past_raw(
        past_raw_actions, action_scaler, action_block
    )
    if len(past_norm) != ctx - 1:
        raise RuntimeError(
            f"Context={ctx} requires {ctx-1} coarse past actions, "
            f"got {len(past_norm)}"
        )

    px = context_px.to(device=device, dtype=torch.float32)
    out = []
    for st in range(0, len(future_norm), int(batch_size)):
        fut = torch.as_tensor(
            future_norm[st:st + int(batch_size)],
            device=device, dtype=torch.float32,
        )
        s = fut.shape[0]
        if ctx > 1:
            past = torch.as_tensor(
                past_norm, device=device, dtype=torch.float32
            )[None].expand(s, -1, -1)
            full = torch.cat([past, fut], dim=1)
        else:
            full = fut
        # model.rollout expects pixels (B,S,T,C,H,W)
        info = {"pixels": px[None, None]}
        rolled = model.rollout(info, full.unsqueeze(0))
        out.append(rolled["predicted_emb"][0, :s, -1].detach())
    return torch.cat(out, dim=0)


def _selection_metrics(score, phys, k):
    score = np.asarray(score, dtype=np.float64)
    phys = np.asarray(phys, dtype=np.float64)
    sel = int(np.argmin(score))
    oracle = int(np.argmin(phys))
    return {
        "rho_phys": _spearman(score, phys),
        "top10_recall": _topk_recall(score, phys, int(k)),
        "oracle_best_rank_pct": _rank_percentile(score, oracle),
        "selected_phys_percentile": _rank_percentile(phys, sel),
        "selection_regret": float(phys[sel] - phys[oracle]),
    }


def _build_solve0_history(
    dataset_state,
    dataset_action,
    dataset_episode,
    dataset_step,
    dataset_row,
    max_context,
    env,
    goal_state,
    variations,
    seed,
    transform,
    exact_current_px,
    action_block,
):
    """Dataset pre-history for initial planner state; unavailable prefixes stay None."""
    histories = {1: {
        "px": exact_current_px[-1:].clone(),
        "past_raw": np.empty((0, 2), dtype=np.float32),
    }}
    step = int(dataset_step[dataset_row])
    ep = dataset_episode[dataset_row]

    for ctx in range(2, int(max_context) + 1):
        need = (ctx - 1) * int(action_block)
        if step < need or dataset_row - need < 0:
            histories[ctx] = None
            continue
        ok = True
        for d in range(need + 1):
            j = dataset_row - need + d
            if (
                dataset_episode[j] != ep
                or int(dataset_step[j]) != step - need + d
            ):
                ok = False
                break
        if not ok:
            histories[ctx] = None
            continue

        states = [
            dataset_state[dataset_row - k * int(action_block)]
            for k in range(ctx - 1, 0, -1)
        ]
        raw_images = [
            _render_state(
                env, s, goal_state, variations,
                seed + 100 * ctx + ii,
            )
            for ii, s in enumerate(states)
        ]
        prior_px = _transform_batch(transform, raw_images)
        px = torch.cat([prior_px, exact_current_px[-1:].cpu()], dim=0)
        past_raw = np.asarray(
            dataset_action[dataset_row - need:dataset_row],
            dtype=np.float32,
        )
        histories[ctx] = {"px": px, "past_raw": past_raw}
    return histories


def _build_solve1_history(
    replay_payload,
    exact_current_px,
    action_block,
    max_context,
):
    """Actual planner-generated history ending at solve-1 state."""
    images = replay_payload["images"]
    raw_plan = replay_payload["raw_actions"]
    if len(images) < int(max_context):
        raise RuntimeError("Insufficient coarse replay history for solve 1.")

    histories = {1: {
        "px": exact_current_px[-1:].clone(),
        "past_raw": np.empty((0, 2), dtype=np.float32),
    }}
    transform = replay_payload["transform"]
    for ctx in range(2, int(max_context) + 1):
        # images include t=0,5,...,25. Need the ctx-1 frames before final,
        # while replacing final with exact processed solver current frame.
        prior_raw_images = images[-ctx:-1]
        prior_px = _transform_batch(transform, prior_raw_images)
        px = torch.cat([prior_px, exact_current_px[-1:].cpu()], dim=0)
        need = (ctx - 1) * int(action_block)
        past_raw = np.asarray(raw_plan[-need:], dtype=np.float32)
        histories[ctx] = {"px": px, "past_raw": past_raw}
    return histories


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    mcfg = cfg.get("mechanism", {})
    outdir = Path(str(mcfg.get(
        "output_dir", "outputs/b3000_critical_jepa_mechanism"
    )))
    outdir.mkdir(parents=True, exist_ok=True)

    lewm_policy = str(mcfg.get("lewm_policy", "lewm_epoch_10"))
    ald_policy = str(mcfg.get(
        "ald_policy",
        "pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10",
    ))
    contexts = sorted(set(map(int, mcfg.get("contexts", [1, 2, 3]))))
    replay_iterations = list(map(int, mcfg.get(
        "replay_iterations", [0, 3, 9]
    )))
    model_batch = int(mcfg.get("model_batch_size", 64))
    reference_manifest = str(mcfg.get("reference_manifest", "")).strip()
    manual_indices = list(map(int, mcfg.get("eval_indices", [])))

    if contexts != [1, 2, 3]:
        raise ValueError("Formal mechanism diagnostic expects contexts [1,2,3].")
    for it in replay_iterations:
        if it < 0 or it >= int(cfg.solver.n_steps):
            raise ValueError(f"Invalid replay iteration {it}")

    cfg.world.max_episode_steps = max(
        2 * int(cfg.eval.eval_budget),
        int(cfg.eval.goal_offset_steps) + 1,
    )

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    _, eval_rows, eval_episodes, eval_start = _prepare_eval_rows(cfg, dataset)
    start_states, goal_states = _load_start_goal_states(
        dataset, eval_episodes, eval_start, cfg.eval.goal_offset_steps
    )
    process = _build_process(cfg, dataset)

    print("============================================================")
    print("STEP 2: Critical-case JEPA mechanism diagnostic")
    print(
        f"N={cfg.solver.num_samples} I={cfg.solver.n_steps} "
        f"K={cfg.solver.topk} "
        f"B={int(cfg.solver.num_samples)*int(cfg.solver.n_steps)}"
    )
    print(f"contexts={contexts} CEM iterations={replay_iterations}")
    print("No training. No CEM modification. Oracle is diagnosis-only.")
    print("============================================================")

    pre_reference_rows = None
    if reference_manifest:
        pre_reference_rows = _read_reference_manifest(
            Path(reference_manifest)
        )
        record_indices = [
            r["eval_index"] for r in pre_reference_rows
            if r["case_type"] != "both_success"
        ]
    elif manual_indices:
        record_indices = manual_indices
    else:
        record_indices = list(range(len(eval_rows)))

    lewm = _run_closed_loop_recording(
        cfg, dataset, process, lewm_policy, "lewm",
        eval_episodes, eval_start, record_indices,
    )
    ald = _run_closed_loop_recording(
        cfg, dataset, process, ald_policy, "ald_tf",
        eval_episodes, eval_start, record_indices,
    )

    current_types = [
        _case_type(bool(lewm["success"][i]), bool(ald["success"][i]))
        for i in range(len(eval_rows))
    ]

    reference_rows = pre_reference_rows
    if reference_manifest:
        ref_path = Path(reference_manifest)
        if len(reference_rows) != len(eval_rows):
            raise RuntimeError(
                f"Reference manifest length {len(reference_rows)} != "
                f"current eval length {len(eval_rows)}"
            )
        mismatches = []
        for r in reference_rows:
            i = r["eval_index"]
            if current_types[i] != r["case_type"]:
                mismatches.append(
                    (i, r["case_type"], current_types[i])
                )
        if mismatches:
            raise RuntimeError(
                "Paired outcome partition does not reproduce Step 1: "
                + repr(mismatches[:20])
            )
        critical = [
            r["eval_index"] for r in reference_rows
            if r["case_type"] != "both_success"
        ]
        print(
            "Reference manifest reproduced exactly. "
            f"critical={critical}"
        )
    elif manual_indices:
        critical = manual_indices
        print(f"Using manual smoke indices: {critical}")
    else:
        critical = [
            i for i, t in enumerate(current_types) if t != "both_success"
        ]
        print(f"Using current critical indices: {critical}")

    if not critical:
        raise RuntimeError("No cases selected for mechanism diagnostic.")

    # Full raw dataset arrays are needed only for solve-0 optional pre-history.
    ep_col = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    ds_ep = np.asarray(dataset.get_col_data(ep_col))
    ds_step = np.asarray(dataset.get_col_data("step_idx"))
    ds_state = np.asarray(dataset.get_col_data("state"), dtype=np.float64)
    ds_action = np.asarray(dataset.get_col_data("action"), dtype=np.float32)

    transform = img_transform(cfg)
    device = torch.device(str(cfg.solver.device))
    action_block = int(cfg.plan_config.action_block)
    k = int(cfg.solver.topk)

    critical_manifest = []
    for i in critical:
        critical_manifest.append({
            "eval_index": int(i),
            "dataset_row": int(eval_rows[i]),
            "episode_idx": int(eval_episodes[i]),
            "start_step": int(eval_start[i]),
            "case_type": current_types[i],
            "lewm_success": bool(lewm["success"][i]),
            "ald_tf_success": bool(ald["success"][i]),
        })
    _write_csv(outdir / "critical_manifest.csv", critical_manifest)

    replay_env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    runs = {"lewm": lewm, "ald_tf": ald}

    # Use the ACTUAL recorded closed-loop trajectory. Do not reconstruct
    # solve-1 history by resetting the simulator from a 7D observation state.
    mean_rows = []
    try:
        for env_i in critical:
            goal = np.asarray(goal_states[env_i], dtype=np.float64)
            for source_label, source in runs.items():
                tr0, li0 = _find_trace_for_env(
                    source["solver"].trace, env_i, 0
                )
                tr1, li1 = _find_trace_for_env(
                    source["solver"].trace, env_i, 1
                )
                if li0 is None or li1 is None:
                    continue

                recorder = source["recorder"]
                key0 = (0, int(env_i))
                key1 = (1, int(env_i))
                if (
                    key0 not in recorder.solve_step_by_env
                    or key1 not in recorder.solve_step_by_env
                ):
                    raise RuntimeError(
                        f"Missing recorded solve timing for "
                        f"env={env_i} source={source_label}"
                    )
                step0 = int(recorder.solve_step_by_env[key0])
                step1 = int(recorder.solve_step_by_env[key1])
                rec = _record_map(recorder, env_i)
                if step0 not in rec or step1 not in rec:
                    raise RuntimeError(
                        f"Missing recorded states at solve boundaries for "
                        f"env={env_i} source={source_label}"
                    )

                info0 = _slice_info(tr0["solver_info"], li0)
                norm_mean = np.asarray(
                    tr0["mean_after"][li0, -1], dtype=np.float32
                )[None]
                raw_mean = _normalized_to_raw(
                    norm_mean, process["action"], action_block
                )[0]

                action_steps = list(range(step0, step1))
                if any(s not in rec for s in action_steps):
                    raise RuntimeError(
                        f"Incomplete executed action history for "
                        f"env={env_i} source={source_label}"
                    )
                actual_actions = np.asarray(
                    [rec[s]["action"] for s in action_steps],
                    dtype=np.float32,
                )
                if actual_actions.shape != raw_mean.shape:
                    raise RuntimeError(
                        f"Executed action shape {actual_actions.shape} != "
                        f"CEM mean shape {raw_mean.shape} for "
                        f"env={env_i} source={source_label}"
                    )
                action_max_abs = float(np.max(np.abs(
                    actual_actions - raw_mean
                )))
                if action_max_abs > 1e-4:
                    raise RuntimeError(
                        "Official policy did not execute the recorded final "
                        "CEM mean as expected: "
                        f"env={env_i} source={source_label} "
                        f"max_abs_action={action_max_abs:.6g}"
                    )

                actual_start_state = np.asarray(
                    rec[step0]["state"], dtype=np.float64
                )
                actual_next_state = np.asarray(
                    rec[step1]["state"], dtype=np.float64
                )
                actual_next_image = np.asarray(rec[step1]["pixels"])

                start_cost = float(_physical_cost(
                    actual_start_state[None], goal
                )[0][0])
                final_cost = float(_physical_cost(
                    actual_next_state[None], goal
                )[0][0])

                current_px = info0["pixels"]
                if not torch.is_tensor(current_px):
                    current_px = torch.as_tensor(current_px)
                current_px = current_px[0, -1:].detach().cpu()

                for model_label, model_run in runs.items():
                    model = model_run["model"]
                    zr = _encode(
                        model, transform, [actual_next_image],
                        device, model_batch,
                    )[0]
                    zg = _goal_embedding_from_solver(
                        model, info0, device
                    )
                    zp = _predict_processed_context(
                        model,
                        current_px,
                        np.empty((0, 2), dtype=np.float32),
                        norm_mean,
                        process["action"],
                        action_block,
                        device,
                        model_batch,
                    )[0]
                    mean_rows.append({
                        "eval_index": int(env_i),
                        "case_type": current_types[env_i],
                        "source_trajectory": source_label,
                        "scoring_model": model_label,
                        "solve0_world_step": step0,
                        "solve1_world_step": step1,
                        "executed_action_count": int(len(actual_actions)),
                        "executed_vs_cem_mean_max_abs": action_max_abs,
                        "start_phys_cost": start_cost,
                        "next_solve_phys_cost": final_cost,
                        "physical_progress": start_cost - final_cost,
                        "pred_endpoint_mse": float(
                            torch.mean((zp - zr) ** 2).cpu()
                        ),
                        "pred_goal_cost": float(
                            torch.sum((zp - zg) ** 2).cpu()
                        ),
                        "enc_goal_cost": float(
                            torch.sum((zr - zg) ** 2).cpu()
                        ),
                    })

        _write_csv(outdir / "mean_plan_causal_chain.csv", mean_rows)

        pop_rows = []
        cand_meta = []
        cand_phys_all = []
        cand_enc_lewm_all = []
        cand_enc_ald_all = []
        cand_pred = {
            ("lewm", c): [] for c in contexts
        }
        cand_pred.update({
            ("ald_tf", c): [] for c in contexts
        })
        cand_mse = {
            ("lewm", c): [] for c in contexts
        }
        cand_mse.update({
            ("ald_tf", c): [] for c in contexts
        })

        t0 = time.time()
        for case_pos, env_i in enumerate(critical):
            print(
                f"mechanism case {case_pos+1}/{len(critical)} "
                f"env={env_i} type={current_types[env_i]}"
            )
            goal = np.asarray(goal_states[env_i], dtype=np.float64)

            for source_label, source in runs.items():
                for solve_idx in [0, 1]:
                    tr, li = _find_trace_for_env(
                        source["solver"].trace, env_i, solve_idx
                    )
                    if tr is None or li is None:
                        continue

                    info_one = _slice_info(tr["solver_info"], li)
                    variations = _extract_variations(info_one)
                    start_state = np.asarray(
                        tr["solve_start_states"][li], dtype=np.float64
                    )

                    exact_current = info_one["pixels"]
                    if not torch.is_tensor(exact_current):
                        exact_current = torch.as_tensor(exact_current)
                    exact_current = exact_current[0, -1:].detach().cpu()

                    if solve_idx == 0:
                        histories = _build_solve0_history(
                            ds_state, ds_action, ds_ep, ds_step,
                            int(eval_rows[env_i]), max(contexts),
                            replay_env, goal, variations,
                            seed=4_000_000 + env_i,
                            transform=transform,
                            exact_current_px=exact_current,
                            action_block=action_block,
                        )
                    else:
                        histories = _build_recorded_solve1_history(
                            source["recorder"],
                            env_i,
                            solve_idx,
                            exact_current,
                            transform,
                            action_block,
                            max(contexts),
                        )

                    goal_z = {
                        ml: _goal_embedding_from_solver(
                            mr["model"], info_one, device
                        )
                        for ml, mr in runs.items()
                    }

                    for it in replay_iterations:
                        candidates = np.asarray(
                            tr["candidates"][li, it], dtype=np.float32
                        )
                        native_source_cost = np.asarray(
                            tr["predicted_costs"][li, it], dtype=np.float64
                        )
                        raw_candidates = _normalized_to_raw(
                            candidates, process["action"], action_block
                        )
                        phys, ever, endpoint, _, terminal_images = (
                            _replay_candidates_with_images(
                                replay_env,
                                start_state,
                                goal,
                                raw_candidates,
                                variations,
                                seed_base=(
                                    5_000_000
                                    + 100_000 * env_i
                                    + 10_000 * solve_idx
                                    + 1_000 * it
                                    + (0 if source_label == "lewm" else 500)
                                ),
                            )
                        )

                        enc_cost = {}
                        real_z = {}
                        for model_label, model_run in runs.items():
                            zr = _encode(
                                model_run["model"], transform,
                                terminal_images, device, model_batch,
                            )
                            real_z[model_label] = zr
                            zg = goal_z[model_label]
                            enc_cost[model_label] = torch.sum(
                                (zr - zg[None]) ** 2, dim=-1
                            ).cpu().numpy().astype(np.float64)

                        # Frozen encoder should be identical; keep this as an audit.
                        enc_diff = float(np.max(np.abs(
                            enc_cost["lewm"] - enc_cost["ald_tf"]
                        )))

                        base_meta = {
                            "eval_index": int(env_i),
                            "case_type": current_types[env_i],
                            "source_trajectory": source_label,
                            "solve_index": int(solve_idx),
                            "cem_iteration": int(it),
                            "num_samples": len(candidates),
                            "encoder_cost_max_abs_lewm_vs_ald": enc_diff,
                            "oracle_success_available": bool(np.any(ever)),
                            "oracle_success_fraction": float(np.mean(ever)),
                        }

                        for model_label, model_run in runs.items():
                            em = _selection_metrics(
                                enc_cost[model_label], phys, k
                            )
                            # One row per model/context. Encoder metrics are
                            # repeated intentionally for easy grouped analysis.
                            for ctx in contexts:
                                hist = histories.get(ctx)
                                if hist is None:
                                    pop_rows.append({
                                        **base_meta,
                                        "scoring_model": model_label,
                                        "context_length": int(ctx),
                                        "context_available": False,
                                        "rho_enc_phys": em["rho_phys"],
                                        "top10_recall_enc": em["top10_recall"],
                                        "oracle_best_rank_pct_enc":
                                            em["oracle_best_rank_pct"],
                                        "selected_phys_percentile_enc":
                                            em["selected_phys_percentile"],
                                        "selection_regret_enc":
                                            em["selection_regret"],
                                        "rho_pred_phys": float("nan"),
                                        "rho_pred_enc": float("nan"),
                                        "top10_recall_pred": float("nan"),
                                        "oracle_best_rank_pct_pred": float("nan"),
                                        "selected_phys_percentile_pred": float("nan"),
                                        "selection_regret_pred": float("nan"),
                                        "mean_pred_enc_mse": float("nan"),
                                        "median_pred_enc_mse": float("nan"),
                                        "native_c1_max_abs": float("nan"),
                                        "native_c1_rho": float("nan"),
                                    })
                                    continue

                                zp = _predict_processed_context(
                                    model_run["model"],
                                    hist["px"],
                                    hist["past_raw"],
                                    candidates,
                                    process["action"],
                                    action_block,
                                    device,
                                    model_batch,
                                )
                                zg = goal_z[model_label]
                                pred_cost = torch.sum(
                                    (zp - zg[None]) ** 2, dim=-1
                                ).cpu().numpy().astype(np.float64)
                                dyn = torch.mean(
                                    (zp - real_z[model_label]) ** 2, dim=-1
                                ).cpu().numpy().astype(np.float64)
                                pm = _selection_metrics(
                                    pred_cost, phys, k
                                )

                                native_c1_max_abs = float("nan")
                                native_c1_rho = float("nan")
                                if (
                                    ctx == 1
                                    and model_label == source_label
                                ):
                                    native_c1_max_abs = float(np.max(
                                        np.abs(
                                            pred_cost
                                            - native_source_cost
                                        )
                                    ))
                                    scale = float(max(
                                        1.0,
                                        np.max(np.abs(native_source_cost)),
                                    ))
                                    native_c1_rho = _spearman(
                                        pred_cost, native_source_cost
                                    )
                                    if (
                                        native_c1_max_abs > 2e-4 * scale
                                        or (
                                            np.isfinite(native_c1_rho)
                                            and native_c1_rho < 0.999999
                                        )
                                    ):
                                        raise RuntimeError(
                                            "C1 mechanism scoring does not "
                                            "reproduce native CEM cost: "
                                            f"env={env_i} "
                                            f"source={source_label} "
                                            f"solve={solve_idx} iter={it} "
                                            f"max_abs={native_c1_max_abs:.6g} "
                                            f"rho={native_c1_rho}"
                                        )

                                pop_rows.append({
                                    **base_meta,
                                    "scoring_model": model_label,
                                    "context_length": int(ctx),
                                    "context_available": True,
                                    "rho_enc_phys": em["rho_phys"],
                                    "top10_recall_enc": em["top10_recall"],
                                    "oracle_best_rank_pct_enc":
                                        em["oracle_best_rank_pct"],
                                    "selected_phys_percentile_enc":
                                        em["selected_phys_percentile"],
                                    "selection_regret_enc":
                                        em["selection_regret"],
                                    "rho_pred_phys": pm["rho_phys"],
                                    "rho_pred_enc":
                                        _spearman(
                                            pred_cost,
                                            enc_cost[model_label]
                                        ),
                                    "top10_recall_pred": pm["top10_recall"],
                                    "oracle_best_rank_pct_pred":
                                        pm["oracle_best_rank_pct"],
                                    "selected_phys_percentile_pred":
                                        pm["selected_phys_percentile"],
                                    "selection_regret_pred":
                                        pm["selection_regret"],
                                    "mean_pred_enc_mse": float(np.mean(dyn)),
                                    "median_pred_enc_mse":
                                        float(np.median(dyn)),
                                    "native_c1_max_abs":
                                        native_c1_max_abs,
                                    "native_c1_rho":
                                        native_c1_rho,
                                })

                                cand_pred[(model_label, ctx)].append(
                                    pred_cost.astype(np.float32)
                                )
                                cand_mse[(model_label, ctx)].append(
                                    dyn.astype(np.float32)
                                )

                        cand_meta.append((
                            env_i, current_types[env_i], source_label,
                            solve_idx, it,
                        ))
                        cand_phys_all.append(phys.astype(np.float32))
                        cand_enc_lewm_all.append(
                            enc_cost["lewm"].astype(np.float32)
                        )
                        cand_enc_ald_all.append(
                            enc_cost["ald_tf"].astype(np.float32)
                        )

                        # Keep NPZ arrays rectangular: unavailable solve-0
                        # prefixes are filled with NaN arrays.
                        for model_label in ["lewm", "ald_tf"]:
                            for ctx in contexts:
                                if len(cand_pred[(model_label, ctx)]) < len(cand_meta):
                                    cand_pred[(model_label, ctx)].append(
                                        np.full(len(candidates), np.nan, dtype=np.float32)
                                    )
                                    cand_mse[(model_label, ctx)].append(
                                        np.full(len(candidates), np.nan, dtype=np.float32)
                                    )

        _write_csv(
            outdir / "population_mechanism_metrics.csv", pop_rows
        )

        # Candidate-level compact arrays for follow-up analysis without rerun.
        np.savez_compressed(
            outdir / "candidate_mechanism_metrics.npz",
            eval_index=np.asarray([x[0] for x in cand_meta], dtype=np.int32),
            case_type=np.asarray([x[1] for x in cand_meta]),
            source_trajectory=np.asarray([x[2] for x in cand_meta]),
            solve_index=np.asarray([x[3] for x in cand_meta], dtype=np.int32),
            cem_iteration=np.asarray([x[4] for x in cand_meta], dtype=np.int32),
            physical_cost=np.stack(cand_phys_all),
            enc_cost_lewm=np.stack(cand_enc_lewm_all),
            enc_cost_ald_tf=np.stack(cand_enc_ald_all),
            **{
                f"pred_cost_{ml}_c{ctx}": np.stack(
                    cand_pred[(ml, ctx)]
                )
                for ml in ["lewm", "ald_tf"] for ctx in contexts
            },
            **{
                f"pred_enc_mse_{ml}_c{ctx}": np.stack(
                    cand_mse[(ml, ctx)]
                )
                for ml in ["lewm", "ald_tf"] for ctx in contexts
            },
        )

    finally:
        replay_env.close()

    # Group summaries focused on exact critical mechanisms.
    metric_keys = [
        "rho_enc_phys",
        "rho_pred_phys",
        "rho_pred_enc",
        "top10_recall_enc",
        "top10_recall_pred",
        "oracle_best_rank_pct_enc",
        "oracle_best_rank_pct_pred",
        "selected_phys_percentile_enc",
        "selected_phys_percentile_pred",
        "selection_regret_enc",
        "selection_regret_pred",
        "mean_pred_enc_mse",
    ]
    grouped = {}
    case_groups = sorted(set(current_types[i] for i in critical))
    for cg in case_groups:
        grouped[cg] = {}
        for source_label in ["lewm", "ald_tf"]:
            grouped[cg][source_label] = {}
            for solve_idx in [0, 1]:
                grouped[cg][source_label][str(solve_idx)] = {}
                for ctx in contexts:
                    rr = [
                        r for r in pop_rows
                        if r["case_type"] == cg
                        and r["source_trajectory"] == source_label
                        and r["solve_index"] == solve_idx
                        and r["context_length"] == ctx
                        and r["context_available"]
                    ]
                    by_model = {}
                    for ml in ["lewm", "ald_tf"]:
                        rm = [r for r in rr if r["scoring_model"] == ml]
                        by_model[ml] = {
                            "count": len(rm),
                            **{
                                k: _numeric_summary(x[k] for x in rm)
                                for k in metric_keys
                            },
                        }
                    grouped[cg][source_label][str(solve_idx)][str(ctx)] = (
                        by_model
                    )

    mean_grouped = {}
    for cg in case_groups:
        mean_grouped[cg] = {}
        for source_label in ["lewm", "ald_tf"]:
            mean_grouped[cg][source_label] = {}
            for ml in ["lewm", "ald_tf"]:
                rr = [
                    r for r in mean_rows
                    if r["case_type"] == cg
                    and r["source_trajectory"] == source_label
                    and r["scoring_model"] == ml
                ]
                mean_grouped[cg][source_label][ml] = {
                    "count": len(rr),
                    "physical_progress": _numeric_summary(
                        r["physical_progress"] for r in rr
                    ),
                    "next_solve_phys_cost": _numeric_summary(
                        r["next_solve_phys_cost"] for r in rr
                    ),
                    "pred_endpoint_mse": _numeric_summary(
                        r["pred_endpoint_mse"] for r in rr
                    ),
                    "pred_goal_cost": _numeric_summary(
                        r["pred_goal_cost"] for r in rr
                    ),
                    "enc_goal_cost": _numeric_summary(
                        r["enc_goal_cost"] for r in rr
                    ),
                }

    closed_audit = {
        "lewm_success_rate": float(lewm["metrics"]["success_rate"]),
        "ald_tf_success_rate": float(ald["metrics"]["success_rate"]),
        "paired_counts": {
            t: int(sum(x == t for x in current_types))
            for t in [
                "both_success", "lewm_fail_ald_success",
                "both_fail", "lewm_success_ald_fail",
            ]
        },
        "critical_eval_indices": critical,
        "reference_manifest": reference_manifest or None,
        "reference_partition_verified": bool(reference_rows is not None),
    }
    (outdir / "closed_loop_audit.json").write_text(
        json.dumps(_jsonable(closed_audit), indent=2)
    )

    summary = {
        "scientific_question": (
            "On the exact B=3000 planner-critical cases, are LeWM/ALD+TF "
            "differences explained by encoder geometry, absolute endpoint "
            "fidelity, or causal-prefix-dependent predictor distortion?"
        ),
        "config": {
            "num_samples": int(cfg.solver.num_samples),
            "cem_iterations": int(cfg.solver.n_steps),
            "topk": int(cfg.solver.topk),
            "budget_B": int(cfg.solver.num_samples)
                        * int(cfg.solver.n_steps),
            "num_eval": int(cfg.eval.num_eval),
            "contexts": contexts,
            "replay_iterations": replay_iterations,
            "action_block": action_block,
            "lewm_policy": lewm_policy,
            "ald_tf_policy": ald_policy,
            "cem_modified": False,
            "physical_oracle_used_for_planning": False,
            "solve0_prefix": (
                "Dataset pre-history only when episode start_step provides it; "
                "missing C2/C3 contexts are left unavailable."
            ),
            "solve1_prefix": (
                "Actual closed-loop observations and raw actions recorded "
                "directly during the official policy run; no simulator-reset "
                "reconstruction is used."
            ),
        },
        "closed_loop_audit": closed_audit,
        "population_groups": grouped,
        "mean_plan_groups": mean_grouped,
        "num_population_rows": len(pop_rows),
        "num_candidate_populations": len(cand_meta),
        "elapsed_mechanism_seconds": float(time.time() - t0),
        "interpretation_rules": {
            "encoder_ceiling": (
                "High rho_enc_phys / top10_recall_enc means real encoded "
                "future observations preserve task geometry; a gap from "
                "predictor metrics is dynamics error."
            ),
            "endpoint_fidelity": (
                "Lower mean_pred_enc_mse means predicted terminal latent is "
                "closer to encoding the physically replayed terminal image."
            ),
            "prefix_mechanism": (
                "If C1 is poor but C2/C3 recover on solve-1 actual closed-loop "
                "history, short-prefix predictor drift is directly implicated."
            ),
            "mean_plan_chain": (
                "The official policy's actually returned raw actions are "
                "compared directly with the solve-0 final CEM mean. The next "
                "solve state/image are taken directly from the same recorded "
                "closed-loop run, so no incomplete-state simulator replay is "
                "used for the causal-chain evidence."
            ),
        },
    }
    (outdir / "mechanism_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2)
    )

    print("===== MECHANISM SUMMARY =====")
    print(json.dumps(_jsonable({
        "closed_loop_audit": closed_audit,
        "critical_cases": critical,
        "num_population_rows": len(pop_rows),
        "num_candidate_populations": len(cand_meta),
        "elapsed_mechanism_seconds": summary["elapsed_mechanism_seconds"],
    }), indent=2))
    print(f"Saved: {outdir / 'closed_loop_audit.json'}")
    print(f"Saved: {outdir / 'critical_manifest.csv'}")
    print(f"Saved: {outdir / 'population_mechanism_metrics.csv'}")
    print(f"Saved: {outdir / 'mean_plan_causal_chain.csv'}")
    print(f"Saved: {outdir / 'candidate_mechanism_metrics.npz'}")
    print(f"Saved: {outdir / 'mechanism_summary.json'}")
    print("=== DONE ===")


if __name__ == "__main__":
    run()
