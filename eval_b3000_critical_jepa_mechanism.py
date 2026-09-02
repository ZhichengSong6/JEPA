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
   Solve 1 uses the ACTUAL preceding closed-loop trajectory reconstructed by
   replaying the exact CEM mean plan returned by solve 0. Solve 0 uses dataset
   pre-history only when it exists; unavailable prefixes are reported, never
   fabricated.

4) Mean-plan causal chain:
   Replay the exact solve-0 CEM mean that was executed for 25 raw steps,
   verify it reaches the recorded solve-1 state, and compare its predicted
   endpoint with the real encoded endpoint.

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
    _case_type,
    _extract_variations,
    _normalized_to_raw,
    _rank_percentile,
    _run_closed_loop,
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

    lewm = _run_closed_loop(
        cfg, dataset, process, lewm_policy, "lewm",
        eval_episodes, eval_start,
    )
    ald = _run_closed_loop(
        cfg, dataset, process, ald_policy, "ald_tf",
        eval_episodes, eval_start,
    )

    current_types = [
        _case_type(bool(lewm["success"][i]), bool(ald["success"][i]))
        for i in range(len(eval_rows))
    ]

    reference_rows = None
    if reference_manifest:
        ref_path = Path(reference_manifest)
        reference_rows = _read_reference_manifest(ref_path)
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

    # Reconstruct the ACTUAL solve-0 executed mean for each source trajectory.
    mean_cache = {}
    mean_rows = []
    mean_candidate_payload = []
    try:
        for env_i in critical:
            goal = np.asarray(goal_states[env_i], dtype=np.float64)
            for source_label, source in runs.items():
                tr0, li0 = _find_trace_for_env(
                    source["solver"].trace, env_i, 0
                )
                if li0 is None:
                    continue
                info0 = _slice_info(tr0["solver_info"], li0)
                variations = _extract_variations(info0)
                norm_mean = np.asarray(
                    tr0["mean_after"][li0, -1], dtype=np.float32
                )[None]
                raw_mean = _normalized_to_raw(
                    norm_mean, process["action"], action_block
                )[0]
                replay = _replay_mean_with_coarse_history(
                    replay_env,
                    np.asarray(tr0["solve_start_states"][li0], dtype=np.float64),
                    goal,
                    raw_mean,
                    variations,
                    seed=3_000_000 + 10_000 * env_i
                         + (0 if source_label == "lewm" else 500),
                    action_block=action_block,
                )
                replay["raw_actions"] = raw_mean
                replay["transform"] = transform
                mean_cache[(source_label, env_i)] = replay

                tr1, li1 = _find_trace_for_env(
                    source["solver"].trace, env_i, 1
                )
                if li1 is not None:
                    recorded_next = np.asarray(
                        tr1["solve_start_states"][li1], dtype=np.float64
                    )
                    pos_err = float(np.max(np.abs(
                        replay["final_state"][:4] - recorded_next[:4]
                    )))
                    theta_err = float(_angle_error_rad(
                        replay["final_state"][4], recorded_next[4]
                    ))
                    if pos_err > 1e-3 or theta_err > 1e-5:
                        raise RuntimeError(
                            "Mean-plan replay does not reproduce solve-1 state: "
                            f"env={env_i} source={source_label} "
                            f"pos_max={pos_err:.6g} theta={theta_err:.6g}"
                        )
                else:
                    recorded_next = None
                    pos_err = float("nan")
                    theta_err = float("nan")

                start_cost = float(_physical_cost(
                    np.asarray(tr0["solve_start_states"][li0])[None], goal
                )[0][0])
                final_cost = float(_physical_cost(
                    replay["final_state"][None], goal
                )[0][0])

                # Exact processed C1 start and exact solver goal.
                current_px = info0["pixels"]
                if not torch.is_tensor(current_px):
                    current_px = torch.as_tensor(current_px)
                current_px = current_px[0, -1:].detach().cpu()

                real_emb = {}
                pred_emb = {}
                goal_emb = {}
                enc_goal_cost = {}
                for model_label, model_run in runs.items():
                    model = model_run["model"]
                    zr = _encode(
                        model, transform, [replay["final_image"]],
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
                    real_emb[model_label] = zr.detach().cpu().numpy()
                    pred_emb[model_label] = zp.detach().cpu().numpy()
                    goal_emb[model_label] = zg.detach().cpu().numpy()
                    enc_goal_cost[model_label] = float(
                        torch.sum((zr - zg) ** 2).cpu()
                    )
                    mean_rows.append({
                        "eval_index": int(env_i),
                        "case_type": current_types[env_i],
                        "source_trajectory": source_label,
                        "scoring_model": model_label,
                        "start_phys_cost": start_cost,
                        "next_solve_phys_cost": final_cost,
                        "physical_progress": start_cost - final_cost,
                        "next_state_replay_pos_max_abs": pos_err,
                        "next_state_replay_theta_abs_rad": theta_err,
                        "pred_endpoint_mse": float(
                            torch.mean((zp - zr) ** 2).cpu()
                        ),
                        "pred_goal_cost": float(
                            torch.sum((zp - zg) ** 2).cpu()
                        ),
                        "enc_goal_cost": enc_goal_cost[model_label],
                    })
                mean_candidate_payload.append({
                    "eval_index": env_i,
                    "source": source_label,
                    "real_emb_lewm": real_emb["lewm"],
                    "real_emb_ald": real_emb["ald_tf"],
                    "pred_emb_lewm": pred_emb["lewm"],
                    "pred_emb_ald": pred_emb["ald_tf"],
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
                        replay = mean_cache.get((source_label, env_i))
                        if replay is None:
                            raise RuntimeError(
                                f"Missing solve-0 history cache for "
                                f"{source_label} env={env_i}"
                            )
                        histories = _build_solve1_history(
                            replay, exact_current,
                            action_block, max(contexts),
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
                "Actual closed-loop prefix reconstructed from the exact solve-0 "
                "CEM mean that the official policy executed."
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
                "next_state replay audit proves the executed CEM mean produces "
                "the recorded next solve state; physical_progress then links "
                "first-solve model choice to the next planning basin."
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
