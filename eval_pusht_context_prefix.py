#!/usr/bin/env python3
"""Planner-prefix diagnostic for full PushT LeWM.

Compare predictor terminal costs for the same future candidates when rollout
starts from C in {1,2,3} observed coarse frames.  C>1 only prepends preceding
expert observations/actions; every context predicts the same future candidates
from the same current state.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing

from eval import get_episodes_length, img_transform
from eval_pusht_horizon_directional import (
    _check_contiguous,
    _current_goal_images,
    _encode,
    _jsonable,
    _label,
    _make_candidates,
    _metrics,
    _physical_cost,
    _rollout,
    _state_factor,
    _transform_batch,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--num-anchors", type=int, default=50)
    p.add_argument("--contexts", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--radius", type=float, default=0.1565)
    p.add_argument("--num-directions", type=int, default=32)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--pair-margin-frac", type=float, default=0.02)
    p.add_argument("--direction-margin-frac", type=float, default=0.02)
    p.add_argument("--replay-factor-good-threshold", type=float, default=0.10)
    p.add_argument("--world-size", type=float, default=512.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _select_rows_with_history(dataset, n, seed, pre_raw, goal_offset):
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col))
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start = episode_len - int(goal_offset) - 1
    by_ep = {ep: max_start[i] for i, ep in enumerate(ep_indices)}
    eps = np.asarray(dataset.get_col_data(col))
    steps = np.asarray(dataset.get_col_data("step_idx"))
    per_row = np.asarray([by_ep[e] for e in eps])
    valid = (steps >= int(pre_raw)) & (steps <= per_row)
    valid_idx = np.nonzero(valid)[0]
    if len(valid_idx) < int(n):
        raise ValueError(f"Need {n} anchors, only {len(valid_idx)} have required history/future.")
    g = np.random.default_rng(seed)
    sel = g.choice(len(valid_idx), size=int(n), replace=False)
    return col, np.sort(valid_idx[sel]).astype(np.int64)


def _render_state(env, state, goal_state, seed):
    try:
        env.reset(
            seed=int(seed),
            options={
                "state": np.asarray(state, dtype=np.float64),
                "goal_state": np.asarray(goal_state, dtype=np.float64),
            },
        )
    except Exception:
        env.reset(seed=int(seed))
        raw = env.unwrapped
        raw._set_goal_state(np.asarray(goal_state, dtype=np.float64))
        raw._set_state(np.asarray(state, dtype=np.float64))
    return np.asarray(env.unwrapped.render())


def _normalize_coarse(raw, scaler, action_block):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[0] % int(action_block) != 0:
        raise ValueError(f"Raw action length {raw.shape[0]} not divisible by block={action_block}.")
    x = scaler.transform(raw.reshape(-1, 2)).astype(np.float32)
    return x.reshape(-1, int(action_block) * 2)


def _normalize_future_candidates(raw_candidates, scaler, horizon, action_block):
    raw = np.asarray(raw_candidates, dtype=np.float32)
    flat = scaler.transform(raw.reshape(-1, 2)).astype(np.float32)
    return flat.reshape(len(raw), int(horizon), int(action_block) * 2)


@torch.inference_mode()
def _predict_with_context(
    model, transform, context_images, past_raw_actions, future_raw_candidates,
    scaler, horizon, action_block, device, batch_size,
):
    ctx = int(len(context_images))
    expected_past = (ctx - 1) * int(action_block)
    if len(past_raw_actions) != expected_past:
        raise ValueError(
            f"Context={ctx} needs {expected_past} past raw actions, got {len(past_raw_actions)}."
        )

    px = _transform_batch(transform, context_images).to(device).float()
    future_norm = _normalize_future_candidates(
        future_raw_candidates, scaler, horizon, action_block
    )
    if ctx > 1:
        past_norm = _normalize_coarse(past_raw_actions, scaler, action_block)
    else:
        past_norm = np.empty((0, int(action_block) * 2), dtype=np.float32)

    out = []
    for st in range(0, len(future_norm), int(batch_size)):
        fut = torch.from_numpy(future_norm[st:st + int(batch_size)]).to(
            device=device, dtype=torch.float32
        )
        s = fut.shape[0]
        if ctx > 1:
            past = torch.from_numpy(past_norm).to(device=device, dtype=torch.float32)
            past = past[None].expand(s, -1, -1)
            full_actions = torch.cat([past, fut], dim=1)
        else:
            full_actions = fut

        info = {"pixels": px[None, None]}
        rolled = model.rollout(info, full_actions.unsqueeze(0))
        out.append(rolled["predicted_emb"][0, :s, -1].detach())
    return torch.cat(out, dim=0)


def _numeric_summary(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


SUMMARY_KEYS = [
    "rho_pred_phys", "rho_enc_phys", "rho_pred_enc",
    "dir_pred_phys", "dir_enc_phys", "dir_delta_rho_pred_phys",
    "dir_delta_rho_enc_phys", "dir_informative_pairs", "dir_total_pairs",
    "dir_informative_fraction", "mean_abs_phys_direction_delta",
    "pair_acc_pred_phys", "pair_acc_enc_phys", "pred_selected_phys_percentile",
    "enc_selected_phys_percentile", "pred_regret_norm", "enc_regret_norm",
    "mean_pred_enc_mse", "replay_endpoint_factor_error",
]


def _aggregate(rows):
    out = {"count": len(rows)}
    for key in SUMMARY_KEYS:
        out[key] = _numeric_summary([r.get(key, np.nan) for r in rows])
    return out


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels length must equal --policies length")
    contexts = sorted(set(map(int, args.contexts)))
    if not contexts or min(contexts) < 1 or max(contexts) > 3:
        raise ValueError("This diagnostic expects contexts chosen from 1,2,3.")
    if args.horizon * args.action_block > args.goal_offset:
        raise ValueError("Future prediction horizon must not exceed goal offset.")

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=["action", "state"], cache_dir=cache_root
    )
    pre_raw = (max(contexts) - 1) * args.action_block
    col, anchors = _select_rows_with_history(
        dataset, args.num_anchors, args.seed, pre_raw, args.goal_offset
    )
    print("Selected paired eval rows:")
    print(anchors)

    episode_idx = np.asarray(dataset.get_col_data(col))
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    action = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    state = np.asarray(dataset.get_col_data("state"), dtype=np.float64)
    finite = action[np.isfinite(action).all(axis=1)]
    scaler = preprocessing.StandardScaler().fit(finite)

    device = torch.device(args.device)
    transform = img_transform(cfg)
    labels = [_label(p, args.labels, i) for i, p in enumerate(args.policies)]
    models = []
    for label, policy in zip(labels, args.policies):
        print(f"Loading [{label}] {policy}")
        m = swm.policy.AutoCostModel(policy).to(device).eval()
        m.requires_grad_(False)
        m.interpolate_pos_encoding = True
        models.append(m)

    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    nA, nD, nM, nCtx = len(anchors), int(args.num_directions), len(models), len(contexts)
    nC = 1 + nD * 2
    raw_h = args.horizon * args.action_block

    future_actions_all = np.empty((nA, nC, raw_h, 2), dtype=np.float32)
    final_states = np.empty((nA, nC, 7), dtype=np.float64)
    physical_cost = np.empty((nA, nC), dtype=np.float64)
    enc_costs = np.empty((nM, nA, nC), dtype=np.float64)
    pred_costs = np.empty((nM, nCtx, nA, nC), dtype=np.float64)
    pred_enc_mse = np.empty((nM, nCtx, nA, nC), dtype=np.float64)

    rows = []
    start = time.time()
    try:
        for ai, row_idx in enumerate(anchors):
            _check_contiguous(
                row_idx - pre_raw, episode_idx, step_idx,
                pre_raw + max(args.goal_offset, raw_h),
            )
            init_state = state[row_idx].copy()
            goal_state = state[row_idx + args.goal_offset].copy()
            controlled_seed = args.env_seed + ai
            _, goal_image = _current_goal_images(
                env, init_state, goal_state, controlled_seed
            )

            expert_future = action[row_idx:row_idx + raw_h].copy()
            rng = np.random.default_rng(args.seed + 1_000_003 * (ai + 1))
            cands, br_meta, _, d_meta, s_meta, _, eqerr = _make_candidates(
                expert_future,
                np.asarray([args.radius], dtype=np.float64),
                np.asarray([args.radius], dtype=np.float64),
                nD, rng,
            )
            future_actions_all[ai] = cands

            terminal_images = []
            for cand in cands:
                fs, fi, _, _ = _rollout(
                    env, init_state, goal_state, cand, controlled_seed
                )
                final_states[ai, len(terminal_images)] = fs
                terminal_images.append(fi)
            pc, _, _, _, _ = _physical_cost(final_states[ai], goal_state)
            physical_cost[ai] = pc

            replay_endpoint = state[row_idx + raw_h].copy()
            replay_endpoint_error = float(np.linalg.norm(
                _state_factor(final_states[ai, 0], args.world_size)
                - _state_factor(replay_endpoint, args.world_size)
            ))
            idx = np.nonzero(
                np.isclose(br_meta, args.radius, atol=1e-6, rtol=0.0)
            )[0]

            history_images = []
            for back in range(max(contexts) - 1, -1, -1):
                hist_row = row_idx - back * args.action_block
                history_images.append(
                    _render_state(env, state[hist_row], goal_state, controlled_seed)
                )

            for mi, (label, policy, model) in enumerate(zip(labels, args.policies, models)):
                zg = _encode(model, transform, [goal_image], device, args.model_batch_size)[0]
                zr = _encode(
                    model, transform, terminal_images, device, args.model_batch_size
                )
                enc = torch.sum((zr - zg[None]) ** 2, dim=-1).cpu().numpy().astype(np.float64)
                enc_costs[mi, ai] = enc

                for ci_ctx, ctx in enumerate(contexts):
                    context_images = history_images[-ctx:]
                    past_start = row_idx - (ctx - 1) * args.action_block
                    past_raw = action[past_start:row_idx].copy()
                    zp = _predict_with_context(
                        model, transform, context_images, past_raw, cands,
                        scaler, args.horizon, args.action_block,
                        device, args.model_batch_size,
                    )
                    pred = torch.sum((zp - zg[None]) ** 2, dim=-1).cpu().numpy().astype(np.float64)
                    dyn = torch.mean((zp - zr) ** 2, dim=-1).cpu().numpy().astype(np.float64)
                    pred_costs[mi, ci_ctx, ai] = pred
                    pred_enc_mse[mi, ci_ctx, ai] = dyn
                    met = _metrics(
                        pc[idx], enc[idx], pred[idx], dyn[idx],
                        d_meta[idx], s_meta[idx], args
                    )
                    rows.append({
                        "model": label, "policy": policy,
                        "context_length": int(ctx),
                        "anchor_index": ai + 1, "dataset_row": int(row_idx),
                        "episode_idx": int(episode_idx[row_idx]),
                        "step_idx": int(step_idx[row_idx]),
                        "horizon_coarse": int(args.horizon),
                        "raw_horizon": int(raw_h), "radius": float(args.radius),
                        "num_directions": int(nD),
                        "replay_endpoint_factor_error": replay_endpoint_error,
                        "equal_norm_error": float(eqerr), **met,
                    })

            elapsed = time.time() - start
            eta = elapsed / (ai + 1) * (nA - ai - 1)
            print(f"anchor {ai+1:3d}/{nA}: row={row_idx} time={elapsed:.1f}s ETA={eta/60:.1f}min")
    finally:
        env.close()

    csv_path = outdir / "context_prefix_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    npz_path = outdir / "candidate_metrics.npz"
    np.savez_compressed(
        npz_path, anchors=anchors, contexts=np.asarray(contexts, dtype=np.int32),
        labels=np.asarray(labels, dtype=object), policies=np.asarray(args.policies, dtype=object),
        future_candidate_actions=future_actions_all, final_states=final_states,
        physical_cost=physical_cost, enc_costs=enc_costs, pred_costs=pred_costs,
        pred_enc_terminal_mse=pred_enc_mse,
    )

    summary = {
        "config": {
            "dataset": dataset_name, "num_anchors": nA, "contexts": contexts,
            "future_horizon_coarse": int(args.horizon),
            "action_block": int(args.action_block),
            "goal_offset_raw": int(args.goal_offset), "radius": float(args.radius),
            "num_directions": nD, "seed": int(args.seed),
            "planner_alignment": (
                "Every context predicts the same future candidates from the same current "
                "state; C>1 only prepends preceding observed frames/actions."
            ),
            "model_cost": (
                "Raw terminal squared Euclidean latent distance; no factor/readout/GT "
                "state in model scoring."
            ),
        },
        "paired_rows": anchors.tolist(),
        "elapsed_seconds": float(time.time() - start), "models": {},
    }
    for label, policy in zip(labels, args.policies):
        by_ctx = {}
        for ctx in contexts:
            rr = [r for r in rows if r["model"] == label and r["context_length"] == ctx]
            by_ctx[str(ctx)] = _aggregate(rr)
        summary["models"][label] = {"policy": policy, "by_context": by_ctx}

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2))

    print("\n===== CONTEXT-PREFIX SUMMARY =====")
    print(f"{'model':<14} {'ctx':>3} {'rhoP':>7} {'rhoPE':>7} {'accP':>7} {'pctP':>7} {'mse':>9}")
    print("-" * 62)
    for label in labels:
        for ctx in contexts:
            s = summary["models"][label]["by_context"][str(ctx)]
            m = lambda k: s[k]["mean"]
            print(
                f"{label:<14} {ctx:>3d} {m('rho_pred_phys'):>7.3f} "
                f"{m('rho_pred_enc'):>7.3f} {m('pair_acc_pred_phys'):>7.3f} "
                f"{m('pred_selected_phys_percentile'):>7.3f} "
                f"{m('mean_pred_enc_mse'):>9.5f}"
            )

    print(f"\nSaved: {csv_path}\nSaved: {npz_path}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
