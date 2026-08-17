#!/usr/bin/env python3
"""
Action-conditioned candidate-ranking diagnostic for official/full PushT LeWM.

For each paired evaluation anchor, this script generates the SAME finite-horizon
raw-action candidate pool for every checkpoint and compares three quantities:

  C_phys(U): real simulator terminal task cost,
  C_enc(U):  raw Euclidean latent cost after encoding the REAL terminal image,
  C_pred(U): raw Euclidean latent cost after LeWM predictor rollout.

This separates two failure modes that are otherwise confounded by planning:

  Physical <-> Encoder : representation / goal-geometry ranking,
  Encoder  <-> Predictor: learned dynamics ranking,
  Physical <-> Predictor: what the unchanged CEM latent objective actually sees.

No factor head is used. Physical state is used only as an oracle diagnostic.
The model-side score is exactly raw terminal squared Euclidean latent distance.

Candidate pool (default K=128):
  * expert future sequence from the dataset,
  * zero-action sequence,
  * Gaussian perturbations around expert at several raw-action noise scales,
  * a small fraction of uniform random sequences.

The default horizon is the official planning horizon: 5 coarse LeWM actions,
each containing 5 raw PushT actions, i.e. 25 raw actions total.

Primary outputs:
  anchor_metrics.csv    one row per anchor x model,
  candidate_metrics.npz all candidate-level physical/model costs and metadata,
  summary.json          grouped aggregate metrics and experiment metadata.
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Full-PushT Physical/Encoder/Predictor action-ranking diagnostic."
    )
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--num-anchors", type=int, default=50)
    p.add_argument("--num-candidates", type=int, default=128)
    p.add_argument(
        "--noise-scales", nargs="+", type=float,
        default=[0.05, 0.10, 0.20, 0.40],
    )
    p.add_argument("--uniform-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--pair-margin-frac", type=float, default=0.02)
    p.add_argument("--replay-factor-good-threshold", type=float, default=0.10)
    p.add_argument("--object-motion-px", type=float, default=1.0)
    p.add_argument("--object-motion-deg", type=float, default=1.0)
    p.add_argument("--world-size", type=float, default=512.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-dir", default=None,
        help="Default: $STABLEWM_HOME/pusht_action_ranking",
    )
    return p.parse_args()


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _label(policy, labels, i):
    if labels is not None:
        return labels[i]
    return Path(policy).name.replace("_epoch_10", "").replace("lewm_", "")


def _angle_error_rad(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def _state_factor(state, world_size=512.0):
    s = np.asarray(state, dtype=np.float64)
    pusher = 2.0 * s[..., 0:2] / float(world_size) - 1.0
    block = 2.0 * s[..., 2:4] / float(world_size) - 1.0
    theta = s[..., 4]
    theta_unit = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
    return np.concatenate([pusher, block, theta_unit], axis=-1)


def _physical_components(states, goal_state):
    s = np.asarray(states, dtype=np.float64)
    g = np.asarray(goal_state, dtype=np.float64)
    pusher = np.linalg.norm(s[..., 0:2] - g[0:2], axis=-1)
    block = np.linalg.norm(s[..., 2:4] - g[2:4], axis=-1)
    joint = np.linalg.norm(s[..., 0:4] - g[0:4], axis=-1)
    theta = _angle_error_rad(s[..., 4], g[4])
    return pusher, block, joint, theta


def _primary_physical_cost(states, goal_state):
    pusher, block, joint, theta = _physical_components(states, goal_state)
    cost = (joint / 20.0) ** 2 + (theta / (np.pi / 9.0)) ** 2
    return cost, pusher, block, joint, theta


def _factor_physical_cost(states, goal_state, world_size):
    f = _state_factor(states, world_size)
    fg = _state_factor(goal_state, world_size)
    return np.sum((f - fg) ** 2, axis=-1)


def _rankdata_average(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    ra = _rankdata_average(a[mask])
    rb = _rankdata_average(b[mask])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(ra, rb) / den)


def _pairwise_accuracy(reference, estimate, margin_frac=0.02):
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    q25, q75 = np.percentile(ref[np.isfinite(ref)], [25, 75])
    margin = float(margin_frac) * max(float(q75 - q25), 1e-12)
    correct = 0.0
    count = 0
    n = len(ref)
    for i in range(n):
        if not np.isfinite(ref[i]) or not np.isfinite(est[i]):
            continue
        for j in range(i + 1, n):
            if not np.isfinite(ref[j]) or not np.isfinite(est[j]):
                continue
            dr = ref[i] - ref[j]
            if abs(dr) <= margin:
                continue
            de = est[i] - est[j]
            count += 1
            if abs(de) <= 1e-12:
                correct += 0.5
            elif np.sign(dr) == np.sign(de):
                correct += 1.0
    return (float(correct / count) if count > 0 else float("nan"), int(count))


def _topk_recall(reference, estimate, k):
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    k = int(min(max(1, k), len(ref)))
    ref_top = set(np.argsort(ref)[:k].tolist())
    est_top = set(np.argsort(est)[:k].tolist())
    return float(len(ref_top & est_top) / k)


def _selected_physical_percentile(physical_cost, selected_idx):
    ranks = _rankdata_average(np.asarray(physical_cost, dtype=np.float64)) - 1.0
    if len(ranks) <= 1:
        return 0.0
    return float(ranks[int(selected_idx)] / (len(ranks) - 1))


def _selection_stats(physical_cost, score_cost):
    p = np.asarray(physical_cost, dtype=np.float64)
    s = np.asarray(score_cost, dtype=np.float64)
    oracle_idx = int(np.nanargmin(p))
    selected_idx = int(np.nanargmin(s))
    oracle = float(p[oracle_idx])
    selected = float(p[selected_idx])
    regret = selected - oracle
    p90 = float(np.nanpercentile(p, 90))
    norm_regret = regret / max(p90 - oracle, 1e-12)
    return {
        "oracle_idx": oracle_idx,
        "selected_idx": selected_idx,
        "oracle_physical_cost": oracle,
        "selected_physical_cost": selected,
        "selection_regret": float(regret),
        "selection_regret_norm": float(norm_regret),
        "selected_physical_percentile": _selected_physical_percentile(p, selected_idx),
    }


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


def _select_official_rows(dataset, num_anchors, seed, goal_offset):
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - int(goal_offset) - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")
    if num_anchors > len(valid_indices) - 1:
        raise ValueError("Requested more anchors than official valid starting points.")
    g = np.random.default_rng(seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=int(num_anchors), replace=False
    )
    rows = np.sort(valid_indices[random_episode_indices]).astype(np.int64)
    return col, rows


def _check_anchor_contiguity(row, episode_idx, step_idx, horizon_raw):
    ep = episode_idx[row]
    t = int(step_idx[row])
    for d in range(horizon_raw + 1):
        j = row + d
        if j >= len(step_idx) or episode_idx[j] != ep or int(step_idx[j]) != t + d:
            raise RuntimeError(
                f"Dataset is not contiguous at row={row}, offset={d}; "
                "cannot form the exact raw-action horizon."
            )


def _make_candidates(expert_actions, num_candidates, noise_scales, uniform_frac, rng):
    expert = np.nan_to_num(
        np.asarray(expert_actions, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0
    )
    expert = np.clip(expert, -1.0, 1.0)
    k = int(num_candidates)
    if k < 4:
        raise ValueError("--num-candidates must be at least 4.")
    if not noise_scales or any(s < 0 for s in noise_scales):
        raise ValueError("--noise-scales must contain non-negative values.")
    if not (0.0 <= uniform_frac < 1.0):
        raise ValueError("--uniform-frac must be in [0,1).")
    candidates = np.empty((k, *expert.shape), dtype=np.float32)
    type_code = np.empty(k, dtype=np.int8)
    noise_scale = np.full(k, np.nan, dtype=np.float32)
    candidates[0] = expert
    type_code[0] = 0
    candidates[1] = 0.0
    type_code[1] = 1
    remaining = k - 2
    n_uniform = min(remaining, int(round(remaining * float(uniform_frac))))
    n_gaussian = remaining - n_uniform
    scales = list(map(float, noise_scales))
    for j in range(n_gaussian):
        idx = 2 + j
        sigma = scales[j % len(scales)]
        noise = rng.normal(0.0, sigma, size=expert.shape).astype(np.float32)
        candidates[idx] = np.clip(expert + noise, -1.0, 1.0)
        type_code[idx] = 2
        noise_scale[idx] = sigma
    for j in range(n_uniform):
        idx = 2 + n_gaussian + j
        candidates[idx] = rng.uniform(-1.0, 1.0, size=expert.shape).astype(np.float32)
        type_code[idx] = 3
    return candidates, type_code, noise_scale


def _reset_anchor(env, init_state, goal_state, seed):
    try:
        obs, info = env.reset(
            seed=int(seed),
            options={
                "state": np.asarray(init_state, dtype=np.float64),
                "goal_state": np.asarray(goal_state, dtype=np.float64),
            },
        )
        return obs, info
    except Exception:
        obs, info = env.reset(seed=int(seed))
        raw = env.unwrapped
        raw._set_goal_state(np.asarray(goal_state, dtype=np.float64))
        raw._set_state(np.asarray(init_state, dtype=np.float64))
        try:
            raw._set_state(np.asarray(goal_state, dtype=np.float64))
            raw._goal = raw.render()
            raw._set_state(np.asarray(init_state, dtype=np.float64))
        except Exception:
            pass
        obs = {"state": np.asarray(init_state, dtype=np.float64)}
        info = raw._get_info()
        return obs, info


def _controlled_current_goal_images(env, init_state, goal_state, seed):
    _, info = _reset_anchor(env, init_state, goal_state, seed)
    raw = env.unwrapped
    current = np.asarray(raw.render())
    goal = info.get("goal", None)
    if goal is None:
        raw._set_state(np.asarray(goal_state, dtype=np.float64))
        goal = np.asarray(raw.render())
        raw._set_state(np.asarray(init_state, dtype=np.float64))
    return current, np.asarray(goal)


def _rollout_candidate(env, init_state, goal_state, raw_actions, seed):
    _reset_anchor(env, init_state, goal_state, seed)
    raw = env.unwrapped
    had_contact = False
    contact_steps = 0
    obs = None
    for a in np.asarray(raw_actions, dtype=np.float32):
        obs, _, _, _, info = raw.step(a)
        n_contacts = int(info.get("n_contacts", 0))
        had_contact = had_contact or n_contacts > 0
        contact_steps += int(n_contacts > 0)
    final_state = np.asarray(obs["state"], dtype=np.float64)
    final_image = np.asarray(raw.render())
    return final_state, final_image, had_contact, contact_steps


def _transform_batch(transform, images):
    xs = []
    for im in images:
        x = transform(im)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        xs.append(x)
    return torch.stack(xs, dim=0)


@torch.inference_mode()
def _encode_images(model, transform, images, device, batch_size):
    outs = []
    for start in range(0, len(images), batch_size):
        px = _transform_batch(transform, images[start : start + batch_size]).to(device)
        z = model.encode({"pixels": px[:, None].float()})["emb"][:, 0]
        outs.append(z.detach())
    return torch.cat(outs, dim=0)


def _normalize_candidate_actions(raw_candidates, action_scaler, horizon, action_block):
    raw = np.asarray(raw_candidates, dtype=np.float32)
    k = raw.shape[0]
    expected = int(horizon) * int(action_block)
    if raw.shape[1:] != (expected, 2):
        raise RuntimeError(f"Expected candidate actions {(k, expected, 2)}, got {raw.shape}.")
    flat = raw.reshape(-1, 2)
    norm = action_scaler.transform(flat).astype(np.float32)
    norm = norm.reshape(k, int(horizon), int(action_block), 2)
    return norm.reshape(k, int(horizon), int(action_block) * 2)


@torch.inference_mode()
def _predict_terminal_latents(
    model, transform, current_image, normalized_actions, device, batch_size
):
    current_px = _transform_batch(transform, [current_image]).to(device).float()[0]
    outs = []
    k = len(normalized_actions)
    for start in range(0, k, batch_size):
        a = torch.from_numpy(normalized_actions[start : start + batch_size]).to(
            device=device, dtype=torch.float32
        )
        s = a.shape[0]
        action_sequence = a.unsqueeze(0)
        info = {
            "pixels": current_px[None, None, None].expand(1, 1, 1, -1, -1, -1)
        }
        rolled = model.rollout(info, action_sequence)
        z = rolled["predicted_emb"][0, :s, -1]
        outs.append(z.detach())
    return torch.cat(outs, dim=0)


def _anchor_model_metrics(
    physical_cost, factor_cost, pusher_err, block_err, joint_err, theta_err,
    enc_cost, pred_cost, terminal_pred_enc_mse, pair_margin_frac,
):
    acc_pred, n_pair_pred = _pairwise_accuracy(physical_cost, pred_cost, pair_margin_frac)
    acc_enc, n_pair_enc = _pairwise_accuracy(physical_cost, enc_cost, pair_margin_frac)
    pred_sel = _selection_stats(physical_cost, pred_cost)
    enc_sel = _selection_stats(physical_cost, enc_cost)
    oracle_idx = pred_sel["oracle_idx"]
    pred_idx = pred_sel["selected_idx"]
    enc_idx = enc_sel["selected_idx"]
    return {
        "spearman_pred_phys": _spearman(pred_cost, physical_cost),
        "spearman_enc_phys": _spearman(enc_cost, physical_cost),
        "spearman_pred_enc": _spearman(pred_cost, enc_cost),
        "spearman_pred_factorphys": _spearman(pred_cost, factor_cost),
        "spearman_enc_factorphys": _spearman(enc_cost, factor_cost),
        "pair_acc_pred_phys": acc_pred,
        "pair_acc_enc_phys": acc_enc,
        "pair_count_pred_phys": n_pair_pred,
        "pair_count_enc_phys": n_pair_enc,
        "top10_recall_pred_phys": _topk_recall(physical_cost, pred_cost, 10),
        "top10_recall_enc_phys": _topk_recall(physical_cost, enc_cost, 10),
        "top25_recall_pred_phys": _topk_recall(physical_cost, pred_cost, 25),
        "top25_recall_enc_phys": _topk_recall(physical_cost, enc_cost, 25),
        "mean_terminal_pred_enc_mse": float(np.mean(terminal_pred_enc_mse)),
        "median_terminal_pred_enc_mse": float(np.median(terminal_pred_enc_mse)),
        "oracle_candidate_idx": int(oracle_idx),
        "pred_selected_idx": int(pred_idx),
        "enc_selected_idx": int(enc_idx),
        "pred_selection_regret": pred_sel["selection_regret"],
        "pred_selection_regret_norm": pred_sel["selection_regret_norm"],
        "pred_selected_physical_percentile": pred_sel["selected_physical_percentile"],
        "enc_selection_regret": enc_sel["selection_regret"],
        "enc_selection_regret_norm": enc_sel["selection_regret_norm"],
        "enc_selected_physical_percentile": enc_sel["selected_physical_percentile"],
        "oracle_physical_cost": float(physical_cost[oracle_idx]),
        "pred_selected_physical_cost": float(physical_cost[pred_idx]),
        "enc_selected_physical_cost": float(physical_cost[enc_idx]),
        "oracle_pusher_error_px": float(pusher_err[oracle_idx]),
        "oracle_block_error_px": float(block_err[oracle_idx]),
        "oracle_joint_error_px": float(joint_err[oracle_idx]),
        "oracle_theta_error_deg": float(np.degrees(theta_err[oracle_idx])),
        "pred_selected_pusher_error_px": float(pusher_err[pred_idx]),
        "pred_selected_block_error_px": float(block_err[pred_idx]),
        "pred_selected_joint_error_px": float(joint_err[pred_idx]),
        "pred_selected_theta_error_deg": float(np.degrees(theta_err[pred_idx])),
        "enc_selected_pusher_error_px": float(pusher_err[enc_idx]),
        "enc_selected_block_error_px": float(block_err[enc_idx]),
        "enc_selected_joint_error_px": float(joint_err[enc_idx]),
        "enc_selected_theta_error_deg": float(np.degrees(theta_err[enc_idx])),
    }


SUMMARY_METRICS = [
    "spearman_pred_phys", "spearman_enc_phys", "spearman_pred_enc",
    "spearman_pred_factorphys", "spearman_enc_factorphys",
    "pair_acc_pred_phys", "pair_acc_enc_phys",
    "top10_recall_pred_phys", "top10_recall_enc_phys",
    "top25_recall_pred_phys", "top25_recall_enc_phys",
    "pred_selection_regret", "pred_selection_regret_norm",
    "pred_selected_physical_percentile",
    "enc_selection_regret", "enc_selection_regret_norm",
    "enc_selected_physical_percentile",
    "mean_terminal_pred_enc_mse", "median_terminal_pred_enc_mse",
    "pred_selected_pusher_error_px", "pred_selected_block_error_px",
    "pred_selected_joint_error_px", "pred_selected_theta_error_deg",
    "enc_selected_pusher_error_px", "enc_selected_block_error_px",
    "enc_selected_joint_error_px", "enc_selected_theta_error_deg",
    "oracle_pusher_error_px", "oracle_block_error_px",
    "oracle_joint_error_px", "oracle_theta_error_deg",
    "replay_factor_error", "replay_joint_error_px", "replay_theta_error_deg",
    "candidate_contact_fraction",
]


def _aggregate(rows):
    out = {"count": len(rows)}
    for key in SUMMARY_METRICS:
        out[key] = _numeric_summary([r.get(key, np.nan) for r in rows])
    return out


def _group_rows(rows):
    groups = {
        "all": rows,
        "expert_contact": [r for r in rows if r["expert_had_contact"]],
        "expert_no_contact": [r for r in rows if not r["expert_had_contact"]],
        "object_active": [r for r in rows if r["expert_object_active"]],
        "object_inactive": [r for r in rows if not r["expert_object_active"]],
        "replay_good": [r for r in rows if r["replay_good"]],
        "replay_bad": [r for r in rows if not r["replay_good"]],
    }
    return {name: _aggregate(group) for name, group in groups.items()}


def _mean(group_summary, key):
    x = group_summary.get(key, {})
    return x.get("mean") if isinstance(x, dict) else None


def _fmt(x, digits=3):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _print_table(summary, group):
    print(f"\n===== ACTION RANKING: {group} =====")
    header = (
        f"{'model':<14} {'rhoP':>7} {'rhoE':>7} {'rhoPE':>7} "
        f"{'accP':>7} {'accE':>7} {'top10P':>7} {'regP':>7} "
        f"{'pctP':>7} {'dynMSE':>10}"
    )
    print(header)
    print("-" * len(header))
    for label, payload in summary["models"].items():
        s = payload["summary"][group]
        print(
            f"{label:<14} "
            f"{_fmt(_mean(s, 'spearman_pred_phys')):>7} "
            f"{_fmt(_mean(s, 'spearman_enc_phys')):>7} "
            f"{_fmt(_mean(s, 'spearman_pred_enc')):>7} "
            f"{_fmt(_mean(s, 'pair_acc_pred_phys')):>7} "
            f"{_fmt(_mean(s, 'pair_acc_enc_phys')):>7} "
            f"{_fmt(_mean(s, 'top10_recall_pred_phys')):>7} "
            f"{_fmt(_mean(s, 'pred_selection_regret_norm')):>7} "
            f"{_fmt(_mean(s, 'pred_selected_physical_percentile')):>7} "
            f"{_fmt(_mean(s, 'mean_terminal_pred_enc_mse'), 5):>10}"
        )
    print(
        "rhoP=Spearman(pred latent cost, physical cost); "
        "rhoE=Spearman(real encoded latent cost, physical cost); "
        "rhoPE=Spearman(pred latent cost, real encoded latent cost);\n"
        "accP/accE=pairwise ranking accuracy; top10P=physical top-10 recall from predictor ranking; "
        "regP=normalized physical regret of predictor-selected candidate; "
        "pctP=physical percentile of predictor-selected candidate (0 best); "
        "dynMSE=terminal predictor-vs-real-encoder latent MSE."
    )


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels must have the same length as --policies.")
    if args.num_anchors <= 0:
        raise ValueError("--num-anchors must be positive.")
    if args.horizon <= 0 or args.action_block <= 0:
        raise ValueError("--horizon and --action-block must be positive.")
    if args.goal_offset != args.horizon * args.action_block:
        raise ValueError(
            "For this diagnostic, --goal-offset must equal horizon*action_block."
        )

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    output_root = (
        Path(args.output_dir) if args.output_dir is not None
        else cache_root / "pusht_action_ranking"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=["action", "state"], cache_dir=cache_root
    )
    col_name, anchors = _select_official_rows(
        dataset, args.num_anchors, args.seed, args.goal_offset
    )
    print("Selected paired eval rows:")
    print(anchors)

    episode_idx = np.asarray(dataset.get_col_data(col_name))
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    action = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    state = np.asarray(dataset.get_col_data("state"), dtype=np.float64)

    finite_action = action[np.isfinite(action).all(axis=1)]
    action_scaler = preprocessing.StandardScaler()
    action_scaler.fit(finite_action)

    device = torch.device(args.device)
    transform = img_transform(cfg)
    labels = [_label(p, args.labels, i) for i, p in enumerate(args.policies)]
    models = []
    for label, policy in zip(labels, args.policies):
        print(f"Loading [{label}] {policy}")
        model = swm.policy.AutoCostModel(policy).to(device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        models.append(model)

    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    nA = len(anchors)
    nC = int(args.num_candidates)
    raw_horizon = int(args.horizon) * int(args.action_block)
    nM = len(models)

    candidate_actions = np.empty((nA, nC, raw_horizon, 2), dtype=np.float32)
    candidate_type = np.empty((nA, nC), dtype=np.int8)
    candidate_noise_scale = np.empty((nA, nC), dtype=np.float32)
    final_states = np.empty((nA, nC, 7), dtype=np.float64)
    candidate_had_contact = np.empty((nA, nC), dtype=bool)
    candidate_contact_steps = np.empty((nA, nC), dtype=np.int16)
    physical_cost = np.empty((nA, nC), dtype=np.float64)
    factor_physical_cost = np.empty((nA, nC), dtype=np.float64)
    pusher_error = np.empty((nA, nC), dtype=np.float64)
    block_error = np.empty((nA, nC), dtype=np.float64)
    joint_error = np.empty((nA, nC), dtype=np.float64)
    theta_error_rad = np.empty((nA, nC), dtype=np.float64)
    enc_costs = np.empty((nM, nA, nC), dtype=np.float64)
    pred_costs = np.empty((nM, nA, nC), dtype=np.float64)
    pred_enc_mse = np.empty((nM, nA, nC), dtype=np.float64)

    rows_by_model = {label: [] for label in labels}
    start_total = time.time()

    try:
        for ai, r in enumerate(anchors):
            anchor_start = time.time()
            _check_anchor_contiguity(r, episode_idx, step_idx, raw_horizon)
            ep = int(episode_idx[r])
            t = int(step_idx[r])
            goal_r = r + args.goal_offset
            init_state = state[r].copy()
            goal_state = state[goal_r].copy()
            expert_actions = action[r : r + raw_horizon].copy()
            expert_has_nan = bool(not np.isfinite(expert_actions).all())

            rng = np.random.default_rng(args.seed + 1_000_003 * (ai + 1))
            candidates, type_code, noise_scale = _make_candidates(
                expert_actions, nC, args.noise_scales, args.uniform_frac, rng
            )
            candidate_actions[ai] = candidates
            candidate_type[ai] = type_code
            candidate_noise_scale[ai] = noise_scale

            controlled_seed = int(args.env_seed + ai)
            current_image, goal_image = _controlled_current_goal_images(
                env, init_state, goal_state, controlled_seed
            )

            final_images = []
            for ci in range(nC):
                fs, fi, hc, cs = _rollout_candidate(
                    env, init_state, goal_state, candidates[ci], controlled_seed
                )
                final_states[ai, ci] = fs
                final_images.append(fi)
                candidate_had_contact[ai, ci] = hc
                candidate_contact_steps[ai, ci] = cs

            (
                physical_cost[ai], pusher_error[ai], block_error[ai],
                joint_error[ai], theta_error_rad[ai],
            ) = _primary_physical_cost(final_states[ai], goal_state)
            factor_physical_cost[ai] = _factor_physical_cost(
                final_states[ai], goal_state, args.world_size
            )

            replay_factor_error = float(
                np.linalg.norm(
                    _state_factor(final_states[ai, 0], args.world_size)
                    - _state_factor(goal_state, args.world_size)
                )
            )
            replay_joint_error = float(joint_error[ai, 0])
            replay_theta_deg = float(np.degrees(theta_error_rad[ai, 0]))
            replay_good = bool(replay_factor_error <= args.replay_factor_good_threshold)
            expert_had_contact = bool(candidate_had_contact[ai, 0])
            candidate_contact_fraction = float(candidate_had_contact[ai].mean())
            expert_block_motion = float(
                np.linalg.norm(final_states[ai, 0, 2:4] - init_state[2:4])
            )
            expert_theta_motion_deg = float(
                np.degrees(_angle_error_rad(final_states[ai, 0, 4], init_state[4]))
            )
            expert_object_active = bool(
                expert_block_motion >= args.object_motion_px
                or expert_theta_motion_deg >= args.object_motion_deg
            )

            normalized_actions = _normalize_candidate_actions(
                candidates, action_scaler, args.horizon, args.action_block
            )

            for mi, (label, policy, model) in enumerate(
                zip(labels, args.policies, models)
            ):
                z_goal = _encode_images(
                    model, transform, [goal_image], device, args.model_batch_size
                )[0]
                z_real = _encode_images(
                    model, transform, final_images, device, args.model_batch_size
                )
                z_pred = _predict_terminal_latents(
                    model, transform, current_image, normalized_actions,
                    device, args.model_batch_size,
                )
                enc = torch.sum((z_real - z_goal[None]) ** 2, dim=-1)
                pred = torch.sum((z_pred - z_goal[None]) ** 2, dim=-1)
                dyn = torch.mean((z_pred - z_real) ** 2, dim=-1)
                enc_np = enc.detach().cpu().numpy().astype(np.float64)
                pred_np = pred.detach().cpu().numpy().astype(np.float64)
                dyn_np = dyn.detach().cpu().numpy().astype(np.float64)
                enc_costs[mi, ai] = enc_np
                pred_costs[mi, ai] = pred_np
                pred_enc_mse[mi, ai] = dyn_np

                metrics = _anchor_model_metrics(
                    physical_cost[ai], factor_physical_cost[ai],
                    pusher_error[ai], block_error[ai], joint_error[ai],
                    theta_error_rad[ai], enc_np, pred_np, dyn_np,
                    args.pair_margin_frac,
                )
                row = {
                    "model": label,
                    "policy": policy,
                    "anchor_index": ai + 1,
                    "dataset_row": int(r),
                    "episode_idx": ep,
                    "step_idx": t,
                    "goal_dataset_row": int(goal_r),
                    "num_candidates": nC,
                    "horizon_coarse": int(args.horizon),
                    "action_block": int(args.action_block),
                    "raw_horizon": raw_horizon,
                    "expert_has_nan": expert_has_nan,
                    "expert_had_contact": expert_had_contact,
                    "candidate_contact_fraction": candidate_contact_fraction,
                    "expert_object_active": expert_object_active,
                    "expert_block_motion_px": expert_block_motion,
                    "expert_theta_motion_deg": expert_theta_motion_deg,
                    "replay_factor_error": replay_factor_error,
                    "replay_joint_error_px": replay_joint_error,
                    "replay_theta_error_deg": replay_theta_deg,
                    "replay_good": replay_good,
                    **metrics,
                }
                rows_by_model[label].append(row)

            elapsed_anchor = time.time() - anchor_start
            done = ai + 1
            elapsed_total = time.time() - start_total
            eta = elapsed_total / done * (nA - done)
            print(
                f"anchor {done:3d}/{nA}: row={r} contact={expert_had_contact} "
                f"active={expert_object_active} replay={replay_factor_error:.4f} "
                f"time={elapsed_anchor:.2f}s ETA={eta/60:.1f}min"
            )
    finally:
        env.close()

    all_rows = [row for label in labels for row in rows_by_model[label]]
    csv_path = output_root / "anchor_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    npz_path = output_root / "candidate_metrics.npz"
    np.savez_compressed(
        npz_path,
        anchors=anchors,
        labels=np.asarray(labels, dtype=object),
        policies=np.asarray(args.policies, dtype=object),
        candidate_actions=candidate_actions,
        candidate_type=candidate_type,
        candidate_noise_scale=candidate_noise_scale,
        final_states=final_states,
        candidate_had_contact=candidate_had_contact,
        candidate_contact_steps=candidate_contact_steps,
        physical_cost=physical_cost,
        factor_physical_cost=factor_physical_cost,
        pusher_error_px=pusher_error,
        block_error_px=block_error,
        joint_error_px=joint_error,
        theta_error_rad=theta_error_rad,
        enc_costs=enc_costs,
        pred_costs=pred_costs,
        pred_enc_terminal_mse=pred_enc_mse,
    )

    total_time = time.time() - start_total
    summary = {
        "config": {
            "dataset": dataset_name,
            "num_anchors": nA,
            "num_candidates": nC,
            "seed": int(args.seed),
            "env_seed": int(args.env_seed),
            "horizon": int(args.horizon),
            "action_block": int(args.action_block),
            "raw_horizon": raw_horizon,
            "goal_offset": int(args.goal_offset),
            "noise_scales": list(map(float, args.noise_scales)),
            "uniform_frac": float(args.uniform_frac),
            "pair_margin_frac": float(args.pair_margin_frac),
            "replay_factor_good_threshold": float(args.replay_factor_good_threshold),
            "physical_primary_cost": (
                "(joint_position_error_px/20)^2 + "
                "(wrapped_theta_error_rad/(pi/9))^2"
            ),
            "model_cost": "raw terminal squared Euclidean latent distance",
            "rendering_note": (
                "Current, goal, and real candidate terminal images are rendered "
                "from a controlled simulator reset with the same state/goal and "
                "same per-anchor variation seed."
            ),
        },
        "paired_rows": anchors.tolist(),
        "elapsed_seconds": float(total_time),
        "models": {},
    }
    for label, policy in zip(labels, args.policies):
        summary["models"][label] = {
            "policy": policy,
            "summary": _group_rows(rows_by_model[label]),
        }

    summary_path = output_root / "summary.json"
    with summary_path.open("w") as f:
        json.dump(_jsonable(summary), f, indent=2)

    for group in ["all", "expert_contact", "expert_no_contact", "object_active"]:
        _print_table(summary, group)

    print(f"\nElapsed: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
