#!/usr/bin/env python3
"""
Unified full-PushT controllability evaluation for LeWM-family checkpoints.

This script compares multiple checkpoints on exactly the same physical anchors
and exactly the same 10-D coarse-action perturbations.  It intentionally keeps
all controllability diagnostics in ONE file.

For one coarse PushT action block
    u = [a_t, ..., a_{t+4}] in R^10,
it estimates
    J_phys = d f(s_{t+5}) / d u,
    J_enc  = d E(o_{t+5}) / d u,
    J_pred = d P(z_hist, u) / d u.

Physical state is used only for diagnostics.  Factor heads are never used.
The official raw-latent planner/cost is not modified.

Main metrics
------------
1. singular spectrum, threshold rank, stable rank, energy rank (90/95%)
2. encoder-predictor Jacobian relative error
3. latent principal angles, including a physical-rank-matched version
4. physical-vs-latent action-space principal angles
5. pullback action metric G = J^T J alignment
6. physical-rank-matched goal controllability
7. relative control authority ||J||_F / ||z_goal-z||
8. true encoded delta-z vs predicted delta-z consistency
9. contact/no-contact and object-active/inactive grouping
10. simulator replay sanity check against the recorded next dataset state

The raw threshold-based latent rank is retained only as a diagnostic.  Cross-
model conclusions should preferentially use energy rank and physical-rank-
matched quantities, because tiny image/latent singular directions can inflate
raw numerical rank.
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from eval import img_transform


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified full PushT local controllability diagnostics."
    )
    p.add_argument(
        "--policies",
        nargs="+",
        required=True,
        help=(
            "Checkpoint paths relative to STABLEWM_HOME, without "
            "_object.ckpt, exactly as accepted by AutoCostModel."
        ),
    )
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--num-anchors", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Central finite-difference perturbation in each raw action scalar.",
    )
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--rank-rel-tol", type=float, default=1e-2)
    p.add_argument("--object-motion-px", type=float, default=1.0)
    p.add_argument("--object-motion-deg", type=float, default=1.0)
    p.add_argument("--world-size", type=float, default=512.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: $STABLEWM_HOME/pusht_controllability",
    )
    p.add_argument("--max-anchor-attempt-factor", type=int, default=20)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


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


def _angle_error_rad(a, b):
    d = np.asarray(a) - np.asarray(b)
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def _safe_cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def _safe_rel_error(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    den = np.linalg.norm(target)
    if den <= 1e-12:
        return float("nan")
    return float(np.linalg.norm(pred - target) / den)


def _safe_ratio(num, den):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 1e-12:
        return float("nan")
    return float(num / den)


def _state_factor(state, world_size=512.0):
    """Continuous 6-D physical diagnostic coordinate."""
    s = np.asarray(state, dtype=np.float64)
    pusher = 2.0 * s[..., 0:2] / float(world_size) - 1.0
    block = 2.0 * s[..., 2:4] / float(world_size) - 1.0
    theta = s[..., 4]
    theta_unit = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
    return np.concatenate([pusher, block, theta_unit], axis=-1)


def _transform_one(transform, image):
    x = transform(image)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x


@torch.inference_mode()
def _encode_images(model, transform, images, device, batch_size=32):
    tensors = [_transform_one(transform, im) for im in images]
    outs = []
    for start in range(0, len(tensors), batch_size):
        px = torch.stack(tensors[start : start + batch_size], dim=0).to(device)
        emb = model.encode({"pixels": px[:, None]})["emb"][:, 0]
        outs.append(emb.detach())
    return torch.cat(outs, dim=0)


def _normalize_action_block(block, mean, std):
    x = (np.asarray(block, dtype=np.float32) - mean[None]) / std[None]
    return x.reshape(-1)


@torch.inference_mode()
def _predict_variants(model, z_hist, prev_blocks, variant_blocks, mean, std, device):
    fixed = [
        _normalize_action_block(prev_blocks[0], mean, std),
        _normalize_action_block(prev_blocks[1], mean, std),
    ]
    ctx_actions = []
    for b in variant_blocks:
        ctx_actions.append(
            np.stack(
                [fixed[0], fixed[1], _normalize_action_block(b, mean, std)],
                axis=0,
            )
        )
    actions = torch.from_numpy(np.stack(ctx_actions)).float().to(device)
    hist = z_hist.expand(len(variant_blocks), -1, -1).contiguous()
    act_emb = model.action_encoder(actions)
    return model.predict(hist, act_emb)[:, -1].detach()


# -----------------------------------------------------------------------------
# Finite differences and simulator
# -----------------------------------------------------------------------------


def _make_variants(base_block, epsilon):
    """base,+e0,-e0,+e1,-e1,... with raw action clipping."""
    base = np.asarray(base_block, dtype=np.float32)
    flat = base.reshape(-1)
    variants = [base.copy()]
    denoms = []
    for j in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[j] = min(1.0, float(plus[j] + epsilon))
        minus[j] = max(-1.0, float(minus[j] - epsilon))
        denom = float(plus[j] - minus[j])
        if denom <= 1e-8:
            raise RuntimeError(f"Finite-difference denominator vanished at dim {j}.")
        variants.extend([plus.reshape(base.shape), minus.reshape(base.shape)])
        denoms.append(denom)
    return variants, np.asarray(denoms, dtype=np.float64)


def _jacobian_from_variants(values, denoms):
    y = np.asarray(values, dtype=np.float64)
    cols = []
    for j, d in enumerate(denoms):
        yp = y[1 + 2 * j]
        ym = y[2 + 2 * j]
        cols.append((yp - ym) / d)
    return np.stack(cols, axis=1)


def _rollout_block(env, init_state, goal_state, action_block):
    env.reset(seed=0)
    raw = env.unwrapped
    raw._set_goal_state(np.asarray(goal_state, dtype=np.float64))
    raw._set_state(np.asarray(init_state, dtype=np.float64))

    had_contact = False
    contact_steps = 0
    obs = None
    for action in np.asarray(action_block, dtype=np.float32):
        obs, _, _, _, info = raw.step(action)
        n_contacts = int(info.get("n_contacts", 0))
        had_contact = had_contact or n_contacts > 0
        contact_steps += int(n_contacts > 0)

    final_state = np.asarray(obs["state"], dtype=np.float64)
    final_image = raw.render()
    return final_state, final_image, had_contact, contact_steps


# -----------------------------------------------------------------------------
# Linear-algebra diagnostics
# -----------------------------------------------------------------------------


def _energy_rank(s, fraction):
    s = np.asarray(s, dtype=np.float64)
    energy = s * s
    total = float(energy.sum())
    if total <= 1e-20:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, fraction, side="left") + 1)


def _energy_fraction_top_r(s, r):
    s = np.asarray(s, dtype=np.float64)
    energy = s * s
    total = float(energy.sum())
    if total <= 1e-20 or r <= 0:
        return float("nan")
    return float(energy[: min(int(r), len(energy))].sum() / total)


def _svd_info(J, rel_tol):
    J = np.asarray(J, dtype=np.float64)
    U, s, Vh = np.linalg.svd(J, full_matrices=False)
    if len(s) == 0 or s[0] <= 1e-12:
        rank = 0
        stable_rank = 0.0
        condition = float("nan")
    else:
        rank = int(np.sum(s / s[0] > rel_tol))
        stable_rank = float(np.sum(s * s) / (s[0] * s[0]))
        condition = (
            float(s[0] / s[rank - 1])
            if rank > 0 and s[rank - 1] > 1e-12
            else float("nan")
        )
    return {
        "U": U,
        "s": s,
        "Vh": Vh,
        "rank": rank,
        "stable_rank": stable_rank,
        "energy_rank90": _energy_rank(s, 0.90),
        "energy_rank95": _energy_rank(s, 0.95),
        "condition_active": condition,
        "fro": float(np.linalg.norm(J, ord="fro")),
    }


def _principal_angles(Q1, Q2):
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return np.array([], dtype=np.float64)
    r = min(Q1.shape[1], Q2.shape[1])
    M = Q1[:, :r].T @ Q2[:, :r]
    c = np.linalg.svd(M, compute_uv=False)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def _subspace_summary(angles):
    if len(angles) == 0:
        return {"mean_deg": float("nan"), "max_deg": float("nan")}
    return {
        "mean_deg": float(np.mean(angles)),
        "max_deg": float(np.max(angles)),
    }


def _gram(J):
    J = np.asarray(J, dtype=np.float64)
    return J.T @ J


def _gram_cosine(G1, G2):
    a = np.asarray(G1, dtype=np.float64)
    b = np.asarray(G2, dtype=np.float64)
    den = np.linalg.norm(a, "fro") * np.linalg.norm(b, "fro")
    if den <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / den)


def _gram_rel_error_trace_normalized(G1, G2):
    a = np.asarray(G1, dtype=np.float64)
    b = np.asarray(G2, dtype=np.float64)
    ta, tb = np.trace(a), np.trace(b)
    if ta <= 1e-12 or tb <= 1e-12:
        return float("nan")
    a = a / ta
    b = b / tb
    return float(
        np.linalg.norm(a - b, "fro") / max(np.linalg.norm(a, "fro"), 1e-12)
    )


def _goal_projection_ratio(goal_direction, U, rank):
    g = np.asarray(goal_direction, dtype=np.float64)
    ng = np.linalg.norm(g)
    if ng <= 1e-12 or rank <= 0:
        return float("nan")
    r = min(int(rank), U.shape[1])
    Q = U[:, :r]
    return float(np.linalg.norm(Q.T @ g) / ng)


def _matched_rank(physical_rank, latent_rank):
    return int(max(0, min(int(physical_rank), int(latent_rank))))


# -----------------------------------------------------------------------------
# Dataset anchor selection
# -----------------------------------------------------------------------------


def _select_anchor_rows(
    episode_idx,
    step_idx,
    action,
    state,
    num_anchors,
    seed,
    history_size,
    action_block,
    goal_offset,
    max_attempt_factor,
):
    if history_size != 3:
        raise ValueError("Current evaluator expects LeWM history_size=3.")
    history_back = (history_size - 1) * action_block

    unique_ep, inv = np.unique(episode_idx, return_inverse=True)
    max_step = np.full(len(unique_ep), -1, dtype=np.int64)
    np.maximum.at(max_step, inv, step_idx.astype(np.int64))
    max_for_row = max_step[inv]

    valid = (
        (step_idx >= history_back)
        & (step_idx + action_block <= max_for_row)
        & (step_idx + goal_offset <= max_for_row)
        & np.isfinite(state).all(axis=1)
    )
    candidates = np.nonzero(valid)[0]

    rng = np.random.default_rng(seed)
    order = rng.permutation(candidates)
    max_try = min(len(order), max(num_anchors * max_attempt_factor, num_anchors))

    selected = []
    for r in order[:max_try]:
        ep = episode_idx[r]
        t = int(step_idx[r])
        required = [
            (r - 2 * action_block, t - 2 * action_block),
            (r - action_block, t - action_block),
            (r, t),
            (r + action_block, t + action_block),
            (r + goal_offset, t + goal_offset),
        ]
        if any(
            idx < 0
            or idx >= len(step_idx)
            or episode_idx[idx] != ep
            or int(step_idx[idx]) != expected
            for idx, expected in required
        ):
            continue

        a0 = r - 2 * action_block
        a1 = r + action_block
        if a0 < 0 or a1 > len(action):
            continue
        if not np.isfinite(action[a0:a1]).all():
            continue

        selected.append(int(r))
        if len(selected) >= num_anchors:
            break

    if len(selected) < num_anchors:
        raise RuntimeError(
            f"Only found {len(selected)} valid anchors; requested {num_anchors}."
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def _load_pixels(dataset, row_indices):
    rows = dataset.get_row_data(np.asarray(row_indices, dtype=np.int64))
    pixels = rows["pixels"]
    if torch.is_tensor(pixels):
        return [pixels[i] for i in range(len(row_indices))]
    arr = np.asarray(pixels)
    return [arr[i] for i in range(len(row_indices))]


# -----------------------------------------------------------------------------
# Summaries / compact table
# -----------------------------------------------------------------------------


def _numeric_summary(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
    }


SUMMARY_KEYS = [
    "phys_rank",
    "enc_rank",
    "pred_rank",
    "phys_stable_rank",
    "enc_stable_rank",
    "pred_stable_rank",
    "phys_energy_rank90",
    "enc_energy_rank90",
    "pred_energy_rank90",
    "phys_energy_rank95",
    "enc_energy_rank95",
    "pred_energy_rank95",
    "enc_energy_top_physrank",
    "pred_energy_top_physrank",
    "enc_pred_jac_rel_fro",
    "latent_subspace_mean_deg",
    "latent_subspace_max_deg",
    "latent_subspace_physrank_mean_deg",
    "latent_subspace_physrank_max_deg",
    "action_subspace_phys_enc_mean_deg",
    "action_subspace_phys_enc_max_deg",
    "action_subspace_phys_pred_mean_deg",
    "action_subspace_phys_pred_max_deg",
    "gram_cos_phys_enc",
    "gram_cos_phys_pred",
    "gram_cos_enc_pred",
    "gram_rel_phys_enc",
    "gram_rel_phys_pred",
    "goal_latent_distance",
    "goal_controllability_enc_rawrank",
    "goal_controllability_pred_rawrank",
    "goal_controllability_enc_physrank",
    "goal_controllability_pred_physrank",
    "enc_control_authority",
    "pred_control_authority",
    "delta_true_norm",
    "delta_pred_norm",
    "delta_pred_true_cosine",
    "delta_pred_true_rel_error",
    "delta_pred_true_norm_ratio",
    "true_goal_progress_cosine",
    "pred_goal_progress_cosine",
    "sim_replay_factor_error",
    "phys_pusher_gain",
    "phys_block_gain",
    "phys_theta_gain",
    "enc_gain",
    "pred_gain",
]


def _aggregate_rows(rows):
    out = {"count": len(rows)}
    for k in SUMMARY_KEYS:
        out[k] = _numeric_summary([r.get(k, np.nan) for r in rows])
    return out


def _group_summary(rows):
    groups = {
        "all": rows,
        "contact": [r for r in rows if r["had_contact"]],
        "no_contact": [r for r in rows if not r["had_contact"]],
        "object_active": [r for r in rows if r["object_active"]],
        "object_inactive": [r for r in rows if not r["object_active"]],
    }
    return {name: _aggregate_rows(group) for name, group in groups.items()}


def _mean(summary, key):
    x = summary.get(key, {})
    return x.get("mean") if isinstance(x, dict) else None


def _fmt(x, digits=3):
    if x is None or not np.isfinite(x):
        return "   n/a"
    return f"{x:.{digits}f}"


def _print_compact_table(comparison, group):
    print(f"\n===== COMPACT COMPARISON: {group} =====")
    header = (
        f"{'model':<14} {'eR95':>5} {'Jp/Je':>7} {'angR':>7} "
        f"{'Gcos':>7} {'Gerr':>7} {'goalR':>7} {'auth':>7} "
        f"{'dcos':>7} {'derr':>7}"
    )
    print(header)
    print("-" * len(header))
    for label, payload in comparison["models"].items():
        s = payload["summary"][group]
        print(
            f"{label:<14} "
            f"{_fmt(_mean(s, 'pred_energy_rank95'), 1):>5} "
            f"{_fmt(_mean(s, 'enc_pred_jac_rel_fro')):>7} "
            f"{_fmt(_mean(s, 'latent_subspace_physrank_mean_deg'), 1):>7} "
            f"{_fmt(_mean(s, 'gram_cos_phys_pred')):>7} "
            f"{_fmt(_mean(s, 'gram_rel_phys_pred')):>7} "
            f"{_fmt(_mean(s, 'goal_controllability_pred_physrank')):>7} "
            f"{_fmt(_mean(s, 'pred_control_authority')):>7} "
            f"{_fmt(_mean(s, 'delta_pred_true_cosine')):>7} "
            f"{_fmt(_mean(s, 'delta_pred_true_rel_error')):>7}"
        )
    print(
        "eR95=pred 95%-energy rank; Jp/Je=relative predictor-vs-encoder "
        "Jacobian error; angR=latent principal angle matched to physical rank;\n"
        "Gcos/Gerr=physical-vs-predictor action-metric alignment; "
        "goalR=goal projection using only top physical-rank predictor directions;\n"
        "auth=||J_pred||F/||z_goal-z||; dcos/derr=predicted vs true encoded delta-z."
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _model_label(policy, labels, idx):
    if labels is not None:
        return labels[idx]
    return Path(policy).name.replace("_epoch_10", "").replace("lewm_", "")


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels must have the same length as --policies.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    if args.action_block != 5:
        raise ValueError("Current LeWM PushT checkpoints use action_block=5.")

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    output_root = (
        Path(args.output_dir)
        if args.output_dir is not None
        else cache_root / "pusht_controllability"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=["action", "state"],
        cache_dir=cache_root,
    )
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = np.asarray(dataset.get_col_data(col_name))
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    action = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    state = np.asarray(dataset.get_col_data("state"), dtype=np.float64)

    finite_action = action[np.isfinite(action).all(axis=1)]
    action_mean = finite_action.mean(axis=0).astype(np.float32)
    # Match training get_column_normalizer(): torch.std correction=1.
    action_std = finite_action.std(axis=0, ddof=1).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6)

    anchors = _select_anchor_rows(
        episode_idx,
        step_idx,
        action,
        state,
        args.num_anchors,
        args.seed,
        args.history_size,
        args.action_block,
        args.goal_offset,
        args.max_anchor_attempt_factor,
    )
    print(f"Selected {len(anchors)} anchors:")
    print(anchors)

    device = torch.device(args.device)
    transform = img_transform(cfg)

    models, labels = [], []
    for i, policy in enumerate(args.policies):
        label = _model_label(policy, args.labels, i)
        print(f"Loading [{label}] {policy}")
        model = swm.policy.AutoCostModel(policy).to(device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        models.append(model)
        labels.append(label)

    rows_by_model = {label: [] for label in labels}
    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")

    try:
        for anchor_i, r in enumerate(anchors, start=1):
            ep = int(episode_idx[r])
            t = int(step_idx[r])
            next_r = r + args.action_block
            goal_r = r + args.goal_offset

            init_state = state[r].copy()
            dataset_next_state = state[next_r].copy()
            goal_state = state[goal_r].copy()

            hist_rows = [
                r - 2 * args.action_block,
                r - args.action_block,
                r,
            ]
            hist_images = _load_pixels(dataset, hist_rows)
            next_image = _load_pixels(dataset, [next_r])[0]
            goal_image = _load_pixels(dataset, [goal_r])[0]

            prev_blocks = [
                action[r - 2 * args.action_block : r - args.action_block].copy(),
                action[r - args.action_block : r].copy(),
            ]
            base_block = action[r : r + args.action_block].copy()
            variants, denoms = _make_variants(base_block, args.epsilon)

            # Shared simulator perturbations for every checkpoint.
            physical_values = []
            final_images = []
            base_contact = False
            base_contact_steps = 0
            base_final_state = None

            for vi, block in enumerate(variants):
                final_state, final_image, had_contact, contact_steps = _rollout_block(
                    env, init_state, goal_state, block
                )
                physical_values.append(
                    _state_factor(final_state, world_size=args.world_size)
                )
                final_images.append(final_image)
                if vi == 0:
                    base_final_state = final_state
                    base_contact = had_contact
                    base_contact_steps = contact_steps

            J_phys = _jacobian_from_variants(physical_values, denoms)
            phys_info = _svd_info(J_phys, args.rank_rel_tol)
            G_phys = _gram(J_phys)

            block_motion_px = float(
                np.linalg.norm(base_final_state[2:4] - init_state[2:4])
            )
            theta_motion_deg = float(
                np.degrees(_angle_error_rad(base_final_state[4], init_state[4]))
            )
            object_active = bool(
                block_motion_px >= args.object_motion_px
                or theta_motion_deg >= args.object_motion_deg
            )
            sim_replay_factor_error = float(
                np.linalg.norm(
                    _state_factor(base_final_state, args.world_size)
                    - _state_factor(dataset_next_state, args.world_size)
                )
            )

            for label, policy, model in zip(labels, args.policies, models):
                # Dataset latents are used for history/current/next/goal so that
                # delta-z prediction diagnostics are not confounded by a render
                # mismatch between recorded data and freshly rendered simulator images.
                z_hist = _encode_images(model, transform, hist_images, device)[None]
                z_next_data = _encode_images(model, transform, [next_image], device)[0]
                z_goal = _encode_images(model, transform, [goal_image], device)[0]

                # Fresh simulator renders are used only to estimate the encoder
                # finite-difference Jacobian under the same physical perturbations.
                z_final_variants = _encode_images(model, transform, final_images, device)
                z_pred_variants = _predict_variants(
                    model,
                    z_hist,
                    prev_blocks,
                    variants,
                    action_mean,
                    action_std,
                    device,
                )

                J_enc = _jacobian_from_variants(
                    z_final_variants.detach().cpu().numpy(), denoms
                )
                J_pred = _jacobian_from_variants(
                    z_pred_variants.detach().cpu().numpy(), denoms
                )

                enc_info = _svd_info(J_enc, args.rank_rel_tol)
                pred_info = _svd_info(J_pred, args.rank_rel_tol)
                G_enc, G_pred = _gram(J_enc), _gram(J_pred)

                # Raw-rank latent subspace comparison (diagnostic only).
                latent_angles = _principal_angles(
                    enc_info["U"][:, : enc_info["rank"]],
                    pred_info["U"][:, : pred_info["rank"]],
                )
                latent_summary = _subspace_summary(latent_angles)

                # Fairer latent comparison: force both models to use no more
                # directions than are physically active at this anchor.
                r_lat_phys = min(
                    phys_info["rank"], enc_info["rank"], pred_info["rank"]
                )
                latent_phys_angles = _principal_angles(
                    enc_info["U"][:, :r_lat_phys],
                    pred_info["U"][:, :r_lat_phys],
                )
                latent_phys_summary = _subspace_summary(latent_phys_angles)

                # All right singular vectors live in the SAME 10-D action space.
                r_phys_enc = _matched_rank(phys_info["rank"], enc_info["rank"])
                r_phys_pred = _matched_rank(phys_info["rank"], pred_info["rank"])
                phys_V = phys_info["Vh"][: phys_info["rank"]].T
                enc_V = enc_info["Vh"][:r_phys_enc].T
                pred_V = pred_info["Vh"][:r_phys_pred].T
                phys_enc_summary = _subspace_summary(_principal_angles(phys_V, enc_V))
                phys_pred_summary = _subspace_summary(_principal_angles(phys_V, pred_V))

                enc_pred_rel = float(
                    np.linalg.norm(J_pred - J_enc, "fro")
                    / max(np.linalg.norm(J_enc, "fro"), 1e-12)
                )

                z_current_np = z_hist[0, -1].detach().cpu().numpy()
                z_next_np = z_next_data.detach().cpu().numpy()
                z_goal_np = z_goal.detach().cpu().numpy()
                z_pred_np = z_pred_variants[0].detach().cpu().numpy()

                goal_direction = z_goal_np - z_current_np
                goal_distance = float(np.linalg.norm(goal_direction))

                # Keep raw-rank versions for debugging, but cross-model analysis
                # should use the physical-rank-matched versions below.
                goal_enc_raw = _goal_projection_ratio(
                    goal_direction, enc_info["U"], enc_info["rank"]
                )
                goal_pred_raw = _goal_projection_ratio(
                    goal_direction, pred_info["U"], pred_info["rank"]
                )
                goal_enc_phys = _goal_projection_ratio(
                    goal_direction, enc_info["U"], r_phys_enc
                )
                goal_pred_phys = _goal_projection_ratio(
                    goal_direction, pred_info["U"], r_phys_pred
                )

                true_delta = z_next_np - z_current_np
                pred_delta = z_pred_np - z_current_np
                true_delta_norm = float(np.linalg.norm(true_delta))
                pred_delta_norm = float(np.linalg.norm(pred_delta))

                row = {
                    "model": label,
                    "policy": policy,
                    "anchor_index": anchor_i,
                    "dataset_row": int(r),
                    "episode_idx": ep,
                    "step_idx": t,
                    "epsilon_raw_action": float(args.epsilon),
                    "had_contact": bool(base_contact),
                    "contact_steps": int(base_contact_steps),
                    "object_active": object_active,
                    "baseline_block_motion_px": block_motion_px,
                    "baseline_theta_motion_deg": theta_motion_deg,
                    "sim_replay_factor_error": sim_replay_factor_error,
                    "phys_rank": phys_info["rank"],
                    "enc_rank": enc_info["rank"],
                    "pred_rank": pred_info["rank"],
                    "phys_stable_rank": phys_info["stable_rank"],
                    "enc_stable_rank": enc_info["stable_rank"],
                    "pred_stable_rank": pred_info["stable_rank"],
                    "phys_energy_rank90": phys_info["energy_rank90"],
                    "enc_energy_rank90": enc_info["energy_rank90"],
                    "pred_energy_rank90": pred_info["energy_rank90"],
                    "phys_energy_rank95": phys_info["energy_rank95"],
                    "enc_energy_rank95": enc_info["energy_rank95"],
                    "pred_energy_rank95": pred_info["energy_rank95"],
                    "enc_energy_top_physrank": _energy_fraction_top_r(
                        enc_info["s"], phys_info["rank"]
                    ),
                    "pred_energy_top_physrank": _energy_fraction_top_r(
                        pred_info["s"], phys_info["rank"]
                    ),
                    "phys_condition_active": phys_info["condition_active"],
                    "enc_condition_active": enc_info["condition_active"],
                    "pred_condition_active": pred_info["condition_active"],
                    "phys_gain": phys_info["fro"],
                    "enc_gain": enc_info["fro"],
                    "pred_gain": pred_info["fro"],
                    "phys_pusher_gain": float(np.linalg.norm(J_phys[0:2], "fro")),
                    "phys_block_gain": float(np.linalg.norm(J_phys[2:4], "fro")),
                    "phys_theta_gain": float(np.linalg.norm(J_phys[4:6], "fro")),
                    "enc_pred_jac_rel_fro": enc_pred_rel,
                    "latent_subspace_mean_deg": latent_summary["mean_deg"],
                    "latent_subspace_max_deg": latent_summary["max_deg"],
                    "latent_subspace_physrank_mean_deg": latent_phys_summary["mean_deg"],
                    "latent_subspace_physrank_max_deg": latent_phys_summary["max_deg"],
                    "action_subspace_phys_enc_mean_deg": phys_enc_summary["mean_deg"],
                    "action_subspace_phys_enc_max_deg": phys_enc_summary["max_deg"],
                    "action_subspace_phys_pred_mean_deg": phys_pred_summary["mean_deg"],
                    "action_subspace_phys_pred_max_deg": phys_pred_summary["max_deg"],
                    "gram_cos_phys_enc": _gram_cosine(G_phys, G_enc),
                    "gram_cos_phys_pred": _gram_cosine(G_phys, G_pred),
                    "gram_cos_enc_pred": _gram_cosine(G_enc, G_pred),
                    "gram_rel_phys_enc": _gram_rel_error_trace_normalized(G_phys, G_enc),
                    "gram_rel_phys_pred": _gram_rel_error_trace_normalized(G_phys, G_pred),
                    "goal_latent_distance": goal_distance,
                    "goal_controllability_enc_rawrank": goal_enc_raw,
                    "goal_controllability_pred_rawrank": goal_pred_raw,
                    "goal_controllability_enc_physrank": goal_enc_phys,
                    "goal_controllability_pred_physrank": goal_pred_phys,
                    "enc_control_authority": _safe_ratio(enc_info["fro"], goal_distance),
                    "pred_control_authority": _safe_ratio(pred_info["fro"], goal_distance),
                    "delta_true_norm": true_delta_norm,
                    "delta_pred_norm": pred_delta_norm,
                    "delta_pred_true_cosine": _safe_cosine(pred_delta, true_delta),
                    "delta_pred_true_rel_error": _safe_rel_error(pred_delta, true_delta),
                    "delta_pred_true_norm_ratio": _safe_ratio(
                        pred_delta_norm, true_delta_norm
                    ),
                    "true_goal_progress_cosine": _safe_cosine(true_delta, goal_direction),
                    "pred_goal_progress_cosine": _safe_cosine(pred_delta, goal_direction),
                }

                for j, sval in enumerate(phys_info["s"]):
                    row[f"phys_s{j+1}"] = float(sval)
                for j, sval in enumerate(enc_info["s"]):
                    row[f"enc_s{j+1}"] = float(sval)
                for j, sval in enumerate(pred_info["s"]):
                    row[f"pred_s{j+1}"] = float(sval)

                rows_by_model[label].append(row)

            print(
                f"[{anchor_i:03d}/{len(anchors):03d}] row={r} ep={ep} step={t} "
                f"contact={base_contact} active={object_active} "
                f"phys_rank={phys_info['rank']} phys_e95={phys_info['energy_rank95']} "
                f"replay_err={sim_replay_factor_error:.4f}"
            )
    finally:
        env.close()

    comparison = {
        "settings": {
            "dataset": dataset_name,
            "num_anchors": len(anchors),
            "anchor_rows": anchors,
            "seed": args.seed,
            "epsilon_raw_action": args.epsilon,
            "action_block": args.action_block,
            "history_size": args.history_size,
            "goal_offset": args.goal_offset,
            "rank_rel_tol": args.rank_rel_tol,
            "object_motion_px": args.object_motion_px,
            "object_motion_deg": args.object_motion_deg,
            "action_mean": action_mean,
            "action_std": action_std,
            "physical_factor": (
                "[2*xp/512-1,2*yp/512-1,2*xT/512-1,2*yT/512-1,"
                "sin(theta),cos(theta)]"
            ),
            "interpretation": {
                "raw_rank": (
                    "Diagnostic only; finite-difference/image discretization can create "
                    "small extra singular directions."
                ),
                "energy_rank95": (
                    "Preferred dimensionality diagnostic: number of singular directions "
                    "explaining 95% of Jacobian energy."
                ),
                "goal_controllability_pred_physrank": (
                    "Goal projection using only the top predictor directions allowed by "
                    "the local physical rank; preferred over raw-rank goal projection."
                ),
                "pred_control_authority": (
                    "||J_pred||_F / ||z_goal-z_current||; scale-aware local action "
                    "authority relative to the current latent goal distance."
                ),
                "delta_pred_true": (
                    "Uses recorded dataset current/next images for true delta-z and the "
                    "predictor output for predicted delta-z."
                ),
            },
        },
        "models": {},
    }

    for label, policy in zip(labels, args.policies):
        rows = rows_by_model[label]
        summary = _group_summary(rows)
        comparison["models"][label] = {"policy": policy, "summary": summary}

        model_dir = output_root / label
        model_dir.mkdir(parents=True, exist_ok=True)
        csv_path = model_dir / "anchors.csv"
        json_path = model_dir / "results.json"

        fieldnames = sorted({k for row in rows for k in row.keys()})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        with json_path.open("w") as f:
            json.dump(
                _jsonable(
                    {
                        "policy": policy,
                        "settings": comparison["settings"],
                        "summary": summary,
                        "anchors": rows,
                    }
                ),
                f,
                indent=2,
            )

        print(f"\n===== {label} =====")
        print(json.dumps(_jsonable(summary), indent=2))
        print(f"Saved: {csv_path}")
        print(f"Saved: {json_path}")

    comparison_path = output_root / "comparison.json"
    with comparison_path.open("w") as f:
        json.dump(_jsonable(comparison), f, indent=2)

    _print_compact_table(comparison, "all")
    _print_compact_table(comparison, "object_active")
    _print_compact_table(comparison, "no_contact")
    print(f"\nSaved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
