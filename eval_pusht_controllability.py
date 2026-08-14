#!/usr/bin/env python3
"""
Full PushT controllability diagnostics for LeWM-family checkpoints.

One script evaluates multiple checkpoints on exactly the same physical anchors
and action perturbations. It measures:

  1) physical controllability Jacobian J_phys = d f(s_{t+5}) / d u
  2) encoder Jacobian J_enc = d E(o_{t+5}) / d u
  3) predictor Jacobian J_pred = d P(z_hist, u) / d u
  4) singular spectra / effective rank / stable rank
  5) encoder-predictor Jacobian consistency
  6) latent controllable-subspace principal angles
  7) physical-vs-latent action-space principal angles
  8) action-space pullback Gram alignment: G = J^T J
  9) local goal controllability ratio
 10) contact / no-contact and object-active / inactive grouping

The 10-D coarse action u is five consecutive raw PushT actions:
    u = [a_t, ..., a_{t+4}] in R^10.

No factor readout is used for planning or for J_enc/J_pred. Physical state is
used only for evaluation/diagnostics.
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Full PushT local controllability diagnostics."
    )
    p.add_argument(
        "--policies",
        nargs="+",
        required=True,
        help=(
            "Checkpoint names/paths relative to STABLEWM_HOME, without "
            "_object.ckpt, exactly as accepted by swm.policy.AutoCostModel."
        ),
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional short labels, one per policy.",
    )
    p.add_argument(
        "--config",
        default="config/eval/pusht.yaml",
        help="PushT eval config used for dataset/img-size defaults.",
    )
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
        help=(
            "Default: $STABLEWM_HOME/pusht_controllability. "
            "Contains per-model CSV/JSON plus a comparison JSON."
        ),
    )
    p.add_argument(
        "--max-anchor-attempt-factor",
        type=int,
        default=20,
        help="How many candidate rows to inspect relative to num_anchors.",
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


def _angle_error_rad(a, b):
    d = np.asarray(a) - np.asarray(b)
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def _state_factor(state, world_size=512.0):
    """6-D physical state coordinates with a continuous theta representation."""
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
    """Encode a list/array of CHW or HWC images to [N,D]."""
    tensors = [_transform_one(transform, im) for im in images]
    outs = []
    for start in range(0, len(tensors), batch_size):
        px = torch.stack(tensors[start : start + batch_size], dim=0).to(device)
        info = {"pixels": px[:, None]}
        emb = model.encode(info)["emb"][:, 0]
        outs.append(emb.detach())
    return torch.cat(outs, dim=0)


def _normalize_action_block(block, mean, std):
    """Normalize five raw [2]-D actions then flatten to one 10-D coarse action."""
    x = (np.asarray(block, dtype=np.float32) - mean[None]) / std[None]
    return x.reshape(-1)


@torch.inference_mode()
def _predict_variants(model, z_hist, prev_blocks, variant_blocks, mean, std, device):
    """Predict z_{t+5} for variants of the current five-action block."""
    n = len(variant_blocks)
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
    hist = z_hist.expand(n, -1, -1).contiguous()
    act_emb = model.action_encoder(actions)
    pred = model.predict(hist, act_emb)[:, -1]
    return pred.detach()


def _make_variants(base_block, epsilon):
    """base,+e_0,-e_0,+e_1,-e_1,... with raw-action clipping."""
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
        variants.append(plus.reshape(base.shape))
        variants.append(minus.reshape(base.shape))
        denoms.append(denom)
    return variants, np.asarray(denoms, dtype=np.float64)


def _jacobian_from_variants(values, denoms):
    """values ordering: base,+0,-0,+1,-1,... -> [output_dim, action_dim]."""
    y = np.asarray(values, dtype=np.float64)
    cols = []
    for j, d in enumerate(denoms):
        yp = y[1 + 2 * j]
        ym = y[2 + 2 * j]
        cols.append((yp - ym) / d)
    return np.stack(cols, axis=1)


def _rollout_block(env, init_state, goal_state, action_block):
    """Reset physics, restore 7-D state, then execute five raw actions."""
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
    d = np.linalg.norm(a, "fro") * np.linalg.norm(b, "fro")
    if d <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / d)


def _gram_rel_error_trace_normalized(G1, G2):
    a = np.asarray(G1, dtype=np.float64)
    b = np.asarray(G2, dtype=np.float64)
    ta = np.trace(a)
    tb = np.trace(b)
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
    Q = U[:, :rank]
    return float(np.linalg.norm(Q.T @ g) / ng)


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
    "enc_pred_jac_rel_fro",
    "latent_subspace_mean_deg",
    "latent_subspace_max_deg",
    "action_subspace_phys_enc_mean_deg",
    "action_subspace_phys_enc_max_deg",
    "action_subspace_phys_pred_mean_deg",
    "action_subspace_phys_pred_max_deg",
    "gram_cos_phys_enc",
    "gram_cos_phys_pred",
    "gram_cos_enc_pred",
    "gram_rel_phys_enc",
    "gram_rel_phys_pred",
    "goal_controllability_enc",
    "goal_controllability_pred",
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
        raise ValueError(
            "This evaluator currently expects the trained LeWM history_size=3."
        )
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
            (r + goal_offset, t + goal_offset),
        ]
        if any(
            idx < 0
            or idx >= len(step_idx)
            or episode_idx[idx] != ep
            or int(step_idx[idx]) != expected_step
            for idx, expected_step in required
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
        raise ValueError(
            "Current LeWM PushT checkpoints were trained with frameskip/action_block=5."
        )

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(
        os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir())
    )
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
    col_name = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )

    episode_idx = np.asarray(dataset.get_col_data(col_name))
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    action = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    state = np.asarray(dataset.get_col_data("state"), dtype=np.float64)

    finite_action = action[np.isfinite(action).all(axis=1)]
    action_mean = finite_action.mean(axis=0).astype(np.float32)
    # Match train.py/get_column_normalizer (torch.std default correction=1).
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

    models = []
    labels = []
    for i, policy in enumerate(args.policies):
        label = _model_label(policy, args.labels, i)
        print(f"Loading [{label}] {policy}")
        model = swm.policy.AutoCostModel(policy).to(device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        models.append(model)
        labels.append(label)

    rows_by_model = {label: [] for label in labels}

    env = gym.make(
        str(cfg.world.env_name),
        render_mode="rgb_array",
    )

    try:
        for anchor_i, r in enumerate(anchors, start=1):
            ep = int(episode_idx[r])
            t = int(step_idx[r])
            init_state = state[r].copy()
            goal_state = state[r + args.goal_offset].copy()

            hist_rows = [
                r - 2 * args.action_block,
                r - args.action_block,
                r,
            ]
            goal_row = r + args.goal_offset
            hist_images = _load_pixels(dataset, hist_rows)
            goal_image = _load_pixels(dataset, [goal_row])[0]

            prev_blocks = [
                action[
                    r - 2 * args.action_block : r - args.action_block
                ].copy(),
                action[r - args.action_block : r].copy(),
            ]
            base_block = action[r : r + args.action_block].copy()
            variants, denoms = _make_variants(base_block, args.epsilon)

            # Simulator perturbations are shared by every model.
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

            for label, policy, model in zip(labels, args.policies, models):
                z_hist = _encode_images(model, transform, hist_images, device)[None]
                z_goal = _encode_images(model, transform, [goal_image], device)[0]
                z_final_variants = _encode_images(
                    model, transform, final_images, device
                )

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
                G_enc = _gram(J_enc)
                G_pred = _gram(J_pred)

                latent_angles = _principal_angles(
                    enc_info["U"][:, : enc_info["rank"]],
                    pred_info["U"][:, : pred_info["rank"]],
                )
                latent_angle_summary = _subspace_summary(latent_angles)

                phys_V = phys_info["Vh"][: phys_info["rank"]].T
                enc_V = enc_info["Vh"][: enc_info["rank"]].T
                pred_V = pred_info["Vh"][: pred_info["rank"]].T

                phys_enc_angles = _principal_angles(phys_V, enc_V)
                phys_pred_angles = _principal_angles(phys_V, pred_V)
                phys_enc_summary = _subspace_summary(phys_enc_angles)
                phys_pred_summary = _subspace_summary(phys_pred_angles)

                jac_den = max(np.linalg.norm(J_enc, "fro"), 1e-12)
                enc_pred_rel = float(
                    np.linalg.norm(J_pred - J_enc, "fro") / jac_den
                )

                goal_direction = (
                    z_goal.detach().cpu().numpy()
                    - z_hist[0, -1].detach().cpu().numpy()
                )
                goal_enc = _goal_projection_ratio(
                    goal_direction, enc_info["U"], enc_info["rank"]
                )
                goal_pred = _goal_projection_ratio(
                    goal_direction, pred_info["U"], pred_info["rank"]
                )

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
                    "phys_rank": phys_info["rank"],
                    "enc_rank": enc_info["rank"],
                    "pred_rank": pred_info["rank"],
                    "phys_stable_rank": phys_info["stable_rank"],
                    "enc_stable_rank": enc_info["stable_rank"],
                    "pred_stable_rank": pred_info["stable_rank"],
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
                    "latent_subspace_mean_deg": latent_angle_summary["mean_deg"],
                    "latent_subspace_max_deg": latent_angle_summary["max_deg"],
                    "action_subspace_phys_enc_mean_deg": phys_enc_summary["mean_deg"],
                    "action_subspace_phys_enc_max_deg": phys_enc_summary["max_deg"],
                    "action_subspace_phys_pred_mean_deg": phys_pred_summary["mean_deg"],
                    "action_subspace_phys_pred_max_deg": phys_pred_summary["max_deg"],
                    "gram_cos_phys_enc": _gram_cosine(G_phys, G_enc),
                    "gram_cos_phys_pred": _gram_cosine(G_phys, G_pred),
                    "gram_cos_enc_pred": _gram_cosine(G_enc, G_pred),
                    "gram_rel_phys_enc": _gram_rel_error_trace_normalized(
                        G_phys, G_enc
                    ),
                    "gram_rel_phys_pred": _gram_rel_error_trace_normalized(
                        G_phys, G_pred
                    ),
                    "goal_controllability_enc": goal_enc,
                    "goal_controllability_pred": goal_pred,
                }

                for j, sval in enumerate(phys_info["s"]):
                    row[f"phys_s{j+1}"] = float(sval)
                for j, sval in enumerate(enc_info["s"]):
                    row[f"enc_s{j+1}"] = float(sval)
                for j, sval in enumerate(pred_info["s"]):
                    row[f"pred_s{j+1}"] = float(sval)

                rows_by_model[label].append(row)

            print(
                f"[{anchor_i:03d}/{len(anchors):03d}] "
                f"row={r} ep={ep} step={t} "
                f"contact={base_contact} active={object_active} "
                f"block_move={block_motion_px:.2f}px "
                f"theta_move={theta_motion_deg:.2f}deg"
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
            "note": (
                "All Jacobians are finite differences with respect to the SAME "
                "raw 10-D action block, so J^T J lives in a common action space. "
                "Ground-truth state is used only for diagnostics."
            ),
        },
        "models": {},
    }

    for label, policy in zip(labels, args.policies):
        rows = rows_by_model[label]
        summary = _group_summary(rows)
        comparison["models"][label] = {
            "policy": policy,
            "summary": summary,
        }

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
    print(f"\nSaved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
