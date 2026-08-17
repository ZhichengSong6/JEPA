#!/usr/bin/env python3
"""
Equal-budget directional controllability diagnostic for official/full PushT LeWM.

For each paired evaluation anchor, use the dataset expert future action sequence U0
as a nominal finite-horizon plan. Sample random directions v_i in the FULL raw
action-sequence space and construct paired candidates

    U_i^+ = U0 + r v_i
    U_i^- = U0 - r v_i

with ||v_i||_2 = 1, so both candidates have the same raw action-sequence L2
perturbation radius r around U0. The same base direction v_i is reused for every
requested radius. Directions are sampled so the largest requested radius remains
inside the environment action bounds [-1, 1]; smaller radii therefore remain
feasible automatically.

IMPORTANT: "equal budget" here means equal L2 norm of the RAW ACTION-SEQUENCE
PERTURBATION around the expert plan. It is not a claim of equal torque, energy,
or physical work.

For every candidate, compare:

  C_phys(U): real simulator terminal task cost,
  C_enc(U):  raw Euclidean latent goal cost after encoding the REAL terminal image,
  C_pred(U): raw Euclidean latent goal cost after LeWM predictor rollout.

No factor head/readout is used for model scoring. Physical state is used only for
oracle diagnostics. The planner-side model cost remains exactly raw terminal
squared Euclidean latent distance.

Primary questions:

1. Fixed-radius ranking:
   Among candidates with exactly the same ||U-U0||_2 = r, can the latent cost
   rank which action directions produce better physical goal progress?

2. Opposite-direction accuracy:
   For each v_i, physics chooses between U0+r v_i and U0-r v_i. Does the latent
   cost choose the same sign? This is a direct finite-horizon directional test
   and does not require latent gradients.

3. Physical <-> Encoder <-> Predictor:
   If the real encoded terminal geometry gets the direction right but the
   predictor does not, the failure is dynamics. If the encoder itself gets it
   wrong, the failure is representation/goal geometry.

Defaults intentionally stay near the expert regime:
    radii = [0.35, 0.70, 1.40]
These approximately match the observed 50-D perturbation norms produced by the
previous sigma=[0.05,0.10,0.20] experiment. sigma=0.4 / uniform-random regimes
are intentionally excluded from the primary test to reduce OOD confounding.

Outputs:
  anchor_radius_metrics.csv
      one row per anchor x radius x model
  candidate_metrics.npz
      all candidate actions, physical outcomes, model costs, and metadata
  summary.json
      grouped aggregate metrics and experiment metadata
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
        description=(
            "Full-PushT equal-norm, opposite-direction Physical/Encoder/Predictor "
            "controllability diagnostic."
        )
    )
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--num-anchors", type=int, default=50)
    p.add_argument(
        "--radii",
        nargs="+",
        type=float,
        default=[0.35, 0.70, 1.40],
        help=(
            "Exact L2 radii in the flattened raw action-sequence space around "
            "the expert future plan."
        ),
    )
    p.add_argument(
        "--num-directions",
        type=int,
        default=32,
        help="Number of +/- direction pairs per radius and anchor.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument(
        "--pair-margin-frac",
        type=float,
        default=0.02,
        help=(
            "Ignore nearly tied physical candidate pairs whose cost difference "
            "is <= this fraction of the fixed-radius physical-cost IQR."
        ),
    )
    p.add_argument(
        "--direction-margin-frac",
        type=float,
        default=0.02,
        help=(
            "For +/- directional accuracy, ignore physics pairs whose |C+ - C-| "
            "is <= this fraction of the fixed-radius physical-cost IQR."
        ),
    )
    p.add_argument("--replay-factor-good-threshold", type=float, default=0.10)
    p.add_argument("--object-motion-px", type=float, default=1.0)
    p.add_argument("--object-motion-deg", type=float, default=1.0)
    p.add_argument("--world-size", type=float, default=512.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: $STABLEWM_HOME/pusht_directional_controllability",
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
    """Diagnostic state coordinate only; never used as the model planning cost."""
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
    finite_ref = ref[np.isfinite(ref)]
    if len(finite_ref) < 2:
        return float("nan"), 0, float("nan")
    q25, q75 = np.percentile(finite_ref, [25, 75])
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
    acc = float(correct / count) if count > 0 else float("nan")
    return acc, int(count), margin


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
        "oracle_local_idx": oracle_idx,
        "selected_local_idx": selected_idx,
        "oracle_physical_cost": oracle,
        "selected_physical_cost": selected,
        "selection_regret": float(regret),
        "selection_regret_norm": float(norm_regret),
        "selected_physical_percentile": _selected_physical_percentile(
            p, selected_idx
        ),
    }


def _numeric_summary(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
        }
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
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")
    if num_anchors > len(valid_indices) - 1:
        raise ValueError("Requested more anchors than official valid starting points.")
    g = np.random.default_rng(seed)
    # Intentionally reproduces the official evaluator's current selection logic.
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
        if (
            j >= len(step_idx)
            or episode_idx[j] != ep
            or int(step_idx[j]) != t + d
        ):
            raise RuntimeError(
                f"Dataset is not contiguous at row={row}, offset={d}; "
                "cannot form the exact raw-action horizon."
            )


def _bounded_direction_at_radius(slack, radius, rng):
    """
    Sample delta with ||delta||_2 == radius and |delta_j| <= slack_j.

    We draw a random Gaussian orientation, then find alpha such that
        |delta_j| = min(alpha * |g_j|, slack_j)
    lies exactly on the requested L2 sphere. Random signs are applied after the
    bounded magnitudes are found. This guarantees BOTH expert+delta and
    expert-delta stay within [-1,1] when slack = 1 - |expert|.
    """
    slack = np.asarray(slack, dtype=np.float64).reshape(-1)
    radius = float(radius)
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    max_norm = float(np.linalg.norm(slack))
    if max_norm + 1e-10 < radius:
        raise ValueError(
            f"Exact equal-norm radius {radius:.6f} is infeasible around this "
            f"expert sequence; maximum symmetric feasible norm is {max_norm:.6f}."
        )

    g = np.abs(rng.normal(size=slack.shape))
    g = np.maximum(g, 1e-12)
    signs = rng.choice(np.array([-1.0, 1.0]), size=slack.shape)

    def norm_at(alpha):
        return float(np.linalg.norm(np.minimum(alpha * g, slack)))

    lo, hi = 0.0, 1.0
    while norm_at(hi) < radius:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Failed to bracket equal-norm direction scale.")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if norm_at(mid) < radius:
            lo = mid
        else:
            hi = mid

    mag = np.minimum(hi * g, slack)
    delta = signs * mag
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or abs(norm - radius) > 1e-8:
        raise RuntimeError(
            f"Equal-norm construction failed: target={radius}, actual={norm}."
        )
    return delta


def _make_equal_norm_candidates(expert_actions, radii, num_directions, rng):
    """
    Candidate 0 is the exact expert plan for replay sanity.

    The remaining candidates are ordered by radius, then direction, then sign:
        +v, -v
    for every direction. The SAME v is reused across all radii.

    Directions are constructed at max(radii), which guarantees that scaling them
    down to every smaller radius preserves both action bounds and direction.
    """
    expert = np.nan_to_num(
        np.asarray(expert_actions, dtype=np.float64),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    expert = np.clip(expert, -1.0, 1.0)
    radii = np.asarray(sorted(set(map(float, radii))), dtype=np.float64)
    if len(radii) == 0 or np.any(radii <= 0):
        raise ValueError("--radii must contain positive values.")
    if int(num_directions) <= 0:
        raise ValueError("--num-directions must be positive.")

    flat = expert.reshape(-1)
    slack = np.maximum(1.0 - np.abs(flat), 0.0)
    rmax = float(np.max(radii))
    if np.linalg.norm(slack) + 1e-10 < rmax:
        raise ValueError(
            f"Requested max radius {rmax:.6f} is infeasible around this expert "
            f"sequence (symmetric feasible norm={np.linalg.norm(slack):.6f})."
        )

    directions = np.empty((int(num_directions), flat.size), dtype=np.float64)
    direction_saturated_frac = np.empty(int(num_directions), dtype=np.float64)
    for di in range(int(num_directions)):
        delta_max = _bounded_direction_at_radius(slack, rmax, rng)
        directions[di] = delta_max / rmax
        direction_saturated_frac[di] = float(
            np.mean(np.isclose(np.abs(delta_max), slack, atol=1e-8))
        )

    n_candidates = 1 + len(radii) * int(num_directions) * 2
    candidates = np.empty((n_candidates, flat.size), dtype=np.float64)
    radius_meta = np.full(n_candidates, np.nan, dtype=np.float64)
    direction_meta = np.full(n_candidates, -1, dtype=np.int32)
    sign_meta = np.zeros(n_candidates, dtype=np.int8)
    perturb_norm = np.zeros(n_candidates, dtype=np.float64)

    candidates[0] = flat
    ci = 1
    for radius in radii:
        for di in range(int(num_directions)):
            v = directions[di]
            for sign in (+1, -1):
                delta = float(sign) * float(radius) * v
                cand = flat + delta
                if np.max(np.abs(cand)) > 1.0 + 1e-7:
                    raise RuntimeError(
                        "Equal-norm candidate exceeded raw action bounds; this "
                        "should be impossible after symmetric feasibility sampling."
                    )
                candidates[ci] = cand
                radius_meta[ci] = radius
                direction_meta[ci] = di
                sign_meta[ci] = sign
                perturb_norm[ci] = np.linalg.norm(delta)
                ci += 1

    candidates = candidates.reshape(
        n_candidates, expert.shape[0], expert.shape[1]
    ).astype(np.float32)
    directions = directions.reshape(
        int(num_directions), expert.shape[0], expert.shape[1]
    ).astype(np.float32)

    # Recompute actual stored float32 radii. They should be equal to requested
    # radii to floating-point precision and are saved for auditing.
    expert32 = candidates[0].astype(np.float64)
    actual_norm = np.linalg.norm(
        candidates.astype(np.float64).reshape(n_candidates, -1)
        - expert32.reshape(1, -1),
        axis=1,
    )
    requested = np.nan_to_num(radius_meta, nan=0.0)
    max_radius_error = float(np.max(np.abs(actual_norm - requested)))
    if max_radius_error > 5e-6:
        raise RuntimeError(
            f"Stored float32 candidates violate equal-radius tolerance: "
            f"max error={max_radius_error:.3e}."
        )

    return {
        "candidates": candidates,
        "radii": radii.astype(np.float32),
        "candidate_radius": radius_meta.astype(np.float32),
        "candidate_direction": direction_meta,
        "candidate_sign": sign_meta,
        "candidate_perturb_norm": actual_norm.astype(np.float32),
        "directions": directions,
        "direction_saturated_fraction_at_rmax": direction_saturated_frac.astype(
            np.float32
        ),
        "max_equal_norm_abs_error": max_radius_error,
    }


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
        px = _transform_batch(transform, images[start : start + batch_size]).to(
            device
        )
        z = model.encode({"pixels": px[:, None].float()})["emb"][:, 0]
        outs.append(z.detach())
    return torch.cat(outs, dim=0)


def _normalize_candidate_actions(
    raw_candidates, action_scaler, horizon, action_block
):
    raw = np.asarray(raw_candidates, dtype=np.float32)
    k = raw.shape[0]
    expected = int(horizon) * int(action_block)
    if raw.shape[1:] != (expected, 2):
        raise RuntimeError(
            f"Expected candidate actions {(k, expected, 2)}, got {raw.shape}."
        )
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
        a = torch.from_numpy(
            normalized_actions[start : start + batch_size]
        ).to(device=device, dtype=torch.float32)
        s = a.shape[0]
        action_sequence = a.unsqueeze(0)
        info = {
            "pixels": current_px[None, None, None].expand(
                1, 1, 1, -1, -1, -1
            )
        }
        rolled = model.rollout(info, action_sequence)
        z = rolled["predicted_emb"][0, :s, -1]
        outs.append(z.detach())
    return torch.cat(outs, dim=0)


def _directional_metrics(
    physical_cost,
    score_cost,
    candidate_direction,
    candidate_sign,
    direction_margin_frac,
):
    p = np.asarray(physical_cost, dtype=np.float64)
    s = np.asarray(score_cost, dtype=np.float64)
    d = np.asarray(candidate_direction, dtype=np.int32)
    sg = np.asarray(candidate_sign, dtype=np.int8)

    q25, q75 = np.percentile(p[np.isfinite(p)], [25, 75])
    margin = float(direction_margin_frac) * max(float(q75 - q25), 1e-12)

    delta_phys = []
    delta_score = []
    correct = 0.0
    informative = 0
    total = 0
    for di in np.unique(d[d >= 0]):
        plus = np.nonzero((d == di) & (sg == 1))[0]
        minus = np.nonzero((d == di) & (sg == -1))[0]
        if len(plus) != 1 or len(minus) != 1:
            raise RuntimeError(f"Malformed +/- pair for direction {di}.")
        ip, im = int(plus[0]), int(minus[0])
        dp = float(p[ip] - p[im])
        ds = float(s[ip] - s[im])
        delta_phys.append(dp)
        delta_score.append(ds)
        total += 1
        if abs(dp) <= margin:
            continue
        informative += 1
        if abs(ds) <= 1e-12:
            correct += 0.5
        elif np.sign(dp) == np.sign(ds):
            correct += 1.0

    delta_phys = np.asarray(delta_phys, dtype=np.float64)
    delta_score = np.asarray(delta_score, dtype=np.float64)
    return {
        "direction_accuracy": (
            float(correct / informative) if informative > 0 else float("nan")
        ),
        "direction_informative_pairs": int(informative),
        "direction_total_pairs": int(total),
        "direction_informative_fraction": (
            float(informative / total) if total > 0 else float("nan")
        ),
        "direction_margin": float(margin),
        "direction_delta_spearman": _spearman(delta_score, delta_phys),
        "direction_mean_abs_physical_delta": float(
            np.mean(np.abs(delta_phys))
        ),
        "direction_median_abs_physical_delta": float(
            np.median(np.abs(delta_phys))
        ),
    }


def _fixed_radius_model_metrics(
    physical_cost,
    pusher_err,
    block_err,
    joint_err,
    theta_err,
    enc_cost,
    pred_cost,
    terminal_pred_enc_mse,
    candidate_direction,
    candidate_sign,
    pair_margin_frac,
    direction_margin_frac,
):
    acc_pred, n_pair_pred, pair_margin = _pairwise_accuracy(
        physical_cost, pred_cost, pair_margin_frac
    )
    acc_enc, n_pair_enc, _ = _pairwise_accuracy(
        physical_cost, enc_cost, pair_margin_frac
    )
    pred_sel = _selection_stats(physical_cost, pred_cost)
    enc_sel = _selection_stats(physical_cost, enc_cost)

    pred_dir = _directional_metrics(
        physical_cost,
        pred_cost,
        candidate_direction,
        candidate_sign,
        direction_margin_frac,
    )
    enc_dir = _directional_metrics(
        physical_cost,
        enc_cost,
        candidate_direction,
        candidate_sign,
        direction_margin_frac,
    )

    pred_idx = pred_sel["selected_local_idx"]
    enc_idx = enc_sel["selected_local_idx"]
    oracle_idx = pred_sel["oracle_local_idx"]

    return {
        "spearman_pred_phys": _spearman(pred_cost, physical_cost),
        "spearman_enc_phys": _spearman(enc_cost, physical_cost),
        "spearman_pred_enc": _spearman(pred_cost, enc_cost),
        "pair_acc_pred_phys": acc_pred,
        "pair_acc_enc_phys": acc_enc,
        "pair_count_pred_phys": n_pair_pred,
        "pair_count_enc_phys": n_pair_enc,
        "pair_margin_physical_cost": float(pair_margin),
        "direction_acc_pred_phys": pred_dir["direction_accuracy"],
        "direction_acc_enc_phys": enc_dir["direction_accuracy"],
        "direction_delta_spearman_pred_phys": pred_dir[
            "direction_delta_spearman"
        ],
        "direction_delta_spearman_enc_phys": enc_dir[
            "direction_delta_spearman"
        ],
        "direction_informative_pairs": pred_dir[
            "direction_informative_pairs"
        ],
        "direction_total_pairs": pred_dir["direction_total_pairs"],
        "direction_informative_fraction": pred_dir[
            "direction_informative_fraction"
        ],
        "direction_margin_physical_cost": pred_dir["direction_margin"],
        "direction_mean_abs_physical_delta": pred_dir[
            "direction_mean_abs_physical_delta"
        ],
        "direction_median_abs_physical_delta": pred_dir[
            "direction_median_abs_physical_delta"
        ],
        "mean_terminal_pred_enc_mse": float(np.mean(terminal_pred_enc_mse)),
        "median_terminal_pred_enc_mse": float(
            np.median(terminal_pred_enc_mse)
        ),
        "pred_selection_regret": pred_sel["selection_regret"],
        "pred_selection_regret_norm": pred_sel["selection_regret_norm"],
        "pred_selected_physical_percentile": pred_sel[
            "selected_physical_percentile"
        ],
        "enc_selection_regret": enc_sel["selection_regret"],
        "enc_selection_regret_norm": enc_sel["selection_regret_norm"],
        "enc_selected_physical_percentile": enc_sel[
            "selected_physical_percentile"
        ],
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
        "pred_selected_theta_error_deg": float(
            np.degrees(theta_err[pred_idx])
        ),
        "enc_selected_pusher_error_px": float(pusher_err[enc_idx]),
        "enc_selected_block_error_px": float(block_err[enc_idx]),
        "enc_selected_joint_error_px": float(joint_err[enc_idx]),
        "enc_selected_theta_error_deg": float(
            np.degrees(theta_err[enc_idx])
        ),
    }


SUMMARY_METRICS = [
    "spearman_pred_phys",
    "spearman_enc_phys",
    "spearman_pred_enc",
    "pair_acc_pred_phys",
    "pair_acc_enc_phys",
    "direction_acc_pred_phys",
    "direction_acc_enc_phys",
    "direction_delta_spearman_pred_phys",
    "direction_delta_spearman_enc_phys",
    "direction_informative_pairs",
    "direction_total_pairs",
    "direction_informative_fraction",
    "direction_mean_abs_physical_delta",
    "direction_median_abs_physical_delta",
    "pred_selection_regret",
    "pred_selection_regret_norm",
    "pred_selected_physical_percentile",
    "enc_selection_regret",
    "enc_selection_regret_norm",
    "enc_selected_physical_percentile",
    "mean_terminal_pred_enc_mse",
    "median_terminal_pred_enc_mse",
    "pred_selected_pusher_error_px",
    "pred_selected_block_error_px",
    "pred_selected_joint_error_px",
    "pred_selected_theta_error_deg",
    "enc_selected_pusher_error_px",
    "enc_selected_block_error_px",
    "enc_selected_joint_error_px",
    "enc_selected_theta_error_deg",
    "oracle_pusher_error_px",
    "oracle_block_error_px",
    "oracle_joint_error_px",
    "oracle_theta_error_deg",
    "replay_factor_error",
    "replay_joint_error_px",
    "replay_theta_error_deg",
    "candidate_contact_fraction",
    "candidate_bound_fraction",
    "max_equal_norm_abs_error",
    "direction_saturated_fraction_at_rmax",
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
        "expert_no_contact": [
            r for r in rows if not r["expert_had_contact"]
        ],
        "object_active": [r for r in rows if r["expert_object_active"]],
        "object_inactive": [
            r for r in rows if not r["expert_object_active"]
        ],
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


def _print_radius_table(summary, radius, group="all"):
    radius_key = f"{float(radius):.6g}"
    print(
        f"\n===== EQUAL-NORM DIRECTIONAL TEST: r={float(radius):.3f} "
        f"group={group} ====="
    )
    header = (
        f"{'model':<14} {'rhoP':>7} {'rhoE':>7} {'rhoPE':>7} "
        f"{'dirP':>7} {'dirE':>7} {'dRhoP':>7} {'accP':>7} "
        f"{'pctP':>7} {'info':>7} {'bound':>7}"
    )
    print(header)
    print("-" * len(header))
    for label, payload in summary["models"].items():
        s = payload["by_radius"][radius_key][group]
        print(
            f"{label:<14} "
            f"{_fmt(_mean(s, 'spearman_pred_phys')):>7} "
            f"{_fmt(_mean(s, 'spearman_enc_phys')):>7} "
            f"{_fmt(_mean(s, 'spearman_pred_enc')):>7} "
            f"{_fmt(_mean(s, 'direction_acc_pred_phys')):>7} "
            f"{_fmt(_mean(s, 'direction_acc_enc_phys')):>7} "
            f"{_fmt(_mean(s, 'direction_delta_spearman_pred_phys')):>7} "
            f"{_fmt(_mean(s, 'pair_acc_pred_phys')):>7} "
            f"{_fmt(_mean(s, 'pred_selected_physical_percentile')):>7} "
            f"{_fmt(_mean(s, 'direction_informative_fraction')):>7} "
            f"{_fmt(_mean(s, 'candidate_bound_fraction')):>7}"
        )
    print(
        "rhoP/rhoE/rhoPE: fixed-radius Spearman for Predictor-Physical / "
        "Encoder-Physical / Predictor-Encoder.\n"
        "dirP/dirE: +/- opposite-direction accuracy on physically informative "
        "pairs (0.5=random).\n"
        "dRhoP: Spearman between (C_pred+ - C_pred-) and "
        "(C_phys+ - C_phys-) across directions.\n"
        "accP: fixed-radius all-pairs ranking accuracy; pctP: physical percentile "
        "of predictor-selected candidate (0=best).\n"
        "info: fraction of +/- pairs with a non-negligible physical preference; "
        "bound: fraction of raw candidate action scalars at |u|~=1."
    )


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels must have the same length as --policies.")
    if args.num_anchors <= 0:
        raise ValueError("--num-anchors must be positive.")
    if args.num_directions <= 0:
        raise ValueError("--num-directions must be positive.")
    if args.horizon <= 0 or args.action_block <= 0:
        raise ValueError("--horizon and --action-block must be positive.")
    if args.goal_offset != args.horizon * args.action_block:
        raise ValueError(
            "For this diagnostic, --goal-offset must equal horizon*action_block."
        )
    radii = np.asarray(sorted(set(map(float, args.radii))), dtype=np.float64)
    if len(radii) == 0 or np.any(radii <= 0):
        raise ValueError("--radii must contain positive values.")

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(
        os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir())
    )
    output_root = (
        Path(args.output_dir)
        if args.output_dir is not None
        else cache_root / "pusht_directional_controllability"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=["action", "state"],
        cache_dir=cache_root,
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
    nR = len(radii)
    nD = int(args.num_directions)
    nC = 1 + nR * nD * 2
    raw_horizon = int(args.horizon) * int(args.action_block)
    nM = len(models)

    candidate_actions = np.empty(
        (nA, nC, raw_horizon, 2), dtype=np.float32
    )
    candidate_radius = np.full((nA, nC), np.nan, dtype=np.float32)
    candidate_direction = np.full((nA, nC), -1, dtype=np.int32)
    candidate_sign = np.zeros((nA, nC), dtype=np.int8)
    candidate_perturb_norm = np.zeros((nA, nC), dtype=np.float32)
    direction_vectors = np.empty(
        (nA, nD, raw_horizon, 2), dtype=np.float32
    )
    direction_saturated_fraction = np.empty((nA, nD), dtype=np.float32)
    max_equal_norm_error = np.empty(nA, dtype=np.float64)

    final_states = np.empty((nA, nC, 7), dtype=np.float64)
    candidate_had_contact = np.empty((nA, nC), dtype=bool)
    candidate_contact_steps = np.empty((nA, nC), dtype=np.int16)
    physical_cost = np.empty((nA, nC), dtype=np.float64)
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
            cand = _make_equal_norm_candidates(
                expert_actions, radii, nD, rng
            )
            candidates = cand["candidates"]
            if len(candidates) != nC:
                raise RuntimeError("Unexpected candidate count.")
            candidate_actions[ai] = candidates
            candidate_radius[ai] = cand["candidate_radius"]
            candidate_direction[ai] = cand["candidate_direction"]
            candidate_sign[ai] = cand["candidate_sign"]
            candidate_perturb_norm[ai] = cand["candidate_perturb_norm"]
            direction_vectors[ai] = cand["directions"]
            direction_saturated_fraction[ai] = cand[
                "direction_saturated_fraction_at_rmax"
            ]
            max_equal_norm_error[ai] = cand["max_equal_norm_abs_error"]

            controlled_seed = int(args.env_seed + ai)
            current_image, goal_image = _controlled_current_goal_images(
                env, init_state, goal_state, controlled_seed
            )

            final_images = []
            for ci in range(nC):
                fs, fi, hc, cs = _rollout_candidate(
                    env,
                    init_state,
                    goal_state,
                    candidates[ci],
                    controlled_seed,
                )
                final_states[ai, ci] = fs
                final_images.append(fi)
                candidate_had_contact[ai, ci] = hc
                candidate_contact_steps[ai, ci] = cs

            (
                physical_cost[ai],
                pusher_error[ai],
                block_error[ai],
                joint_error[ai],
                theta_error_rad[ai],
            ) = _primary_physical_cost(final_states[ai], goal_state)

            # Candidate 0 is exactly the expert future sequence and is used only
            # for replay sanity / anchor classification, not fixed-radius ranking.
            replay_factor_error = float(
                np.linalg.norm(
                    _state_factor(final_states[ai, 0], args.world_size)
                    - _state_factor(goal_state, args.world_size)
                )
            )
            replay_joint_error = float(joint_error[ai, 0])
            replay_theta_deg = float(
                np.degrees(theta_error_rad[ai, 0])
            )
            replay_good = bool(
                replay_factor_error <= args.replay_factor_good_threshold
            )
            expert_had_contact = bool(candidate_had_contact[ai, 0])
            expert_block_motion = float(
                np.linalg.norm(
                    final_states[ai, 0, 2:4] - init_state[2:4]
                )
            )
            expert_theta_motion_deg = float(
                np.degrees(
                    _angle_error_rad(
                        final_states[ai, 0, 4], init_state[4]
                    )
                )
            )
            expert_object_active = bool(
                expert_block_motion >= args.object_motion_px
                or expert_theta_motion_deg >= args.object_motion_deg
            )

            normalized_actions = _normalize_candidate_actions(
                candidates,
                action_scaler,
                args.horizon,
                args.action_block,
            )

            for mi, (label, policy, model) in enumerate(
                zip(labels, args.policies, models)
            ):
                z_goal = _encode_images(
                    model,
                    transform,
                    [goal_image],
                    device,
                    args.model_batch_size,
                )[0]
                z_real = _encode_images(
                    model,
                    transform,
                    final_images,
                    device,
                    args.model_batch_size,
                )
                z_pred = _predict_terminal_latents(
                    model,
                    transform,
                    current_image,
                    normalized_actions,
                    device,
                    args.model_batch_size,
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

                for radius in radii:
                    mask = np.isclose(
                        candidate_radius[ai],
                        radius,
                        rtol=0.0,
                        atol=1e-6,
                    )
                    idx = np.nonzero(mask)[0]
                    if len(idx) != 2 * nD:
                        raise RuntimeError(
                            f"Expected {2*nD} candidates at radius {radius}, "
                            f"got {len(idx)}."
                        )

                    metrics = _fixed_radius_model_metrics(
                        physical_cost[ai, idx],
                        pusher_error[ai, idx],
                        block_error[ai, idx],
                        joint_error[ai, idx],
                        theta_error_rad[ai, idx],
                        enc_np[idx],
                        pred_np[idx],
                        dyn_np[idx],
                        candidate_direction[ai, idx],
                        candidate_sign[ai, idx],
                        args.pair_margin_frac,
                        args.direction_margin_frac,
                    )
                    candidate_contact_fraction = float(
                        candidate_had_contact[ai, idx].mean()
                    )
                    candidate_bound_fraction = float(
                        np.mean(np.abs(candidate_actions[ai, idx]) >= 0.999999)
                    )
                    actual_radius = candidate_perturb_norm[ai, idx]
                    radius_error = float(
                        np.max(np.abs(actual_radius - float(radius)))
                    )

                    row = {
                        "model": label,
                        "policy": policy,
                        "anchor_index": ai + 1,
                        "dataset_row": int(r),
                        "episode_idx": ep,
                        "step_idx": t,
                        "goal_dataset_row": int(goal_r),
                        "radius": float(radius),
                        "num_directions": nD,
                        "num_radius_candidates": int(len(idx)),
                        "horizon_coarse": int(args.horizon),
                        "action_block": int(args.action_block),
                        "raw_horizon": raw_horizon,
                        "expert_has_nan": expert_has_nan,
                        "expert_had_contact": expert_had_contact,
                        "candidate_contact_fraction": candidate_contact_fraction,
                        "candidate_bound_fraction": candidate_bound_fraction,
                        "expert_object_active": expert_object_active,
                        "expert_block_motion_px": expert_block_motion,
                        "expert_theta_motion_deg": expert_theta_motion_deg,
                        "replay_factor_error": replay_factor_error,
                        "replay_joint_error_px": replay_joint_error,
                        "replay_theta_error_deg": replay_theta_deg,
                        "replay_good": replay_good,
                        "max_equal_norm_abs_error": radius_error,
                        "direction_saturated_fraction_at_rmax": float(
                            np.mean(direction_saturated_fraction[ai])
                        ),
                        **metrics,
                    }
                    rows_by_model[label].append(row)

            elapsed_anchor = time.time() - anchor_start
            done = ai + 1
            elapsed_total = time.time() - start_total
            eta = elapsed_total / done * (nA - done)
            bound_rmax = float(
                np.mean(
                    np.abs(
                        candidate_actions[
                            ai,
                            np.isclose(
                                candidate_radius[ai],
                                float(np.max(radii)),
                                rtol=0.0,
                                atol=1e-6,
                            ),
                        ]
                    )
                    >= 0.999999
                )
            )
            print(
                f"anchor {done:3d}/{nA}: row={r} "
                f"contact={expert_had_contact} active={expert_object_active} "
                f"replay={replay_factor_error:.4f} "
                f"eqerr={max_equal_norm_error[ai]:.1e} "
                f"bound@rmax={bound_rmax:.3f} "
                f"time={elapsed_anchor:.2f}s ETA={eta/60:.1f}min"
            )
    finally:
        env.close()

    all_rows = [
        row for label in labels for row in rows_by_model[label]
    ]
    csv_path = output_root / "anchor_radius_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(all_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_rows)

    npz_path = output_root / "candidate_metrics.npz"
    np.savez_compressed(
        npz_path,
        anchors=anchors,
        radii=radii.astype(np.float32),
        labels=np.asarray(labels, dtype=object),
        policies=np.asarray(args.policies, dtype=object),
        candidate_actions=candidate_actions,
        candidate_radius=candidate_radius,
        candidate_direction=candidate_direction,
        candidate_sign=candidate_sign,
        candidate_perturb_norm=candidate_perturb_norm,
        direction_vectors=direction_vectors,
        direction_saturated_fraction_at_rmax=direction_saturated_fraction,
        max_equal_norm_abs_error=max_equal_norm_error,
        final_states=final_states,
        candidate_had_contact=candidate_had_contact,
        candidate_contact_steps=candidate_contact_steps,
        physical_cost=physical_cost,
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
            "num_directions_per_radius": nD,
            "num_candidates_per_anchor_including_expert": nC,
            "radii": radii.tolist(),
            "seed": int(args.seed),
            "env_seed": int(args.env_seed),
            "horizon": int(args.horizon),
            "action_block": int(args.action_block),
            "raw_horizon": raw_horizon,
            "flattened_action_dimension": raw_horizon * 2,
            "goal_offset": int(args.goal_offset),
            "pair_margin_frac": float(args.pair_margin_frac),
            "direction_margin_frac": float(args.direction_margin_frac),
            "replay_factor_good_threshold": float(
                args.replay_factor_good_threshold
            ),
            "equal_budget_definition": (
                "Exact L2 norm of flattened raw action-sequence perturbation "
                "around the dataset expert future plan; not torque/energy/work."
            ),
            "direction_sampling": (
                "Random directions are made symmetrically feasible at max(radii) "
                "under raw Box[-1,1] bounds, then the same unit directions are "
                "reused at every smaller radius."
            ),
            "candidate_zero": (
                "Candidate index 0 is the exact expert future plan and is used "
                "only for replay sanity/anchor classification; it is excluded "
                "from every fixed-radius ranking metric."
            ),
            "physical_primary_cost": (
                "(joint_position_error_px/20)^2 + "
                "(wrapped_theta_error_rad/(pi/9))^2"
            ),
            "model_cost": (
                "raw terminal squared Euclidean latent distance; no factor head, "
                "readout, or ground-truth state enters model scoring"
            ),
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
        model_rows = rows_by_model[label]
        by_radius = {}
        for radius in radii:
            radius_key = f"{float(radius):.6g}"
            rr = [
                row
                for row in model_rows
                if abs(float(row["radius"]) - float(radius)) < 1e-8
            ]
            by_radius[radius_key] = _group_rows(rr)
        summary["models"][label] = {
            "policy": policy,
            "by_radius": by_radius,
        }

    summary_path = output_root / "summary.json"
    with summary_path.open("w") as f:
        json.dump(_jsonable(summary), f, indent=2)

    for radius in radii:
        _print_radius_table(summary, radius, "all")
    if any(
        row["expert_had_contact"]
        for row in rows_by_model[labels[0]]
    ):
        for radius in radii:
            _print_radius_table(summary, radius, "expert_contact")

    print(
        f"\nMax equal-norm absolute error across all anchors: "
        f"{np.max(max_equal_norm_error):.3e}"
    )
    print(f"Elapsed: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
