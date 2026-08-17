#!/usr/bin/env python3
"""
Horizon-conditioned equal-budget directional diagnostic for full PushT LeWM.

Scientific question
-------------------
Does action-direction fidelity fail already at a short prediction horizon, or is
it mainly lost through multi-step autoregressive rollout?

For one fixed current state and ONE FIXED final task goal (default dataset state
at t+25 raw steps), evaluate several predictor horizons H in coarse LeWM steps.
For each H, the nominal plan U0,H is the corresponding prefix of the dataset
expert future actions (H * action_block raw 2-D actions). We construct paired
candidates

    U+ = U0,H + r_H v
    U- = U0,H - r_H v

with exact equal L2 perturbation norm and symmetric action-bound feasibility.

IMPORTANT radius control
------------------------
The action-sequence dimensionality changes with H. Holding total L2 radius fixed
would make a perturbation much larger per action scalar at H=1 than at H=5.
Therefore the default is RMS-matched scaling

    r_H = r_ref * sqrt(H / H_ref)

with H_ref=5. This keeps r_H / sqrt(2*action_block*H) constant, i.e. the same
RMS perturbation per raw action scalar across horizons. Use --radius-scaling
fixed only as an explicit ablation.

The final TASK GOAL is fixed across all H (default t+25), so only the prediction
horizon changes. Replay sanity for each H is checked against the DATASET STATE
at that horizon t + H*action_block, not against the final task goal.

For every candidate:
  C_phys : real simulator terminal task cost to the fixed final goal
  C_enc  : raw Euclidean latent cost of the REAL terminal image to fixed goal
  C_pred : raw Euclidean latent cost of the predicted terminal latent to goal

No factor head/readout is used for model scoring. Ground-truth state is oracle
only. The model-side planning cost remains raw terminal squared Euclidean latent
distance.

Outputs (single run):
  anchor_horizon_radius_metrics.csv
  candidate_metrics.npz
  summary.json
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
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--num-anchors", type=int, default=50)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 5])
    p.add_argument("--reference-horizon", type=int, default=5)
    p.add_argument(
        "--radii", nargs="+", type=float, default=[0.35, 0.70, 1.40],
        help="Reference-horizon exact flattened raw-action L2 radii.",
    )
    p.add_argument(
        "--radius-scaling", choices=["rms", "fixed"], default="rms",
        help="rms: r_H=r_ref*sqrt(H/H_ref); fixed: use r_ref at every H.",
    )
    p.add_argument("--num-directions", type=int, default=32)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument(
        "--goal-offset", type=int, default=25,
        help="Fixed final task goal offset in RAW dataset steps for every H.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--pair-margin-frac", type=float, default=0.02)
    p.add_argument("--direction-margin-frac", type=float, default=0.02)
    p.add_argument("--replay-factor-good-threshold", type=float, default=0.10)
    p.add_argument("--object-motion-px", type=float, default=1.0)
    p.add_argument("--object-motion-deg", type=float, default=1.0)
    p.add_argument("--world-size", type=float, default=512.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-dir", default=None,
        help="Default: $STABLEWM_HOME/pusht_horizon_directional_controllability",
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
    th = s[..., 4]
    th2 = np.stack([np.sin(th), np.cos(th)], axis=-1)
    return np.concatenate([pusher, block, th2], axis=-1)


def _physical_components(states, goal_state):
    s = np.asarray(states, dtype=np.float64)
    g = np.asarray(goal_state, dtype=np.float64)
    pusher = np.linalg.norm(s[..., 0:2] - g[0:2], axis=-1)
    block = np.linalg.norm(s[..., 2:4] - g[2:4], axis=-1)
    joint = np.linalg.norm(s[..., 0:4] - g[0:4], axis=-1)
    theta = _angle_error_rad(s[..., 4], g[4])
    return pusher, block, joint, theta


def _physical_cost(states, goal_state):
    pusher, block, joint, theta = _physical_components(states, goal_state)
    cost = (joint / 20.0) ** 2 + (theta / (np.pi / 9.0)) ** 2
    return cost, pusher, block, joint, theta


def _rankdata(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = _rankdata(a[m]), _rankdata(b[m])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(np.dot(ra, rb) / den) if den > 1e-12 else float("nan")


def _pairwise_accuracy(reference, estimate, margin_frac):
    r = np.asarray(reference, dtype=np.float64)
    e = np.asarray(estimate, dtype=np.float64)
    rf = r[np.isfinite(r)]
    q25, q75 = np.percentile(rf, [25, 75])
    margin = float(margin_frac) * max(float(q75 - q25), 1e-12)
    good = 0.0
    n = 0
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            if not (np.isfinite(r[i]) and np.isfinite(r[j]) and np.isfinite(e[i]) and np.isfinite(e[j])):
                continue
            dr = r[i] - r[j]
            if abs(dr) <= margin:
                continue
            de = e[i] - e[j]
            n += 1
            if abs(de) <= 1e-12:
                good += 0.5
            elif np.sign(dr) == np.sign(de):
                good += 1.0
    return (float(good / n) if n else float("nan")), int(n)


def _selection_percentile(physical, score):
    p = np.asarray(physical, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    idx = int(np.nanargmin(s))
    ranks = _rankdata(p) - 1.0
    pct = float(ranks[idx] / max(len(p) - 1, 1))
    oracle = float(np.nanmin(p))
    selected = float(p[idx])
    p90 = float(np.nanpercentile(p, 90))
    regret_norm = (selected - oracle) / max(p90 - oracle, 1e-12)
    return pct, float(regret_norm)


def _numeric_summary(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(x)), "mean": float(np.mean(x)),
        "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def _select_rows(dataset, n, seed, goal_offset):
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col))
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start = episode_len - int(goal_offset) - 1
    by_ep = {ep: max_start[i] for i, ep in enumerate(ep_indices)}
    per_row = np.asarray([by_ep[e] for e in dataset.get_col_data(col)])
    valid = np.asarray(dataset.get_col_data("step_idx")) <= per_row
    valid_idx = np.nonzero(valid)[0]
    print(f"{valid.sum()} valid starting points found for evaluation.")
    g = np.random.default_rng(seed)
    sel = g.choice(len(valid_idx) - 1, size=int(n), replace=False)
    return col, np.sort(valid_idx[sel]).astype(np.int64)


def _check_contiguous(row, episode_idx, step_idx, max_raw):
    ep, t = episode_idx[row], int(step_idx[row])
    for d in range(max_raw + 1):
        j = row + d
        if j >= len(step_idx) or episode_idx[j] != ep or int(step_idx[j]) != t + d:
            raise RuntimeError(f"Dataset not contiguous at row={row}, offset={d}.")


def _bounded_delta(slack, radius, rng):
    slack = np.asarray(slack, dtype=np.float64).reshape(-1)
    if np.linalg.norm(slack) + 1e-10 < radius:
        raise ValueError(
            f"Requested symmetric radius {radius:.6f} exceeds feasible norm {np.linalg.norm(slack):.6f}."
        )
    g = np.maximum(np.abs(rng.normal(size=slack.shape)), 1e-12)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=slack.shape)

    def nrm(alpha):
        return float(np.linalg.norm(np.minimum(alpha * g, slack)))

    lo, hi = 0.0, 1.0
    while nrm(hi) < radius:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if nrm(mid) < radius:
            lo = mid
        else:
            hi = mid
    delta = signs * np.minimum(hi * g, slack)
    if abs(np.linalg.norm(delta) - radius) > 1e-8:
        raise RuntimeError("Equal-norm construction failed.")
    return delta


def _make_candidates(expert_actions, actual_radii, base_radii, num_directions, rng):
    expert = np.nan_to_num(np.asarray(expert_actions, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
    expert = np.clip(expert, -1.0, 1.0)
    flat = expert.reshape(-1)
    slack = np.maximum(1.0 - np.abs(flat), 0.0)
    actual_radii = np.asarray(actual_radii, dtype=np.float64)
    base_radii = np.asarray(base_radii, dtype=np.float64)
    rmax = float(np.max(actual_radii))

    dirs = np.empty((num_directions, flat.size), dtype=np.float64)
    for di in range(num_directions):
        dirs[di] = _bounded_delta(slack, rmax, rng) / rmax

    nc = 1 + len(actual_radii) * num_directions * 2
    cand = np.empty((nc, flat.size), dtype=np.float64)
    meta_base = np.full(nc, np.nan, dtype=np.float64)
    meta_actual = np.full(nc, np.nan, dtype=np.float64)
    meta_dir = np.full(nc, -1, dtype=np.int32)
    meta_sign = np.zeros(nc, dtype=np.int8)
    cand[0] = flat
    ci = 1
    for br, ar in zip(base_radii, actual_radii):
        for di in range(num_directions):
            for sg in (+1, -1):
                delta = float(sg) * float(ar) * dirs[di]
                c = flat + delta
                if np.max(np.abs(c)) > 1.0 + 1e-7:
                    raise RuntimeError("Symmetric candidate violated action bounds.")
                cand[ci] = c
                meta_base[ci], meta_actual[ci] = br, ar
                meta_dir[ci], meta_sign[ci] = di, sg
                ci += 1
    cand = cand.reshape(nc, *expert.shape).astype(np.float32)
    actual_norm = np.linalg.norm(
        cand.astype(np.float64).reshape(nc, -1) - cand[0].astype(np.float64).reshape(1, -1), axis=1
    )
    requested = np.nan_to_num(meta_actual, nan=0.0)
    eqerr = float(np.max(np.abs(actual_norm - requested)))
    if eqerr > 5e-6:
        raise RuntimeError(f"Stored equal-norm error too large: {eqerr:.3e}")
    return cand, meta_base.astype(np.float32), meta_actual.astype(np.float32), meta_dir, meta_sign, actual_norm.astype(np.float32), eqerr


def _reset_anchor(env, init_state, goal_state, seed):
    try:
        return env.reset(
            seed=int(seed),
            options={"state": np.asarray(init_state, dtype=np.float64), "goal_state": np.asarray(goal_state, dtype=np.float64)},
        )
    except Exception:
        obs, info = env.reset(seed=int(seed))
        raw = env.unwrapped
        raw._set_goal_state(np.asarray(goal_state, dtype=np.float64))
        raw._set_state(np.asarray(init_state, dtype=np.float64))
        return obs, raw._get_info()


def _current_goal_images(env, init_state, goal_state, seed):
    _, info = _reset_anchor(env, init_state, goal_state, seed)
    raw = env.unwrapped
    cur = np.asarray(raw.render())
    goal = info.get("goal", None)
    if goal is None:
        raw._set_state(np.asarray(goal_state, dtype=np.float64))
        goal = np.asarray(raw.render())
        raw._set_state(np.asarray(init_state, dtype=np.float64))
    return cur, np.asarray(goal)


def _rollout(env, init_state, goal_state, actions, seed):
    _reset_anchor(env, init_state, goal_state, seed)
    raw = env.unwrapped
    had_contact, contact_steps, obs = False, 0, None
    for a in np.asarray(actions, dtype=np.float32):
        obs, _, _, _, info = raw.step(a)
        nc = int(info.get("n_contacts", 0))
        had_contact = had_contact or nc > 0
        contact_steps += int(nc > 0)
    return np.asarray(obs["state"], dtype=np.float64), np.asarray(raw.render()), had_contact, contact_steps


def _transform_batch(transform, images):
    xs = []
    for im in images:
        x = transform(im)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        xs.append(x)
    return torch.stack(xs, dim=0)


@torch.inference_mode()
def _encode(model, transform, images, device, bs):
    out = []
    for st in range(0, len(images), bs):
        px = _transform_batch(transform, images[st:st+bs]).to(device)
        out.append(model.encode({"pixels": px[:, None].float()})["emb"][:, 0].detach())
    return torch.cat(out, dim=0)


def _normalize_actions(raw_candidates, scaler, horizon, action_block):
    raw = np.asarray(raw_candidates, dtype=np.float32)
    flat = raw.reshape(-1, 2)
    x = scaler.transform(flat).astype(np.float32)
    x = x.reshape(len(raw), horizon, action_block, 2)
    return x.reshape(len(raw), horizon, action_block * 2)


@torch.inference_mode()
def _predict(model, transform, current_image, normalized_actions, device, bs):
    cur = _transform_batch(transform, [current_image]).to(device).float()[0]
    out = []
    for st in range(0, len(normalized_actions), bs):
        a = torch.from_numpy(normalized_actions[st:st+bs]).to(device=device, dtype=torch.float32)
        s = a.shape[0]
        info = {"pixels": cur[None, None, None].expand(1, 1, 1, -1, -1, -1)}
        rolled = model.rollout(info, a.unsqueeze(0))
        out.append(rolled["predicted_emb"][0, :s, -1].detach())
    return torch.cat(out, dim=0)


def _direction_metrics(physical, score, direction, sign, margin_frac):
    p = np.asarray(physical, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    d = np.asarray(direction)
    sg = np.asarray(sign)
    q25, q75 = np.percentile(p[np.isfinite(p)], [25, 75])
    margin = float(margin_frac) * max(float(q75 - q25), 1e-12)
    dp_all, ds_all = [], []
    correct, informative = 0.0, 0
    dirs = np.unique(d[d >= 0])
    for di in dirs:
        ip = np.nonzero((d == di) & (sg == 1))[0]
        im = np.nonzero((d == di) & (sg == -1))[0]
        if len(ip) != 1 or len(im) != 1:
            raise RuntimeError(f"Malformed +/- pair direction={di}")
        dp = float(p[ip[0]] - p[im[0]])
        ds = float(s[ip[0]] - s[im[0]])
        dp_all.append(dp); ds_all.append(ds)
        if abs(dp) <= margin:
            continue
        informative += 1
        if abs(ds) <= 1e-12:
            correct += 0.5
        elif np.sign(dp) == np.sign(ds):
            correct += 1.0
    return {
        "dir_acc": float(correct / informative) if informative else float("nan"),
        "dir_delta_spearman": _spearman(ds_all, dp_all),
        "dir_informative_pairs": int(informative),
        "dir_total_pairs": int(len(dirs)),
        "dir_informative_fraction": float(informative / len(dirs)) if len(dirs) else float("nan"),
        "mean_abs_phys_delta": float(np.mean(np.abs(dp_all))),
    }


def _metrics(physical, enc, pred, dyn, direction, sign, args):
    pred_dir = _direction_metrics(physical, pred, direction, sign, args.direction_margin_frac)
    enc_dir = _direction_metrics(physical, enc, direction, sign, args.direction_margin_frac)
    accp, _ = _pairwise_accuracy(physical, pred, args.pair_margin_frac)
    acce, _ = _pairwise_accuracy(physical, enc, args.pair_margin_frac)
    pctp, regp = _selection_percentile(physical, pred)
    pcte, rege = _selection_percentile(physical, enc)
    return {
        "rho_pred_phys": _spearman(pred, physical),
        "rho_enc_phys": _spearman(enc, physical),
        "rho_pred_enc": _spearman(pred, enc),
        "dir_pred_phys": pred_dir["dir_acc"],
        "dir_enc_phys": enc_dir["dir_acc"],
        "dir_delta_rho_pred_phys": pred_dir["dir_delta_spearman"],
        "dir_delta_rho_enc_phys": enc_dir["dir_delta_spearman"],
        "dir_informative_pairs": pred_dir["dir_informative_pairs"],
        "dir_total_pairs": pred_dir["dir_total_pairs"],
        "dir_informative_fraction": pred_dir["dir_informative_fraction"],
        "mean_abs_phys_direction_delta": pred_dir["mean_abs_phys_delta"],
        "pair_acc_pred_phys": accp,
        "pair_acc_enc_phys": acce,
        "pred_selected_phys_percentile": pctp,
        "enc_selected_phys_percentile": pcte,
        "pred_regret_norm": regp,
        "enc_regret_norm": rege,
        "mean_pred_enc_mse": float(np.mean(dyn)),
    }


SUMMARY_KEYS = [
    "rho_pred_phys", "rho_enc_phys", "rho_pred_enc",
    "dir_pred_phys", "dir_enc_phys", "dir_delta_rho_pred_phys",
    "dir_delta_rho_enc_phys", "dir_informative_pairs", "dir_total_pairs",
    "dir_informative_fraction", "mean_abs_phys_direction_delta",
    "pair_acc_pred_phys", "pair_acc_enc_phys", "pred_selected_phys_percentile",
    "enc_selected_phys_percentile", "pred_regret_norm", "enc_regret_norm",
    "mean_pred_enc_mse", "replay_endpoint_factor_error",
    "candidate_contact_fraction", "candidate_bound_fraction", "equal_norm_error",
]


def _aggregate(rows):
    out = {"count": len(rows)}
    for k in SUMMARY_KEYS:
        out[k] = _numeric_summary([r.get(k, np.nan) for r in rows])
    return out


def _groups(rows):
    return {
        "all": _aggregate(rows),
        "expert_contact": _aggregate([r for r in rows if r["expert_had_contact"]]),
        "expert_no_contact": _aggregate([r for r in rows if not r["expert_had_contact"]]),
        "object_active": _aggregate([r for r in rows if r["expert_object_active"]]),
        "object_inactive": _aggregate([r for r in rows if not r["expert_object_active"]]),
        "replay_good": _aggregate([r for r in rows if r["replay_good"]]),
    }


def _m(summary, key):
    x = summary.get(key, {})
    return x.get("mean") if isinstance(x, dict) else None


def _fmt(x, d=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _print_table(summary, h, base_radius, actual_radius, group="all"):
    hk, rk = str(int(h)), f"{float(base_radius):.6g}"
    print(
        f"\n===== HORIZON DIRECTIONAL: H={h} raw={h*summary['config']['action_block']} "
        f"base_r={base_radius:.3f} actual_r={actual_radius:.3f} group={group} ====="
    )
    header = (
        f"{'model':<14} {'rhoP':>7} {'rhoE':>7} {'rhoPE':>7} {'dirP':>7} "
        f"{'dirE':>7} {'dRhoP':>7} {'accP':>7} {'pctP':>7} {'info':>7}"
    )
    print(header); print("-" * len(header))
    for label, payload in summary["models"].items():
        s = payload["by_horizon"][hk]["by_base_radius"][rk][group]
        print(
            f"{label:<14} {_fmt(_m(s,'rho_pred_phys')):>7} {_fmt(_m(s,'rho_enc_phys')):>7} "
            f"{_fmt(_m(s,'rho_pred_enc')):>7} {_fmt(_m(s,'dir_pred_phys')):>7} "
            f"{_fmt(_m(s,'dir_enc_phys')):>7} {_fmt(_m(s,'dir_delta_rho_pred_phys')):>7} "
            f"{_fmt(_m(s,'pair_acc_pred_phys')):>7} {_fmt(_m(s,'pred_selected_phys_percentile')):>7} "
            f"{_fmt(_m(s,'dir_informative_fraction')):>7}"
        )


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels length must equal --policies length")
    horizons = sorted(set(map(int, args.horizons)))
    if not horizons or min(horizons) <= 0:
        raise ValueError("--horizons must be positive")
    if max(horizons) * args.action_block > args.goal_offset:
        raise ValueError("Every tested horizon endpoint must be <= fixed --goal-offset")
    if args.reference_horizon <= 0:
        raise ValueError("--reference-horizon must be positive")
    base_radii = np.asarray(sorted(set(map(float, args.radii))), dtype=np.float64)
    if len(base_radii) == 0 or np.any(base_radii <= 0):
        raise ValueError("--radii must be positive")

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    outdir = Path(args.output_dir) if args.output_dir else cache_root / "pusht_horizon_directional_controllability"
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(dataset_name, keys_to_cache=["action", "state"], cache_dir=cache_root)
    col, anchors = _select_rows(dataset, args.num_anchors, args.seed, args.goal_offset)
    print("Selected paired eval rows:"); print(anchors)
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
        m.requires_grad_(False); m.interpolate_pos_encoding = True
        models.append(m)

    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    nH, nA, nR, nD, nM = len(horizons), len(anchors), len(base_radii), args.num_directions, len(models)
    nC = 1 + nR * nD * 2
    max_raw = max(horizons) * args.action_block

    # Padded candidate action tensors; only first raw_horizon entries are valid.
    candidate_actions = np.full((nH, nA, nC, max_raw, 2), np.nan, dtype=np.float32)
    candidate_base_radius = np.full((nH, nA, nC), np.nan, dtype=np.float32)
    candidate_actual_radius = np.full((nH, nA, nC), np.nan, dtype=np.float32)
    candidate_direction = np.full((nH, nA, nC), -1, dtype=np.int32)
    candidate_sign = np.zeros((nH, nA, nC), dtype=np.int8)
    candidate_perturb_norm = np.zeros((nH, nA, nC), dtype=np.float32)
    final_states = np.empty((nH, nA, nC, 7), dtype=np.float64)
    had_contact = np.empty((nH, nA, nC), dtype=bool)
    contact_steps = np.empty((nH, nA, nC), dtype=np.int16)
    phys_cost = np.empty((nH, nA, nC), dtype=np.float64)
    enc_cost = np.empty((nM, nH, nA, nC), dtype=np.float64)
    pred_cost = np.empty((nM, nH, nA, nC), dtype=np.float64)
    pred_enc_mse = np.empty((nM, nH, nA, nC), dtype=np.float64)

    rows_by_model = {l: [] for l in labels}
    start = time.time()
    max_eqerr = 0.0

    try:
        for ai, row_idx in enumerate(anchors):
            anchor_start = time.time()
            _check_contiguous(row_idx, episode_idx, step_idx, max(args.goal_offset, max_raw))
            init_state = state[row_idx].copy()
            fixed_goal_state = state[row_idx + args.goal_offset].copy()
            controlled_seed = args.env_seed + ai
            current_image, fixed_goal_image = _current_goal_images(env, init_state, fixed_goal_state, controlled_seed)

            for hi, h in enumerate(horizons):
                raw_h = h * args.action_block
                expert_actions = action[row_idx:row_idx + raw_h].copy()
                endpoint_state = state[row_idx + raw_h].copy()
                scale = np.sqrt(h / args.reference_horizon) if args.radius_scaling == "rms" else 1.0
                actual_radii = base_radii * scale
                rng = np.random.default_rng(args.seed + 1_000_003 * (ai + 1) + 10_007 * h)
                cands, br_meta, ar_meta, d_meta, s_meta, pn_meta, eqerr = _make_candidates(
                    expert_actions, actual_radii, base_radii, nD, rng
                )
                max_eqerr = max(max_eqerr, eqerr)
                candidate_actions[hi, ai, :, :raw_h] = cands
                candidate_base_radius[hi, ai] = br_meta
                candidate_actual_radius[hi, ai] = ar_meta
                candidate_direction[hi, ai] = d_meta
                candidate_sign[hi, ai] = s_meta
                candidate_perturb_norm[hi, ai] = pn_meta

                images = []
                for ci in range(nC):
                    fs, fi, hc, cs = _rollout(env, init_state, fixed_goal_state, cands[ci], controlled_seed)
                    final_states[hi, ai, ci] = fs
                    images.append(fi)
                    had_contact[hi, ai, ci] = hc
                    contact_steps[hi, ai, ci] = cs

                pc, pusher, block, joint, theta = _physical_cost(final_states[hi, ai], fixed_goal_state)
                phys_cost[hi, ai] = pc

                # Replay sanity is relative to the dataset endpoint AT THIS HORIZON.
                replay_endpoint_error = float(np.linalg.norm(
                    _state_factor(final_states[hi, ai, 0], args.world_size)
                    - _state_factor(endpoint_state, args.world_size)
                ))
                replay_good = replay_endpoint_error <= args.replay_factor_good_threshold
                expert_contact = bool(had_contact[hi, ai, 0])
                expert_block_motion = float(np.linalg.norm(final_states[hi, ai, 0, 2:4] - init_state[2:4]))
                expert_theta_motion = float(np.degrees(_angle_error_rad(final_states[hi, ai, 0, 4], init_state[4])))
                expert_active = expert_block_motion >= args.object_motion_px or expert_theta_motion >= args.object_motion_deg
                normalized = _normalize_actions(cands, scaler, h, args.action_block)

                for mi, (label, policy, model) in enumerate(zip(labels, args.policies, models)):
                    zg = _encode(model, transform, [fixed_goal_image], device, args.model_batch_size)[0]
                    zr = _encode(model, transform, images, device, args.model_batch_size)
                    zp = _predict(model, transform, current_image, normalized, device, args.model_batch_size)
                    enc = torch.sum((zr - zg[None]) ** 2, dim=-1).cpu().numpy().astype(np.float64)
                    pred = torch.sum((zp - zg[None]) ** 2, dim=-1).cpu().numpy().astype(np.float64)
                    dyn = torch.mean((zp - zr) ** 2, dim=-1).cpu().numpy().astype(np.float64)
                    enc_cost[mi, hi, ai] = enc
                    pred_cost[mi, hi, ai] = pred
                    pred_enc_mse[mi, hi, ai] = dyn

                    for bri, br in enumerate(base_radii):
                        ar = actual_radii[bri]
                        idx = np.nonzero(np.isclose(br_meta, br, atol=1e-6, rtol=0.0))[0]
                        met = _metrics(pc[idx], enc[idx], pred[idx], dyn[idx], d_meta[idx], s_meta[idx], args)
                        bound_frac = float(np.mean(np.abs(cands[idx]) >= 0.999999))
                        row = {
                            "model": label, "policy": policy, "anchor_index": ai + 1,
                            "dataset_row": int(row_idx), "episode_idx": int(episode_idx[row_idx]),
                            "step_idx": int(step_idx[row_idx]), "fixed_goal_dataset_row": int(row_idx + args.goal_offset),
                            "horizon_coarse": int(h), "raw_horizon": int(raw_h),
                            "base_radius": float(br), "actual_radius": float(ar),
                            "radius_scaling": args.radius_scaling, "reference_horizon": int(args.reference_horizon),
                            "num_directions": int(nD), "replay_endpoint_factor_error": replay_endpoint_error,
                            "replay_good": bool(replay_good), "expert_had_contact": expert_contact,
                            "expert_object_active": bool(expert_active), "expert_block_motion_px": expert_block_motion,
                            "expert_theta_motion_deg": expert_theta_motion,
                            "candidate_contact_fraction": float(had_contact[hi, ai, idx].mean()),
                            "candidate_bound_fraction": bound_frac, "equal_norm_error": float(eqerr),
                            **met,
                        }
                        rows_by_model[label].append(row)

            elapsed = time.time() - anchor_start
            eta = (time.time() - start) / (ai + 1) * (nA - ai - 1)
            print(f"anchor {ai+1:3d}/{nA}: row={row_idx} eqerr<={max_eqerr:.1e} time={elapsed:.2f}s ETA={eta/60:.1f}min")
    finally:
        env.close()

    all_rows = [r for l in labels for r in rows_by_model[l]]
    csv_path = outdir / "anchor_horizon_radius_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)

    npz_path = outdir / "candidate_metrics.npz"
    np.savez_compressed(
        npz_path, anchors=anchors, horizons=np.asarray(horizons, dtype=np.int32),
        base_radii=base_radii.astype(np.float32), labels=np.asarray(labels, dtype=object),
        policies=np.asarray(args.policies, dtype=object), candidate_actions=candidate_actions,
        candidate_base_radius=candidate_base_radius, candidate_actual_radius=candidate_actual_radius,
        candidate_direction=candidate_direction, candidate_sign=candidate_sign,
        candidate_perturb_norm=candidate_perturb_norm, final_states=final_states,
        candidate_had_contact=had_contact, candidate_contact_steps=contact_steps,
        physical_cost=phys_cost, enc_costs=enc_cost, pred_costs=pred_cost,
        pred_enc_terminal_mse=pred_enc_mse,
    )

    summary = {
        "config": {
            "dataset": dataset_name, "num_anchors": nA, "horizons": horizons,
            "reference_horizon": args.reference_horizon, "base_radii": base_radii.tolist(),
            "radius_scaling": args.radius_scaling, "action_block": args.action_block,
            "goal_offset_raw": args.goal_offset, "num_directions": nD, "seed": args.seed,
            "equal_budget_definition": "Exact flattened raw-action-sequence L2 perturbation; RMS matched across H by default.",
            "goal_definition": "One fixed final dataset goal at t+goal_offset for every tested prediction horizon.",
            "replay_definition": "Expert replay is compared with the dataset state at t+H*action_block for each H.",
            "model_cost": "Raw terminal squared Euclidean latent distance; no factor/readout/GT state in model scoring.",
        },
        "paired_rows": anchors.tolist(), "elapsed_seconds": float(time.time() - start), "models": {},
    }
    for label, policy in zip(labels, args.policies):
        mh = {}
        for h in horizons:
            by_r = {}
            for br in base_radii:
                rr = [x for x in rows_by_model[label] if x["horizon_coarse"] == h and abs(x["base_radius"] - br) < 1e-8]
                by_r[f"{br:.6g}"] = _groups(rr)
            mh[str(h)] = {"by_base_radius": by_r}
        summary["models"][label] = {"policy": policy, "by_horizon": mh}

    summary_path = outdir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(_jsonable(summary), f, indent=2)

    for h in horizons:
        scale = np.sqrt(h / args.reference_horizon) if args.radius_scaling == "rms" else 1.0
        for br in base_radii:
            _print_table(summary, h, br, br * scale, "all")
    print(f"\nMax equal-norm error: {max_eqerr:.3e}")
    print(f"Elapsed: {summary['elapsed_seconds']:.1f}s ({summary['elapsed_seconds']/60:.2f} min)")
    print(f"Saved: {csv_path}\nSaved: {npz_path}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
