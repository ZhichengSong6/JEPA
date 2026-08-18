#!/usr/bin/env python3
"""
Fixed-action-dimension horizon diagnostic for full PushT LeWM-family models.

Purpose
-------
Previous finite-horizon Jacobian diagnostics changed TWO things with horizon:
  (1) rollout length H,
  (2) perturbed action-space dimension (10 at H=1 -> 50 at H=5).

This evaluator isolates rollout propagation.  For every anchor we perturb ONLY
the FIRST coarse action block (5 raw 2-D actions = 10 action scalars), and use
the EXACT SAME 10-D perturbation directions and radius for H in {1,2,3,5}.
Later actions remain the expert nominal sequence.

Thus the only changing variable is how far the effect of the same early action
perturbation must propagate through the predictor rollout.

For symmetric probes
    U+ = U0 + [delta, 0, ..., 0]
    U- = U0 - [delta, 0, ..., 0]
we compare
    r_enc(H,v)  = [E(o_H(U+)) - E(o_H(U-))] / (2 r)
    r_pred(H,v) = [P_H(z,U+)   - P_H(z,U-)]   / (2 r)

We reconstruct a 10-D finite-difference action Jacobian and Gram geometry at
each horizon, and additionally compute CEM-facing proxies using the same
candidate set:
  * elite-set overlap (predicted vs encoder / physical top candidates),
  * cosine between predicted and physical CEM elite mean updates,
  * physical percentile/regret of the candidate selected by predicted cost.

These planning proxies connect response-geometry distortion to what CEM would
actually do: choose elites, move its sampling distribution, and select an action.
They are not a replacement for a final closed-loop CEM budget sweep, but they
are a direct diagnostic of why geometry distortion can reduce CEM efficiency.

Outputs
-------
  pair_response_metrics.csv
  anchor_horizon_metrics.csv
  summary.csv
  fixed_action_response_vectors.npz
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
from eval_pusht_horizon_directional import (
    _bounded_delta,
    _check_contiguous,
    _current_goal_images,
    _encode,
    _label,
    _normalize_actions,
    _physical_cost,
    _reset_anchor,
    _select_rows,
    _state_factor,
    _predict,
    _spearman,
)
from eval_pusht_latent_response_calibration import (
    _cosine,
    _energy_rank,
    _gram_metrics,
    _jacobian_geometry,
    _null_energy_fraction,
    _stable_rank,
    _subspace_overlap,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--num-anchors", type=int, default=50)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 5])
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument(
        "--radius", type=float, default=0.1565,
        help="Exact L2 radius in the FIRST 10-D action block, fixed for every H.",
    )
    p.add_argument("--num-directions", type=int, default=64)
    p.add_argument("--elite-frac", type=float, default=0.10)
    p.add_argument("--weak-quantile", type=float, default=0.25)
    p.add_argument("--strong-quantile", type=float, default=0.75)
    p.add_argument("--null-sv-frac", type=float, default=0.05)
    p.add_argument("--pinv-rcond", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--world-size", type=float, default=512.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None)
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


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def _median(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _slope_origin(ref, est):
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    m = np.isfinite(ref) & np.isfinite(est)
    if m.sum() < 2:
        return float("nan")
    r, e = ref[m], est[m]
    den = float(np.dot(r, r))
    return float(np.dot(r, e) / den) if den > 1e-12 else float("nan")


def _make_fixed_first_block_candidates(expert_max, radius, num_directions, rng, action_block):
    """Perturb only the first action_block raw 2-D actions; later actions stay nominal."""
    expert = np.asarray(expert_max, dtype=np.float64).copy()
    expert = np.nan_to_num(expert, nan=0.0, posinf=1.0, neginf=-1.0)
    expert = np.clip(expert, -1.0, 1.0)

    first = expert[:action_block].reshape(-1)
    action_dim = first.size
    slack = np.maximum(1.0 - np.abs(first), 0.0)
    if np.linalg.norm(slack) + 1e-10 < radius:
        raise ValueError(
            f"Fixed first-block radius {radius:.6f} exceeds feasible symmetric norm "
            f"{np.linalg.norm(slack):.6f}."
        )

    dirs = np.empty((num_directions, action_dim), dtype=np.float64)
    for di in range(num_directions):
        dirs[di] = _bounded_delta(slack, radius, rng) / radius

    nc = 1 + 2 * num_directions
    cands = np.repeat(expert[None], nc, axis=0)
    signs = np.zeros(nc, dtype=np.int8)
    direction = np.full(nc, -1, dtype=np.int32)
    first_deltas = np.zeros((nc, action_dim), dtype=np.float64)

    ci = 1
    for di in range(num_directions):
        for sg in (+1, -1):
            d = float(sg) * float(radius) * dirs[di]
            f = first + d
            if np.max(np.abs(f)) > 1.0 + 1e-7:
                raise RuntimeError("Fixed-block candidate violated action bounds")
            cands[ci, :action_block] = f.reshape(action_block, 2)
            signs[ci] = sg
            direction[ci] = di
            first_deltas[ci] = d
            ci += 1

    norms = np.linalg.norm(first_deltas, axis=1)
    err = float(np.max(np.abs(norms[1:] - radius))) if nc > 1 else 0.0
    if err > 5e-6:
        raise RuntimeError(f"Equal-norm construction failed: {err:.3e}")

    return (
        cands.astype(np.float32),
        dirs.astype(np.float32),
        direction,
        signs,
        first_deltas.astype(np.float32),
        err,
    )


def _rollout_checkpoints(env, init_state, goal_state, actions, checkpoints, seed):
    """One rollout to max H; return state/image/contact summary at requested raw steps."""
    _reset_anchor(env, init_state, goal_state, seed)
    raw = env.unwrapped
    checkpoints = set(map(int, checkpoints))
    out = {}
    had_contact = False
    contact_steps = 0

    for k, a in enumerate(np.asarray(actions, dtype=np.float32), start=1):
        obs, _, _, _, info = raw.step(a)
        nc = int(info.get("n_contacts", 0))
        had_contact = had_contact or nc > 0
        contact_steps += int(nc > 0)
        if k in checkpoints:
            out[k] = {
                "state": np.asarray(obs["state"], dtype=np.float64).copy(),
                "image": np.asarray(raw.render()).copy(),
                "had_contact": bool(had_contact),
                "contact_steps": int(contact_steps),
            }
    return out


def _pair_metrics(enc_jvp, pred_jvp, phys_jvp):
    en = float(np.linalg.norm(enc_jvp))
    pn = float(np.linalg.norm(pred_jvp))
    ph = float(np.linalg.norm(phys_jvp))
    eps = max(1e-12, en * 1e-8)
    cos = _cosine(enc_jvp, pred_jvp)
    gain = pn / max(en, eps)
    rel = float(np.linalg.norm(pred_jvp - enc_jvp) / max(en, eps))
    den = float(np.dot(enc_jvp, enc_jvp))
    if den > 1e-20:
        parallel_gain = float(np.dot(pred_jvp, enc_jvp) / den)
        residual = pred_jvp - parallel_gain * enc_jvp
        orth = float(np.linalg.norm(residual) / np.sqrt(den))
    else:
        parallel_gain, orth = float("nan"), float("nan")
    return {
        "enc_response_norm": en,
        "pred_response_norm": pn,
        "phys_response_norm": ph,
        "pred_enc_cosine": cos,
        "pred_enc_gain_ratio": gain,
        "pred_enc_log2_gain": float(np.log2(max(gain, 1e-12))),
        "pred_enc_rel_vector_error": rel,
        "pred_parallel_gain": parallel_gain,
        "pred_orthogonal_leakage_ratio": orth,
    }


def _summarize_pair_rows(rows, weak_q, strong_q):
    if not rows:
        return {"n_pairs": 0}
    en = np.asarray([r["enc_response_norm"] for r in rows], dtype=np.float64)
    pn = np.asarray([r["pred_response_norm"] for r in rows], dtype=np.float64)
    cos = np.asarray([r["pred_enc_cosine"] for r in rows], dtype=np.float64)
    gain = np.asarray([r["pred_enc_gain_ratio"] for r in rows], dtype=np.float64)
    rel = np.asarray([r["pred_enc_rel_vector_error"] for r in rows], dtype=np.float64)
    ortho = np.asarray([r["pred_orthogonal_leakage_ratio"] for r in rows], dtype=np.float64)
    qw = float(np.quantile(en, weak_q))
    qs = float(np.quantile(en, strong_q))
    weak = en <= qw
    strong = en >= qs
    return {
        "n_pairs": int(len(rows)),
        "gain_spearman": _spearman(en, pn),
        "gain_slope_origin": _slope_origin(en, pn),
        "median_enc_response_norm": _median(en),
        "median_pred_response_norm": _median(pn),
        "median_gain_ratio": _median(gain),
        "median_cosine": _median(cos),
        "mean_cosine": _mean(cos),
        "median_rel_vector_error": _median(rel),
        "median_orthogonal_leakage_ratio": _median(ortho),
        "weak_overresponse_rate": float(np.mean(pn[weak] >= qs)) if weak.any() else float("nan"),
        "strong_underresponse_rate": float(np.mean(pn[strong] <= qw)) if strong.any() else float("nan"),
        "strong_median_cosine": _median(cos[strong]),
    }


def _rank_percentile(reference, idx):
    r = np.asarray(reference, dtype=np.float64)
    order = np.argsort(r, kind="mergesort")
    rank = int(np.where(order == int(idx))[0][0])
    return float(rank / max(len(r) - 1, 1))


def _normalized_regret(reference, idx):
    r = np.asarray(reference, dtype=np.float64)
    oracle = float(np.min(r))
    selected = float(r[int(idx)])
    p90 = float(np.percentile(r, 90))
    return float((selected - oracle) / max(p90 - oracle, 1e-12))


def _elite_metrics(phys_cost, enc_cost, pred_cost, first_deltas, elite_frac):
    n = len(pred_cost)
    k = max(2, int(np.ceil(float(elite_frac) * n)))
    ip = np.argsort(pred_cost)[:k]
    ie = np.argsort(enc_cost)[:k]
    iph = np.argsort(phys_cost)[:k]

    def overlap(a, b):
        return float(len(set(map(int, a)) & set(map(int, b))) / k)

    def mean_update(idx):
        return np.mean(np.asarray(first_deltas, dtype=np.float64)[idx], axis=0)

    up = mean_update(ip)
    ue = mean_update(ie)
    uo = mean_update(iph)
    best_pred = int(np.argmin(pred_cost))
    best_enc = int(np.argmin(enc_cost))

    return {
        "elite_k": int(k),
        "elite_overlap_pred_phys": overlap(ip, iph),
        "elite_overlap_pred_enc": overlap(ip, ie),
        "elite_overlap_enc_phys": overlap(ie, iph),
        "cem_update_cos_pred_phys": _cosine(up, uo),
        "cem_update_cos_pred_enc": _cosine(up, ue),
        "cem_update_cos_enc_phys": _cosine(ue, uo),
        "pred_selected_phys_percentile": _rank_percentile(phys_cost, best_pred),
        "pred_selected_phys_regret_norm": _normalized_regret(phys_cost, best_pred),
        "enc_selected_phys_percentile": _rank_percentile(phys_cost, best_enc),
        "enc_selected_phys_regret_norm": _normalized_regret(phys_cost, best_enc),
        "rho_pred_phys": _spearman(pred_cost, phys_cost),
        "rho_enc_phys": _spearman(enc_cost, phys_cost),
        "rho_pred_enc": _spearman(pred_cost, enc_cost),
    }


def _aggregate_anchor_rows(rows):
    if not rows:
        return {}
    keys = [
        "gram_cos_pred_enc", "gram_rel_error_pred_enc", "gram_trace_ratio_pred_enc",
        "gram_shape_rel_error_pred_enc", "top3_action_subspace_overlap",
        "enc_stable_rank", "pred_stable_rank", "enc_rank95", "pred_rank95",
        "pred_energy_in_enc_null_fraction", "enc_energy_in_pred_null_fraction",
        "gram_cos_enc_phys", "gram_cos_pred_phys",
        "elite_overlap_pred_phys", "elite_overlap_pred_enc", "elite_overlap_enc_phys",
        "cem_update_cos_pred_phys", "cem_update_cos_pred_enc", "cem_update_cos_enc_phys",
        "pred_selected_phys_percentile", "pred_selected_phys_regret_norm",
        "enc_selected_phys_percentile", "enc_selected_phys_regret_norm",
        "rho_pred_phys", "rho_enc_phys", "rho_pred_enc",
    ]
    out = {"n_anchors": len(rows)}
    for k in keys:
        vals = [r.get(k, np.nan) for r in rows]
        out[f"{k}_mean"] = _mean(vals)
        out[f"{k}_median"] = _median(vals)
    return out


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels length must equal --policies length")
    horizons = sorted(set(map(int, args.horizons)))
    if not horizons or min(horizons) <= 0:
        raise ValueError("--horizons must be positive")
    if max(horizons) * args.action_block > args.goal_offset:
        raise ValueError("max horizon endpoint must be <= --goal-offset")
    if args.radius <= 0:
        raise ValueError("--radius must be positive")
    if args.num_directions < args.action_block * 2:
        print("WARNING: fewer probe directions than fixed 10-D action space; Jacobian is underdetermined")

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    outdir = Path(args.output_dir) if args.output_dir else cache_root / "pusht_fixed_action_response_horizon"
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(dataset_name, keys_to_cache=["action", "state"], cache_dir=cache_root)
    col, anchors = _select_rows(dataset, args.num_anchors, args.seed, args.goal_offset)
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
    max_h = max(horizons)
    max_raw = max_h * args.action_block
    checkpoints = [h * args.action_block for h in horizons]
    action_dim = args.action_block * 2
    nC = 1 + 2 * args.num_directions

    # Determine common latent dim.
    first_init = state[anchors[0]].copy()
    first_goal = state[anchors[0] + args.goal_offset].copy()
    first_img, _ = _current_goal_images(env, first_init, first_goal, args.env_seed)
    latent_dims = [int(_encode(m, transform, [first_img], device, args.model_batch_size).shape[-1]) for m in models]
    if len(set(latent_dims)) != 1:
        raise ValueError(f"Models have different latent dims: {dict(zip(labels, latent_dims))}")
    latent_dim = latent_dims[0]
    print(f"Latent dim: {latent_dim}; fixed perturbed action dim: {action_dim}")

    enc_store = np.full((len(models), len(horizons), len(anchors), args.num_directions, latent_dim), np.nan, dtype=np.float32)
    pred_store = np.full_like(enc_store, np.nan)
    phys_store = np.full((len(horizons), len(anchors), args.num_directions, 6), np.nan, dtype=np.float32)
    probe_store = np.full((len(anchors), args.num_directions, action_dim), np.nan, dtype=np.float32)

    pair_rows = []
    anchor_rows = []
    start = time.time()
    max_eqerr = 0.0

    try:
        for ai, row_idx in enumerate(anchors):
            t0 = time.time()
            _check_contiguous(row_idx, episode_idx, step_idx, max(args.goal_offset, max_raw))
            init_state = state[row_idx].copy()
            goal_state = state[row_idx + args.goal_offset].copy()
            seed = args.env_seed + ai
            current_image, goal_image = _current_goal_images(env, init_state, goal_state, seed)
            expert_max = action[row_idx:row_idx + max_raw].copy()

            rng = np.random.default_rng(args.seed + 1_000_003 * (ai + 1))
            cands, dirs, dmeta, smeta, first_deltas, eqerr = _make_fixed_first_block_candidates(
                expert_max, args.radius, args.num_directions, rng, args.action_block
            )
            max_eqerr = max(max_eqerr, eqerr)
            probe_store[ai] = dirs

            # One real rollout per candidate to max horizon; capture all H checkpoints.
            real_states = {h: np.empty((nC, state.shape[-1]), dtype=np.float64) for h in horizons}
            real_images = {h: [None] * nC for h in horizons}
            real_contact = {h: np.zeros(nC, dtype=bool) for h in horizons}
            for ci in range(nC):
                out = _rollout_checkpoints(env, init_state, goal_state, cands[ci], checkpoints, seed)
                for h in horizons:
                    raw_h = h * args.action_block
                    real_states[h][ci] = out[raw_h]["state"]
                    real_images[h][ci] = out[raw_h]["image"]
                    real_contact[h][ci] = out[raw_h]["had_contact"]

            for mi, (label, model) in enumerate(zip(labels, models)):
                zg = _encode(model, transform, [goal_image], device, args.model_batch_size)[0]

                for hi, h in enumerate(horizons):
                    raw_h = h * args.action_block
                    zr = _encode(model, transform, real_images[h], device, args.model_batch_size)
                    normalized = _normalize_actions(cands[:, :raw_h], scaler, h, args.action_block)
                    zp = _predict(model, transform, current_image, normalized, device, args.model_batch_size)

                    enc_cost = torch.sum((zr - zg[None]) ** 2, dim=-1).cpu().numpy().astype(np.float64)
                    pred_cost = torch.sum((zp - zg[None]) ** 2, dim=-1).cpu().numpy().astype(np.float64)
                    phys_cost, *_ = _physical_cost(real_states[h], goal_state)

                    enc_jvps = []
                    pred_jvps = []
                    phys_jvps = []
                    local_pair_rows = []

                    for di in range(args.num_directions):
                        ip = np.where((dmeta == di) & (smeta > 0))[0]
                        im = np.where((dmeta == di) & (smeta < 0))[0]
                        if len(ip) != 1 or len(im) != 1:
                            raise RuntimeError(f"Malformed +/- pair direction={di}")
                        ip, im = int(ip[0]), int(im[0])
                        enc_jvp = ((zr[ip] - zr[im]) / (2.0 * args.radius)).detach().cpu().numpy().astype(np.float64)
                        pred_jvp = ((zp[ip] - zp[im]) / (2.0 * args.radius)).detach().cpu().numpy().astype(np.float64)
                        phip = _state_factor(real_states[h][ip], args.world_size)
                        phim = _state_factor(real_states[h][im], args.world_size)
                        phys_jvp = (phip - phim) / (2.0 * args.radius)

                        enc_store[mi, hi, ai, di] = enc_jvp.astype(np.float32)
                        pred_store[mi, hi, ai, di] = pred_jvp.astype(np.float32)
                        if mi == 0:
                            phys_store[hi, ai, di] = phys_jvp.astype(np.float32)

                        enc_jvps.append(enc_jvp)
                        pred_jvps.append(pred_jvp)
                        phys_jvps.append(phys_jvp)
                        pm = _pair_metrics(enc_jvp, pred_jvp, phys_jvp)
                        row = {
                            "model": label,
                            "anchor_index": ai + 1,
                            "dataset_row": int(row_idx),
                            "horizon": int(h),
                            "raw_horizon": int(raw_h),
                            "direction": int(di),
                            "radius": float(args.radius),
                            "plus_contact": int(real_contact[h][ip]),
                            "minus_contact": int(real_contact[h][im]),
                            **pm,
                        }
                        pair_rows.append(row)
                        local_pair_rows.append(row)

                    V = dirs.astype(np.float64)
                    Renc = np.asarray(enc_jvps, dtype=np.float64)
                    Rpred = np.asarray(pred_jvps, dtype=np.float64)
                    Rphys = np.asarray(phys_jvps, dtype=np.float64)
                    ge = _jacobian_geometry(V, Renc, args.pinv_rcond)
                    gp = _jacobian_geometry(V, Rpred, args.pinv_rcond)
                    gph = _jacobian_geometry(V, Rphys, args.pinv_rcond)

                    gc_pe, gr_pe, tr_pe, shape_pe = _gram_metrics(ge["G"], gp["G"])
                    gc_eph, _, _, _ = _gram_metrics(gph["G"], ge["G"])
                    gc_pph, _, _, _ = _gram_metrics(gph["G"], gp["G"])
                    k = min(3, action_dim)
                    sub = _subspace_overlap(ge["Vt"], gp["Vt"], k)
                    pred_in_enc_null, enc_null_dim = _null_energy_fraction(
                        gp["G"], ge["Vt"], ge["s"], args.null_sv_frac
                    )
                    enc_in_pred_null, pred_null_dim = _null_energy_fraction(
                        ge["G"], gp["Vt"], gp["s"], args.null_sv_frac
                    )

                    elite = _elite_metrics(
                        phys_cost, enc_cost, pred_cost, first_deltas, args.elite_frac
                    )
                    pair_summary = _summarize_pair_rows(
                        local_pair_rows, args.weak_quantile, args.strong_quantile
                    )

                    anchor_rows.append({
                        "model": label,
                        "anchor_index": ai + 1,
                        "dataset_row": int(row_idx),
                        "horizon": int(h),
                        "raw_horizon": int(raw_h),
                        "radius": float(args.radius),
                        "action_dim": int(action_dim),
                        "num_directions": int(args.num_directions),
                        "probe_rank": int(ge["probe_rank"]),
                        "probe_condition": float(ge["probe_condition"]),
                        "gram_cos_pred_enc": gc_pe,
                        "gram_rel_error_pred_enc": gr_pe,
                        "gram_trace_ratio_pred_enc": tr_pe,
                        "gram_shape_rel_error_pred_enc": shape_pe,
                        "top3_action_subspace_overlap": sub,
                        "enc_stable_rank": _stable_rank(ge["s"]),
                        "pred_stable_rank": _stable_rank(gp["s"]),
                        "enc_rank95": _energy_rank(ge["s"], 0.95),
                        "pred_rank95": _energy_rank(gp["s"], 0.95),
                        "pred_energy_in_enc_null_fraction": pred_in_enc_null,
                        "enc_energy_in_pred_null_fraction": enc_in_pred_null,
                        "enc_null_dim": int(enc_null_dim),
                        "pred_null_dim": int(pred_null_dim),
                        "gram_cos_enc_phys": gc_eph,
                        "gram_cos_pred_phys": gc_pph,
                        **pair_summary,
                        **elite,
                    })

            elapsed = time.time() - t0
            eta = (time.time() - start) / (ai + 1) * (len(anchors) - ai - 1)
            print(
                f"anchor {ai+1:3d}/{len(anchors)} row={row_idx} "
                f"eqerr<={max_eqerr:.1e} time={elapsed:.2f}s ETA={eta/60:.1f}min"
            )
    finally:
        env.close()

    pair_csv = outdir / "pair_response_metrics.csv"
    anchor_csv = outdir / "anchor_horizon_metrics.csv"
    _write_csv(pair_csv, pair_rows)
    _write_csv(anchor_csv, anchor_rows)

    summary_rows = []
    for label in labels:
        for h in horizons:
            ar = [r for r in anchor_rows if r["model"] == label and r["horizon"] == h]
            pr = [r for r in pair_rows if r["model"] == label and r["horizon"] == h]
            s = {
                "model": label,
                "horizon": int(h),
                "raw_horizon": int(h * args.action_block),
                "radius": float(args.radius),
                **_aggregate_anchor_rows(ar),
                **{f"pair_{k}": v for k, v in _summarize_pair_rows(pr, args.weak_quantile, args.strong_quantile).items()},
            }
            summary_rows.append(s)

    summary_csv = outdir / "summary.csv"
    _write_csv(summary_csv, summary_rows)

    npz_path = outdir / "fixed_action_response_vectors.npz"
    np.savez_compressed(
        npz_path,
        anchors=anchors,
        horizons=np.asarray(horizons, dtype=np.int32),
        labels=np.asarray(labels, dtype=object),
        radius=np.asarray(args.radius, dtype=np.float32),
        probe_directions=probe_store,
        enc_jvps=enc_store,
        pred_jvps=pred_store,
        phys_jvps=phys_store,
    )

    summary = {
        "config": {
            "dataset": dataset_name,
            "num_anchors": int(len(anchors)),
            "horizons": horizons,
            "action_block": int(args.action_block),
            "fixed_perturbed_action_dim": int(action_dim),
            "radius": float(args.radius),
            "num_directions": int(args.num_directions),
            "elite_frac": float(args.elite_frac),
            "goal_offset": int(args.goal_offset),
            "definition": "Only first coarse action block is perturbed; identical 10-D probes/radius are used at every horizon.",
        },
        "elapsed_seconds": float(time.time() - start),
        "summary_rows": summary_rows,
    }
    summary_json = outdir / "summary.json"
    with summary_json.open("w") as f:
        json.dump(_jsonable(summary), f, indent=2)

    print()
    hdr = (
        f"{'model':<12} {'H':>2} {'cos':>7} {'gain':>7} {'gramPE':>8} "
        f"{'GphysE':>8} {'GphysP':>8} {'eliteP':>7} {'updP':>7} {'pctP':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summary_rows:
        def f(k):
            x = s.get(k, np.nan)
            return "nan" if not np.isfinite(x) else f"{x:.3f}"
        print(
            f"{s['model']:<12} {s['horizon']:>2d} "
            f"{f('pair_median_cosine'):>7} {f('pair_median_gain_ratio'):>7} "
            f"{f('gram_cos_pred_enc_mean'):>8} {f('gram_cos_enc_phys_mean'):>8} "
            f"{f('gram_cos_pred_phys_mean'):>8} {f('elite_overlap_pred_phys_mean'):>7} "
            f"{f('cem_update_cos_pred_phys_mean'):>7} {f('pred_selected_phys_percentile_mean'):>7}"
        )

    print(f"\nMax equal-norm error: {max_eqerr:.3e}")
    print(f"Saved: {pair_csv}")
    print(f"Saved: {anchor_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
