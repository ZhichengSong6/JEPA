#!/usr/bin/env python3
"""
Direct latent action-response / finite-horizon Jacobian calibration for full PushT.

Scientific question
-------------------
Existing action-sensitive objectives mainly ask whether different actions remain
distinguishable. This evaluator asks the stronger question:

    Does the predictor have the same state-, direction-, strength-, and
    horizon-dependent action response as the REAL environment followed by the
    model encoder?

For the same symmetric equal-budget probes used by
eval_pusht_horizon_directional.py,

    U+ = U0 + r_H v
    U- = U0 - r_H v,

we compare endpoint response vectors directly, WITHOUT any goal-distance cost:

    r_enc(v)  = [E(o_H(U+)) - E(o_H(U-))] / (2 r_H)
    r_pred(v) = [P_H(z,U+)   - P_H(z,U-)]   / (2 r_H)

and also a normalized physical-state response

    r_phys(v) = [phi(s_H(U+)) - phi(s_H(U-))] / (2 r_H),

where phi=[pusher_xy, block_xy, sin(theta), cos(theta)] in normalized units.

With >= action_dim probe directions, we additionally reconstruct a least-squares
finite-difference Jacobian on action space:

    R ~= V J^T,
    J^T = pinv(V) R,

then compare the action-space pullback/controllability Gram matrices

    G = J^T J.

This allows us to distinguish:
  * response collapse / missing sensitivity,
  * over-response / sensitivity leakage,
  * wrong response direction,
  * gain miscalibration,
  * response-geometry distortion,
  * predicted sensitivity in encoder near-null action directions.

IMPORTANT
---------
The primary "real" reference is E(real terminal image), not a goal cost.
Therefore H=5 is NOT confounded by the nominal expert endpoint coinciding with
the fixed task goal.

The Jacobian/Gram reconstruction is local finite-difference analysis. Use a
small radius (default reference radius 0.35) and enough directions (default 64).
At H=5 the raw action dimension is 50, so 64 directions makes the least-squares
system overdetermined if the probe matrix has full rank.

Outputs
-------
  pair_response_metrics.csv
  anchor_geometry_metrics.csv
  summary.csv
  latent_response_vectors.npz
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

from eval import img_transform
from eval_pusht_horizon_directional import (
    _check_contiguous,
    _current_goal_images,
    _encode,
    _jsonable,
    _label,
    _make_candidates,
    _normalize_actions,
    _predict,
    _rollout,
    _select_rows,
    _state_factor,
)


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
        "--radii", nargs="+", type=float, default=[0.35],
        help="Reference-horizon exact flattened raw-action L2 radii. "
             "Use 0.35 first for local Jacobian analysis.",
    )
    p.add_argument(
        "--radius-scaling", choices=["rms", "fixed"], default="rms",
        help="rms: r_H=r_ref*sqrt(H/H_ref), matching RMS perturbation per action scalar.",
    )
    p.add_argument(
        "--num-directions", type=int, default=64,
        help="Use >=50 for full-column-rank probing at H=5 (action dim=50 by default).",
    )
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument(
        "--goal-offset", type=int, default=25,
        help="Used only to keep the same rendered goal / anchor selection as prior eval. "
             "No goal cost is used in the metrics.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--world-size", type=float, default=512.0)
    p.add_argument(
        "--weak-quantile", type=float, default=0.25,
        help="Bottom quantile of true encoder JVP norm used for over-response rate.",
    )
    p.add_argument(
        "--strong-quantile", type=float, default=0.75,
        help="Top quantile of true encoder JVP norm used for under-response/strong-direction metrics.",
    )
    p.add_argument(
        "--null-sv-frac", type=float, default=0.10,
        help="Encoder/predictor action singular values below frac * max(sigma) are near-null.",
    )
    p.add_argument(
        "--pinv-rcond", type=float, default=1e-6,
        help="Relative cutoff for least-squares Jacobian reconstruction.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-dir", default=None,
        help="Default: $STABLEWM_HOME/pusht_latent_response_calibration",
    )
    return p.parse_args()


def _cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > eps else float("nan")


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
    a, b = _rankdata(a[m]), _rankdata(b[m])
    a -= a.mean()
    b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


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


def _contact_mode(cp, cm):
    cp, cm = bool(cp), bool(cm)
    if not cp and not cm:
        return "no_contact"
    if cp and cm:
        return "both_contact"
    return "contact_switch"


def _energy_rank(s, frac=0.95):
    s = np.asarray(s, dtype=np.float64)
    e = s ** 2
    total = float(e.sum())
    if total <= 1e-20:
        return 0
    return int(np.searchsorted(np.cumsum(e) / total, frac, side="left") + 1)


def _stable_rank(s):
    s = np.asarray(s, dtype=np.float64)
    if len(s) == 0 or s[0] <= 1e-12:
        return 0.0
    return float(np.sum(s ** 2) / (s[0] ** 2))


def _gram_metrics(G_ref, G_est):
    G_ref = np.asarray(G_ref, dtype=np.float64)
    G_est = np.asarray(G_est, dtype=np.float64)
    nr = float(np.linalg.norm(G_ref))
    ne = float(np.linalg.norm(G_est))
    gram_cos = float(np.sum(G_ref * G_est) / (nr * ne)) if nr > 1e-20 and ne > 1e-20 else float("nan")
    gram_rel = float(np.linalg.norm(G_est - G_ref) / nr) if nr > 1e-20 else float("nan")

    tr = float(np.trace(G_ref))
    te = float(np.trace(G_est))
    trace_ratio = te / tr if tr > 1e-20 else float("nan")

    if tr > 1e-20 and te > 1e-20:
        shape_rel = float(np.linalg.norm(G_est / te - G_ref / tr) / np.linalg.norm(G_ref / tr))
    else:
        shape_rel = float("nan")

    return gram_cos, gram_rel, trace_ratio, shape_rel


def _subspace_overlap(V_ref, V_est, k):
    if k <= 0:
        return float("nan")
    A = np.asarray(V_ref[:k], dtype=np.float64).T
    B = np.asarray(V_est[:k], dtype=np.float64).T
    return float(np.linalg.norm(A.T @ B, ord="fro") ** 2 / k)


def _jacobian_geometry(V, R, rcond):
    V = np.asarray(V, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    pinv = np.linalg.pinv(V, rcond=rcond)
    JT = pinv @ R
    J = JT.T
    G = JT @ JT.T
    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    probe_s = np.linalg.svd(V, compute_uv=False)
    tol = rcond * (probe_s[0] if len(probe_s) else 1.0)
    probe_rank = int(np.sum(probe_s > tol))
    probe_cond = float(probe_s[0] / probe_s[-1]) if len(probe_s) and probe_s[-1] > 1e-12 else float("inf")
    return {
        "J": J,
        "G": G,
        "s": s,
        "Vt": Vt,
        "probe_rank": probe_rank,
        "probe_condition": probe_cond,
    }


def _null_energy_fraction(G, Vt_ref, s_ref, threshold_frac):
    G = np.asarray(G, dtype=np.float64)
    s_ref = np.asarray(s_ref, dtype=np.float64)
    Vt_ref = np.asarray(Vt_ref, dtype=np.float64)
    total = float(np.trace(G))
    if total <= 1e-20 or len(s_ref) == 0 or s_ref[0] <= 1e-12:
        return float("nan"), 0
    action_dim = Vt_ref.shape[0]
    sval_full = np.zeros(action_dim, dtype=np.float64)
    sval_full[:len(s_ref)] = s_ref
    null = sval_full < float(threshold_frac) * s_ref[0]
    if not np.any(null):
        return 0.0, 0
    N = Vt_ref[null]
    e = float(np.trace(N @ G @ N.T))
    return float(e / total), int(np.sum(null))


def _pair_metrics(enc_jvp, pred_jvp, phys_jvp):
    en = float(np.linalg.norm(enc_jvp))
    pn = float(np.linalg.norm(pred_jvp))
    phn = float(np.linalg.norm(phys_jvp))
    eps = max(1e-12, en * 1e-8)

    cos_pe = _cosine(enc_jvp, pred_jvp)
    gain_ratio = pn / max(en, eps)
    rel_err = float(np.linalg.norm(pred_jvp - enc_jvp) / max(en, eps))

    den = float(np.dot(enc_jvp, enc_jvp))
    if den > 1e-20:
        parallel_gain = float(np.dot(pred_jvp, enc_jvp) / den)
        residual = pred_jvp - parallel_gain * enc_jvp
        orth_ratio = float(np.linalg.norm(residual) / np.sqrt(den))
    else:
        parallel_gain = float("nan")
        orth_ratio = float("nan")

    return {
        "enc_response_norm": en,
        "pred_response_norm": pn,
        "phys_response_norm": phn,
        "pred_enc_cosine": cos_pe,
        "pred_enc_gain_ratio": gain_ratio,
        "pred_enc_log2_gain": float(np.log2(max(gain_ratio, 1e-12))),
        "pred_enc_rel_vector_error": rel_err,
        "pred_parallel_gain": parallel_gain,
        "pred_orthogonal_leakage_ratio": orth_ratio,
    }


def _summarize_pairs(rows, weak_q, strong_q):
    if not rows:
        return {"n_pairs": 0}

    en = np.asarray([r["enc_response_norm"] for r in rows], dtype=np.float64)
    pn = np.asarray([r["pred_response_norm"] for r in rows], dtype=np.float64)
    cos = np.asarray([r["pred_enc_cosine"] for r in rows], dtype=np.float64)
    ratio = np.asarray([r["pred_enc_gain_ratio"] for r in rows], dtype=np.float64)
    log2g = np.asarray([r["pred_enc_log2_gain"] for r in rows], dtype=np.float64)
    rel = np.asarray([r["pred_enc_rel_vector_error"] for r in rows], dtype=np.float64)
    ortho = np.asarray([r["pred_orthogonal_leakage_ratio"] for r in rows], dtype=np.float64)

    qw = float(np.quantile(en, weak_q))
    qs = float(np.quantile(en, strong_q))
    weak = en <= qw
    strong = en >= qs

    weak_over = float(np.mean(pn[weak] >= qs)) if weak.any() else float("nan")
    strong_under = float(np.mean(pn[strong] <= qw)) if strong.any() else float("nan")

    return {
        "n_pairs": int(len(rows)),
        "gain_spearman": _spearman(en, pn),
        "gain_slope_origin": _slope_origin(en, pn),
        "median_enc_response_norm": _median(en),
        "median_pred_response_norm": _median(pn),
        "median_gain_ratio": _median(ratio),
        "median_log2_gain": _median(log2g),
        "median_cosine": _median(cos),
        "mean_cosine": _mean(cos),
        "negative_cosine_fraction": float(np.mean(cos[np.isfinite(cos)] < 0.0)) if np.isfinite(cos).any() else float("nan"),
        "median_rel_vector_error": _median(rel),
        "median_orthogonal_leakage_ratio": _median(ortho),
        "weak_ref_threshold": qw,
        "strong_ref_threshold": qs,
        "weak_overresponse_rate": weak_over,
        "strong_underresponse_rate": strong_under,
        "strong_median_cosine": _median(cos[strong]),
        "strong_mean_cosine": _mean(cos[strong]),
        "weak_median_pred_norm": _median(pn[weak]),
        "strong_median_pred_norm": _median(pn[strong]),
    }


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()

    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels length must equal --policies length")
    if not (0.0 < args.weak_quantile < args.strong_quantile < 1.0):
        raise ValueError("Require 0 < weak_quantile < strong_quantile < 1")
    if not (0.0 < args.null_sv_frac < 1.0):
        raise ValueError("--null-sv-frac must be in (0,1)")

    horizons = sorted(set(map(int, args.horizons)))
    if not horizons or min(horizons) <= 0:
        raise ValueError("--horizons must be positive")
    if max(horizons) * args.action_block > args.goal_offset:
        raise ValueError("Every tested horizon endpoint must be <= --goal-offset")
    base_radii = np.asarray(sorted(set(map(float, args.radii))), dtype=np.float64)
    if len(base_radii) == 0 or np.any(base_radii <= 0):
        raise ValueError("--radii must be positive")

    max_action_dim = max(horizons) * args.action_block * 2
    if args.num_directions < max_action_dim:
        print(
            f"WARNING: num_directions={args.num_directions} < max action_dim={max_action_dim}. "
            "H=max Jacobian reconstruction will be underdetermined. "
            "Pairwise JVP metrics remain valid."
        )

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    outdir = Path(args.output_dir) if args.output_dir else cache_root / "pusht_latent_response_calibration"
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=["action", "state"], cache_dir=cache_root
    )
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

    nM = len(models)
    nH = len(horizons)
    nA = len(anchors)
    nR = len(base_radii)
    nD = args.num_directions
    max_raw = max(horizons) * args.action_block

    first_init = state[anchors[0]].copy()
    first_goal = state[anchors[0] + args.goal_offset].copy()
    first_img, _ = _current_goal_images(env, first_init, first_goal, args.env_seed)
    latent_dims = [
        int(_encode(m, transform, [first_img], device, args.model_batch_size).shape[-1])
        for m in models
    ]
    if len(set(latent_dims)) != 1:
        raise ValueError(f"Models have different latent dims: {dict(zip(labels, latent_dims))}")
    latent_dim = latent_dims[0]
    print(f"Latent dim: {latent_dim}")

    enc_jvp_store = np.full((nM, nH, nA, nR, nD, latent_dim), np.nan, dtype=np.float32)
    pred_jvp_store = np.full_like(enc_jvp_store, np.nan)
    phys_jvp_store = np.full((nH, nA, nR, nD, 6), np.nan, dtype=np.float32)
    probe_dirs = np.full((nH, nA, nD, max_action_dim), np.nan, dtype=np.float32)
    pair_contact_mode = np.full((nH, nA, nR, nD), -1, dtype=np.int8)
    contact_code = {"no_contact": 0, "both_contact": 1, "contact_switch": 2}

    pair_rows = []
    geometry_rows = []
    start = time.time()
    max_eqerr = 0.0

    try:
        for ai, row_idx in enumerate(anchors):
            t0 = time.time()
            _check_contiguous(
                row_idx, episode_idx, step_idx, max(args.goal_offset, max_raw)
            )
            init_state = state[row_idx].copy()
            fixed_goal_state = state[row_idx + args.goal_offset].copy()
            controlled_seed = args.env_seed + ai
            current_image, _ = _current_goal_images(
                env, init_state, fixed_goal_state, controlled_seed
            )

            for hi, h in enumerate(horizons):
                raw_h = h * args.action_block
                action_dim = raw_h * 2
                expert_actions = action[row_idx:row_idx + raw_h].copy()

                scale = (
                    np.sqrt(h / args.reference_horizon)
                    if args.radius_scaling == "rms" else 1.0
                )
                actual_radii = base_radii * scale
                rng = np.random.default_rng(
                    args.seed + 1_000_003 * (ai + 1) + 10_007 * h
                )
                cands, br_meta, ar_meta, d_meta, s_meta, pn_meta, eqerr = _make_candidates(
                    expert_actions, actual_radii, base_radii, nD, rng
                )
                max_eqerr = max(max_eqerr, eqerr)

                for di in range(nD):
                    ip = np.where((d_meta == di) & (s_meta == 1) & np.isclose(br_meta, base_radii[0]))[0]
                    im = np.where((d_meta == di) & (s_meta == -1) & np.isclose(br_meta, base_radii[0]))[0]
                    if len(ip) != 1 or len(im) != 1:
                        raise RuntimeError(f"Malformed +/- pair h={h}, dir={di}")
                    ar0 = float(actual_radii[0])
                    v = (cands[ip[0]].reshape(-1) - cands[im[0]].reshape(-1)) / (2.0 * ar0)
                    probe_dirs[hi, ai, di, :action_dim] = v.astype(np.float32)

                images = []
                final_states = np.empty((len(cands), 7), dtype=np.float64)
                had_contact = np.empty(len(cands), dtype=bool)

                for ci in range(len(cands)):
                    fs, fi, hc, _ = _rollout(
                        env, init_state, fixed_goal_state, cands[ci], controlled_seed
                    )
                    final_states[ci] = fs
                    images.append(fi)
                    had_contact[ci] = hc

                phys_factor = _state_factor(final_states, args.world_size).astype(np.float64)
                normalized = _normalize_actions(cands, scaler, h, args.action_block)

                for mi, (label, model) in enumerate(zip(labels, models)):
                    zr = _encode(
                        model, transform, images, device, args.model_batch_size
                    ).cpu().numpy().astype(np.float64)
                    zp = _predict(
                        model, transform, current_image, normalized,
                        device, args.model_batch_size
                    ).cpu().numpy().astype(np.float64)

                    for ri, (br, ar) in enumerate(zip(base_radii, actual_radii)):
                        V = np.asarray(
                            probe_dirs[hi, ai, :, :action_dim], dtype=np.float64
                        )
                        Renc = np.empty((nD, latent_dim), dtype=np.float64)
                        Rpred = np.empty_like(Renc)
                        Rphys = np.empty((nD, 6), dtype=np.float64)

                        for di in range(nD):
                            ip = np.where(
                                np.isclose(br_meta, br)
                                & (d_meta == di) & (s_meta == 1)
                            )[0]
                            im = np.where(
                                np.isclose(br_meta, br)
                                & (d_meta == di) & (s_meta == -1)
                            )[0]
                            if len(ip) != 1 or len(im) != 1:
                                raise RuntimeError(
                                    f"Malformed +/- pair h={h}, r={br}, dir={di}"
                                )
                            ip, im = int(ip[0]), int(im[0])

                            enc_v = (zr[ip] - zr[im]) / (2.0 * float(ar))
                            pred_v = (zp[ip] - zp[im]) / (2.0 * float(ar))
                            phys_v = (phys_factor[ip] - phys_factor[im]) / (2.0 * float(ar))

                            Renc[di] = enc_v
                            Rpred[di] = pred_v
                            Rphys[di] = phys_v

                            enc_jvp_store[mi, hi, ai, ri, di] = enc_v.astype(np.float32)
                            pred_jvp_store[mi, hi, ai, ri, di] = pred_v.astype(np.float32)
                            if mi == 0:
                                phys_jvp_store[hi, ai, ri, di] = phys_v.astype(np.float32)

                            mode = _contact_mode(had_contact[ip], had_contact[im])
                            if mi == 0:
                                pair_contact_mode[hi, ai, ri, di] = contact_code[mode]

                            met = _pair_metrics(enc_v, pred_v, phys_v)
                            pair_rows.append({
                                "model": label,
                                "model_idx": mi,
                                "anchor": ai,
                                "dataset_row": int(row_idx),
                                "episode_idx": int(episode_idx[row_idx]),
                                "step_idx": int(step_idx[row_idx]),
                                "horizon": int(h),
                                "raw_horizon": int(raw_h),
                                "action_dim": int(action_dim),
                                "base_radius": float(br),
                                "actual_radius": float(ar),
                                "direction": int(di),
                                "plus_contact": int(had_contact[ip]),
                                "minus_contact": int(had_contact[im]),
                                "contact_mode": mode,
                                **met,
                            })

                        je = _jacobian_geometry(V, Renc, args.pinv_rcond)
                        jp = _jacobian_geometry(V, Rpred, args.pinv_rcond)
                        jph = _jacobian_geometry(V, Rphys, args.pinv_rcond)

                        gcos_pe, grel_pe, tr_pe, shape_pe = _gram_metrics(je["G"], jp["G"])
                        gcos_phys_e, _, _, _ = _gram_metrics(jph["G"], je["G"])
                        gcos_phys_p, _, _, _ = _gram_metrics(jph["G"], jp["G"])

                        pred_in_enc_null, enc_null_dim = _null_energy_fraction(
                            jp["G"], je["Vt"], je["s"], args.null_sv_frac
                        )
                        enc_in_pred_null, pred_null_dim = _null_energy_fraction(
                            je["G"], jp["Vt"], jp["s"], args.null_sv_frac
                        )

                        k = min(3, action_dim)
                        topk_overlap = _subspace_overlap(
                            je["Vt"], jp["Vt"], k
                        )

                        geometry_rows.append({
                            "model": label,
                            "model_idx": mi,
                            "anchor": ai,
                            "dataset_row": int(row_idx),
                            "horizon": int(h),
                            "raw_horizon": int(raw_h),
                            "action_dim": int(action_dim),
                            "base_radius": float(br),
                            "actual_radius": float(ar),
                            "num_directions": int(nD),
                            "probe_rank": int(je["probe_rank"]),
                            "probe_condition": float(je["probe_condition"]),
                            "gram_cos_pred_enc": gcos_pe,
                            "gram_rel_error_pred_enc": grel_pe,
                            "gram_trace_ratio_pred_enc": tr_pe,
                            "gram_shape_rel_error_pred_enc": shape_pe,
                            "gram_cos_enc_phys": gcos_phys_e,
                            "gram_cos_pred_phys": gcos_phys_p,
                            "top3_action_subspace_overlap": topk_overlap,
                            "enc_stable_rank": _stable_rank(je["s"]),
                            "pred_stable_rank": _stable_rank(jp["s"]),
                            "phys_stable_rank": _stable_rank(jph["s"]),
                            "enc_rank90": _energy_rank(je["s"], 0.90),
                            "pred_rank90": _energy_rank(jp["s"], 0.90),
                            "phys_rank90": _energy_rank(jph["s"], 0.90),
                            "enc_rank95": _energy_rank(je["s"], 0.95),
                            "pred_rank95": _energy_rank(jp["s"], 0.95),
                            "phys_rank95": _energy_rank(jph["s"], 0.95),
                            "enc_null_dim": int(enc_null_dim),
                            "pred_null_dim": int(pred_null_dim),
                            "pred_energy_in_enc_null_fraction": pred_in_enc_null,
                            "enc_energy_in_pred_null_fraction": enc_in_pred_null,
                            "enc_top_sv": float(je["s"][0]) if len(je["s"]) else 0.0,
                            "pred_top_sv": float(jp["s"][0]) if len(jp["s"]) else 0.0,
                            "phys_top_sv": float(jph["s"][0]) if len(jph["s"]) else 0.0,
                        })

            elapsed = time.time() - t0
            eta = (time.time() - start) / (ai + 1) * (nA - ai - 1)
            print(
                f"anchor {ai+1:3d}/{nA}: row={row_idx} "
                f"eqerr<={max_eqerr:.1e} time={elapsed:.2f}s ETA={eta/60:.1f}min"
            )
    finally:
        env.close()

    pair_csv = outdir / "pair_response_metrics.csv"
    geom_csv = outdir / "anchor_geometry_metrics.csv"
    _write_csv(pair_csv, pair_rows)
    _write_csv(geom_csv, geometry_rows)

    summary_rows = []
    groups = ["all", "no_contact", "both_contact", "contact_switch"]
    for label in labels:
        for h in horizons:
            for br in base_radii:
                base = [
                    r for r in pair_rows
                    if r["model"] == label
                    and r["horizon"] == h
                    and abs(r["base_radius"] - br) < 1e-8
                ]
                for group in groups:
                    rows = base if group == "all" else [r for r in base if r["contact_mode"] == group]
                    if not rows:
                        continue
                    summary_rows.append({
                        "model": label,
                        "horizon": int(h),
                        "base_radius": float(br),
                        "group": group,
                        **_summarize_pairs(
                            rows, args.weak_quantile, args.strong_quantile
                        ),
                    })

    summary_csv = outdir / "summary.csv"
    _write_csv(summary_csv, summary_rows)

    npz_path = outdir / "latent_response_vectors.npz"
    np.savez_compressed(
        npz_path,
        anchors=anchors,
        horizons=np.asarray(horizons, dtype=np.int32),
        base_radii=base_radii.astype(np.float32),
        labels=np.asarray(labels, dtype=object),
        policies=np.asarray(args.policies, dtype=object),
        probe_action_directions=probe_dirs,
        enc_jvp=enc_jvp_store,
        pred_jvp=pred_jvp_store,
        phys_jvp=phys_jvp_store,
        pair_contact_mode=pair_contact_mode,
        contact_mode_labels=np.asarray(
            ["no_contact", "both_contact", "contact_switch"], dtype=object
        ),
    )

    meta = {
        "config": {
            "dataset": dataset_name,
            "num_anchors": nA,
            "horizons": horizons,
            "reference_horizon": args.reference_horizon,
            "base_radii": base_radii.tolist(),
            "radius_scaling": args.radius_scaling,
            "action_block": args.action_block,
            "goal_offset_raw": args.goal_offset,
            "num_directions": nD,
            "seed": args.seed,
            "weak_quantile": args.weak_quantile,
            "strong_quantile": args.strong_quantile,
            "null_sv_frac": args.null_sv_frac,
            "pinv_rcond": args.pinv_rcond,
            "primary_reference": (
                "Direct symmetric endpoint latent response: "
                "[E(real U+) - E(real U-)]/(2r). No goal cost."
            ),
            "jacobian_reconstruction": (
                "Least-squares J from R=V J^T using identical symmetric action probes."
            ),
        },
        "elapsed_seconds": float(time.time() - start),
        "max_equal_norm_error": float(max_eqerr),
        "outputs": {
            "pair_response_metrics": str(pair_csv),
            "anchor_geometry_metrics": str(geom_csv),
            "summary": str(summary_csv),
            "vectors": str(npz_path),
        },
    }
    with (outdir / "summary.json").open("w") as f:
        json.dump(_jsonable(meta), f, indent=2)

    print()
    print("===== DIRECT LATENT RESPONSE SUMMARY (group=all) =====")
    hdr = (
        f"{'model':<14} {'H':>2} {'r':>5} {'n':>6} "
        f"{'rhoGain':>8} {'slope':>7} {'cosMed':>7} {'gainMed':>8} "
        f"{'weakOver':>9} {'strongUnder':>11} {'strongCos':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary_rows:
        if r["group"] != "all":
            continue

        def ff(x):
            return "nan" if x is None or not np.isfinite(float(x)) else f"{float(x):.3f}"

        print(
            f"{r['model']:<14} {int(r['horizon']):>2d} {r['base_radius']:>5.2f} "
            f"{int(r['n_pairs']):>6d} "
            f"{ff(r['gain_spearman']):>8} {ff(r['gain_slope_origin']):>7} "
            f"{ff(r['median_cosine']):>7} {ff(r['median_gain_ratio']):>8} "
            f"{ff(r['weak_overresponse_rate']):>9} "
            f"{ff(r['strong_underresponse_rate']):>11} "
            f"{ff(r['strong_median_cosine']):>9}"
        )

    print()
    print(f"Max equal-norm error: {max_eqerr:.3e}")
    print(f"Elapsed: {meta['elapsed_seconds']:.1f}s ({meta['elapsed_seconds']/60:.2f} min)")
    print(f"Saved: {pair_csv}")
    print(f"Saved: {geom_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
