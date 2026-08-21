#!/usr/bin/env python3
"""Goal-conditioned local gradient-error decomposition for full PushT.

Scientific question
-------------------
Stage I substantially reduces planner-facing endpoint bias and action-projected
bias, yet the H=5 CEM-facing landscape remains poor.  This evaluator asks what
first-order term remains.

For a nominal expert action sequence U0, let

    d = z_enc(U0) - z_goal
    e = z_pred(U0) - z_enc(U0)
    J = d z_enc / dU
    Jhat = d z_pred / dU

and define the latent goal costs

    C(U)    = ||z_enc(U)  - z_goal||^2
    Chat(U) = ||z_pred(U) - z_goal||^2.

Their linearized action gradients are

    g_true = 2 J^T d
    g_pred = 2 Jhat^T (d + e)

so the linearized gradient error decomposes exactly as

    g_pred - g_true = A + B

with

    A = 2 Jhat^T e
    B = 2 (Jhat - J)^T d.

A is the endpoint-bias/action-response interaction targeted by Stage I APB.
B is the goal-conditioned Jacobian mismatch term not explicitly constrained by
Stage I.

The evaluator uses the same fixed first 10-D action block, radius=0.1565,
anchors, and symmetric probe directions as
``eval_pusht_fixed_action_response_horizon.py``.

A necessary finite-difference sanity check is also reported.  From the same
plus/minus candidate latent goal costs we reconstruct direct local cost
gradients g_true_fd and g_pred_fd.  This tests whether the first-order
J^T d decomposition actually describes the finite-radius optimizer-facing
landscape rather than merely satisfying an algebraic identity.

Primary outputs
---------------
For each model / anchor / horizon:
  * cosine(g_pred_linear, g_true_linear)
  * cosine(g_pred_fd, g_true_fd)
  * ||A|| / ||g_true_linear||
  * ||B|| / ||g_true_linear||
  * ||B|| / ||A||
  * cosine(A, linear_gradient_error)
  * cosine(B, linear_gradient_error)
  * algebraic closure relative error
  * encoder/predictor linear-vs-FD gradient cosine
  * cosine(A+B, FD gradient error)
  * relative residual of A+B against FD gradient error

No physical state is used in any model-side metric.  Simulator state is used
only to reset and replay counterfactual trajectories so their terminal images
can be encoded.
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
from eval_pusht_fixed_action_response_horizon import (
    _make_fixed_first_block_candidates,
    _rollout_checkpoints,
)
from eval_pusht_horizon_directional import (
    _check_contiguous,
    _current_goal_images,
    _encode,
    _label,
    _normalize_actions,
    _predict,
    _select_rows,
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
    p.add_argument("--radius", type=float, default=0.1565)
    p.add_argument("--num-directions", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--pinv-rcond", type=float, default=1e-6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > eps else float("nan")


def _safe_ratio(num, den, eps=1e-12):
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > eps else float("nan")


def _finite(values):
    x = np.asarray(values, dtype=np.float64)
    return x[np.isfinite(x)]


def _mean(values):
    x = _finite(values)
    return float(np.mean(x)) if len(x) else float("nan")


def _median(values):
    x = _finite(values)
    return float(np.median(x)) if len(x) else float("nan")


def _quantile(values, q):
    x = _finite(values)
    return float(np.quantile(x, q)) if len(x) else float("nan")


def _reconstruct_jt(V, responses, rcond):
    """Least-squares J^T from directional responses R ~= V J^T."""
    V = np.asarray(V, dtype=np.float64)
    R = np.asarray(responses, dtype=np.float64)
    pinv = np.linalg.pinv(V, rcond=rcond)
    return pinv @ R  # [action_dim, latent_dim]


def _reconstruct_gradient(V, directional_derivatives, rcond):
    """Least-squares g from directional derivatives y ~= V g."""
    V = np.asarray(V, dtype=np.float64)
    y = np.asarray(directional_derivatives, dtype=np.float64).reshape(-1)
    return np.linalg.pinv(V, rcond=rcond) @ y


def _summarize(rows):
    metrics = [
        "grad_cos_pred_true_linear",
        "grad_cos_pred_true_fd",
        "termA_norm_over_true",
        "termB_norm_over_true",
        "termB_norm_over_A",
        "termA_cos_linear_error",
        "termB_cos_linear_error",
        "closure_rel_error",
        "true_linear_vs_fd_cos",
        "pred_linear_vs_fd_cos",
        "linear_error_vs_fd_error_cos",
        "linear_error_vs_fd_error_rel_residual",
        "true_grad_linear_norm",
        "pred_grad_linear_norm",
        "true_grad_fd_norm",
        "pred_grad_fd_norm",
        "termA_norm",
        "termB_norm",
        "linear_grad_error_norm",
        "fd_grad_error_norm",
        "endpoint_mse",
    ]
    out = {"n_anchors": len(rows)}
    for key in metrics:
        vals = [r[key] for r in rows]
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_median"] = _median(vals)
        out[f"{key}_p10"] = _quantile(vals, 0.10)
        out[f"{key}_p90"] = _quantile(vals, 0.90)
    return out


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels length must equal --policies length")
    if args.radius <= 0:
        raise ValueError("--radius must be positive")
    if args.num_directions < args.action_block * 2:
        raise ValueError(
            "Need at least as many probe directions as the fixed 10-D action space "
            f"for a determined reconstruction; got {args.num_directions}."
        )

    horizons = sorted(set(map(int, args.horizons)))
    if not horizons or min(horizons) <= 0:
        raise ValueError("--horizons must contain positive integers")
    if max(horizons) * args.action_block > args.goal_offset:
        raise ValueError("max horizon endpoint must be <= --goal-offset")

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else cache_root / "goal_gradient_error_decomposition"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=["action", "state"],
        cache_dir=cache_root,
    )
    col, anchors = _select_rows(dataset, args.num_anchors, args.seed, args.goal_offset)
    print("Selected paired eval rows:")
    print(anchors)

    episode_idx = np.asarray(dataset.get_col_data(col))
    step_idx = np.asarray(dataset.get_col_data("step_idx"))
    action = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    state = np.asarray(dataset.get_col_data("state"), dtype=np.float64)
    finite_actions = action[np.isfinite(action).all(axis=1)]
    scaler = preprocessing.StandardScaler().fit(finite_actions)

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
    max_h = max(horizons)
    max_raw = max_h * args.action_block
    checkpoints = [h * args.action_block for h in horizons]
    nC = 1 + 2 * args.num_directions

    rows = []
    start = time.time()
    max_eqerr = 0.0

    try:
        for ai, row_idx in enumerate(anchors):
            t0 = time.time()
            _check_contiguous(
                row_idx, episode_idx, step_idx, max(args.goal_offset, max_raw)
            )
            init_state = state[row_idx].copy()
            goal_state = state[row_idx + args.goal_offset].copy()
            seed = args.env_seed + ai
            current_image, goal_image = _current_goal_images(
                env, init_state, goal_state, seed
            )
            expert_max = action[row_idx : row_idx + max_raw].copy()

            rng = np.random.default_rng(args.seed + 1_000_003 * (ai + 1))
            cands, dirs, dmeta, smeta, _, eqerr = _make_fixed_first_block_candidates(
                expert_max,
                args.radius,
                args.num_directions,
                rng,
                args.action_block,
            )
            max_eqerr = max(max_eqerr, eqerr)
            V = np.asarray(dirs, dtype=np.float64)
            probe_s = np.linalg.svd(V, compute_uv=False)
            probe_tol = args.pinv_rcond * probe_s[0]
            probe_rank = int(np.sum(probe_s > probe_tol))
            probe_condition = float(probe_s[0] / probe_s[-1])

            # Replay every candidate once to max H and retain terminal images at
            # all requested horizons.  This is the same physical reference used
            # by the fixed-action landscape evaluator.
            real_images = {h: [None] * nC for h in horizons}
            for ci in range(nC):
                out = _rollout_checkpoints(
                    env, init_state, goal_state, cands[ci], checkpoints, seed
                )
                for h in horizons:
                    real_images[h][ci] = out[h * args.action_block]["image"]

            for label, model in zip(labels, models):
                zg = _encode(
                    model, transform, [goal_image], device, args.model_batch_size
                )[0].detach().cpu().numpy().astype(np.float64)

                for h in horizons:
                    raw_h = h * args.action_block
                    zr = _encode(
                        model,
                        transform,
                        real_images[h],
                        device,
                        args.model_batch_size,
                    ).detach().cpu().numpy().astype(np.float64)
                    normalized = _normalize_actions(
                        cands[:, :raw_h], scaler, h, args.action_block
                    )
                    zp = _predict(
                        model,
                        transform,
                        current_image,
                        normalized,
                        device,
                        args.model_batch_size,
                    ).detach().cpu().numpy().astype(np.float64)

                    z_enc0 = zr[0]
                    z_pred0 = zp[0]
                    d = z_enc0 - zg
                    e = z_pred0 - z_enc0

                    enc_responses = []
                    pred_responses = []
                    enc_cost_dir_deriv = []
                    pred_cost_dir_deriv = []

                    enc_cost = np.sum((zr - zg[None]) ** 2, axis=-1)
                    pred_cost = np.sum((zp - zg[None]) ** 2, axis=-1)

                    for di in range(args.num_directions):
                        ip = np.where((dmeta == di) & (smeta > 0))[0]
                        im = np.where((dmeta == di) & (smeta < 0))[0]
                        if len(ip) != 1 or len(im) != 1:
                            raise RuntimeError(
                                f"Malformed +/- pair direction={di}"
                            )
                        ip, im = int(ip[0]), int(im[0])

                        enc_responses.append(
                            (zr[ip] - zr[im]) / (2.0 * args.radius)
                        )
                        pred_responses.append(
                            (zp[ip] - zp[im]) / (2.0 * args.radius)
                        )
                        enc_cost_dir_deriv.append(
                            (enc_cost[ip] - enc_cost[im]) / (2.0 * args.radius)
                        )
                        pred_cost_dir_deriv.append(
                            (pred_cost[ip] - pred_cost[im]) / (2.0 * args.radius)
                        )

                    Renc = np.asarray(enc_responses, dtype=np.float64)
                    Rpred = np.asarray(pred_responses, dtype=np.float64)
                    JT = _reconstruct_jt(V, Renc, args.pinv_rcond)
                    JThat = _reconstruct_jt(V, Rpred, args.pinv_rcond)

                    # Full gradients include the common factor 2 from squared
                    # Euclidean latent goal cost.
                    g_true = 2.0 * (JT @ d)
                    g_pred = 2.0 * (JThat @ (d + e))
                    termA = 2.0 * (JThat @ e)
                    termB = 2.0 * ((JThat - JT) @ d)
                    linear_error = g_pred - g_true
                    closure = termA + termB

                    g_true_fd = _reconstruct_gradient(
                        V, enc_cost_dir_deriv, args.pinv_rcond
                    )
                    g_pred_fd = _reconstruct_gradient(
                        V, pred_cost_dir_deriv, args.pinv_rcond
                    )
                    fd_error = g_pred_fd - g_true_fd

                    true_norm = float(np.linalg.norm(g_true))
                    pred_norm = float(np.linalg.norm(g_pred))
                    a_norm = float(np.linalg.norm(termA))
                    b_norm = float(np.linalg.norm(termB))
                    lin_err_norm = float(np.linalg.norm(linear_error))
                    fd_err_norm = float(np.linalg.norm(fd_error))
                    closure_residual = float(np.linalg.norm(linear_error - closure))
                    fd_residual = float(np.linalg.norm(fd_error - closure))

                    rows.append(
                        {
                            "model": label,
                            "anchor_index": ai + 1,
                            "dataset_row": int(row_idx),
                            "horizon": int(h),
                            "raw_horizon": int(raw_h),
                            "radius": float(args.radius),
                            "action_dim": int(V.shape[1]),
                            "num_directions": int(args.num_directions),
                            "probe_rank": probe_rank,
                            "probe_condition": probe_condition,
                            "endpoint_mse": float(np.mean(e ** 2)),
                            "true_grad_linear_norm": true_norm,
                            "pred_grad_linear_norm": pred_norm,
                            "true_grad_fd_norm": float(np.linalg.norm(g_true_fd)),
                            "pred_grad_fd_norm": float(np.linalg.norm(g_pred_fd)),
                            "grad_cos_pred_true_linear": _cosine(g_pred, g_true),
                            "grad_cos_pred_true_fd": _cosine(g_pred_fd, g_true_fd),
                            "termA_norm": a_norm,
                            "termB_norm": b_norm,
                            "termA_norm_over_true": _safe_ratio(a_norm, true_norm),
                            "termB_norm_over_true": _safe_ratio(b_norm, true_norm),
                            "termB_norm_over_A": _safe_ratio(b_norm, a_norm),
                            "linear_grad_error_norm": lin_err_norm,
                            "fd_grad_error_norm": fd_err_norm,
                            "termA_cos_linear_error": _cosine(termA, linear_error),
                            "termB_cos_linear_error": _cosine(termB, linear_error),
                            "termA_cos_fd_error": _cosine(termA, fd_error),
                            "termB_cos_fd_error": _cosine(termB, fd_error),
                            "closure_rel_error": _safe_ratio(
                                closure_residual, lin_err_norm
                            ),
                            "true_linear_vs_fd_cos": _cosine(g_true, g_true_fd),
                            "pred_linear_vs_fd_cos": _cosine(g_pred, g_pred_fd),
                            "linear_error_vs_fd_error_cos": _cosine(
                                closure, fd_error
                            ),
                            "linear_error_vs_fd_error_rel_residual": _safe_ratio(
                                fd_residual, fd_err_norm
                            ),
                        }
                    )

            elapsed = time.time() - t0
            eta = (time.time() - start) / (ai + 1) * (len(anchors) - ai - 1)
            print(
                f"anchor {ai+1:3d}/{len(anchors)} row={row_idx} "
                f"eqerr<={max_eqerr:.1e} time={elapsed:.2f}s ETA={eta/60:.1f}min"
            )
    finally:
        env.close()

    anchor_csv = outdir / "anchor_gradient_decomposition.csv"
    _write_csv(anchor_csv, rows)

    summary_rows = []
    for label in labels:
        for h in horizons:
            subset = [
                r for r in rows if r["model"] == label and r["horizon"] == h
            ]
            summary_rows.append(
                {
                    "model": label,
                    "horizon": int(h),
                    "raw_horizon": int(h * args.action_block),
                    "radius": float(args.radius),
                    **_summarize(subset),
                }
            )

    summary_csv = outdir / "summary.csv"
    _write_csv(summary_csv, summary_rows)

    summary_json = outdir / "summary.json"
    with summary_json.open("w") as f:
        json.dump(
            {
                "config": {
                    "dataset": dataset_name,
                    "num_anchors": int(len(anchors)),
                    "horizons": horizons,
                    "action_block": int(args.action_block),
                    "fixed_perturbed_action_dim": int(args.action_block * 2),
                    "radius": float(args.radius),
                    "num_directions": int(args.num_directions),
                    "seed": int(args.seed),
                    "env_seed": int(args.env_seed),
                    "pinv_rcond": float(args.pinv_rcond),
                },
                "anchors": anchors.tolist(),
                "elapsed_seconds": float(time.time() - start),
                "summary_rows": summary_rows,
            },
            f,
            indent=2,
        )

    print()
    hdr = (
        f"{'model':<12} {'H':>2} {'gCosLin':>8} {'gCosFD':>8} "
        f"{'A/true':>8} {'B/true':>8} {'B/A':>8} "
        f"{'ABcosFD':>8} {'FDres':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summary_rows:
        print(
            f"{s['model']:<12} {s['horizon']:>2d} "
            f"{s['grad_cos_pred_true_linear_mean']:>8.3f} "
            f"{s['grad_cos_pred_true_fd_mean']:>8.3f} "
            f"{s['termA_norm_over_true_mean']:>8.3f} "
            f"{s['termB_norm_over_true_mean']:>8.3f} "
            f"{s['termB_norm_over_A_mean']:>8.3f} "
            f"{s['linear_error_vs_fd_error_cos_mean']:>8.3f} "
            f"{s['linear_error_vs_fd_error_rel_residual_mean']:>8.3f}"
        )

    print("\nLinearization sanity (1.0 cosine is ideal):")
    hdr2 = (
        f"{'model':<12} {'H':>2} {'encLinFD':>10} {'predLinFD':>10} "
        f"{'AcosErr':>9} {'BcosErr':>9} {'closure':>9}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for s in summary_rows:
        print(
            f"{s['model']:<12} {s['horizon']:>2d} "
            f"{s['true_linear_vs_fd_cos_mean']:>10.3f} "
            f"{s['pred_linear_vs_fd_cos_mean']:>10.3f} "
            f"{s['termA_cos_linear_error_mean']:>9.3f} "
            f"{s['termB_cos_linear_error_mean']:>9.3f} "
            f"{s['closure_rel_error_mean']:>9.2e}"
        )

    print(f"\nMax equal-norm error: {max_eqerr:.3e}")
    print(f"Saved: {anchor_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
