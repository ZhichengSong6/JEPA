#!/usr/bin/env python3
"""Planner-facing Stage-I mechanism comparison for full PushT.

This evaluator is intentionally narrow.  It tests whether Stage I actually
reduces the two quantities its training objective is designed to reduce:

  e_H = z_pred,H(U0) - z_enc,H(U0)

and the component of that nominal endpoint bias along local predicted
action-response directions

  q(v) = normalize((z_pred,H(U0+r v)-z_pred,H(U0-r v))/(2r)).

For every anchor and horizon we report

  endpoint_mse              = mean_d e_H[d]^2
  projected_bias_raw        = mean_v (e_H^T q(v))^2
  projected_bias_per_dim    = projected_bias_raw / latent_dim
  projected_bias_abs        = mean_v |e_H^T q(v)|
  projected_alignment_cos2  = projected_bias_raw / ||e_H||^2

The final metric is scale-free: it is the average squared cosine between the
endpoint-bias vector and sampled local predicted response directions.

To make the comparison directly compatible with
``eval_pusht_fixed_action_response_horizon.py``, this script uses the same:
  * anchor selection (seed=42 by default),
  * first coarse action block only (5 raw 2-D actions = 10-D),
  * exact raw-action L2 radius 0.1565,
  * bounded symmetric perturbations,
  * horizons H in {1,2,3,5}, and
  * planner-facing one-image rollout path used by ``model.rollout``.

No simulator state is used in any model-side metric.  Simulator state is used
only to reset the environment so the nominal expert rollout can be rendered.
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
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--response-eps", type=float, default=1e-8)
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


def _finite(values):
    x = np.asarray(values, dtype=np.float64)
    return x[np.isfinite(x)]


def _mean(values):
    x = _finite(values)
    return float(np.mean(x)) if len(x) else float("nan")


def _median(values):
    x = _finite(values)
    return float(np.median(x)) if len(x) else float("nan")


def _q(values, percentile):
    x = _finite(values)
    return float(np.percentile(x, percentile)) if len(x) else float("nan")


def _summarize(rows):
    metrics = [
        "endpoint_mse",
        "endpoint_l2",
        "projected_bias_raw",
        "projected_bias_per_dim",
        "projected_bias_abs",
        "projected_alignment_cos2",
        "response_norm_mean",
        "response_norm_median",
    ]
    out = {"n_anchors": len(rows)}
    for key in metrics:
        vals = [r[key] for r in rows]
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_median"] = _median(vals)
        out[f"{key}_p10"] = _q(vals, 10)
        out[f"{key}_p90"] = _q(vals, 90)
    return out


def _paired_comparison(base_rows, test_rows, base_label, test_label, horizon):
    b = {int(r["anchor_index"]): r for r in base_rows}
    t = {int(r["anchor_index"]): r for r in test_rows}
    ids = sorted(set(b) & set(t))
    metrics = [
        "endpoint_mse",
        "projected_bias_per_dim",
        "projected_bias_abs",
        "projected_alignment_cos2",
        "response_norm_mean",
    ]
    out = {
        "horizon": int(horizon),
        "baseline": base_label,
        "comparison": test_label,
        "n_paired_anchors": len(ids),
    }
    for key in metrics:
        bv = np.asarray([b[i][key] for i in ids], dtype=np.float64)
        tv = np.asarray([t[i][key] for i in ids], dtype=np.float64)
        mask = np.isfinite(bv) & np.isfinite(tv)
        bv, tv = bv[mask], tv[mask]
        bm = float(np.mean(bv)) if len(bv) else float("nan")
        tm = float(np.mean(tv)) if len(tv) else float("nan")
        out[f"{key}_baseline_mean"] = bm
        out[f"{key}_comparison_mean"] = tm
        out[f"{key}_mean_ratio"] = tm / bm if np.isfinite(bm) and abs(bm) > 1e-20 else float("nan")
        out[f"{key}_relative_change_pct"] = (
            100.0 * (tm - bm) / bm
            if np.isfinite(bm) and abs(bm) > 1e-20
            else float("nan")
        )
        paired_ratio = tv / np.maximum(np.abs(bv), 1e-20)
        out[f"{key}_paired_ratio_median"] = _median(paired_ratio)
    return out


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels length must equal --policies length")
    if len(args.policies) < 2:
        raise ValueError("Provide at least two policies for a comparison.")
    if args.radius <= 0:
        raise ValueError("--radius must be positive")
    if args.num_directions < 1:
        raise ValueError("--num-directions must be positive")

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
        else cache_root / "stage1_bias_mechanism_eval"
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

    rows = []
    start = time.time()
    max_eqerr = 0.0

    try:
        for ai, row_idx in enumerate(anchors):
            t0 = time.time()
            _check_contiguous(row_idx, episode_idx, step_idx, max(args.goal_offset, max_raw))
            init_state = state[row_idx].copy()
            goal_state = state[row_idx + args.goal_offset].copy()
            seed = args.env_seed + ai
            current_image, _ = _current_goal_images(env, init_state, goal_state, seed)
            expert_max = action[row_idx : row_idx + max_raw].copy()

            # Same direction construction and per-anchor RNG as the fixed-action
            # landscape evaluator, so the probes match exactly for the same seed.
            rng = np.random.default_rng(args.seed + 1_000_003 * (ai + 1))
            cands, _, dmeta, smeta, _, eqerr = _make_fixed_first_block_candidates(
                expert_max,
                args.radius,
                args.num_directions,
                rng,
                args.action_block,
            )
            max_eqerr = max(max_eqerr, eqerr)

            # Only the nominal physical rollout is required.  Counterfactual
            # +/- simulator endpoints are deliberately not used in these metrics.
            nominal = _rollout_checkpoints(
                env,
                init_state,
                goal_state,
                expert_max,
                checkpoints,
                seed,
            )

            for label, model in zip(labels, models):
                for h in horizons:
                    raw_h = h * args.action_block
                    z_enc = _encode(
                        model,
                        transform,
                        [nominal[raw_h]["image"]],
                        device,
                        args.model_batch_size,
                    )[0]

                    normalized = _normalize_actions(
                        cands[:, :raw_h], scaler, h, args.action_block
                    )
                    z_pred_all = _predict(
                        model,
                        transform,
                        current_image,
                        normalized,
                        device,
                        args.model_batch_size,
                    )
                    z_pred = z_pred_all[0]
                    endpoint_error = z_pred - z_enc
                    latent_dim = int(endpoint_error.numel())
                    endpoint_l2 = float(torch.linalg.vector_norm(endpoint_error).item())
                    endpoint_mse = float(endpoint_error.pow(2).mean().item())

                    projections = []
                    response_norms = []
                    for di in range(args.num_directions):
                        ip = np.where((dmeta == di) & (smeta > 0))[0]
                        im = np.where((dmeta == di) & (smeta < 0))[0]
                        if len(ip) != 1 or len(im) != 1:
                            raise RuntimeError(f"Malformed +/- pair direction={di}")
                        ip, im = int(ip[0]), int(im[0])
                        response = (z_pred_all[ip] - z_pred_all[im]) / (2.0 * args.radius)
                        response_norm = torch.linalg.vector_norm(response)
                        rn = float(response_norm.item())
                        response_norms.append(rn)
                        if rn <= args.response_eps:
                            continue
                        q = response / response_norm
                        projections.append(float(torch.dot(endpoint_error, q).item()))

                    if not projections:
                        raise RuntimeError(
                            f"No nonzero response directions for model={label}, H={h}, anchor={ai+1}."
                        )

                    proj = np.asarray(projections, dtype=np.float64)
                    raw_apb = float(np.mean(proj ** 2))
                    apb_per_dim = raw_apb / float(latent_dim)
                    alignment_cos2 = raw_apb / max(endpoint_l2 ** 2, 1e-20)

                    rows.append(
                        {
                            "model": label,
                            "anchor_index": ai + 1,
                            "dataset_row": int(row_idx),
                            "horizon": int(h),
                            "raw_horizon": int(raw_h),
                            "latent_dim": latent_dim,
                            "radius": float(args.radius),
                            "num_directions": int(args.num_directions),
                            "num_valid_response_directions": int(len(projections)),
                            "nominal_had_contact": int(nominal[raw_h]["had_contact"]),
                            "endpoint_mse": endpoint_mse,
                            "endpoint_l2": endpoint_l2,
                            "projected_bias_raw": raw_apb,
                            "projected_bias_per_dim": apb_per_dim,
                            "projected_bias_abs": float(np.mean(np.abs(proj))),
                            "projected_alignment_cos2": alignment_cos2,
                            "response_norm_mean": float(np.mean(response_norms)),
                            "response_norm_median": float(np.median(response_norms)),
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

    anchor_csv = outdir / "anchor_bias_metrics.csv"
    _write_csv(anchor_csv, rows)

    summary_rows = []
    for label in labels:
        for h in horizons:
            subset = [r for r in rows if r["model"] == label and r["horizon"] == h]
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

    comparison_rows = []
    base_label = labels[0]
    for test_label in labels[1:]:
        for h in horizons:
            base_rows = [r for r in rows if r["model"] == base_label and r["horizon"] == h]
            test_rows = [r for r in rows if r["model"] == test_label and r["horizon"] == h]
            comparison_rows.append(
                _paired_comparison(base_rows, test_rows, base_label, test_label, h)
            )

    comparison_csv = outdir / "comparison.csv"
    _write_csv(comparison_csv, comparison_rows)

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
                    "rollout_context": "planner-facing one current image via model.rollout",
                },
                "anchors": anchors.tolist(),
                "elapsed_seconds": float(time.time() - start),
                "summary_rows": summary_rows,
                "comparison_rows": comparison_rows,
            },
            f,
            indent=2,
        )

    print()
    hdr = (
        f"{'model':<12} {'H':>2} {'endMSE':>10} {'APB/D':>10} "
        f"{'|proj|':>10} {'cos2':>9} {'resp':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summary_rows:
        print(
            f"{s['model']:<12} {s['horizon']:>2d} "
            f"{s['endpoint_mse_mean']:>10.5f} "
            f"{s['projected_bias_per_dim_mean']:>10.5f} "
            f"{s['projected_bias_abs_mean']:>10.5f} "
            f"{s['projected_alignment_cos2_mean']:>9.5f} "
            f"{s['response_norm_mean_mean']:>9.4f}"
        )

    print("\nPaired relative change vs first policy (negative means reduction):")
    for c in comparison_rows:
        print(
            f"H={c['horizon']}: "
            f"endpoint_mse {c['endpoint_mse_relative_change_pct']:+.1f}% | "
            f"APB/D {c['projected_bias_per_dim_relative_change_pct']:+.1f}% | "
            f"alignment cos2 {c['projected_alignment_cos2_relative_change_pct']:+.1f}% | "
            f"response norm {c['response_norm_mean_relative_change_pct']:+.1f}%"
        )

    print(f"\nMax equal-norm error: {max_eqerr:.3e}")
    print(f"Saved: {anchor_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {comparison_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
