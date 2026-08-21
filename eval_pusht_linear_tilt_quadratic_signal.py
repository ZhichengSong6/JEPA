#!/usr/bin/env python3
"""Evaluate spurious first-order tilt versus true quadratic signal in PushT.

This is a narrow, mechanism-focused diagnostic for LeWM-family models.
It reuses the same fixed 10-D first-action-block perturbations used by
``eval_pusht_fixed_action_response_horizon.py`` and asks:

  1) Near the nominal expert endpoint, how large is the predictor's rollout-bias
     induced linear tilt

        L_spur(delta) = 2 e^T Jhat delta,
        e = zhat(U0) - z(U0),

     relative to the true quadratic action signal

        Q_true(delta) = delta^T J^T J delta ?

  2) If we remove ONLY that first-order rollout-bias tilt from the predicted
     latent goal cost,

        C_pred_debiased(delta)
          = C_pred(delta) - L_spur(delta),

     how much do planner-facing metrics recover?

The debiased cost is an oracle diagnostic, NOT a deployable planner: computing
``e`` requires the real nominal endpoint encoder embedding.  Its purpose is to
establish mechanism, not to propose test-time privileged information.

Outputs
-------
  anchor_metrics.csv
  summary.csv
  summary.json

Primary reported quantities
---------------------------
  tilt_to_quadvar_rms
      RMS(L_spur) / RMS(Q_true - mean(Q_true)).  The denominator measures the
      across-candidate quadratic variation that can actually affect ranking.

  rho_qtrue_enc
      Spearman correlation between the quadratic approximation Q_true and the
      exact encoder latent candidate cost.

  rho_pred_phys / rho_debiased_phys
  elite_overlap_pred_phys / elite_overlap_debiased_phys
  cem_update_cos_pred_phys / cem_update_cos_debiased_phys
  pred_selected_phys_percentile / debiased_selected_phys_percentile

The same exact candidate set is used for all metrics.
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
    _elite_metrics,
    _make_fixed_first_block_candidates,
    _rollout_checkpoints,
)
from eval_pusht_horizon_directional import (
    _check_contiguous,
    _current_goal_images,
    _encode,
    _label,
    _normalize_actions,
    _physical_cost,
    _predict,
    _select_rows,
    _spearman,
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
    p.add_argument("--elite-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--model-batch-size", type=int, default=64)
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
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _finite(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def _mean(x):
    x = _finite(x)
    return float(np.mean(x)) if len(x) else float("nan")


def _median(x):
    x = _finite(x)
    return float(np.median(x)) if len(x) else float("nan")


def _rms(x):
    x = _finite(x)
    return float(np.sqrt(np.mean(x * x))) if len(x) else float("nan")


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


def _score_metrics(phys_cost, enc_cost, score, first_deltas, elite_frac, prefix):
    """Planner-facing metrics for an arbitrary candidate score."""
    n = len(score)
    k = max(2, int(np.ceil(float(elite_frac) * n)))
    iscore = np.argsort(score)[:k]
    iphys = np.argsort(phys_cost)[:k]
    ienc = np.argsort(enc_cost)[:k]

    def overlap(a, b):
        return float(len(set(map(int, a)) & set(map(int, b))) / k)

    def mean_update(idx):
        return np.mean(np.asarray(first_deltas, dtype=np.float64)[idx], axis=0)

    uscore = mean_update(iscore)
    uphys = mean_update(iphys)
    uenc = mean_update(ienc)

    def cosine(a, b):
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")

    best = int(np.argmin(score))
    return {
        f"rho_{prefix}_phys": _spearman(score, phys_cost),
        f"rho_{prefix}_enc": _spearman(score, enc_cost),
        f"elite_overlap_{prefix}_phys": overlap(iscore, iphys),
        f"elite_overlap_{prefix}_enc": overlap(iscore, ienc),
        f"cem_update_cos_{prefix}_phys": cosine(uscore, uphys),
        f"cem_update_cos_{prefix}_enc": cosine(uscore, uenc),
        f"{prefix}_selected_phys_percentile": _rank_percentile(phys_cost, best),
        f"{prefix}_selected_phys_regret_norm": _normalized_regret(phys_cost, best),
    }


def _aggregate(rows):
    if not rows:
        return {}
    keys = [
        "endpoint_mse",
        "d_l2",
        "e_l2",
        "rms_spurious_linear",
        "rms_true_linear",
        "rms_pred_total_linear",
        "rms_true_quadratic",
        "rms_centered_true_quadratic",
        "rms_exact_true_even",
        "rms_centered_exact_true_even",
        "tilt_to_quad_rms",
        "tilt_to_quadvar_rms",
        "tilt_to_exact_evenvar_rms",
        "rho_qtrue_enc",
        "rho_qtrue_phys",
        "rho_exact_even_enc",
        "rho_pred_phys",
        "rho_pred_enc",
        "elite_overlap_pred_phys",
        "elite_overlap_pred_enc",
        "cem_update_cos_pred_phys",
        "cem_update_cos_pred_enc",
        "pred_selected_phys_percentile",
        "pred_selected_phys_regret_norm",
        "rho_debiased_phys",
        "rho_debiased_enc",
        "elite_overlap_debiased_phys",
        "elite_overlap_debiased_enc",
        "cem_update_cos_debiased_phys",
        "cem_update_cos_debiased_enc",
        "debiased_selected_phys_percentile",
        "debiased_selected_phys_regret_norm",
    ]
    out = {"n_anchors": len(rows)}
    for key in keys:
        vals = [r.get(key, np.nan) for r in rows]
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_median"] = _median(vals)
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
    if args.num_directions < 1:
        raise ValueError("--num-directions must be positive")

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else cache_root / "linear_tilt_quadratic_signal"
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
            _check_contiguous(row_idx, episode_idx, step_idx, max(args.goal_offset, max_raw))
            init_state = state[row_idx].copy()
            goal_state = state[row_idx + args.goal_offset].copy()
            seed = args.env_seed + ai
            current_image, goal_image = _current_goal_images(env, init_state, goal_state, seed)
            expert_max = action[row_idx : row_idx + max_raw].copy()

            rng = np.random.default_rng(args.seed + 1_000_003 * (ai + 1))
            cands, dirs, dmeta, smeta, first_deltas, eqerr = _make_fixed_first_block_candidates(
                expert_max,
                args.radius,
                args.num_directions,
                rng,
                args.action_block,
            )
            max_eqerr = max(max_eqerr, eqerr)

            # We need exact real endpoints for every +/- candidate because the
            # debiased-score diagnostic is evaluated against the same physical
            # candidate landscape used by the previous fixed-horizon evaluator.
            real_states = {
                h: np.empty((nC, state.shape[-1]), dtype=np.float64)
                for h in horizons
            }
            real_images = {h: [None] * nC for h in horizons}
            for ci in range(nC):
                out = _rollout_checkpoints(
                    env,
                    init_state,
                    goal_state,
                    cands[ci],
                    checkpoints,
                    seed,
                )
                for h in horizons:
                    raw_h = h * args.action_block
                    real_states[h][ci] = out[raw_h]["state"]
                    real_images[h][ci] = out[raw_h]["image"]

            for label, model in zip(labels, models):
                zg = _encode(
                    model,
                    transform,
                    [goal_image],
                    device,
                    args.model_batch_size,
                )[0]

                for h in horizons:
                    raw_h = h * args.action_block
                    zr = _encode(
                        model,
                        transform,
                        real_images[h],
                        device,
                        args.model_batch_size,
                    )
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
                    )

                    enc_cost = (
                        torch.sum((zr - zg[None]) ** 2, dim=-1)
                        .detach().cpu().numpy().astype(np.float64)
                    )
                    pred_cost = (
                        torch.sum((zp - zg[None]) ** 2, dim=-1)
                        .detach().cpu().numpy().astype(np.float64)
                    )
                    phys_cost, *_ = _physical_cost(real_states[h], goal_state)

                    z0 = zr[0]
                    zhat0 = zp[0]
                    d = z0 - zg
                    e = zhat0 - z0
                    endpoint_mse = float(e.pow(2).mean().item())
                    d_l2 = float(torch.linalg.vector_norm(d).item())
                    e_l2 = float(torch.linalg.vector_norm(e).item())

                    spurious_linear = np.zeros(nC, dtype=np.float64)
                    true_linear = np.zeros(nC, dtype=np.float64)
                    pred_total_linear = np.zeros(nC, dtype=np.float64)
                    q_true = np.zeros(nC, dtype=np.float64)
                    exact_true_even = np.zeros(nC, dtype=np.float64)

                    # Nominal candidate has zero delta and therefore zero local
                    # linear/quadratic increment by construction.
                    for di in range(args.num_directions):
                        ip = np.where((dmeta == di) & (smeta > 0))[0]
                        im = np.where((dmeta == di) & (smeta < 0))[0]
                        if len(ip) != 1 or len(im) != 1:
                            raise RuntimeError(f"Malformed +/- pair direction={di}")
                        ip, im = int(ip[0]), int(im[0])

                        jv = (zr[ip] - zr[im]) / (2.0 * args.radius)
                        jhatv = (zp[ip] - zp[im]) / (2.0 * args.radius)

                        l_spur_plus = 2.0 * args.radius * float(torch.dot(e, jhatv).item())
                        l_true_plus = 2.0 * args.radius * float(torch.dot(d, jv).item())
                        l_pred_plus = 2.0 * args.radius * float(torch.dot(d + e, jhatv).item())
                        q = (args.radius ** 2) * float(torch.dot(jv, jv).item())

                        spurious_linear[ip] = +l_spur_plus
                        spurious_linear[im] = -l_spur_plus
                        true_linear[ip] = +l_true_plus
                        true_linear[im] = -l_true_plus
                        pred_total_linear[ip] = +l_pred_plus
                        pred_total_linear[im] = -l_pred_plus
                        q_true[ip] = q
                        q_true[im] = q

                        even = 0.5 * (
                            (enc_cost[ip] - enc_cost[0])
                            + (enc_cost[im] - enc_cost[0])
                        )
                        exact_true_even[ip] = even
                        exact_true_even[im] = even

                    pred_debiased = pred_cost - spurious_linear

                    # Exclude nominal from RMS/correlation of local increment
                    # terms; it is deterministically zero and would bias them.
                    sl = slice(1, None)
                    q_centered = q_true[sl] - np.mean(q_true[sl])
                    even_centered = exact_true_even[sl] - np.mean(exact_true_even[sl])
                    rms_spur = _rms(spurious_linear[sl])
                    rms_tlin = _rms(true_linear[sl])
                    rms_plin = _rms(pred_total_linear[sl])
                    rms_q = _rms(q_true[sl])
                    rms_qvar = _rms(q_centered)
                    rms_even = _rms(exact_true_even[sl])
                    rms_evenvar = _rms(even_centered)

                    row = {
                        "model": label,
                        "anchor_index": ai + 1,
                        "dataset_row": int(row_idx),
                        "horizon": int(h),
                        "raw_horizon": int(raw_h),
                        "radius": float(args.radius),
                        "num_directions": int(args.num_directions),
                        "endpoint_mse": endpoint_mse,
                        "d_l2": d_l2,
                        "e_l2": e_l2,
                        "rms_spurious_linear": rms_spur,
                        "rms_true_linear": rms_tlin,
                        "rms_pred_total_linear": rms_plin,
                        "rms_true_quadratic": rms_q,
                        "rms_centered_true_quadratic": rms_qvar,
                        "rms_exact_true_even": rms_even,
                        "rms_centered_exact_true_even": rms_evenvar,
                        "tilt_to_quad_rms": rms_spur / max(rms_q, 1e-12),
                        "tilt_to_quadvar_rms": rms_spur / max(rms_qvar, 1e-12),
                        "tilt_to_exact_evenvar_rms": rms_spur / max(rms_evenvar, 1e-12),
                        "rho_qtrue_enc": _spearman(q_true[sl], enc_cost[sl]),
                        "rho_qtrue_phys": _spearman(q_true[sl], phys_cost[sl]),
                        "rho_exact_even_enc": _spearman(exact_true_even[sl], enc_cost[sl]),
                    }
                    row.update(
                        _score_metrics(
                            phys_cost,
                            enc_cost,
                            pred_cost,
                            first_deltas,
                            args.elite_frac,
                            "pred",
                        )
                    )
                    row.update(
                        _score_metrics(
                            phys_cost,
                            enc_cost,
                            pred_debiased,
                            first_deltas,
                            args.elite_frac,
                            "debiased",
                        )
                    )
                    rows.append(row)

            elapsed = time.time() - t0
            eta = (time.time() - start) / (ai + 1) * (len(anchors) - ai - 1)
            print(
                f"anchor {ai+1:3d}/{len(anchors)} row={row_idx} "
                f"eqerr<={max_eqerr:.1e} time={elapsed:.2f}s ETA={eta/60:.1f}min"
            )
    finally:
        env.close()

    anchor_csv = outdir / "anchor_metrics.csv"
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
                    **_aggregate(subset),
                }
            )

    summary_csv = outdir / "summary.csv"
    _write_csv(summary_csv, summary_rows)

    summary_json = outdir / "summary.json"
    with summary_json.open("w") as f:
        json.dump(
            _jsonable(
                {
                    "config": {
                        "dataset": dataset_name,
                        "num_anchors": int(len(anchors)),
                        "horizons": horizons,
                        "action_block": int(args.action_block),
                        "fixed_perturbed_action_dim": int(args.action_block * 2),
                        "radius": float(args.radius),
                        "num_directions": int(args.num_directions),
                        "elite_frac": float(args.elite_frac),
                        "goal_offset": int(args.goal_offset),
                        "note": "Debiased score is oracle diagnostic only; it uses the real nominal endpoint embedding to compute e.",
                    },
                    "anchors": anchors.tolist(),
                    "elapsed_seconds": float(time.time() - start),
                    "summary_rows": summary_rows,
                }
            ),
            f,
            indent=2,
        )

    print()
    hdr = (
        f"{'model':<12} {'H':>2} {'tilt/Qv':>9} {'rhoQ/E':>8} "
        f"{'rhoRaw':>8} {'rhoDeb':>8} {'eliteR':>8} {'eliteD':>8} "
        f"{'updR':>8} {'updD':>8} {'pctR':>8} {'pctD':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summary_rows:
        print(
            f"{s['model']:<12} {s['horizon']:>2d} "
            f"{s['tilt_to_quadvar_rms_mean']:>9.3f} "
            f"{s['rho_qtrue_enc_mean']:>8.3f} "
            f"{s['rho_pred_phys_mean']:>8.3f} "
            f"{s['rho_debiased_phys_mean']:>8.3f} "
            f"{s['elite_overlap_pred_phys_mean']:>8.3f} "
            f"{s['elite_overlap_debiased_phys_mean']:>8.3f} "
            f"{s['cem_update_cos_pred_phys_mean']:>8.3f} "
            f"{s['cem_update_cos_debiased_phys_mean']:>8.3f} "
            f"{s['pred_selected_phys_percentile_mean']:>8.3f} "
            f"{s['debiased_selected_phys_percentile_mean']:>8.3f}"
        )

    print("\nSignal magnitudes (mean over anchors):")
    hdr2 = (
        f"{'model':<12} {'H':>2} {'|d|':>8} {'|e|':>8} {'Lspur':>9} "
        f"{'Ltrue':>9} {'Qtrue':>9} {'Qvar':>9} {'EvenVar':>9}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for s in summary_rows:
        print(
            f"{s['model']:<12} {s['horizon']:>2d} "
            f"{s['d_l2_mean']:>8.3f} "
            f"{s['e_l2_mean']:>8.3f} "
            f"{s['rms_spurious_linear_mean']:>9.4f} "
            f"{s['rms_true_linear_mean']:>9.4f} "
            f"{s['rms_true_quadratic_mean']:>9.4f} "
            f"{s['rms_centered_true_quadratic_mean']:>9.4f} "
            f"{s['rms_centered_exact_true_even_mean']:>9.4f}"
        )

    print(f"\nMax equal-norm error: {max_eqerr:.3e}")
    print(f"Saved: {anchor_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
