#!/usr/bin/env python3
"""Fixed-horizon / varying-goal-offset control for PushT.

Scientific question
-------------------
The previous H={1,2,3,5} diagnostics used the official PushT goal offset of 25
raw steps.  Since horizon=5 and action_block=5, the H=5 nominal expert endpoint
is also t+25, so H and goal proximity changed together.

This evaluator removes that confound.  It FIXES the model rollout horizon and
candidate action neighborhood and changes only the task goal offset:

    rollout endpoint = t + fixed_horizon * action_block  (default t+25)
    goal             = t + goal_offset                   (15,20,25,30,35)

For every anchor, every goal offset uses the same:
  * physical start state,
  * nominal expert action sequence,
  * first-block 10-D +/- perturbations,
  * perturbation radius and random directions,
  * rollout horizon.

Changing the task goal follows the official PushT rendering/evaluation semantics,
so the rendered goal marker and goal image change with the goal.  This is therefore
a control over task-goal proximity at fixed horizon, not an artificial latent-only
replacement of z_g.

Primary hypotheses
------------------
Let

    d = z(U0) - z_g,
    e = zhat(U0) - z(U0).

Near the goal, the exact encoder cost should become increasingly even in +/-
action perturbations and increasingly well described by

    Q_true(delta) = delta^T J^T J delta.

At the same time, rollout bias creates the spurious odd/linear term

    L_spur(delta) = 2 e^T Jhat delta.

If goal proximity is the mechanism rather than H=5 itself, then as ||d|| becomes
small we expect:
  * exact encoder odd/even ratio to decrease,
  * rho(Q_true, encoder candidate cost) to increase,
  * oracle removal of L_spur to give a larger planning-landscape improvement.

The oracle-debiased score

    C_pred_debiased = C_pred - L_spur

requires the real nominal endpoint embedding and is only a mechanism diagnostic,
not a deployable planner.

Outputs
-------
  anchor_goal_metrics.csv
  summary.csv
  proximity_correlations.csv
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
    _physical_cost,
    _predict,
    _select_rows,
    _spearman,
)
from eval_pusht_linear_tilt_quadratic_signal import _score_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--num-anchors", type=int, default=50)
    p.add_argument("--fixed-horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument(
        "--goal-offsets",
        nargs="+",
        type=int,
        default=[15, 20, 25, 30, 35],
        help="Raw-step goal offsets. Fixed rollout endpoint is H*action_block.",
    )
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


def _aggregate(rows):
    keys = [
        "d_l2",
        "e_l2",
        "endpoint_mse",
        "rms_exact_odd",
        "rms_exact_even",
        "rms_centered_exact_even",
        "exact_odd_to_even_rms",
        "exact_odd_to_evenvar_rms",
        "rms_spurious_linear",
        "rms_true_linear",
        "rms_true_quadratic",
        "rms_centered_true_quadratic",
        "tilt_to_quadvar_rms",
        "rho_qtrue_enc",
        "rho_qtrue_phys",
        "rho_pred_phys",
        "rho_pred_enc",
        "elite_overlap_pred_phys",
        "cem_update_cos_pred_phys",
        "pred_selected_phys_percentile",
        "rho_debiased_phys",
        "rho_debiased_enc",
        "elite_overlap_debiased_phys",
        "cem_update_cos_debiased_phys",
        "debiased_selected_phys_percentile",
        "debias_rho_phys_gain",
        "debias_elite_phys_gain",
        "debias_update_phys_gain",
        "debias_percentile_improvement",
    ]
    out = {"n_anchors": len(rows)}
    for key in keys:
        vals = [r.get(key, np.nan) for r in rows]
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_median"] = _median(vals)
    return out


def _proximity_correlations(rows, label):
    """Pooled per-anchor/per-goal relationships for one model."""
    subset = [r for r in rows if r["model"] == label]
    d = [r["d_l2"] for r in subset]
    return {
        "model": label,
        "n_points": len(subset),
        # Expected negative: closer goals should be more quadratic/even.
        "rho_d_vs_qtrue_enc": _spearman(d, [r["rho_qtrue_enc"] for r in subset]),
        # Expected positive: farther goals should have stronger odd/linear signal.
        "rho_d_vs_exact_odd_to_even": _spearman(
            d, [r["exact_odd_to_even_rms"] for r in subset]
        ),
        "rho_d_vs_exact_odd_to_evenvar": _spearman(
            d, [r["exact_odd_to_evenvar_rms"] for r in subset]
        ),
        # Expected negative if oracle debias matters most near the goal.
        "rho_d_vs_debias_rho_gain": _spearman(
            d, [r["debias_rho_phys_gain"] for r in subset]
        ),
        "rho_d_vs_debias_elite_gain": _spearman(
            d, [r["debias_elite_phys_gain"] for r in subset]
        ),
        "rho_d_vs_debias_update_gain": _spearman(
            d, [r["debias_update_phys_gain"] for r in subset]
        ),
        "rho_d_vs_debias_percentile_improvement": _spearman(
            d, [r["debias_percentile_improvement"] for r in subset]
        ),
    }


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels length must equal --policies length")
    if args.fixed_horizon <= 0 or args.action_block <= 0:
        raise ValueError("--fixed-horizon and --action-block must be positive")
    if args.radius <= 0 or args.num_directions <= 0:
        raise ValueError("--radius and --num-directions must be positive")

    goal_offsets = sorted(set(map(int, args.goal_offsets)))
    if not goal_offsets or min(goal_offsets) <= 0:
        raise ValueError("--goal-offsets must contain positive integers")

    raw_horizon = int(args.fixed_horizon * args.action_block)
    max_required_offset = max(raw_horizon, max(goal_offsets))

    cfg = OmegaConf.load(args.config)
    dataset_name = args.dataset or str(cfg.eval.dataset_name)
    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    outdir = (
        Path(args.output_dir)
        if args.output_dir
        else cache_root / "fixed_horizon_goal_proximity"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=["action", "state"],
        cache_dir=cache_root,
    )

    # Select ONE common anchor set that is valid for every requested goal and
    # for the fixed rollout endpoint. This is essential for a paired control.
    col, anchors = _select_rows(
        dataset, args.num_anchors, args.seed, max_required_offset
    )
    print("Selected common paired eval rows:")
    print(anchors)
    print(
        f"Fixed rollout: H={args.fixed_horizon}, action_block={args.action_block}, "
        f"raw endpoint offset={raw_horizon}; goal offsets={goal_offsets}"
    )

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
    nC = 1 + 2 * args.num_directions
    rows = []
    start = time.time()
    max_eqerr = 0.0

    try:
        for ai, row_idx in enumerate(anchors):
            t0 = time.time()
            _check_contiguous(
                row_idx, episode_idx, step_idx, max_required_offset
            )
            init_state = state[row_idx].copy()
            expert = action[row_idx : row_idx + raw_horizon].copy()
            seed = args.env_seed + ai

            # Same candidate neighborhood for EVERY goal offset at this anchor.
            rng = np.random.default_rng(args.seed + 1_000_003 * (ai + 1))
            cands, _, dmeta, smeta, first_deltas, eqerr = (
                _make_fixed_first_block_candidates(
                    expert,
                    args.radius,
                    args.num_directions,
                    rng,
                    args.action_block,
                )
            )
            max_eqerr = max(max_eqerr, eqerr)

            normalized = _normalize_actions(
                cands, scaler, args.fixed_horizon, args.action_block
            )

            for goal_offset in goal_offsets:
                goal_state = state[row_idx + goal_offset].copy()
                current_image, goal_image = _current_goal_images(
                    env, init_state, goal_state, seed
                )

                # Render exact candidate endpoints under this task goal.  The
                # physical dynamics are unchanged, but the official PushT image
                # rendering contains task-goal information, so images are kept
                # goal-consistent instead of reusing a different-goal render.
                real_states = np.empty((nC, state.shape[-1]), dtype=np.float64)
                real_images = [None] * nC
                for ci in range(nC):
                    out = _rollout_checkpoints(
                        env,
                        init_state,
                        goal_state,
                        cands[ci],
                        [raw_horizon],
                        seed,
                    )
                    real_states[ci] = out[raw_horizon]["state"]
                    real_images[ci] = out[raw_horizon]["image"]

                phys_cost, *_ = _physical_cost(real_states, goal_state)

                for label, model in zip(labels, models):
                    zg = _encode(
                        model, transform, [goal_image], device, args.model_batch_size
                    )[0]
                    zr = _encode(
                        model, transform, real_images, device, args.model_batch_size
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

                    z0 = zr[0]
                    zhat0 = zp[0]
                    d = z0 - zg
                    e = zhat0 - z0
                    d_l2 = float(torch.linalg.vector_norm(d).item())
                    e_l2 = float(torch.linalg.vector_norm(e).item())
                    endpoint_mse = float(e.pow(2).mean().item())

                    spurious_linear = np.zeros(nC, dtype=np.float64)
                    true_linear = np.zeros(nC, dtype=np.float64)
                    q_true = np.zeros(nC, dtype=np.float64)
                    exact_odd = np.zeros(nC, dtype=np.float64)
                    exact_even = np.zeros(nC, dtype=np.float64)

                    for di in range(args.num_directions):
                        ip = np.where((dmeta == di) & (smeta > 0))[0]
                        im = np.where((dmeta == di) & (smeta < 0))[0]
                        if len(ip) != 1 or len(im) != 1:
                            raise RuntimeError(f"Malformed +/- pair direction={di}")
                        ip, im = int(ip[0]), int(im[0])

                        jv = (zr[ip] - zr[im]) / (2.0 * args.radius)
                        jhatv = (zp[ip] - zp[im]) / (2.0 * args.radius)

                        l_spur_plus = 2.0 * args.radius * float(
                            torch.dot(e, jhatv).item()
                        )
                        l_true_plus = 2.0 * args.radius * float(
                            torch.dot(d, jv).item()
                        )
                        q = (args.radius ** 2) * float(torch.dot(jv, jv).item())

                        spurious_linear[ip] = +l_spur_plus
                        spurious_linear[im] = -l_spur_plus
                        true_linear[ip] = +l_true_plus
                        true_linear[im] = -l_true_plus
                        q_true[ip] = q
                        q_true[im] = q

                        # Exact encoder cost decomposition for a symmetric pair:
                        # odd(+delta) = [C(+delta)-C(-delta)]/2
                        # even         = [C(+delta)+C(-delta)]/2 - C(0)
                        odd = 0.5 * (enc_cost[ip] - enc_cost[im])
                        even = 0.5 * (enc_cost[ip] + enc_cost[im]) - enc_cost[0]
                        exact_odd[ip] = +odd
                        exact_odd[im] = -odd
                        exact_even[ip] = even
                        exact_even[im] = even

                    pred_debiased = pred_cost - spurious_linear
                    sl = slice(1, None)

                    q_centered = q_true[sl] - np.mean(q_true[sl])
                    even_centered = exact_even[sl] - np.mean(exact_even[sl])
                    rms_odd = _rms(exact_odd[sl])
                    rms_even = _rms(exact_even[sl])
                    rms_evenvar = _rms(even_centered)
                    rms_spur = _rms(spurious_linear[sl])
                    rms_tlin = _rms(true_linear[sl])
                    rms_q = _rms(q_true[sl])
                    rms_qvar = _rms(q_centered)

                    row = {
                        "model": label,
                        "anchor_index": ai + 1,
                        "dataset_row": int(row_idx),
                        "fixed_horizon": int(args.fixed_horizon),
                        "raw_horizon": int(raw_horizon),
                        "goal_offset": int(goal_offset),
                        "goal_temporal_delta_from_endpoint": int(
                            goal_offset - raw_horizon
                        ),
                        "goal_temporal_abs_delta": int(
                            abs(goal_offset - raw_horizon)
                        ),
                        "radius": float(args.radius),
                        "num_directions": int(args.num_directions),
                        "d_l2": d_l2,
                        "e_l2": e_l2,
                        "endpoint_mse": endpoint_mse,
                        "rms_exact_odd": rms_odd,
                        "rms_exact_even": rms_even,
                        "rms_centered_exact_even": rms_evenvar,
                        "exact_odd_to_even_rms": rms_odd / max(rms_even, 1e-12),
                        "exact_odd_to_evenvar_rms": rms_odd / max(rms_evenvar, 1e-12),
                        "rms_spurious_linear": rms_spur,
                        "rms_true_linear": rms_tlin,
                        "rms_true_quadratic": rms_q,
                        "rms_centered_true_quadratic": rms_qvar,
                        "tilt_to_quadvar_rms": rms_spur / max(rms_qvar, 1e-12),
                        "rho_qtrue_enc": _spearman(q_true[sl], enc_cost[sl]),
                        "rho_qtrue_phys": _spearman(q_true[sl], phys_cost[sl]),
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
                    row["debias_rho_phys_gain"] = (
                        row["rho_debiased_phys"] - row["rho_pred_phys"]
                    )
                    row["debias_elite_phys_gain"] = (
                        row["elite_overlap_debiased_phys"]
                        - row["elite_overlap_pred_phys"]
                    )
                    row["debias_update_phys_gain"] = (
                        row["cem_update_cos_debiased_phys"]
                        - row["cem_update_cos_pred_phys"]
                    )
                    # Positive means debias selected a physically better-ranked candidate.
                    row["debias_percentile_improvement"] = (
                        row["pred_selected_phys_percentile"]
                        - row["debiased_selected_phys_percentile"]
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

    anchor_csv = outdir / "anchor_goal_metrics.csv"
    _write_csv(anchor_csv, rows)

    summary_rows = []
    for label in labels:
        for go in goal_offsets:
            subset = [
                r for r in rows
                if r["model"] == label and r["goal_offset"] == go
            ]
            summary_rows.append(
                {
                    "model": label,
                    "fixed_horizon": int(args.fixed_horizon),
                    "raw_horizon": int(raw_horizon),
                    "goal_offset": int(go),
                    "goal_temporal_delta_from_endpoint": int(go - raw_horizon),
                    **_aggregate(subset),
                }
            )

    summary_csv = outdir / "summary.csv"
    _write_csv(summary_csv, summary_rows)

    proximity_rows = [_proximity_correlations(rows, label) for label in labels]
    proximity_csv = outdir / "proximity_correlations.csv"
    _write_csv(proximity_csv, proximity_rows)

    summary_json = outdir / "summary.json"
    with summary_json.open("w") as f:
        json.dump(
            _jsonable(
                {
                    "config": {
                        "dataset": dataset_name,
                        "num_anchors": int(len(anchors)),
                        "fixed_horizon": int(args.fixed_horizon),
                        "action_block": int(args.action_block),
                        "raw_horizon": int(raw_horizon),
                        "goal_offsets": goal_offsets,
                        "radius": float(args.radius),
                        "num_directions": int(args.num_directions),
                        "elite_frac": float(args.elite_frac),
                        "seed": int(args.seed),
                        "env_seed": int(args.env_seed),
                        "control_note": (
                            "Physical start/action neighborhood/horizon fixed; task goal "
                            "varies using official PushT goal-consistent rendering."
                        ),
                    },
                    "anchors": anchors.tolist(),
                    "elapsed_seconds": float(time.time() - start),
                    "max_equal_norm_error": float(max_eqerr),
                    "summary_rows": summary_rows,
                    "proximity_correlations": proximity_rows,
                }
            ),
            f,
            indent=2,
        )

    print()
    hdr = (
        f"{'model':<12} {'goal':>4} {'dT':>4} {'|d|':>7} {'odd/even':>9} "
        f"{'rhoQ/E':>7} {'rhoRaw':>7} {'rhoDeb':>7} {'dRho':>7} "
        f"{'eliteR':>7} {'eliteD':>7} {'updR':>7} {'updD':>7} "
        f"{'pctR':>7} {'pctD':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summary_rows:
        print(
            f"{s['model']:<12} {s['goal_offset']:>4d} "
            f"{s['goal_temporal_delta_from_endpoint']:>+4d} "
            f"{s['d_l2_mean']:>7.3f} "
            f"{s['exact_odd_to_even_rms_mean']:>9.3f} "
            f"{s['rho_qtrue_enc_mean']:>7.3f} "
            f"{s['rho_pred_phys_mean']:>7.3f} "
            f"{s['rho_debiased_phys_mean']:>7.3f} "
            f"{s['debias_rho_phys_gain_mean']:>7.3f} "
            f"{s['elite_overlap_pred_phys_mean']:>7.3f} "
            f"{s['elite_overlap_debiased_phys_mean']:>7.3f} "
            f"{s['cem_update_cos_pred_phys_mean']:>7.3f} "
            f"{s['cem_update_cos_debiased_phys_mean']:>7.3f} "
            f"{s['pred_selected_phys_percentile_mean']:>7.3f} "
            f"{s['debiased_selected_phys_percentile_mean']:>7.3f}"
        )

    print("\nPooled proximity correlations across anchors x goal offsets:")
    for p in proximity_rows:
        print(
            f"{p['model']:<12} "
            f"rho(|d|,rhoQ/E)={p['rho_d_vs_qtrue_enc']:+.3f} | "
            f"rho(|d|,odd/even)={p['rho_d_vs_exact_odd_to_even']:+.3f} | "
            f"rho(|d|,debias dRho)={p['rho_d_vs_debias_rho_gain']:+.3f} | "
            f"rho(|d|,debias dElite)={p['rho_d_vs_debias_elite_gain']:+.3f} | "
            f"rho(|d|,debias dUpd)={p['rho_d_vs_debias_update_gain']:+.3f} | "
            f"rho(|d|,pct improvement)={p['rho_d_vs_debias_percentile_improvement']:+.3f}"
        )

    print(f"\nMax equal-norm error: {max_eqerr:.3e}")
    print(f"Saved: {anchor_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {proximity_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
