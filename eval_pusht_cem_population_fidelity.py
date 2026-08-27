#!/usr/bin/env python3
"""CEM-population-scale planner fidelity diagnostic for full PushT.

Unlike the small fixed-radius local probe, this evaluator replays the ACTUAL
candidate population sampled by CEM at selected refinement iterations.  It asks
whether a model that is locally faithful is also faithful at the search scale
that drives elite selection / basin choice.

Requires traces produced with traced_cem.py + solver.save_candidates=true.

For every sampled candidate:
  * planner score: model cost on the exact normalized CEM candidate;
  * planner score: model cost on the exact normalized CEM candidate;
  * clipped-counterfactual score: model cost after inverse-transform + Box
    projection + re-normalization;
  * encoder cost: goal distance of the REAL official-environment terminal image;
  * physical cost: official PushT terminal state cost.

Important: official stable-worldmodel currently inverse-transforms actions
without clipping, and PushT step() accepts those values even when they exceed
the declared Box[-1,1]. Therefore the physical replay here intentionally uses
the UNCLIPPED inverse-transformed action. OOB statistics measure planner-induced
action-support extrapolation; clipped scores are diagnostic counterfactuals only.
"""
import argparse
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
from eval_pusht_fixed_action_response_horizon import _rollout_checkpoints
from eval_pusht_horizon_directional import (
    _current_goal_images,
    _encode,
    _label,
    _normalize_actions,
    _physical_cost,
    _predict,
)
from pusht_trace_eval_utils import (
    decode_normalized_candidates,
    elite_metrics,
    load_traces,
    maybe_inverse_state,
    numeric_summary,
    selection_percentile,
    spearman,
    write_csv,
    write_csv_gz,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--trace-label", default="trace")
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--iterations", nargs="+", type=int, default=[0, 1, 3, 5, 10, 20, 29])
    p.add_argument("--max-solves", type=int, default=20)
    p.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="0 = all traced candidates. If subsampling, all original top-k elites are retained.",
    )
    p.add_argument("--state-space", choices=["auto", "raw", "standardized"], default="auto")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def choose_indices(n, topk_idx, max_candidates, seed):
    if max_candidates <= 0 or max_candidates >= n:
        return np.arange(n, dtype=np.int64)
    topk_idx = np.unique(np.asarray(topk_idx, dtype=np.int64))
    if max_candidates < len(topk_idx):
        raise ValueError(
            f"--max-candidates={max_candidates} is smaller than traced top-k={len(topk_idx)}"
        )
    keep = set(topk_idx.tolist())
    pool = np.asarray([i for i in range(n) if i not in keep], dtype=np.int64)
    rng = np.random.default_rng(seed)
    extra = rng.choice(pool, size=max_candidates - len(keep), replace=False)
    return np.sort(np.asarray(list(keep) + extra.tolist(), dtype=np.int64))


def aggregate(rows):
    keys = [
        "candidate_radius_mean",
        "candidate_radius_median",
        "sigma_l2",
        "raw_oob_candidate_fraction",
        "raw_oob_scalar_fraction",
        "clip_l2_mean",
        "rho_pred_phys",
        "rho_pred_enc",
        "rho_execpred_phys",
        "rho_execpred_enc",
        "rho_enc_phys",
        "rho_tracecost_pred",
        "pred_elite_overlap_phys",
        "pred_elite_update_cos_phys",
        "pred_selected_phys_percentile",
        "execpred_elite_overlap_phys",
        "execpred_selected_phys_percentile",
        "trace_topk_phys_mean",
        "oracle_topk_phys_mean",
        "trace_topk_overlap_phys",
    ]
    out = {"n_rows": len(rows)}
    for k in keys:
        vals = np.asarray([r.get(k, np.nan) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[k + "_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[k + "_median"] = float(np.median(vals)) if len(vals) else float("nan")
    return out


def main():
    a = parse_args()
    if a.labels is not None and len(a.labels) != len(a.policies):
        raise ValueError("--labels length must equal --policies length")

    cfg = OmegaConf.load(a.config)
    dataset_name = a.dataset or str(cfg.eval.dataset_name)
    cache = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    out = (
        Path(a.output_dir)
        if a.output_dir
        else Path(a.trace_dir) / "cem_population_fidelity_official"
    )
    out.mkdir(parents=True, exist_ok=True)

    ds = swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=["action", "state"], cache_dir=cache
    )
    action = np.asarray(ds.get_col_data("action"), dtype=np.float32)
    state = np.asarray(ds.get_col_data("state"), dtype=np.float64)
    action_scaler = preprocessing.StandardScaler().fit(
        action[np.isfinite(action).all(axis=1)]
    )
    state_scaler = preprocessing.StandardScaler().fit(
        state[np.isfinite(state).all(axis=1)]
    )

    device = torch.device(a.device)
    transform = img_transform(cfg)
    labels = [_label(p, a.labels, i) for i, p in enumerate(a.policies)]
    models = []
    for label, policy in zip(labels, a.policies):
        print(f"Loading [{label}] {policy}")
        m = swm.policy.AutoCostModel(policy).to(device).eval()
        m.requires_grad_(False)
        m.interpolate_pos_encoding = True
        models.append(m)

    traces = load_traces(a.trace_dir, a.max_solves)
    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    rows = []
    candidate_rows = []
    start = time.time()

    try:
        for si, path in enumerate(traces):
            tr = np.load(path, allow_pickle=True)
            required = {"candidates", "candidate_costs", "topk_indices", "info_state", "info_goal_state"}
            missing = sorted(required - set(tr.files))
            if missing:
                raise RuntimeError(
                    f"{path} missing {missing}. Re-run CEM trace with solver.save_candidates=true."
                )

            init_state = maybe_inverse_state(tr["info_state"], state_scaler, a.state_space)
            goal_state = maybe_inverse_state(tr["info_goal_state"], state_scaler, a.state_space)
            means = np.asarray(tr["mean"], dtype=np.float32)
            vars_ = np.asarray(tr["var"], dtype=np.float32)
            populations = np.asarray(tr["candidates"], dtype=np.float32)
            trace_costs_all = np.asarray(tr["candidate_costs"], dtype=np.float64)
            topk_all = np.asarray(tr["topk_indices"], dtype=np.int64)
            action_block = int(np.asarray(tr["action_block"]).item())
            horizon = int(np.asarray(tr["horizon"]).item())
            raw_horizon = horizon * action_block
            topk = int(np.asarray(tr["topk"]).item())
            seed = a.env_seed + si
            current_image, goal_image = _current_goal_images(
                env, init_state, goal_state, seed
            )

            valid_iters = [
                i for i in sorted(set(a.iterations))
                if 0 <= i < populations.shape[0]
            ]
            for it in valid_iters:
                full_norm = populations[it]
                full_trace_cost = trace_costs_all[it]
                full_topk = topk_all[it]
                idx = choose_indices(
                    len(full_norm),
                    full_topk,
                    a.max_candidates,
                    seed=9_100_003 * (si + 1) + it,
                )
                c_norm = full_norm[idx]
                trace_cost = full_trace_cost[idx]
                remap = {int(old): j for j, old in enumerate(idx)}
                traced_topk_sub = np.asarray(
                    [remap[int(x)] for x in full_topk if int(x) in remap],
                    dtype=np.int64,
                )

                raw_official, raw_clipped = decode_normalized_candidates(
                    c_norm, action_scaler, action_block
                )
                clipped_norm = _normalize_actions(
                    raw_clipped, action_scaler, horizon, action_block
                )

                center = means[it]
                deltas = (c_norm - center[None]).reshape(len(c_norm), -1)
                radii = np.linalg.norm(deltas, axis=1)
                oob = np.abs(raw_official) > 1.0
                clip_l2 = np.linalg.norm(
                    (raw_official - raw_clipped).reshape(len(raw_official), -1), axis=1
                )

                real_states = np.empty(
                    (len(raw_official), state.shape[-1]), dtype=np.float64
                )
                real_images = [None] * len(raw_official)
                for ci in range(len(raw_official)):
                    rr = _rollout_checkpoints(
                        env,
                        init_state,
                        goal_state,
                        raw_official[ci],
                        [raw_horizon],
                        seed,
                    )[raw_horizon]
                    real_states[ci] = rr["state"]
                    real_images[ci] = rr["image"]

                phys_cost, *_ = _physical_cost(real_states, goal_state)
                phys_topk = np.argsort(phys_cost)[: min(topk, len(phys_cost))]
                trace_topk_eval = traced_topk_sub[: min(topk, len(traced_topk_sub))]
                trace_overlap = (
                    len(set(trace_topk_eval.tolist()) & set(phys_topk.tolist()))
                    / max(len(trace_topk_eval), 1)
                )
                trace_topk_phys_mean = (
                    float(np.mean(phys_cost[trace_topk_eval]))
                    if len(trace_topk_eval) else float("nan")
                )
                oracle_topk_phys_mean = float(np.mean(phys_cost[phys_topk]))

                for label, model in zip(labels, models):
                    zg = _encode(
                        model, transform, [goal_image], device, a.model_batch_size
                    )[0]
                    zr = _encode(
                        model, transform, real_images, device, a.model_batch_size
                    )
                    zp = _predict(
                        model, transform, current_image, c_norm, device, a.model_batch_size
                    )
                    zp_exec = _predict(
                        model,
                        transform,
                        current_image,
                        clipped_norm,
                        device,
                        a.model_batch_size,
                    )

                    enc_cost = (
                        torch.sum((zr - zg[None]) ** 2, dim=-1)
                        .cpu().numpy().astype(np.float64)
                    )
                    pred_cost = (
                        torch.sum((zp - zg[None]) ** 2, dim=-1)
                        .cpu().numpy().astype(np.float64)
                    )
                    exec_pred_cost = (
                        torch.sum((zp_exec - zg[None]) ** 2, dim=-1)
                        .cpu().numpy().astype(np.float64)
                    )

                    pe = elite_metrics(phys_cost, pred_cost, deltas, min(topk, len(c_norm)))
                    ee = elite_metrics(phys_cost, exec_pred_cost, deltas, min(topk, len(c_norm)))
                    pred_pct, _ = selection_percentile(phys_cost, pred_cost)
                    exec_pct, _ = selection_percentile(phys_cost, exec_pred_cost)

                    row = {
                        "trace_source": a.trace_label,
                        "model": label,
                        "trace_file": path.name,
                        "solve_index": int(np.asarray(tr["solve_index"]).item()),
                        "cem_iteration": int(it),
                        "num_candidates_evaluated": int(len(c_norm)),
                        "num_candidates_full": int(len(full_norm)),
                        "topk": int(topk),
                        "candidate_radius_mean": float(np.mean(radii)),
                        "candidate_radius_median": float(np.median(radii)),
                        "sigma_l2": float(np.linalg.norm(vars_[it])),
                        "physical_execution_mode": "official_unclipped",
                        "raw_oob_candidate_fraction": float(
                            np.mean(
                                np.any(
                                    oob.reshape(len(raw_official), -1), axis=1
                                )
                            )
                        ),
                        "raw_oob_scalar_fraction": float(np.mean(oob)),
                        "clip_l2_mean": float(np.mean(clip_l2)),
                        "rho_pred_phys": spearman(pred_cost, phys_cost),
                        "rho_pred_enc": spearman(pred_cost, enc_cost),
                        "rho_execpred_phys": spearman(exec_pred_cost, phys_cost),
                        "rho_execpred_enc": spearman(exec_pred_cost, enc_cost),
                        "rho_enc_phys": spearman(enc_cost, phys_cost),
                        "rho_tracecost_pred": spearman(trace_cost, pred_cost),
                        "pred_elite_overlap_phys": pe["elite_overlap"],
                        "pred_elite_update_cos_phys": pe["elite_update_cosine"],
                        "pred_selected_phys_percentile": pred_pct,
                        "execpred_elite_overlap_phys": ee["elite_overlap"],
                        "execpred_elite_update_cos_phys": ee["elite_update_cosine"],
                        "execpred_selected_phys_percentile": exec_pct,
                        "trace_topk_phys_mean": trace_topk_phys_mean,
                        "oracle_topk_phys_mean": oracle_topk_phys_mean,
                        "trace_topk_overlap_phys": float(trace_overlap),
                    }
                    rows.append(row)

                    for cj in range(len(c_norm)):
                        candidate_rows.append({
                            "trace_source": a.trace_label,
                            "model": label,
                            "trace_file": path.name,
                            "solve_index": int(np.asarray(tr["solve_index"]).item()),
                            "cem_iteration": int(it),
                            "candidate_index_full": int(idx[cj]),
                            "candidate_radius": float(radii[cj]),
                            "raw_oob_fraction": float(np.mean(oob[cj])),
                            "clip_l2": float(clip_l2[cj]),
                            "trace_cost": float(trace_cost[cj]),
                            "pred_cost": float(pred_cost[cj]),
                            "exec_pred_cost": float(exec_pred_cost[cj]),
                            "physical_execution_mode": "official_unclipped",
                            "enc_cost": float(enc_cost[cj]),
                            "physical_cost": float(phys_cost[cj]),
                        })

            print(f"trace {si+1}/{len(traces)} {path.name} done")
    finally:
        env.close()

    write_csv(out / "population_metrics.csv", rows)
    write_csv_gz(out / "population_candidate_metrics.csv.gz", candidate_rows)

    summary = []
    for label in labels:
        for it in sorted(set(r["cem_iteration"] for r in rows if r["model"] == label)):
            rr = [r for r in rows if r["model"] == label and r["cem_iteration"] == it]
            summary.append({
                "trace_source": a.trace_label,
                "model": label,
                "cem_iteration": int(it),
                **aggregate(rr),
            })
    write_csv(out / "summary.csv", summary)
    with open(out / "summary.json", "w") as f:
        json.dump(
            {
                "trace_source": a.trace_label,
                "config": vars(a),
                "summary": summary,
                "elapsed_seconds": time.time() - start,
            },
            f,
            indent=2,
        )

    print("\nCEM-population fidelity")
    print(f"{'model':<10} {'it':>3} {'radius':>8} {'oob':>7} {'rhoP':>7} {'rhoClip':>8} {'rhoE/P':>8} {'elite':>7} {'pct':>7}")
    for s in summary:
        print(
            f"{s['model']:<10} {s['cem_iteration']:>3} "
            f"{s['candidate_radius_mean_mean']:>8.3f} "
            f"{s['raw_oob_candidate_fraction_mean']:>7.3f} "
            f"{s['rho_pred_phys_mean']:>7.3f} "
            f"{s['rho_execpred_phys_mean']:>8.3f} "
            f"{s['rho_enc_phys_mean']:>8.3f} "
            f"{s['pred_elite_overlap_phys_mean']:>7.3f} "
            f"{s['pred_selected_phys_percentile_mean']:>7.3f}"
        )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
