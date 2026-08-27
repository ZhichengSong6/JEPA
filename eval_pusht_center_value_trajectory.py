#!/usr/bin/env python3
"""Center-value trajectory diagnostic along CEM refinement for PushT.

For every traced solve, physically execute EVERY CEM mean mu_0 ... mu_I and
record four notions of value:
  C_pred       model score on the exact planner-space mean;
  C_pred       model score on the exact planner-space mean;
  C_pred_clip  clipped-action counterfactual model score;
  C_enc        latent goal cost of the REAL official-environment terminal image;
  C_phys       official PushT terminal cost.

Official stable-worldmodel inverse-transforms planned actions without clipping,
and PushT accepts those values even outside the declared Box[-1,1]. Therefore
the physical replay intentionally executes the UNCLIPPED inverse-transformed
CEM mean. clip_l2 / C_pred_clip are diagnostics for action-support extrapolation.
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
    decode_normalized_plan,
    load_traces,
    maybe_inverse_state,
    spearman,
    write_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--trace-label", default="trace")
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--max-solves", type=int, default=20)
    p.add_argument("--state-space", choices=["auto", "raw", "standardized"], default="auto")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def aggregate(rows):
    keys = [
        "center_shift_l2",
        "sigma_l2",
        "raw_oob_fraction",
        "clip_l2",
        "physical_cost",
        "enc_cost",
        "pred_cost",
        "exec_pred_cost",
        "endpoint_mse",
        "d_l2",
        "e_l2",
    ]
    out = {"n": len(rows)}
    for k in keys:
        x = np.asarray([r.get(k, np.nan) for r in rows], dtype=np.float64)
        x = x[np.isfinite(x)]
        out[k + "_mean"] = float(np.mean(x)) if len(x) else float("nan")
        out[k + "_median"] = float(np.median(x)) if len(x) else float("nan")
    return out


def solve_summary(rows):
    rows = sorted(rows, key=lambda r: r["cem_iteration"])
    it = np.asarray([r["cem_iteration"] for r in rows], dtype=int)
    pred = np.asarray([r["pred_cost"] for r in rows], dtype=float)
    pred_exec = np.asarray([r["exec_pred_cost"] for r in rows], dtype=float)
    enc = np.asarray([r["enc_cost"] for r in rows], dtype=float)
    phys = np.asarray([r["physical_cost"] for r in rows], dtype=float)

    dp = np.diff(pred)
    dpe = np.diff(pred_exec)
    dphys = np.diff(phys)
    mismatch = (dp < 0) & (dphys > 0)
    mismatch_exec = (dpe < 0) & (dphys > 0)

    ibp = int(np.nanargmin(pred))
    ibpe = int(np.nanargmin(pred_exec))
    ibphys = int(np.nanargmin(phys))
    pspan = max(float(np.nanpercentile(phys, 90) - np.nanmin(phys)), 1e-12)

    return {
        "n_centers": int(len(rows)),
        "rho_path_pred_phys": spearman(pred, phys),
        "rho_path_execpred_phys": spearman(pred_exec, phys),
        "rho_path_enc_phys": spearman(enc, phys),
        "best_pred_iteration": int(it[ibp]),
        "best_execpred_iteration": int(it[ibpe]),
        "best_physical_iteration": int(it[ibphys]),
        "best_physical_cost": float(phys[ibphys]),
        "final_physical_cost": float(phys[-1]),
        "final_vs_best_phys_regret_norm": float((phys[-1] - phys[ibphys]) / pspan),
        "pred_improvement_after_phys_best": float(pred[ibphys] - pred[-1]),
        "execpred_improvement_after_phys_best": float(pred_exec[ibphys] - pred_exec[-1]),
        "physical_worsening_after_best": float(phys[-1] - phys[ibphys]),
        "mismatch_step_count": int(np.sum(mismatch)),
        "mismatch_step_fraction": float(np.mean(mismatch)) if len(mismatch) else 0.0,
        "exec_mismatch_step_count": int(np.sum(mismatch_exec)),
        "exec_mismatch_step_fraction": float(np.mean(mismatch_exec)) if len(mismatch_exec) else 0.0,
        "max_phys_increase_while_pred_decreases": (
            float(np.max(dphys[mismatch])) if np.any(mismatch) else 0.0
        ),
    }


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
        else Path(a.trace_dir) / "center_value_trajectory_official"
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
    start = time.time()

    try:
        for si, path in enumerate(traces):
            tr = np.load(path, allow_pickle=True)
            init_state = maybe_inverse_state(
                tr["info_state"], state_scaler, a.state_space
            )
            goal_state = maybe_inverse_state(
                tr["info_goal_state"], state_scaler, a.state_space
            )
            means = np.asarray(tr["mean"], dtype=np.float32)
            vars_ = np.asarray(tr["var"], dtype=np.float32)
            action_block = int(np.asarray(tr["action_block"]).item())
            horizon = int(np.asarray(tr["horizon"]).item())
            raw_horizon = horizon * action_block
            seed = a.env_seed + si
            current_image, goal_image = _current_goal_images(
                env, init_state, goal_state, seed
            )

            real_states = []
            real_images = []
            raw_official_plans = []
            raw_clipped_plans = []
            clip_norm_plans = []
            for it in range(len(means)):
                raw_official = decode_normalized_plan(
                    means[it], action_scaler, action_block, clip=False
                ).astype(np.float32)
                raw_clipped = np.clip(
                    raw_official, -1.0, 1.0
                ).astype(np.float32)
                rr = _rollout_checkpoints(
                    env,
                    init_state,
                    goal_state,
                    raw_official,
                    [raw_horizon],
                    seed,
                )[raw_horizon]
                real_states.append(rr["state"])
                real_images.append(rr["image"])
                raw_official_plans.append(raw_official)
                raw_clipped_plans.append(raw_clipped)
                clip_norm_plans.append(
                    _normalize_actions(
                        raw_clipped[None],
                        action_scaler,
                        horizon,
                        action_block,
                    )[0]
                )

            real_states = np.asarray(real_states, dtype=np.float64)
            phys_cost, *_ = _physical_cost(real_states, goal_state)
            clip_norm_plans = np.asarray(clip_norm_plans, dtype=np.float32)

            for label, model in zip(labels, models):
                zg = _encode(
                    model, transform, [goal_image], device, a.model_batch_size
                )[0]
                zr = _encode(
                    model, transform, real_images, device, a.model_batch_size
                )
                zp = _predict(
                    model,
                    transform,
                    current_image,
                    means,
                    device,
                    a.model_batch_size,
                )
                zp_exec = _predict(
                    model,
                    transform,
                    current_image,
                    clip_norm_plans,
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
                endpoint_mse = (
                    torch.mean((zp - zr) ** 2, dim=-1)
                    .cpu().numpy().astype(np.float64)
                )
                d_l2 = (
                    torch.linalg.vector_norm(zr - zg[None], dim=-1)
                    .cpu().numpy().astype(np.float64)
                )
                e_l2 = (
                    torch.linalg.vector_norm(zp - zr, dim=-1)
                    .cpu().numpy().astype(np.float64)
                )

                for it in range(len(means)):
                    raw_official = np.asarray(raw_official_plans[it])
                    raw_clipped = np.asarray(raw_clipped_plans[it])
                    rows.append({
                        "trace_source": a.trace_label,
                        "model": label,
                        "trace_file": path.name,
                        "solve_index": int(np.asarray(tr["solve_index"]).item()),
                        "env_index": int(np.asarray(tr["env_index"]).item()),
                        "cem_iteration": int(it),
                        "center_shift_l2": float(np.linalg.norm(means[it] - means[0])),
                        "sigma_l2": float(np.linalg.norm(vars_[it])),
                        "physical_execution_mode": "official_unclipped",
                        "raw_oob_fraction": float(
                            np.mean(np.abs(raw_official) > 1.0)
                        ),
                        "clip_l2": float(
                            np.linalg.norm(raw_official - raw_clipped)
                        ),
                        "physical_cost": float(phys_cost[it]),
                        "enc_cost": float(enc_cost[it]),
                        "pred_cost": float(pred_cost[it]),
                        "exec_pred_cost": float(exec_pred_cost[it]),
                        "endpoint_mse": float(endpoint_mse[it]),
                        "d_l2": float(d_l2[it]),
                        "e_l2": float(e_l2[it]),
                    })
            print(f"trace {si+1}/{len(traces)} {path.name} done")
    finally:
        env.close()

    write_csv(out / "center_value_metrics.csv", rows)

    iteration_rows = []
    for label in labels:
        for it in sorted(set(r["cem_iteration"] for r in rows if r["model"] == label)):
            rr = [
                r for r in rows
                if r["model"] == label and r["cem_iteration"] == it
            ]
            iteration_rows.append({
                "trace_source": a.trace_label,
                "model": label,
                "cem_iteration": int(it),
                **aggregate(rr),
            })
    write_csv(out / "iteration_summary.csv", iteration_rows)

    solve_rows = []
    for label in labels:
        files = sorted(set(r["trace_file"] for r in rows if r["model"] == label))
        for tf in files:
            rr = [r for r in rows if r["model"] == label and r["trace_file"] == tf]
            base = rr[0]
            solve_rows.append({
                "trace_source": a.trace_label,
                "model": label,
                "trace_file": tf,
                "solve_index": int(base["solve_index"]),
                "env_index": int(base["env_index"]),
                **solve_summary(rr),
            })
    write_csv(out / "solve_summary.csv", solve_rows)

    with open(out / "summary.json", "w") as f:
        json.dump(
            {
                "trace_source": a.trace_label,
                "config": vars(a),
                "iteration_summary": iteration_rows,
                "solve_summary": solve_rows,
                "elapsed_seconds": time.time() - start,
            },
            f,
            indent=2,
        )

    worst = sorted(
        solve_rows,
        key=lambda r: r["final_vs_best_phys_regret_norm"],
        reverse=True,
    )[:10]
    print("\nWorst basin-divergence solves")
    print(f"{'model':<9} {'solve':>5} {'rho':>7} {'bestI':>5} {'finalReg':>9} {'mismatch':>8}")
    for r in worst:
        print(
            f"{r['model']:<9} {r['solve_index']:>5} "
            f"{r['rho_path_pred_phys']:>7.3f} "
            f"{r['best_physical_iteration']:>5} "
            f"{r['final_vs_best_phys_regret_norm']:>9.3f} "
            f"{r['mismatch_step_count']:>8}"
        )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
