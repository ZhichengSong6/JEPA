#!/usr/bin/env python3
"""Decompose near-null action responses at CEM-visited PushT centers.

The previous local evaluator can report enormous pred/encoder gain when the
REAL encoded symmetric response is nearly zero.  This script separates:
  (1) physical state-factor response,
  (2) encoder response of the real terminal images,
  (3) predictor response,
and records contact asymmetry plus pusher/block/rotation components.

Centers are selected from planner_center_metrics.csv by response_gain, with a
top-k fallback.  Simulator information is diagnostic-only.
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
    _current_goal_images,
    _encode,
    _label,
    _normalize_actions,
    _predict,
    _state_factor,
)
from pusht_trace_eval_utils import (
    cosine,
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
    p.add_argument("--landscape-csv", default=None)
    p.add_argument("--policies", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default=None)
    p.add_argument("--gain-threshold", type=float, default=1.0e4)
    p.add_argument("--top-centers", type=int, default=10)
    p.add_argument("--radius", type=float, default=0.1565)
    p.add_argument("--num-directions", type=int, default=64)
    p.add_argument("--physical-null-threshold", type=float, default=1.0e-3)
    p.add_argument("--encoder-null-threshold", type=float, default=1.0e-6)
    p.add_argument("--pred-active-threshold", type=float, default=1.0e-3)
    p.add_argument("--state-space", choices=["auto", "raw", "standardized"], default="auto")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--model-batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select_centers(rows, threshold, top_centers):
    best = {}
    for r in rows:
        key = (r["trace_file"], int(r["cem_iteration"]))
        g = float(r.get("response_gain", "nan"))
        if not np.isfinite(g):
            continue
        if key not in best or g > best[key]["max_gain"]:
            best[key] = {
                "trace_file": key[0],
                "cem_iteration": key[1],
                "max_gain": g,
            }
    vals = sorted(best.values(), key=lambda x: x["max_gain"], reverse=True)
    selected = [x for x in vals if x["max_gain"] >= threshold]
    if len(selected) < min(top_centers, len(vals)):
        have = {(x["trace_file"], x["cem_iteration"]) for x in selected}
        for x in vals:
            key = (x["trace_file"], x["cem_iteration"])
            if key not in have:
                selected.append(x)
                have.add(key)
            if len(selected) >= top_centers:
                break
    return selected


def signed_angle_delta(a, b):
    d = float(a) - float(b)
    return float(np.arctan2(np.sin(d), np.cos(d)))


def summarize(rows):
    phys = np.asarray([r["physical_factor_response_norm"] for r in rows], float)
    enc = np.asarray([r["encoder_response_norm"] for r in rows], float)
    pred = np.asarray([r["pred_response_norm"] for r in rows], float)
    gain = np.asarray([r["pred_enc_gain"] for r in rows], float)
    return {
        "n_directions": int(len(rows)),
        "physical_factor_min": float(np.min(phys)),
        "physical_factor_median": float(np.median(phys)),
        "encoder_min": float(np.min(enc)),
        "encoder_median": float(np.median(enc)),
        "pred_min": float(np.min(pred)),
        "pred_median": float(np.median(pred)),
        "gain_median": float(np.median(gain)),
        "gain_max": float(np.max(gain)),
        "rho_physical_encoder": spearman(phys, enc),
        "rho_physical_pred": spearman(phys, pred),
        "physical_null_fraction": float(np.mean([r["physical_null"] for r in rows])),
        "encoder_null_fraction": float(np.mean([r["encoder_null"] for r in rows])),
        "phantom_pred_fraction": float(np.mean([r["phantom_pred"] for r in rows])),
        "encoder_collapse_fraction": float(np.mean([r["encoder_collapse"] for r in rows])),
        "contact_asymmetry_fraction": float(np.mean([r["contact_asymmetry"] for r in rows])),
    }


def main():
    a = parse_args()
    if a.labels is not None and len(a.labels) != len(a.policies):
        raise ValueError("--labels length must equal --policies length")

    landscape_csv = (
        Path(a.landscape_csv)
        if a.landscape_csv
        else Path(a.trace_dir) / "planner_center_landscape" / "planner_center_metrics.csv"
    )
    if not landscape_csv.exists():
        raise FileNotFoundError(landscape_csv)

    selected = select_centers(
        read_csv(landscape_csv), a.gain_threshold, a.top_centers
    )
    if not selected:
        raise RuntimeError("No centers selected.")

    cfg = OmegaConf.load(a.config)
    dataset_name = a.dataset or str(cfg.eval.dataset_name)
    cache = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    out = (
        Path(a.output_dir)
        if a.output_dir
        else Path(a.trace_dir) / "null_response_decomposition"
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

    trace_files = load_traces(a.trace_dir, 0)
    trace_index = {p.name: i for i, p in enumerate(trace_files)}
    selected_map = {(x["trace_file"], x["cem_iteration"]): x for x in selected}
    selected_rows = []
    direction_rows = []
    start = time.time()
    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")

    try:
        for item in selected:
            tf = item["trace_file"]
            it = int(item["cem_iteration"])
            path = Path(a.trace_dir) / tf
            if not path.exists():
                raise FileNotFoundError(path)
            tr = np.load(path, allow_pickle=True)
            means = np.asarray(tr["mean"], dtype=np.float32)
            if not (0 <= it < len(means)):
                continue

            si = trace_index[tf]
            init_state = maybe_inverse_state(
                tr["info_state"], state_scaler, a.state_space
            )
            goal_state = maybe_inverse_state(
                tr["info_goal_state"], state_scaler, a.state_space
            )
            action_block = int(np.asarray(tr["action_block"]).item())
            horizon = int(np.asarray(tr["horizon"]).item())
            raw_horizon = horizon * action_block
            seed = a.env_seed + si
            current_image, _ = _current_goal_images(
                env, init_state, goal_state, seed
            )

            center_raw = decode_normalized_plan(
                means[it], action_scaler, action_block, clip=True
            )
            rng = np.random.default_rng(31_337_001 * (si + 1) + it)
            cands, _, dmeta, smeta, _, eqerr = _make_fixed_first_block_candidates(
                center_raw, a.radius, a.num_directions, rng, action_block
            )
            normalized = _normalize_actions(
                cands, action_scaler, horizon, action_block
            )

            real_states = np.empty((len(cands), state.shape[-1]), dtype=np.float64)
            real_images = [None] * len(cands)
            contacts = [None] * len(cands)
            for ci in range(len(cands)):
                rr = _rollout_checkpoints(
                    env, init_state, goal_state, cands[ci], [raw_horizon], seed
                )[raw_horizon]
                real_states[ci] = rr["state"]
                real_images[ci] = rr["image"]
                contacts[ci] = (bool(rr["had_contact"]), int(rr["contact_steps"]))

            for label, model in zip(labels, models):
                zr = _encode(
                    model, transform, real_images, device, a.model_batch_size
                )
                zp = _predict(
                    model,
                    transform,
                    current_image,
                    normalized,
                    device,
                    a.model_batch_size,
                )
                center_model_rows = []

                for di in range(a.num_directions):
                    ip = np.where((dmeta == di) & (smeta > 0))[0]
                    im = np.where((dmeta == di) & (smeta < 0))[0]
                    if len(ip) != 1 or len(im) != 1:
                        raise RuntimeError(f"Malformed pair direction={di}")
                    ip, im = int(ip[0]), int(im[0])

                    fp = _state_factor(real_states[ip])
                    fm = _state_factor(real_states[im])
                    phys_vec = (fp - fm) / (2.0 * a.radius)
                    pusher = (
                        real_states[ip, 0:2] - real_states[im, 0:2]
                    ) / (2.0 * a.radius)
                    block = (
                        real_states[ip, 2:4] - real_states[im, 2:4]
                    ) / (2.0 * a.radius)
                    theta = signed_angle_delta(
                        real_states[ip, 4], real_states[im, 4]
                    ) / (2.0 * a.radius)

                    enc_vec = (
                        (zr[ip] - zr[im]) / (2.0 * a.radius)
                    ).cpu().numpy()
                    pred_vec = (
                        (zp[ip] - zp[im]) / (2.0 * a.radius)
                    ).cpu().numpy()
                    phys_norm = float(np.linalg.norm(phys_vec))
                    enc_norm = float(np.linalg.norm(enc_vec))
                    pred_norm = float(np.linalg.norm(pred_vec))

                    physical_null = phys_norm <= a.physical_null_threshold
                    encoder_null = enc_norm <= a.encoder_null_threshold
                    pred_active = pred_norm >= a.pred_active_threshold
                    row = {
                        "trace_source": a.trace_label,
                        "model": label,
                        "trace_file": tf,
                        "solve_index": int(np.asarray(tr["solve_index"]).item()),
                        "cem_iteration": it,
                        "source_max_gain": float(item["max_gain"]),
                        "direction": int(di),
                        "radius": float(a.radius),
                        "physical_factor_response_norm": phys_norm,
                        "physical_pusher_response_norm": float(np.linalg.norm(pusher)),
                        "physical_block_response_norm": float(np.linalg.norm(block)),
                        "physical_theta_response_abs": float(abs(theta)),
                        "encoder_response_norm": enc_norm,
                        "pred_response_norm": pred_norm,
                        "pred_enc_gain": pred_norm / max(enc_norm, 1e-12),
                        "pred_enc_cosine": cosine(pred_vec, enc_vec),
                        "physical_null": bool(physical_null),
                        "encoder_null": bool(encoder_null),
                        "pred_active": bool(pred_active),
                        "phantom_pred": bool(physical_null and pred_active),
                        "encoder_collapse": bool((not physical_null) and encoder_null),
                        "plus_had_contact": bool(contacts[ip][0]),
                        "minus_had_contact": bool(contacts[im][0]),
                        "plus_contact_steps": int(contacts[ip][1]),
                        "minus_contact_steps": int(contacts[im][1]),
                        "contact_asymmetry": bool(contacts[ip][0] != contacts[im][0]),
                    }
                    direction_rows.append(row)
                    center_model_rows.append(row)

                selected_rows.append({
                    "trace_source": a.trace_label,
                    "model": label,
                    "trace_file": tf,
                    "solve_index": int(np.asarray(tr["solve_index"]).item()),
                    "cem_iteration": it,
                    "source_max_gain": float(item["max_gain"]),
                    "max_equal_norm_error": float(eqerr),
                    **summarize(center_model_rows),
                })
            print(f"{tf} iteration={it} done")
    finally:
        env.close()

    write_csv(out / "selected_centers.csv", selected_rows)
    write_csv(out / "direction_metrics.csv", direction_rows)
    with open(out / "summary.json", "w") as f:
        json.dump(
            {
                "trace_source": a.trace_label,
                "config": vars(a),
                "selected_from": str(landscape_csv),
                "selected_centers": selected,
                "center_summary": selected_rows,
                "elapsed_seconds": time.time() - start,
            },
            f,
            indent=2,
        )

    print("\nNear-null response decomposition")
    print(f"{'model':<9} {'solve':>5} {'it':>3} {'physMin':>9} {'encMin':>9} {'gainMax':>10} {'phantom':>8} {'encCol':>7}")
    for r in selected_rows:
        print(
            f"{r['model']:<9} {r['solve_index']:>5} {r['cem_iteration']:>3} "
            f"{r['physical_factor_min']:>9.3e} {r['encoder_min']:>9.3e} "
            f"{r['gain_max']:>10.3e} {r['phantom_pred_fraction']:>8.3f} "
            f"{r['encoder_collapse_fraction']:>7.3f}"
        )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
