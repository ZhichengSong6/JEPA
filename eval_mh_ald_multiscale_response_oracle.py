#!/usr/bin/env python3
"""Problem 3: is one local probe radius insufficient for PushT contact dynamics?

DIAGNOSTIC ONLY.  No training or CEM change.

We keep the same formal MH-ALD anchor/probe structure and the official planner
rollout semantics, but evaluate the same seeded block-wise probe directions at
multiple raw-action radii (default 0.08, 0.1565, 0.30).

For each radius we compare teacher/student finite responses with real simulator
counterfactual responses encoded in the same frozen visual frame.  We also ask
whether the real response itself changes with radius and whether +/- probes
cross different contact regimes.  The middle radius reproduces the formal
MH-ALD probe radius.

A strong scale effect would support finite/multi-scale response calibration
instead of treating the local dynamics as one-radius/Jacobian-like.
"""
from __future__ import annotations

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
from eval_mh_ald_context_gap_oracle import (
    _assert_planner_semantics,
    _planner_warmup_rollout,
)
from eval_mh_ald_goal_projection_oracle import _ranking_metrics
from eval_mh_ald_teacher_response_oracle import (
    _candidate_pair_indices,
    _cos,
    _gain,
    _make_blockwise_candidates,
    _pack_and_normalize,
    _relerr,
    _render_state,
    _rollout_multihorizon,
    _select_anchor_rows,
    _summary,
)
from eval_pusht_horizon_directional import (
    _encode,
    _jsonable,
    _physical_cost,
    _spearman,
    _state_factor,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default="pusht_expert_train")
    p.add_argument("--teacher-policy", default="lewm_epoch_10")
    p.add_argument(
        "--student-policy",
        default="pusht_mh_ald_h5_seed3072_ep10_ddp4/"
        "lewm_mh_ald_h5_ddp4_epoch_10",
    )
    p.add_argument("--num-anchors", type=int, default=40)
    p.add_argument("--radii", nargs="+", type=float, default=[0.08, 0.1565, 0.30])
    p.add_argument("--reference-radius", type=float, default=0.1565)
    p.add_argument("--train-observation-history", type=int, default=3)
    p.add_argument("--planner-observation-history", type=int, default=1)
    p.add_argument("--predictor-max-history", type=int, default=3)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--directions-per-position", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=12000)
    p.add_argument("--model-batch-size", type=int, default=128)
    p.add_argument("--replay-good-threshold", type=float, default=0.10)
    p.add_argument("--pair-margin-frac", type=float, default=0.02)
    p.add_argument("--semantic-check-atol", type=float, default=2e-5)
    p.add_argument("--reference-context-summary", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _radius_key(x):
    return f"{float(x):.6f}"


def _same_probe_structure(meta_a, meta_b):
    if len(meta_a) != len(meta_b):
        return False
    for a, b in zip(meta_a, meta_b):
        if (a["position"], a["direction"], a["sign"]) != (
            b["position"], b["direction"], b["sign"]
        ):
            return False
    return True


def _action_direction(candidates, meta, center, position, direction, action_block):
    ip, _ = _candidate_pair_indices(meta, position, direction)
    sl = slice(position * int(action_block), (position + 1) * int(action_block))
    v = (
        np.asarray(candidates[ip][sl], dtype=np.float64)
        - np.asarray(center[sl], dtype=np.float64)
    ).reshape(-1)
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)


def _agg_response(rows):
    keys = [
        "teacher_oracle_cosine", "student_oracle_cosine",
        "teacher_oracle_relerr", "student_oracle_relerr",
        "teacher_oracle_gain", "student_oracle_gain",
        "oracle_cosine_to_small", "oracle_relerr_to_small",
        "teacher_cosine_to_small", "student_cosine_to_small",
    ]
    return {k: _summary(r[k] for r in rows) for k in keys}


def _agg_anchor(rows):
    keys = [
        "teacher_rho_exact_encoder", "student_rho_exact_encoder",
        "teacher_rho_physical", "student_rho_physical",
        "teacher_top10_exact_encoder_overlap", "student_top10_exact_encoder_overlap",
        "teacher_center_endpoint_mse", "student_center_endpoint_mse",
        "rho_exact_encoder_physical",
    ]
    return {k: _summary(r[k] for r in rows) for k in keys}


def _agg_pairs(rows):
    return {
        "pairs": len(rows),
        "cross_contact_fraction": (
            float(np.mean([r["cross_contact"] for r in rows])) if rows else None
        ),
        "any_contact_fraction": (
            float(np.mean([r["any_contact"] for r in rows])) if rows else None
        ),
        "direction_cosine_to_small": _summary(
            r["direction_cosine_to_small"] for r in rows
        ),
        "effective_radius": _summary(r["effective_radius"] for r in rows),
    }


def _group_summary(response_rows, anchor_rows, pair_rows, radii):
    out = {}
    for radius in radii:
        rk = _radius_key(radius)
        rr = [r for r in response_rows if r["radius_key"] == rk]
        aa = [r for r in anchor_rows if r["radius_key"] == rk]
        pp = [r for r in pair_rows if r["radius_key"] == rk]
        out[rk] = {
            "response": _agg_response(rr),
            "anchor": _agg_anchor(aa),
            "pairs": _agg_pairs(pp),
        }
    return out


def main():
    args = parse_args()
    radii = sorted({float(r) for r in args.radii})
    if len(radii) < 2 or min(radii) <= 0:
        raise ValueError("Need at least two positive radii")
    if not any(abs(r - float(args.reference_radius)) < 1e-10 for r in radii):
        raise ValueError("reference-radius must be one of --radii")
    if int(args.train_observation_history) != 3:
        raise ValueError("Formal MH-ALD train observation history is 3")
    if int(args.planner_observation_history) != 1:
        raise ValueError("Formal PushT planner observation history is 1")
    if int(args.predictor_max_history) != 3:
        raise ValueError("Formal predictor max history is 3")
    if int(args.horizon) != 5 or int(args.action_block) != 5:
        raise ValueError("Formal PushT expects horizon=5, action_block=5")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    if int(cfg.world.history_size) != 1:
        raise RuntimeError("Official PushT eval config must have world.history_size=1")

    cache_root = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    dataset = swm.data.HDF5Dataset(
        args.dataset, keys_to_cache=["action", "state"], cache_dir=cache_root
    )
    action = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    state = np.asarray(dataset.get_col_data("state"), dtype=np.float64)
    finite = action[np.isfinite(action).all(axis=1)]
    scaler = preprocessing.StandardScaler().fit(finite)

    history_raw = (int(args.train_observation_history) - 1) * int(args.action_block)
    future_raw = int(args.horizon) * int(args.action_block)
    ep_col, anchors = _select_anchor_rows(
        dataset, args.num_anchors, args.seed, history_raw, future_raw
    )
    if args.reference_context_summary:
        ref = json.loads(Path(args.reference_context_summary).read_text())
        if list(map(int, ref.get("selected_rows", []))) != anchors.tolist():
            raise RuntimeError("Problem-3 anchor rows differ from context-gap oracle")

    episodes = np.asarray(dataset.get_col_data(ep_col))
    steps = np.asarray(dataset.get_col_data("step_idx"))
    device = torch.device(args.device)
    transform = img_transform(cfg)

    teacher = swm.policy.AutoCostModel(args.teacher_policy).to(device).eval()
    teacher.requires_grad_(False)
    teacher.interpolate_pos_encoding = True
    student = swm.policy.AutoCostModel(args.student_policy).to(device).eval()
    student.requires_grad_(False)
    student.interpolate_pos_encoding = True
    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")

    response_rows, anchor_rows, pair_rows = [], [], []
    frame_max_abs = 0.0
    semantic_check = {"teacher": None, "student": None}
    t0 = time.time()

    try:
        for ai, row in enumerate(anchors):
            seed = int(args.env_seed) + ai
            goal_row = int(row) + future_raw
            goal_state = state[goal_row].copy()
            current_state = state[int(row)].copy()
            hist_rows = [
                int(row) - history_raw + k * int(args.action_block)
                for k in range(int(args.train_observation_history))
            ]
            hist_images = [_render_state(env, state[r], goal_state, seed) for r in hist_rows]
            current_image = hist_images[-1]
            goal_image = _render_state(env, goal_state, goal_state, seed)
            hist_s = _encode(student, transform, hist_images, device, args.model_batch_size)
            hist_t = _encode(teacher, transform, hist_images, device, args.model_batch_size)
            fdiff = float(torch.max(torch.abs(hist_s - hist_t)).detach().cpu())
            frame_max_abs = max(frame_max_abs, fdiff)
            if fdiff > 2e-5:
                raise RuntimeError(f"Teacher/student latent frames differ: {fdiff:.3e}")
            goal_z = _encode(student, transform, [goal_image], device, args.model_batch_size)[0]

            future_raw_center = np.nan_to_num(
                action[int(row): int(row) + future_raw],
                nan=0.0, posinf=1.0, neginf=-1.0,
            ).astype(np.float32)
            probe_seed = int(args.seed) + 10000019 * (ai + 1)

            by_radius = {}
            base_meta = None
            replay_reference = None
            for radius in radii:
                candidates_raw, meta = _make_blockwise_candidates(
                    future_raw_center,
                    positions=list(range(int(args.horizon))),
                    directions_per_position=int(args.directions_per_position),
                    radius=radius,
                    seed=probe_seed,
                    action_block=int(args.action_block),
                )
                if base_meta is None:
                    base_meta = meta
                elif not _same_probe_structure(base_meta, meta):
                    raise RuntimeError("Probe structure changed across radii")
                future_packed = np.stack([
                    _pack_and_normalize(c, scaler, args.action_block)
                    for c in candidates_raw
                ])
                teacher_roll = _planner_warmup_rollout(
                    teacher, hist_t[-1].detach().cpu().numpy(), future_packed,
                    args.predictor_max_history, device,
                )
                student_roll = _planner_warmup_rollout(
                    student, hist_s[-1].detach().cpu().numpy(), future_packed,
                    args.predictor_max_history, device,
                )
                if ai == 0 and abs(radius - args.reference_radius) < 1e-10:
                    for name, model, emb in (
                        ("teacher", teacher, hist_t[-1]),
                        ("student", student, hist_s[-1]),
                    ):
                        semantic_check[name] = _assert_planner_semantics(
                            model, current_image, transform, emb, future_packed[0],
                            args.predictor_max_history, device, args.semantic_check_atol,
                        )

                oracle_images, final_states, contacts = [], [], []
                center_states = None
                center_contact = False
                for ci, cand in enumerate(candidates_raw):
                    states_h, images_h, contact = _rollout_multihorizon(
                        env, current_state, goal_state, cand, seed, args.action_block
                    )
                    final_states.append(states_h[-1])
                    contacts.append(bool(contact))
                    oracle_images.extend(images_h)
                    if ci == 0:
                        center_states = states_h
                        center_contact = bool(contact)
                oracle_z = _encode(
                    student, transform, oracle_images, device, args.model_batch_size
                ).reshape(len(candidates_raw), int(args.horizon), -1)
                final_states = np.stack(final_states)
                replay_error = float(np.linalg.norm(
                    _state_factor(center_states[-1]) - _state_factor(goal_state)
                ))
                replay_good = replay_error <= float(args.replay_good_threshold)
                if replay_reference is None:
                    replay_reference = replay_error
                elif abs(replay_error - replay_reference) > 1e-8:
                    raise RuntimeError("Center replay changed across radii")

                exact_cost = (
                    torch.sum((oracle_z[:, -1] - goal_z[None]) ** 2, dim=-1)
                    .detach().cpu().numpy().astype(np.float64)
                )
                physical_cost = _physical_cost(final_states, goal_state)[0]
                tc = (
                    torch.sum((teacher_roll[:, -1] - goal_z[None]) ** 2, dim=-1)
                    .detach().cpu().numpy().astype(np.float64)
                )
                sc = (
                    torch.sum((student_roll[:, -1] - goal_z[None]) ** 2, dim=-1)
                    .detach().cpu().numpy().astype(np.float64)
                )
                tm = _ranking_metrics(tc, exact_cost, physical_cost, args.pair_margin_frac)
                sm = _ranking_metrics(sc, exact_cost, physical_cost, args.pair_margin_frac)
                rk = _radius_key(radius)
                anchor_rows.append({
                    "anchor_index": ai,
                    "dataset_row": int(row),
                    "episode_idx": int(episodes[row]),
                    "step_idx": int(steps[row]),
                    "radius": radius,
                    "radius_key": rk,
                    "replay_good": replay_good,
                    "center_contact": center_contact,
                    "replay_endpoint_factor_error": replay_error,
                    "rho_exact_encoder_physical": _spearman(exact_cost, physical_cost),
                    "teacher_rho_exact_encoder": tm["rho_exact_encoder"],
                    "student_rho_exact_encoder": sm["rho_exact_encoder"],
                    "teacher_rho_physical": tm["rho_physical"],
                    "student_rho_physical": sm["rho_physical"],
                    "teacher_top10_exact_encoder_overlap": tm["top10_exact_encoder_overlap"],
                    "student_top10_exact_encoder_overlap": sm["top10_exact_encoder_overlap"],
                    "teacher_center_endpoint_mse": float(torch.mean(
                        (teacher_roll[0, -1] - oracle_z[0, -1]) ** 2
                    ).detach().cpu()),
                    "student_center_endpoint_mse": float(torch.mean(
                        (student_roll[0, -1] - oracle_z[0, -1]) ** 2
                    ).detach().cpu()),
                })
                by_radius[rk] = {
                    "radius": radius, "candidates": candidates_raw, "meta": meta,
                    "teacher": teacher_roll, "student": student_roll,
                    "oracle": oracle_z, "contacts": contacts,
                    "replay_good": replay_good, "center_contact": center_contact,
                }

            small_key = _radius_key(radii[0])
            small = by_radius[small_key]
            for radius in radii:
                rk = _radius_key(radius)
                cur = by_radius[rk]
                for p in range(int(args.horizon)):
                    for d in range(int(args.directions_per_position)):
                        ip, im = _candidate_pair_indices(cur["meta"], p, d)
                        sip, sim = _candidate_pair_indices(small["meta"], p, d)
                        eff = float(cur["meta"][ip]["effective_radius"])
                        seff = float(small["meta"][sip]["effective_radius"])
                        direction = _action_direction(
                            cur["candidates"], cur["meta"], future_raw_center,
                            p, d, args.action_block,
                        )
                        sdir = _action_direction(
                            small["candidates"], small["meta"], future_raw_center,
                            p, d, args.action_block,
                        )
                        pair_rows.append({
                            "anchor_index": ai, "dataset_row": int(row),
                            "radius": radius, "radius_key": rk,
                            "position": p, "direction": d,
                            "effective_radius": eff,
                            "direction_cosine_to_small": float(np.dot(direction, sdir)),
                            "plus_contact": bool(cur["contacts"][ip]),
                            "minus_contact": bool(cur["contacts"][im]),
                            "any_contact": bool(cur["contacts"][ip] or cur["contacts"][im]),
                            "cross_contact": bool(cur["contacts"][ip] != cur["contacts"][im]),
                            "replay_good": cur["replay_good"],
                            "center_contact": cur["center_contact"],
                        })
                        if eff <= 1e-8 or seff <= 1e-8:
                            continue
                        for h in range(p, int(args.horizon)):
                            oracle = (
                                cur["oracle"][ip, h] - cur["oracle"][im, h]
                            ).detach().cpu().numpy() / (2.0 * eff)
                            teacher_r = (
                                cur["teacher"][ip, h] - cur["teacher"][im, h]
                            ).detach().cpu().numpy() / (2.0 * eff)
                            student_r = (
                                cur["student"][ip, h] - cur["student"][im, h]
                            ).detach().cpu().numpy() / (2.0 * eff)
                            oracle_small = (
                                small["oracle"][sip, h] - small["oracle"][sim, h]
                            ).detach().cpu().numpy() / (2.0 * seff)
                            teacher_small = (
                                small["teacher"][sip, h] - small["teacher"][sim, h]
                            ).detach().cpu().numpy() / (2.0 * seff)
                            student_small = (
                                small["student"][sip, h] - small["student"][sim, h]
                            ).detach().cpu().numpy() / (2.0 * seff)
                            response_rows.append({
                                "anchor_index": ai, "dataset_row": int(row),
                                "radius": radius, "radius_key": rk,
                                "position": p, "direction": d,
                                "horizon_index": h,
                                "horizon_from_perturb": h - p,
                                "effective_radius": eff,
                                "replay_good": cur["replay_good"],
                                "center_contact": cur["center_contact"],
                                "pair_contact": bool(cur["contacts"][ip] or cur["contacts"][im]),
                                "cross_contact": bool(cur["contacts"][ip] != cur["contacts"][im]),
                                "teacher_oracle_cosine": _cos(teacher_r, oracle),
                                "student_oracle_cosine": _cos(student_r, oracle),
                                "teacher_oracle_relerr": _relerr(teacher_r, oracle),
                                "student_oracle_relerr": _relerr(student_r, oracle),
                                "teacher_oracle_gain": _gain(teacher_r, oracle),
                                "student_oracle_gain": _gain(student_r, oracle),
                                "oracle_cosine_to_small": _cos(oracle, oracle_small),
                                "oracle_relerr_to_small": _relerr(oracle, oracle_small),
                                "teacher_cosine_to_small": _cos(teacher_r, teacher_small),
                                "student_cosine_to_small": _cos(student_r, student_small),
                            })

            elapsed = time.time() - t0
            eta = elapsed / (ai + 1) * (len(anchors) - ai - 1)
            print(
                f"anchor {ai+1:3d}/{len(anchors)} row={row} "
                f"good={int(by_radius[_radius_key(args.reference_radius)]['replay_good'])} "
                f"ETA={eta/60:.1f}m",
                flush=True,
            )
    finally:
        env.close()

    response_csv = outdir / "multiscale_response_cells.csv"
    anchor_csv = outdir / "multiscale_anchor_metrics.csv"
    pair_csv = outdir / "multiscale_pair_metrics.csv"
    _write_csv(response_csv, response_rows)
    _write_csv(anchor_csv, anchor_rows)
    _write_csv(pair_csv, pair_rows)

    good_r = [r for r in response_rows if r["replay_good"]]
    good_a = [r for r in anchor_rows if r["replay_good"]]
    good_p = [r for r in pair_rows if r["replay_good"]]
    contact_r = [r for r in good_r if r["pair_contact"]]
    contact_a = [r for r in good_a if r["center_contact"]]
    contact_p = [r for r in good_p if r["any_contact"]]
    crossing_r = [r for r in good_r if r["cross_contact"]]
    crossing_p = [r for r in good_p if r["cross_contact"]]

    summary = {
        "question": (
            "Does the real/planner action response vary enough with perturbation "
            "radius, especially around contact boundaries, that one-radius MH-ALD "
            "is an insufficient local model?"
        ),
        "config": vars(args),
        "selected_rows": anchors.tolist(),
        "teacher_policy": args.teacher_policy,
        "student_policy": args.student_policy,
        "latent_frame_max_abs_teacher_vs_student": frame_max_abs,
        "planner_semantic_check_max_abs": semantic_check,
        "replay_good": _group_summary(good_r, good_a, good_p, radii),
        "replay_good_contact": _group_summary(contact_r, contact_a, contact_p, radii),
        "replay_good_cross_contact": _group_summary(crossing_r, [], crossing_p, radii),
        "protocol_notes": [
            "Official planner context is used for both models: one observed latent then self-generated history capped at 3.",
            "Each radius reuses the same per-anchor/per-position/per-direction RNG seed; action direction cosine to the smallest radius is reported to reveal bound-induced bending.",
            "The exact simulator/encoder response is oracle-only and never changes CEM or training.",
            "The formal 0.1565 radius remains included as the reference MH-ALD scale.",
            "cross_contact means the +/- members of one symmetric pair differ in whether any contact occurred during the 25-step rollout.",
        ],
        "elapsed_seconds": float(time.time() - t0),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print("\n===== MULTISCALE RESPONSE ORACLE: REPLAY-GOOD =====")
    print(json.dumps(_jsonable(summary["replay_good"]), indent=2))
    print("\n===== REPLAY-GOOD + CONTACT =====")
    print(json.dumps(_jsonable(summary["replay_good_contact"]), indent=2))
    print(f"Saved: {response_csv}")
    print(f"Saved: {anchor_csv}")
    print(f"Saved: {pair_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
