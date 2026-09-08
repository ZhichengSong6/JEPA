#!/usr/bin/env python3
"""Diagnose the core ALD premise: is the frozen LeWM teacher response trustworthy?

We compare, on the SAME symmetric block-wise probes,

    teacher response  : T_h(U+) - T_h(U-)
    student response  : S_h(U+) - S_h(U-)
    oracle response   : E(o_h^real(U+)) - E(o_h^real(U-))

where E is the frozen visual encoder frame shared by the official LeWM teacher
and the current MH-ALD student.

The construction mirrors MH-ALD training:
  * history_size = 3 coarse steps
  * horizon = 5 coarse steps
  * action_block = 5 raw PushT actions
  * one action block p is perturbed at a time
  * only causal cells h >= p are evaluated
  * symmetric bounded perturbations use the MH-ALD raw-action radius

No privileged state is used on the model side. Simulator state is oracle-only.
Because the public 7-D PushT state does not contain block momentum, each anchor
also reports an expert replay sanity error. Results are summarized both over all
anchors and over replay-good anchors only.
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
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn import preprocessing

from eval import img_transform
from eval_pusht_horizon_directional import (
    _bounded_delta,
    _encode,
    _jsonable,
    _physical_cost,
    _spearman,
    _state_factor,
)
from stage1_bias_calibration import _autoregressive_rollout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--dataset", default="pusht_expert_train")
    p.add_argument("--teacher-policy", default="lewm_epoch_10")
    p.add_argument(
        "--student-policy",
        default="pusht_mh_ald_h5_seed3072_ep10_ddp4/"
        "lewm_mh_ald_h5_ddp4_epoch_10",
    )
    p.add_argument("--num-anchors", type=int, default=40)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--directions-per-position", type=int, default=4)
    p.add_argument("--perturb-radius", type=float, default=0.1565)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=12000)
    p.add_argument("--model-batch-size", type=int, default=128)
    p.add_argument("--replay-good-threshold", type=float, default=0.10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _summary(values):
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
        }
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def _cos(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    return float(F.cosine_similarity(a[None], b[None], dim=-1, eps=1e-8)[0])


def _relerr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-8))


def _gain(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-8))


def _select_anchor_rows(dataset, num_anchors, seed, history_raw, future_raw):
    ep_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = np.asarray(dataset.get_col_data(ep_col))
    step = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    valid = []
    for i in range(len(step)):
        if int(step[i]) < int(history_raw):
            continue
        j0 = i - int(history_raw)
        j1 = i + int(future_raw)
        if j0 < 0 or j1 >= len(step):
            continue
        epi = ep[i]
        ok = True
        for j in range(j0, j1 + 1):
            expected = int(step[i]) + (j - i)
            if ep[j] != epi or int(step[j]) != expected:
                ok = False
                break
        if ok:
            valid.append(i)
    if len(valid) < int(num_anchors):
        raise RuntimeError(
            f"Only {len(valid)} valid anchors, requested {num_anchors}."
        )
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(np.asarray(valid), size=int(num_anchors), replace=False)
    return ep_col, np.sort(pick.astype(np.int64))


def _reset_state(env, state, goal_state, seed):
    env.reset(seed=int(seed))
    raw = env.unwrapped
    raw._set_goal_state(np.asarray(goal_state, dtype=np.float64))
    raw._set_state(np.asarray(state, dtype=np.float64))
    return raw


def _render_state(env, state, goal_state, seed):
    raw = _reset_state(env, state, goal_state, seed)
    return np.asarray(raw.render())


def _rollout_multihorizon(
    env,
    state,
    goal_state,
    raw_actions,
    seed,
    action_block,
):
    raw = _reset_state(env, state, goal_state, seed)
    images = []
    states = []
    contact = False
    for t, action in enumerate(np.asarray(raw_actions, dtype=np.float32)):
        obs, _, _, _, info = raw.step(action)
        contact = contact or int(info.get("n_contacts", 0)) > 0
        if (t + 1) % int(action_block) == 0:
            states.append(np.asarray(obs["state"], dtype=np.float64))
            images.append(np.asarray(raw.render()))
    return np.stack(states), images, bool(contact)


def _pack_and_normalize(raw_actions, scaler, action_block):
    x = np.asarray(raw_actions, dtype=np.float32)
    if len(x) % int(action_block) != 0:
        raise ValueError("raw action sequence is not divisible by action_block")
    n = len(x) // int(action_block)
    z = scaler.transform(x.reshape(-1, x.shape[-1])).astype(np.float32)
    return z.reshape(n, int(action_block) * x.shape[-1])


def _make_blockwise_candidates(
    future_actions,
    positions,
    directions_per_position,
    radius,
    seed,
    action_block,
):
    base = np.asarray(future_actions, dtype=np.float32).copy()
    if len(base) != len(positions) * int(action_block):
        raise ValueError("future action length does not match positions/action_block")

    candidates = [base.copy()]
    meta = [{
        "position": -1,
        "direction": -1,
        "sign": 0,
        "effective_radius": 0.0,
    }]

    for p in positions:
        sl = slice(p * int(action_block), (p + 1) * int(action_block))
        active = np.nan_to_num(
            base[sl].astype(np.float64),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        active = np.clip(active, -1.0, 1.0)
        flat = active.reshape(-1)
        slack = np.maximum(1.0 - np.abs(flat), 0.0)
        feasible = float(np.linalg.norm(slack))
        effective = min(float(radius), 0.999 * feasible)
        if effective <= 1e-8:
            continue

        for d in range(int(directions_per_position)):
            rng = np.random.default_rng(
                int(seed) + 1000003 * (int(p) + 1) + 10007 * (d + 1)
            )
            delta = _bounded_delta(slack, effective, rng).reshape(active.shape)
            for sign in (+1, -1):
                cand = base.copy()
                cand[sl] = (active + float(sign) * delta).astype(np.float32)
                candidates.append(cand)
                meta.append({
                    "position": int(p),
                    "direction": int(d),
                    "sign": int(sign),
                    "effective_radius": float(effective),
                })

    return np.stack(candidates), meta


@torch.inference_mode()
def _model_rollout(
    model,
    history_emb,
    full_normalized_actions,
    history_size,
    horizon,
    device,
):
    n = full_normalized_actions.shape[0]
    h = torch.as_tensor(history_emb, device=device, dtype=torch.float32)
    h = h[None].expand(n, -1, -1)
    pad = h[:, -1:].expand(n, int(horizon), -1)
    initial_emb = torch.cat([h, pad], dim=1)
    actions = torch.as_tensor(
        full_normalized_actions,
        device=device,
        dtype=torch.float32,
    )
    return _autoregressive_rollout(
        model,
        initial_emb,
        actions,
        history_size=int(history_size),
        horizon=int(horizon),
    ).detach()


def _candidate_pair_indices(meta, position, direction):
    ip = im = None
    for i, m in enumerate(meta):
        if m["position"] != position or m["direction"] != direction:
            continue
        if m["sign"] == 1:
            ip = i
        elif m["sign"] == -1:
            im = i
    if ip is None or im is None:
        raise RuntimeError(
            f"missing +/- probe pair at position={position} direction={direction}"
        )
    return ip, im


def _aggregate_response(rows):
    return {
        "cells": len(rows),
        "teacher_oracle_cosine": _summary(
            r["teacher_oracle_cosine"] for r in rows
        ),
        "student_oracle_cosine": _summary(
            r["student_oracle_cosine"] for r in rows
        ),
        "student_teacher_cosine": _summary(
            r["student_teacher_cosine"] for r in rows
        ),
        "teacher_oracle_gain": _summary(
            r["teacher_oracle_gain"] for r in rows
        ),
        "student_oracle_gain": _summary(
            r["student_oracle_gain"] for r in rows
        ),
        "teacher_oracle_relerr": _summary(
            r["teacher_oracle_relerr"] for r in rows
        ),
        "student_oracle_relerr": _summary(
            r["student_oracle_relerr"] for r in rows
        ),
        "student_teacher_relerr": _summary(
            r["student_teacher_relerr"] for r in rows
        ),
    }


def _aggregate_anchor(rows):
    return {
        "anchors": len(rows),
        "replay_good_fraction": float(np.mean([r["replay_good"] for r in rows]))
        if rows else None,
        "teacher_center_endpoint_mse": _summary(
            r["teacher_center_endpoint_mse"] for r in rows
        ),
        "student_center_endpoint_mse": _summary(
            r["student_center_endpoint_mse"] for r in rows
        ),
        "rho_teacher_exact_encoder": _summary(
            r["rho_teacher_exact_encoder"] for r in rows
        ),
        "rho_student_exact_encoder": _summary(
            r["rho_student_exact_encoder"] for r in rows
        ),
        "rho_exact_encoder_physical": _summary(
            r["rho_exact_encoder_physical"] for r in rows
        ),
        "rho_teacher_physical": _summary(
            r["rho_teacher_physical"] for r in rows
        ),
        "rho_student_physical": _summary(
            r["rho_student_physical"] for r in rows
        ),
        "replay_endpoint_factor_error": _summary(
            r["replay_endpoint_factor_error"] for r in rows
        ),
    }


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if int(args.history_size) != 3 or int(args.horizon) != 5:
        print(
            "WARNING: current formal MH-ALD was trained with history=3,horizon=5."
        )
    if int(args.action_block) != 5:
        print("WARNING: current PushT planner uses action_block=5.")

    cfg = OmegaConf.load(args.config)
    cache_root = Path(
        os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir())
    )
    dataset = swm.data.HDF5Dataset(
        args.dataset,
        keys_to_cache=["action", "state"],
        cache_dir=cache_root,
    )

    action = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    state = np.asarray(dataset.get_col_data("state"), dtype=np.float64)
    finite = action[np.isfinite(action).all(axis=1)]
    scaler = preprocessing.StandardScaler().fit(finite)

    history_raw = (int(args.history_size) - 1) * int(args.action_block)
    future_raw = int(args.horizon) * int(args.action_block)
    ep_col, anchors = _select_anchor_rows(
        dataset,
        args.num_anchors,
        args.seed,
        history_raw,
        future_raw,
    )
    episodes = np.asarray(dataset.get_col_data(ep_col))
    steps = np.asarray(dataset.get_col_data("step_idx"))

    device = torch.device(args.device)
    transform = img_transform(cfg)

    print(f"Loading teacher: {args.teacher_policy}")
    teacher = swm.policy.AutoCostModel(args.teacher_policy).to(device).eval()
    teacher.requires_grad_(False)
    teacher.interpolate_pos_encoding = True

    print(f"Loading student: {args.student_policy}")
    student = swm.policy.AutoCostModel(args.student_policy).to(device).eval()
    student.requires_grad_(False)
    student.interpolate_pos_encoding = True

    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")

    response_rows = []
    anchor_rows = []
    frame_max_abs = 0.0
    t0 = time.time()

    try:
        for ai, row in enumerate(anchors):
            seed = int(args.env_seed) + int(ai)
            goal_row = int(row) + future_raw
            goal_state = state[goal_row].copy()
            current_state = state[int(row)].copy()

            hist_rows = [
                int(row) - history_raw + k * int(args.action_block)
                for k in range(int(args.history_size))
            ]
            hist_images = [
                _render_state(env, state[r], goal_state, seed)
                for r in hist_rows
            ]
            goal_image = _render_state(env, goal_state, goal_state, seed)

            hist_s = _encode(
                student,
                transform,
                hist_images,
                device,
                args.model_batch_size,
            )
            hist_t = _encode(
                teacher,
                transform,
                hist_images,
                device,
                args.model_batch_size,
            )
            this_frame_diff = float(
                torch.max(torch.abs(hist_s - hist_t)).detach().cpu()
            )
            frame_max_abs = max(frame_max_abs, this_frame_diff)
            if this_frame_diff > 2e-5:
                raise RuntimeError(
                    "Teacher/student visual latent frames differ; ALD response "
                    f"comparison is not in one coordinate system. max_abs={this_frame_diff}"
                )

            goal_z = _encode(
                student,
                transform,
                [goal_image],
                device,
                args.model_batch_size,
            )[0]

            history_actions_raw = action[
                int(row) - history_raw : int(row)
            ].copy()
            future_actions_raw = action[
                int(row) : int(row) + future_raw
            ].copy()
            center_future = np.nan_to_num(
                future_actions_raw,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).astype(np.float32)

            candidates_raw, meta = _make_blockwise_candidates(
                center_future,
                positions=list(range(int(args.horizon))),
                directions_per_position=int(args.directions_per_position),
                radius=float(args.perturb_radius),
                seed=int(args.seed) + 10000019 * (ai + 1),
                action_block=int(args.action_block),
            )
            nc = len(candidates_raw)

            hist_packed = _pack_and_normalize(
                history_actions_raw,
                scaler,
                args.action_block,
            )
            future_packed = np.stack([
                _pack_and_normalize(c, scaler, args.action_block)
                for c in candidates_raw
            ])
            hist_repeat = np.broadcast_to(
                hist_packed[None],
                (nc, *hist_packed.shape),
            ).copy()
            full_actions = np.concatenate(
                [hist_repeat, future_packed],
                axis=1,
            )

            teacher_roll = _model_rollout(
                teacher,
                hist_t.detach().cpu().numpy(),
                full_actions,
                args.history_size,
                args.horizon,
                device,
            )
            student_roll = _model_rollout(
                student,
                hist_s.detach().cpu().numpy(),
                full_actions,
                args.history_size,
                args.horizon,
                device,
            )

            oracle_images = []
            final_states = []
            center_states = None
            center_contact = False
            for ci, cand in enumerate(candidates_raw):
                states_h, images_h, contact = _rollout_multihorizon(
                    env,
                    current_state,
                    goal_state,
                    cand,
                    seed,
                    args.action_block,
                )
                final_states.append(states_h[-1])
                oracle_images.extend(images_h)
                if ci == 0:
                    center_states = states_h
                    center_contact = contact

            oracle_z = _encode(
                student,
                transform,
                oracle_images,
                device,
                args.model_batch_size,
            ).reshape(nc, int(args.horizon), -1)

            final_states = np.stack(final_states)
            replay_error = float(
                np.linalg.norm(
                    _state_factor(center_states[-1])
                    - _state_factor(goal_state)
                )
            )
            replay_good = replay_error <= float(args.replay_good_threshold)

            teacher_center_mse = float(
                torch.mean(
                    (teacher_roll[0, -1] - oracle_z[0, -1]) ** 2
                ).detach().cpu()
            )
            student_center_mse = float(
                torch.mean(
                    (student_roll[0, -1] - oracle_z[0, -1]) ** 2
                ).detach().cpu()
            )

            exact_enc_cost = (
                torch.sum((oracle_z[:, -1] - goal_z[None]) ** 2, dim=-1)
                .detach().cpu().numpy().astype(np.float64)
            )
            teacher_cost = (
                torch.sum((teacher_roll[:, -1] - goal_z[None]) ** 2, dim=-1)
                .detach().cpu().numpy().astype(np.float64)
            )
            student_cost = (
                torch.sum((student_roll[:, -1] - goal_z[None]) ** 2, dim=-1)
                .detach().cpu().numpy().astype(np.float64)
            )
            physical_cost = _physical_cost(final_states, goal_state)[0]

            anchor_rows.append({
                "anchor_index": int(ai),
                "dataset_row": int(row),
                "episode_idx": int(episodes[row]),
                "step_idx": int(steps[row]),
                "num_candidates": int(nc),
                "center_contact": bool(center_contact),
                "replay_endpoint_factor_error": replay_error,
                "replay_good": bool(replay_good),
                "teacher_center_endpoint_mse": teacher_center_mse,
                "student_center_endpoint_mse": student_center_mse,
                "rho_teacher_exact_encoder": _spearman(
                    teacher_cost, exact_enc_cost
                ),
                "rho_student_exact_encoder": _spearman(
                    student_cost, exact_enc_cost
                ),
                "rho_exact_encoder_physical": _spearman(
                    exact_enc_cost, physical_cost
                ),
                "rho_teacher_physical": _spearman(
                    teacher_cost, physical_cost
                ),
                "rho_student_physical": _spearman(
                    student_cost, physical_cost
                ),
            })

            for p in range(int(args.horizon)):
                for d in range(int(args.directions_per_position)):
                    try:
                        ip, im = _candidate_pair_indices(meta, p, d)
                    except RuntimeError:
                        continue
                    radius = float(meta[ip]["effective_radius"])
                    if radius <= 1e-8:
                        continue
                    for h in range(p, int(args.horizon)):
                        oracle_resp = (
                            oracle_z[ip, h] - oracle_z[im, h]
                        ).detach().cpu().numpy() / (2.0 * radius)
                        teacher_resp = (
                            teacher_roll[ip, h] - teacher_roll[im, h]
                        ).detach().cpu().numpy() / (2.0 * radius)
                        student_resp = (
                            student_roll[ip, h] - student_roll[im, h]
                        ).detach().cpu().numpy() / (2.0 * radius)

                        response_rows.append({
                            "anchor_index": int(ai),
                            "dataset_row": int(row),
                            "replay_good": bool(replay_good),
                            "position": int(p),
                            "direction": int(d),
                            "horizon_index": int(h),
                            "horizon_from_perturb": int(h - p),
                            "effective_radius": radius,
                            "oracle_response_norm": float(
                                np.linalg.norm(oracle_resp)
                            ),
                            "teacher_response_norm": float(
                                np.linalg.norm(teacher_resp)
                            ),
                            "student_response_norm": float(
                                np.linalg.norm(student_resp)
                            ),
                            "teacher_oracle_cosine": _cos(
                                teacher_resp, oracle_resp
                            ),
                            "student_oracle_cosine": _cos(
                                student_resp, oracle_resp
                            ),
                            "student_teacher_cosine": _cos(
                                student_resp, teacher_resp
                            ),
                            "teacher_oracle_gain": _gain(
                                teacher_resp, oracle_resp
                            ),
                            "student_oracle_gain": _gain(
                                student_resp, oracle_resp
                            ),
                            "teacher_oracle_relerr": _relerr(
                                teacher_resp, oracle_resp
                            ),
                            "student_oracle_relerr": _relerr(
                                student_resp, oracle_resp
                            ),
                            "student_teacher_relerr": _relerr(
                                student_resp, teacher_resp
                            ),
                        })

            elapsed = time.time() - t0
            eta = elapsed / (ai + 1) * (len(anchors) - ai - 1)
            print(
                f"anchor {ai+1:3d}/{len(anchors)} row={row} "
                f"replay_err={replay_error:.4f} good={int(replay_good)} "
                f"ETA={eta/60:.1f}m",
                flush=True,
            )

    finally:
        env.close()

    response_csv = outdir / "response_cells.csv"
    anchor_csv = outdir / "anchor_metrics.csv"
    _write_csv(response_csv, response_rows)
    _write_csv(anchor_csv, anchor_rows)

    good_response = [r for r in response_rows if r["replay_good"]]
    good_anchor = [r for r in anchor_rows if r["replay_good"]]

    by_offset = {}
    for off in range(int(args.horizon)):
        rr = [
            r for r in good_response
            if r["horizon_from_perturb"] == off
        ]
        by_offset[str(off)] = _aggregate_response(rr)

    by_position = {}
    for p in range(int(args.horizon)):
        rr = [
            r for r in good_response
            if r["position"] == p
        ]
        by_position[str(p)] = _aggregate_response(rr)

    summary = {
        "question": (
            "Does the frozen official LeWM teacher provide accurate relative "
            "multi-horizon action responses for MH-ALD, compared with real "
            "simulator counterfactuals encoded in the same frozen latent frame?"
        ),
        "config": vars(args),
        "teacher_policy": args.teacher_policy,
        "student_policy": args.student_policy,
        "selected_rows": anchors.tolist(),
        "latent_frame_max_abs_teacher_vs_student": frame_max_abs,
        "all": {
            "response": _aggregate_response(response_rows),
            "anchor": _aggregate_anchor(anchor_rows),
        },
        "replay_good": {
            "response": _aggregate_response(good_response),
            "anchor": _aggregate_anchor(good_anchor),
        },
        "replay_good_by_horizon_from_perturb": by_offset,
        "replay_good_by_position": by_position,
        "protocol_notes": [
            "Teacher is the frozen official LeWM epoch-10 used by ALD/MH-ALD.",
            "Student is the current formal MH-ALD model.",
            "Teacher and student visual latent frames must numerically match.",
            "Block-wise symmetric probes mirror MH-ALD radius/causal structure.",
            "Real counterfactual trajectories share one reset seed/variation per anchor.",
            "Public PushT state omits block momentum; replay_good filters anchors where the center expert rollout is sufficiently reproducible.",
            "Model ranking uses same-variation rendered goal to isolate dynamics/representation quality from cross-appearance mismatch.",
        ],
        "elapsed_seconds": float(time.time() - t0),
    }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2))

    rg = summary["replay_good"]
    print("\n===== TEACHER RESPONSE ORACLE SUMMARY (REPLAY-GOOD) =====")
    print(json.dumps(_jsonable(rg), indent=2))
    print(
        f"latent_frame_max_abs={frame_max_abs:.3e} "
        f"replay_good={len(good_anchor)}/{len(anchor_rows)}"
    )
    print(f"Saved: {response_csv}")
    print(f"Saved: {anchor_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
