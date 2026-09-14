#!/usr/bin/env python3
"""Measure the MH-ALD train-context vs planner-context response gap on PushT.

Formal MH-ALD calibrates multi-horizon action responses after THREE real visual
latents. Official LeWM PushT planning starts each candidate rollout from ONE
observed visual latent because config/eval/pusht.yaml has world.history_size=1.
The predictor itself still has a maximum history window of 3: its inference
context therefore grows as

    [z_t^real]
    [z_t^real, z_{t+1}^pred]
    [z_t^real, z_{t+1}^pred, z_{t+2}^pred]
    [z_{t+1}^pred, z_{t+2}^pred, z_{t+3}^pred], ...

This diagnostic asks whether MH-ALD response fidelity survives that context
shift. It holds fixed, per anchor:
  * anchor row and reset variation,
  * demonstrated center action sequence,
  * symmetric block-wise probes,
  * real simulator counterfactuals,
  * frozen encoder oracle and goal latent.

Only model rollout context changes:

  train_context   : 3 real latents, then autoregressive rollout
  planner_context : 1 real latent, predictor warm-up to max history 3

The planner-context helper is numerically checked against the checkpoint's
actual JEPA.rollout() on the first anchor. Simulator counterfactuals are
DIAGNOSIS ONLY and never modify training, CEM, or benchmark evaluation.
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
from eval_mh_ald_teacher_response_oracle import (
    _aggregate_response,
    _candidate_pair_indices,
    _cos,
    _gain,
    _make_blockwise_candidates,
    _model_rollout,
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
    p.add_argument("--train-observation-history", type=int, default=3)
    p.add_argument("--planner-observation-history", type=int, default=1)
    p.add_argument("--predictor-max-history", type=int, default=3)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--directions-per-position", type=int, default=4)
    p.add_argument("--perturb-radius", type=float, default=0.1565)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env-seed", type=int, default=12000)
    p.add_argument("--model-batch-size", type=int, default=128)
    p.add_argument("--replay-good-threshold", type=float, default=0.10)
    p.add_argument("--semantic-check-atol", type=float, default=2e-5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def _planner_warmup_rollout(
    model,
    current_emb,
    normalized_plans,
    predictor_max_history,
    device,
):
    """Exact latent recursion of JEPA.rollout starting from one observed frame.

    current_emb: [D]
    normalized_plans: [N,H,A]
    returns: [N,H,D] predicted future latents
    """
    actions = torch.as_tensor(
        normalized_plans, device=device, dtype=torch.float32
    )
    n, horizon = actions.shape[:2]
    current = torch.as_tensor(
        current_emb, device=device, dtype=torch.float32
    )[None, None].expand(n, 1, -1).clone()
    predicted = []
    max_history = int(predictor_max_history)

    for step in range(int(horizon)):
        # At planner warm-up step k, only action blocks 0..k exist in the
        # autoregressive history. JEPA.rollout recomputes action embeddings and
        # takes the last <= max_history entries exactly this way.
        action_hist = actions[:, : step + 1]
        act_emb = model.action_encoder(action_hist)
        emb_trunc = current[:, -max_history:]
        act_trunc = act_emb[:, -max_history:]
        next_emb = model.predict(emb_trunc, act_trunc)[:, -1:]
        predicted.append(next_emb)
        current = torch.cat([current, next_emb], dim=1)

    return torch.cat(predicted, dim=1).detach()


@torch.inference_mode()
def _single_frame_official_rollout(
    model,
    raw_image,
    transform,
    normalized_plan,
    predictor_max_history,
    device,
):
    """Call checkpoint JEPA.rollout using the official one-frame input shape."""
    px = transform(raw_image)
    if not torch.is_tensor(px):
        px = torch.as_tensor(px)
    info = {
        "pixels": px.to(device=device, dtype=torch.float32)[None, None, None]
    }
    acts = torch.as_tensor(
        normalized_plan, device=device, dtype=torch.float32
    )[None, None]
    out = model.rollout(
        info, acts, history_size=int(predictor_max_history)
    )
    pred = out["predicted_emb"][0, 0]
    return pred[1:].detach()


def _assert_planner_semantics(
    model,
    raw_image,
    transform,
    current_emb,
    normalized_plan,
    predictor_max_history,
    device,
    atol,
):
    helper = _planner_warmup_rollout(
        model,
        current_emb,
        normalized_plan[None],
        predictor_max_history,
        device,
    )[0]
    official = _single_frame_official_rollout(
        model,
        raw_image,
        transform,
        normalized_plan,
        predictor_max_history,
        device,
    )
    if helper.shape != official.shape:
        raise RuntimeError(
            f"Planner semantic check shape mismatch: {helper.shape} vs {official.shape}"
        )
    max_abs = float(torch.max(torch.abs(helper - official)).detach().cpu())
    if max_abs > float(atol):
        raise RuntimeError(
            "Planner warm-up helper does not match JEPA.rollout: "
            f"max_abs={max_abs:.3e} > atol={float(atol):.3e}"
        )
    return max_abs


def _aggregate_context_anchor(rows, prefix):
    return {
        "anchors": len(rows),
        "replay_good_fraction": (
            float(np.mean([r["replay_good"] for r in rows])) if rows else None
        ),
        "teacher_center_endpoint_mse": _summary(
            r[f"{prefix}_teacher_center_endpoint_mse"] for r in rows
        ),
        "student_center_endpoint_mse": _summary(
            r[f"{prefix}_student_center_endpoint_mse"] for r in rows
        ),
        "rho_teacher_exact_encoder": _summary(
            r[f"rho_{prefix}_teacher_exact_encoder"] for r in rows
        ),
        "rho_student_exact_encoder": _summary(
            r[f"rho_{prefix}_student_exact_encoder"] for r in rows
        ),
        "rho_exact_encoder_physical": _summary(
            r["rho_exact_encoder_physical"] for r in rows
        ),
        "rho_teacher_physical": _summary(
            r[f"rho_{prefix}_teacher_physical"] for r in rows
        ),
        "rho_student_physical": _summary(
            r[f"rho_{prefix}_student_physical"] for r in rows
        ),
        "replay_endpoint_factor_error": _summary(
            r["replay_endpoint_factor_error"] for r in rows
        ),
    }


def _response_view(rows, prefix):
    return [{
        "teacher_oracle_cosine": r[f"{prefix}_teacher_oracle_cosine"],
        "student_oracle_cosine": r[f"{prefix}_student_oracle_cosine"],
        "student_teacher_cosine": r[f"{prefix}_student_teacher_cosine"],
        "teacher_oracle_gain": r[f"{prefix}_teacher_oracle_gain"],
        "student_oracle_gain": r[f"{prefix}_student_oracle_gain"],
        "teacher_oracle_relerr": r[f"{prefix}_teacher_oracle_relerr"],
        "student_oracle_relerr": r[f"{prefix}_student_oracle_relerr"],
        "student_teacher_relerr": r[f"{prefix}_student_teacher_relerr"],
    } for r in rows]


def _aggregate_gap(rows):
    return {
        "cells": len(rows),
        "teacher_cosine_planner_minus_train": _summary(
            r["teacher_cosine_planner_minus_train"] for r in rows
        ),
        "student_cosine_planner_minus_train": _summary(
            r["student_cosine_planner_minus_train"] for r in rows
        ),
        "teacher_relerr_planner_minus_train": _summary(
            r["teacher_relerr_planner_minus_train"] for r in rows
        ),
        "student_relerr_planner_minus_train": _summary(
            r["student_relerr_planner_minus_train"] for r in rows
        ),
        "train_student_minus_teacher_cosine": _summary(
            r["train_student_minus_teacher_cosine"] for r in rows
        ),
        "planner_student_minus_teacher_cosine": _summary(
            r["planner_student_minus_teacher_cosine"] for r in rows
        ),
    }


def _response_summary(rows):
    return {
        "train_context": _aggregate_response(_response_view(rows, "train")),
        "planner_context": _aggregate_response(_response_view(rows, "planner")),
        "planner_minus_train": _aggregate_gap(rows),
    }


def _group_summary(response_rows, anchor_rows):
    return {
        "train_context": {
            "response": _aggregate_response(
                _response_view(response_rows, "train")
            ),
            "anchor": _aggregate_context_anchor(anchor_rows, "train"),
        },
        "planner_context": {
            "response": _aggregate_response(
                _response_view(response_rows, "planner")
            ),
            "anchor": _aggregate_context_anchor(anchor_rows, "planner"),
        },
        "planner_minus_train": _aggregate_gap(response_rows),
    }


def main():
    args = parse_args()
    if int(args.train_observation_history) != 3:
        raise ValueError("Formal MH-ALD train observation history is 3.")
    if int(args.planner_observation_history) != 1:
        raise ValueError("Formal PushT planner observation history is 1.")
    if int(args.predictor_max_history) != 3:
        raise ValueError("Formal LeWM predictor max history is 3.")
    if int(args.horizon) != 5 or int(args.action_block) != 5:
        raise ValueError("Formal PushT expects horizon=5 and action_block=5.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    if int(cfg.world.history_size) != int(args.planner_observation_history):
        raise RuntimeError(
            "Eval config does not match requested planner observation history: "
            f"cfg={cfg.world.history_size}, requested={args.planner_observation_history}"
        )

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

    train_history_raw = (
        (int(args.train_observation_history) - 1) * int(args.action_block)
    )
    future_raw = int(args.horizon) * int(args.action_block)
    # Keep exactly the established history=3 oracle eligibility. With identical
    # seed/count this selects identical anchor rows and probe RNGs.
    ep_col, anchors = _select_anchor_rows(
        dataset,
        args.num_anchors,
        args.seed,
        train_history_raw,
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
    semantic_check_max_abs = {"teacher": 0.0, "student": 0.0}
    t0 = time.time()

    try:
        for ai, row in enumerate(anchors):
            seed = int(args.env_seed) + int(ai)
            goal_row = int(row) + future_raw
            goal_state = state[goal_row].copy()
            current_state = state[int(row)].copy()
            hist_rows = [
                int(row) - train_history_raw + k * int(args.action_block)
                for k in range(int(args.train_observation_history))
            ]
            hist_images = [
                _render_state(env, state[r], goal_state, seed)
                for r in hist_rows
            ]
            current_image = hist_images[-1]
            goal_image = _render_state(env, goal_state, goal_state, seed)
            hist_s = _encode(
                student, transform, hist_images, device, args.model_batch_size
            )
            hist_t = _encode(
                teacher, transform, hist_images, device, args.model_batch_size
            )
            frame_diff = float(
                torch.max(torch.abs(hist_s - hist_t)).detach().cpu()
            )
            frame_max_abs = max(frame_max_abs, frame_diff)
            if frame_diff > 2e-5:
                raise RuntimeError(
                    f"Teacher/student visual frames differ: {frame_diff:.3e}"
                )
            goal_z = _encode(
                student, transform, [goal_image], device, args.model_batch_size
            )[0]

            history_actions_raw = action[
                int(row) - train_history_raw : int(row)
            ].copy()
            future_actions_raw = action[
                int(row) : int(row) + future_raw
            ].copy()
            center_future = np.nan_to_num(
                future_actions_raw, nan=0.0, posinf=1.0, neginf=-1.0
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
            future_packed = np.stack([
                _pack_and_normalize(c, scaler, args.action_block)
                for c in candidates_raw
            ])
            hist_packed = _pack_and_normalize(
                history_actions_raw, scaler, args.action_block
            )
            hist_repeat = np.broadcast_to(
                hist_packed[None], (nc, *hist_packed.shape)
            ).copy()
            train_actions = np.concatenate([hist_repeat, future_packed], axis=1)

            teacher_train = _model_rollout(
                teacher,
                hist_t.detach().cpu().numpy(),
                train_actions,
                history_size=3,
                horizon=args.horizon,
                device=device,
            )
            student_train = _model_rollout(
                student,
                hist_s.detach().cpu().numpy(),
                train_actions,
                history_size=3,
                horizon=args.horizon,
                device=device,
            )
            teacher_planner = _planner_warmup_rollout(
                teacher,
                hist_t[-1].detach().cpu().numpy(),
                future_packed,
                args.predictor_max_history,
                device,
            )
            student_planner = _planner_warmup_rollout(
                student,
                hist_s[-1].detach().cpu().numpy(),
                future_packed,
                args.predictor_max_history,
                device,
            )

            if ai == 0:
                for name, model, emb in (
                    ("teacher", teacher, hist_t[-1]),
                    ("student", student, hist_s[-1]),
                ):
                    diff = _assert_planner_semantics(
                        model,
                        current_image,
                        transform,
                        emb,
                        future_packed[0],
                        args.predictor_max_history,
                        device,
                        args.semantic_check_atol,
                    )
                    semantic_check_max_abs[name] = diff
                    print(
                        f"planner semantic check [{name}] max_abs={diff:.3e}",
                        flush=True,
                    )

            oracle_images = []
            final_states = []
            candidate_contacts = []
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
                candidate_contacts.append(bool(contact))
                oracle_images.extend(images_h)
                if ci == 0:
                    center_states = states_h
                    center_contact = bool(contact)
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
                    _state_factor(center_states[-1]) - _state_factor(goal_state)
                )
            )
            replay_good = replay_error <= float(args.replay_good_threshold)
            exact_enc_cost = (
                torch.sum((oracle_z[:, -1] - goal_z[None]) ** 2, dim=-1)
                .detach().cpu().numpy().astype(np.float64)
            )
            physical_cost = _physical_cost(final_states, goal_state)[0]

            anchor = {
                "anchor_index": int(ai),
                "dataset_row": int(row),
                "episode_idx": int(episodes[row]),
                "step_idx": int(steps[row]),
                "num_candidates": int(nc),
                "center_contact": bool(center_contact),
                "replay_endpoint_factor_error": replay_error,
                "replay_good": bool(replay_good),
                "rho_exact_encoder_physical": _spearman(
                    exact_enc_cost, physical_cost
                ),
            }
            for prefix, teacher_roll, student_roll in (
                ("train", teacher_train, student_train),
                ("planner", teacher_planner, student_planner),
            ):
                teacher_cost = (
                    torch.sum((teacher_roll[:, -1] - goal_z[None]) ** 2, dim=-1)
                    .detach().cpu().numpy().astype(np.float64)
                )
                student_cost = (
                    torch.sum((student_roll[:, -1] - goal_z[None]) ** 2, dim=-1)
                    .detach().cpu().numpy().astype(np.float64)
                )
                anchor.update({
                    f"{prefix}_teacher_center_endpoint_mse": float(
                        torch.mean(
                            (teacher_roll[0, -1] - oracle_z[0, -1]) ** 2
                        ).detach().cpu()
                    ),
                    f"{prefix}_student_center_endpoint_mse": float(
                        torch.mean(
                            (student_roll[0, -1] - oracle_z[0, -1]) ** 2
                        ).detach().cpu()
                    ),
                    f"rho_{prefix}_teacher_exact_encoder": _spearman(
                        teacher_cost, exact_enc_cost
                    ),
                    f"rho_{prefix}_student_exact_encoder": _spearman(
                        student_cost, exact_enc_cost
                    ),
                    f"rho_{prefix}_teacher_physical": _spearman(
                        teacher_cost, physical_cost
                    ),
                    f"rho_{prefix}_student_physical": _spearman(
                        student_cost, physical_cost
                    ),
                })
            anchor.update({
                "delta_student_rho_encoder_planner_minus_train": (
                    anchor["rho_planner_student_exact_encoder"]
                    - anchor["rho_train_student_exact_encoder"]
                ),
                "delta_teacher_rho_encoder_planner_minus_train": (
                    anchor["rho_planner_teacher_exact_encoder"]
                    - anchor["rho_train_teacher_exact_encoder"]
                ),
            })
            anchor_rows.append(anchor)

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
                        out = {
                            "anchor_index": int(ai),
                            "dataset_row": int(row),
                            "replay_good": bool(replay_good),
                            "center_contact": bool(center_contact),
                            "pair_contact": bool(
                                candidate_contacts[ip] or candidate_contacts[im]
                            ),
                            "position": int(p),
                            "direction": int(d),
                            "horizon_index": int(h),
                            "horizon_from_perturb": int(h - p),
                            "effective_radius": radius,
                            "oracle_response_norm": float(np.linalg.norm(oracle_resp)),
                        }
                        for prefix, tr, sr in (
                            ("train", teacher_train, student_train),
                            ("planner", teacher_planner, student_planner),
                        ):
                            t_resp = (
                                tr[ip, h] - tr[im, h]
                            ).detach().cpu().numpy() / (2.0 * radius)
                            s_resp = (
                                sr[ip, h] - sr[im, h]
                            ).detach().cpu().numpy() / (2.0 * radius)
                            out.update({
                                f"{prefix}_teacher_response_norm": float(np.linalg.norm(t_resp)),
                                f"{prefix}_student_response_norm": float(np.linalg.norm(s_resp)),
                                f"{prefix}_teacher_oracle_cosine": _cos(t_resp, oracle_resp),
                                f"{prefix}_student_oracle_cosine": _cos(s_resp, oracle_resp),
                                f"{prefix}_student_teacher_cosine": _cos(s_resp, t_resp),
                                f"{prefix}_teacher_oracle_gain": _gain(t_resp, oracle_resp),
                                f"{prefix}_student_oracle_gain": _gain(s_resp, oracle_resp),
                                f"{prefix}_teacher_oracle_relerr": _relerr(t_resp, oracle_resp),
                                f"{prefix}_student_oracle_relerr": _relerr(s_resp, oracle_resp),
                                f"{prefix}_student_teacher_relerr": _relerr(s_resp, t_resp),
                            })
                        out.update({
                            "teacher_cosine_planner_minus_train": (
                                out["planner_teacher_oracle_cosine"]
                                - out["train_teacher_oracle_cosine"]
                            ),
                            "student_cosine_planner_minus_train": (
                                out["planner_student_oracle_cosine"]
                                - out["train_student_oracle_cosine"]
                            ),
                            "teacher_relerr_planner_minus_train": (
                                out["planner_teacher_oracle_relerr"]
                                - out["train_teacher_oracle_relerr"]
                            ),
                            "student_relerr_planner_minus_train": (
                                out["planner_student_oracle_relerr"]
                                - out["train_student_oracle_relerr"]
                            ),
                            "train_student_minus_teacher_cosine": (
                                out["train_student_oracle_cosine"]
                                - out["train_teacher_oracle_cosine"]
                            ),
                            "planner_student_minus_teacher_cosine": (
                                out["planner_student_oracle_cosine"]
                                - out["planner_teacher_oracle_cosine"]
                            ),
                        })
                        response_rows.append(out)

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

    response_csv = outdir / "context_response_cells.csv"
    anchor_csv = outdir / "context_anchor_metrics.csv"
    _write_csv(response_csv, response_rows)
    _write_csv(anchor_csv, anchor_rows)

    good_r = [r for r in response_rows if r["replay_good"]]
    good_a = [r for r in anchor_rows if r["replay_good"]]
    contact_r = [r for r in good_r if r["pair_contact"]]
    contact_a = [r for r in good_a if r["center_contact"]]
    no_contact_r = [r for r in good_r if not r["pair_contact"]]
    no_contact_a = [r for r in good_a if not r["center_contact"]]

    by_offset = {
        str(off): _response_summary([
            r for r in good_r if r["horizon_from_perturb"] == off
        ])
        for off in range(int(args.horizon))
    }
    by_position = {
        str(pos): _response_summary([
            r for r in good_r if r["position"] == pos
        ])
        for pos in range(int(args.horizon))
    }

    summary = {
        "question": (
            "Does MH-ALD action-response fidelity degrade when candidate rollouts "
            "start from one observed frame and warm up self-generated predictor "
            "history, as in official LeWM planning, rather than three real latents?"
        ),
        "config": vars(args),
        "teacher_policy": args.teacher_policy,
        "student_policy": args.student_policy,
        "selected_rows": anchors.tolist(),
        "same_anchor_probe_protocol_as_history3_oracle": True,
        "latent_frame_max_abs_teacher_vs_student": frame_max_abs,
        "planner_semantic_check_max_abs": semantic_check_max_abs,
        "all": _group_summary(response_rows, anchor_rows),
        "replay_good": _group_summary(good_r, good_a),
        "replay_good_contact": _group_summary(contact_r, contact_a),
        "replay_good_no_contact": _group_summary(no_contact_r, no_contact_a),
        "replay_good_by_horizon_from_perturb": by_offset,
        "replay_good_by_position": by_position,
        "protocol_notes": [
            "A/B uses identical anchors, probes, real counterfactuals, goal, and frozen visual frame.",
            "Train context starts from three real observation latents.",
            "Planner context starts from one real latent; predictor history grows 1->2->3 and then rolls with max history 3.",
            "The planner-context helper is numerically checked against checkpoint JEPA.rollout().",
            "Anchors retain established history=3 eligibility so identical seed/count selects identical rows/probes.",
            "Same-variation rendered current/goal images are used in both A/B arms to isolate context mismatch from appearance mismatch.",
            "Simulator counterfactuals are diagnosis-only and never used for training or planning.",
        ],
        "elapsed_seconds": float(time.time() - t0),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    print("\n===== CONTEXT GAP ORACLE: REPLAY-GOOD =====")
    print(json.dumps(_jsonable(summary["replay_good"]), indent=2))
    print("\n===== CONTEXT GAP ORACLE: REPLAY-GOOD + CONTACT =====")
    print(json.dumps(_jsonable(summary["replay_good_contact"]), indent=2))
    print(
        "semantic_check_max_abs="
        + json.dumps(semantic_check_max_abs, sort_keys=True)
    )
    print(f"Saved: {response_csv}")
    print(f"Saved: {anchor_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
