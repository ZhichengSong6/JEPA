#!/usr/bin/env python3
"""Measure the MH-ALD train-context vs planner-context response gap on PushT.

Scientific question
-------------------
Formal MH-ALD calibrates multi-horizon action responses after a 3-frame REAL
latent history.  Official LeWM PushT planning uses world.history_size=1, so a
candidate rollout starts from one real observation and its later predictor
contexts are self-generated.  Does the response fidelity gained by MH-ALD
survive that inference context shift?

This diagnostic holds EVERYTHING else fixed for an anchor:
  * the same anchor row and reset variation,
  * the same demonstrated center action sequence,
  * the same symmetric block-wise probes,
  * the same real counterfactual simulator rollouts,
  * the same frozen encoder oracle and goal latent.

It scores teacher and MH student under two rollout contexts:

  train_context   : [z_{t-2}^real, z_{t-1}^real, z_t^real], history_size=3
  planner_context : [z_t^real], history_size=1, then self-generated history

The planner-context helper is numerically checked against JEPA.rollout() itself
on the first anchor before any scientific result is accepted.

The oracle uses simulator counterfactuals for DIAGNOSIS ONLY.  It does not train
or modify a model, planner, CEM cost, or benchmark evaluation.
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
    _aggregate_anchor,
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
    p.add_argument("--train-history-size", type=int, default=3)
    p.add_argument("--planner-history-size", type=int, default=1)
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


def _single_frame_official_rollout(
    model,
    raw_image: np.ndarray,
    transform,
    normalized_plan: np.ndarray,
    device: torch.device,
):
    """Call the checkpoint's actual JEPA.rollout with one observed frame."""
    px = transform(raw_image)
    if not torch.is_tensor(px):
        px = torch.as_tensor(px)
    # JEPA.rollout expects pixels [B,S,T,C,H,W] and actions [B,S,T,A].
    info = {
        "pixels": px.to(device=device, dtype=torch.float32)[None, None, None]
    }
    acts = torch.as_tensor(
        normalized_plan,
        device=device,
        dtype=torch.float32,
    )[None, None]
    out = model.rollout(info, acts, history_size=3)
    pred = out["predicted_emb"][0, 0]
    # One observed latent followed by exactly H predicted latents.
    return pred[1:].detach()


def _assert_planner_semantics(
    model,
    raw_image,
    transform,
    current_emb,
    normalized_plan,
    device,
    atol,
):
    """Prove history_size=1 helper matches the checkpoint's inference rollout."""
    helper = _model_rollout(
        model,
        current_emb[None],
        normalized_plan[None],
        history_size=1,
        horizon=int(normalized_plan.shape[0]),
        device=device,
    )[0]
    official = _single_frame_official_rollout(
        model,
        raw_image,
        transform,
        normalized_plan,
        device,
    )
    if helper.shape != official.shape:
        raise RuntimeError(
            f"Planner semantic check shape mismatch: helper={helper.shape}, "
            f"official={official.shape}"
        )
    max_abs = float(torch.max(torch.abs(helper - official)).detach().cpu())
    if max_abs > float(atol):
        raise RuntimeError(
            "history_size=1 diagnostic rollout does not match JEPA.rollout: "
            f"max_abs={max_abs:.3e} > atol={float(atol):.3e}"
        )
    return max_abs


def _context_metrics(prefix, roll, oracle_z, goal_z, physical_cost):
    pred_cost = (
        torch.sum((roll[:, -1] - goal_z[None]) ** 2, dim=-1)
        .detach().cpu().numpy().astype(np.float64)
    )
    return {
        f"{prefix}_center_endpoint_mse": float(
            torch.mean((roll[0, -1] - oracle_z[0, -1]) ** 2).detach().cpu()
        ),
        f"rho_{prefix}_exact_encoder": _spearman(pred_cost, (
            torch.sum((oracle_z[:, -1] - goal_z[None]) ** 2, dim=-1)
            .detach().cpu().numpy().astype(np.float64)
        )),
        f"rho_{prefix}_physical": _spearman(pred_cost, physical_cost),
    }


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
    out = []
    for r in rows:
        out.append({
            "teacher_oracle_cosine": r[f"{prefix}_teacher_oracle_cosine"],
            "student_oracle_cosine": r[f"{prefix}_student_oracle_cosine"],
            "student_teacher_cosine": r[f"{prefix}_student_teacher_cosine"],
            "teacher_oracle_gain": r[f"{prefix}_teacher_oracle_gain"],
            "student_oracle_gain": r[f"{prefix}_student_oracle_gain"],
            "teacher_oracle_relerr": r[f"{prefix}_teacher_oracle_relerr"],
            "student_oracle_relerr": r[f"{prefix}_student_oracle_relerr"],
            "student_teacher_relerr": r[f"{prefix}_student_teacher_relerr"],
        })
    return out


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
        "student_planner_advantage_over_teacher_cosine": _summary(
            r["planner_student_minus_teacher_cosine"] for r in rows
        ),
        "student_train_advantage_over_teacher_cosine": _summary(
            r["train_student_minus_teacher_cosine"] for r in rows
        ),
    }


def main():
    args = parse_args()
    if int(args.train_history_size) != 3:
        raise ValueError("Formal MH-ALD train history must be 3 for this A/B test.")
    if int(args.planner_history_size) != 1:
        raise ValueError("Formal PushT planner history must be 1 for this A/B test.")
    if int(args.horizon) != 5 or int(args.action_block) != 5:
        raise ValueError("Formal PushT diagnostic expects horizon=5, action_block=5.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    if int(cfg.world.history_size) != 1:
        raise RuntimeError(
            f"Config world.history_size={cfg.world.history_size}, expected formal planner value 1."
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

    # Deliberately retain the SAME eligibility as the established history=3
    # teacher-response oracle so seed/num_anchors choose the same rows/probes.
    train_history_raw = (
        (int(args.train_history_size) - 1) * int(args.action_block)
    )
    future_raw = int(args.horizon) * int(args.action_block)
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
    response_rows: list[dict] = []
    anchor_rows: list[dict] = []
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
                for k in range(int(args.train_history_size))
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
            this_frame_diff = float(
                torch.max(torch.abs(hist_s - hist_t)).detach().cpu()
            )
            frame_max_abs = max(frame_max_abs, this_frame_diff)
            if this_frame_diff > 2e-5:
                raise RuntimeError(
                    "Teacher/student visual frames differ: "
                    f"max_abs={this_frame_diff:.3e}"
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
            future_packed = np.stack([
                _pack_and_normalize(c, scaler, args.action_block)
                for c in candidates_raw
            ])

            # Train-context action sequence includes two preceding action blocks.
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

            # Planner-context starts from current real latent only. The exact
            # same five candidate action blocks are then rolled autoregressively.
            teacher_planner = _model_rollout(
                teacher,
                hist_t[-1:].detach().cpu().numpy(),
                future_packed,
                history_size=1,
                horizon=args.horizon,
                device=device,
            )
            student_planner = _model_rollout(
                student,
                hist_s[-1:].detach().cpu().numpy(),
                future_packed,
                history_size=1,
                horizon=args.horizon,
                device=device,
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
                        device,
                        args.semantic_check_atol,
                    )
                    semantic_check_max_abs[name] = max(
                        semantic_check_max_abs[name], diff
                    )
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
                tcost = (
                    torch.sum((teacher_roll[:, -1] - goal_z[None]) ** 2, dim=-1)
                    .detach().cpu().numpy().astype(np.float64)
                )
                scost = (
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
                        tcost, exact_enc_cost
                    ),
                    f"rho_{prefix}_student_exact_encoder": _spearman(
                        scost, exact_enc_cost
                    ),
                    f"rho_{prefix}_teacher_physical": _spearman(
                        tcost, physical_cost
                    ),
                    f"rho_{prefix}_student_physical": _spearman(
                        scost, physical_cost
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
                        row_out = {
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
                            teacher_resp = (
                                tr[ip, h] - tr[im, h]
                            ).detach().cpu().numpy() / (2.0 * radius)
                            student_resp = (
                                sr[ip, h] - sr[im, h]
                            ).detach().cpu().numpy() / (2.0 * radius)
                            row_out.update({
                                f"{prefix}_teacher_response_norm": float(
                                    np.linalg.norm(teacher_resp)
                                ),
                                f"{prefix}_student_response_norm": float(
                                    np.linalg.norm(student_resp)
                                ),
                                f"{prefix}_teacher_oracle_cosine": _cos(
                                    teacher_resp, oracle_resp
                                ),
                                f"{prefix}_student_oracle_cosine": _cos(
                                    student_resp, oracle_resp
                                ),
                                f"{prefix}_student_teacher_cosine": _cos(
                                    student_resp, teacher_resp
                                ),
                                f"{prefix}_teacher_oracle_gain": _gain(
                                    teacher_resp, oracle_resp
                                ),
                                f"{prefix}_student_oracle_gain": _gain(
                                    student_resp, oracle_resp
                                ),
                                f"{prefix}_teacher_oracle_relerr": _relerr(
                                    teacher_resp, oracle_resp
                                ),
                                f"{prefix}_student_oracle_relerr": _relerr(
                                    student_resp, oracle_resp
                                ),
                                f"{prefix}_student_teacher_relerr": _relerr(
                                    student_resp, teacher_resp
                                ),
                            })
                        row_out.update({
                            "teacher_cosine_planner_minus_train": (
                                row_out["planner_teacher_oracle_cosine"]
                                - row_out["train_teacher_oracle_cosine"]
                            ),
                            "student_cosine_planner_minus_train": (
                                row_out["planner_student_oracle_cosine"]
                                - row_out["train_student_oracle_cosine"]
                            ),
                            "teacher_relerr_planner_minus_train": (
                                row_out["planner_teacher_oracle_relerr"]
                                - row_out["train_teacher_oracle_relerr"]
                            ),
                            "student_relerr_planner_minus_train": (
                                row_out["planner_student_oracle_relerr"]
                                - row_out["train_student_oracle_relerr"]
                            ),
                            "planner_student_minus_teacher_cosine": (
                                row_out["planner_student_oracle_cosine"]
                                - row_out["planner_teacher_oracle_cosine"]
                            ),
                            "train_student_minus_teacher_cosine": (
                                row_out["train_student_oracle_cosine"]
                                - row_out["train_teacher_oracle_cosine"]
                            ),
                        })
                        response_rows.append(row_out)

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

    _write_csv(outdir / "context_response_cells.csv", response_rows)
    _write_csv(outdir / "context_anchor_metrics.csv", anchor_rows)

    good_r = [r for r in response_rows if r["replay_good"]]
    good_a = [r for r in anchor_rows if r["replay_good"]]
    good_contact_r = [r for r in good_r if r["pair_contact"]]
    good_contact_a = [r for r in good_a if r["center_contact"]]

    def pack(rows_r, rows_a):
        return {
            "train_context": {
                "response": _aggregate_response(_response_view(rows_r, "train")),
                "anchor": _aggregate_context_anchor(rows_a, "train"),
            },
            "planner_context": {
                "response": _aggregate_response(_response_view(rows_r, "planner")),
                "anchor": _aggregate_context_anchor(rows_a, "planner"),
            },
            "planner_minus_train": _aggregate_gap(rows_r),
        }

    by_offset = {}
    for off in range(int(args.horizon)):
        rr = [r for r in good_r if r["horizon_from_perturb"] == off]
        by_offset[str(off)] = pack(rr, good_a)

    by_position = {}
    for p in range(int(args.horizon)):
        rr = [r for r in good_r if r["position"] == p]
        by_position[str(p)] = pack(rr, good_a)

    summary = {
        "question": (
            "Does MH-ALD action-response fidelity degrade when the model is "
            "rolled out with official planner context (one observed frame, then "
            "self-generated latent history) instead of its 3-real-frame training context?"
        ),
        "config": vars(args),
        "teacher_policy": args.teacher_policy,
        "student_policy": args.student_policy,
        "selected_rows": anchors.tolist(),
        "same_anchor_probe_protocol_as_history3_oracle": True,
        "latent_frame_max_abs_teacher_vs_student": frame_max_abs,
        "planner_semantic_check_max_abs": semantic_check_max_abs,
        "all": pack(response_rows, anchor_rows),
        "replay_good": pack(good_r, good_a),
        "replay_good_contact": pack(good_contact_r, good_contact_a),
        "replay_good_by_horizon_from_perturb": by_offset,
        "replay_good_by_position": by_position,
        "protocol_notes": [
            "The train/planner A/B uses identical anchors, candidate actions, real counterfactuals, goal, and latent frame.",
            "Only model rollout context changes: history=3 real latents versus history=1 real latent followed by self-generated predictions.",
            "Anchors retain the established history=3 oracle eligibility so identical seed/count select identical formal rows and probes.",
            "Planner history=1 helper is numerically checked against checkpoint JEPA.rollout on the first anchor.",
            "Rendered same-variation current/goal images are used in both contexts to isolate latent-context mismatch rather than appearance mismatch.",
            "Simulator counterfactuals are diagnostic oracle data only and never used for training or planning.",
        ],
        "elapsed_seconds": float(time.time() - t0),
    }
    (outdir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n"
    )

    rg = summary["replay_good"]
    rgc = summary["replay_good_contact"]
    print("\n===== CONTEXT GAP ORACLE: REPLAY-GOOD =====")
    print(json.dumps(_jsonable(rg), indent=2))
    print("\n===== CONTEXT GAP ORACLE: REPLAY-GOOD + CONTACT =====")
    print(json.dumps(_jsonable(rgc), indent=2))
    print(
        "semantic_check_max_abs="
        + json.dumps(semantic_check_max_abs, sort_keys=True)
    )
    print(f"Saved: {outdir / 'context_response_cells.csv'}")
    print(f"Saved: {outdir / 'context_anchor_metrics.csv'}")
    print(f"Saved: {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
