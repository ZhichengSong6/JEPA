#!/usr/bin/env python3
"""Problem 2: does goal-parallel latent error dominate planner ranking failures?

This is DIAGNOSTIC ONLY.  It does not train a model or modify CEM.

For the same PushT anchor/probe protocol as the MH response oracle, models are
rolled out with the OFFICIAL planner semantics: one observed visual latent,
then self-generated predictor history (max predictor history = 3).

For every candidate endpoint define

    r = z_real - z_goal          exact frozen-encoder goal residual
    e = z_pred - z_real          world-model endpoint error

so the planner cost error is exactly

    ||r + e||^2 - ||r||^2 = 2 r^T e + ||e||^2.

Decompose e into the component parallel to r and the orthogonal component:

    e_parallel = proj_r(e),       e_perp = e - e_parallel.

We then perform two offline counterfactual cost interventions on the SAME
candidate population:

    full              : ||r + e||^2
    remove_parallel   : ||r + e_perp||^2
    remove_orthogonal : ||r + e_parallel||^2

If removing goal-parallel error recovers exact-encoder candidate ordering much
more strongly than removing orthogonal error, this supports a goal-sensitive
training metric rather than uniform latent MSE.

The real simulator and frozen encoder are oracle-only.  Official benchmark
success, CEM, and the checkpoint cost function are unchanged.
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
from eval_mh_ald_teacher_response_oracle import (
    _make_blockwise_candidates,
    _pack_and_normalize,
    _render_state,
    _rollout_multihorizon,
    _select_anchor_rows,
    _summary,
)
from eval_pusht_horizon_directional import (
    _encode,
    _jsonable,
    _pairwise_accuracy,
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
    p.add_argument("--pair-margin-frac", type=float, default=0.02)
    p.add_argument("--semantic-check-atol", type=float, default=2e-5)
    p.add_argument("--decomposition-atol", type=float, default=2e-5)
    p.add_argument("--reference-context-summary", default=None)
    p.add_argument("--reference-context-anchor-csv", default=None)
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _topk_overlap(score, reference, k=10):
    score = np.asarray(score, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    kk = min(int(k), len(score))
    a = set(np.argsort(score, kind="stable")[:kk].tolist())
    b = set(np.argsort(reference, kind="stable")[:kk].tolist())
    return float(len(a & b) / max(kk, 1))


def _selection_regret(score, reference):
    score = np.asarray(score, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    idx = int(np.argmin(score))
    return float(reference[idx] - np.min(reference)), idx


def _decompose_endpoint_error(pred_z, real_z, goal_z, eps=1e-12):
    """Exact radial/orthogonal error decomposition for [N,D] tensors."""
    pred = torch.as_tensor(pred_z)
    real = torch.as_tensor(real_z, device=pred.device, dtype=pred.dtype)
    goal = torch.as_tensor(goal_z, device=pred.device, dtype=pred.dtype)
    if pred.ndim != 2 or real.shape != pred.shape or goal.ndim != 1:
        raise ValueError(
            f"Expected pred/real [N,D], goal [D], got "
            f"{pred.shape}, {real.shape}, {goal.shape}"
        )
    if goal.shape[0] != pred.shape[1]:
        raise ValueError("Goal latent dimension mismatch")

    residual = real - goal[None]
    error = pred - real
    r2 = torch.sum(residual * residual, dim=-1)
    dot = torch.sum(residual * error, dim=-1)
    coeff = torch.zeros_like(dot)
    good = r2 > float(eps)
    coeff[good] = dot[good] / r2[good]
    e_parallel = coeff[:, None] * residual
    e_perp = error - e_parallel

    exact_cost = r2
    full_cost = torch.sum((residual + error) ** 2, dim=-1)
    remove_parallel_cost = torch.sum((residual + e_perp) ** 2, dim=-1)
    remove_orthogonal_cost = torch.sum((residual + e_parallel) ** 2, dim=-1)

    cross_term = 2.0 * dot
    error_sq = torch.sum(error * error, dim=-1)
    parallel_sq = torch.sum(e_parallel * e_parallel, dim=-1)
    perp_sq = torch.sum(e_perp * e_perp, dim=-1)

    return {
        "residual": residual,
        "error": error,
        "e_parallel": e_parallel,
        "e_perp": e_perp,
        "exact_cost": exact_cost,
        "full_cost": full_cost,
        "remove_parallel_cost": remove_parallel_cost,
        "remove_orthogonal_cost": remove_orthogonal_cost,
        "cross_term": cross_term,
        "error_sq": error_sq,
        "parallel_sq": parallel_sq,
        "perp_sq": perp_sq,
        "residual_sq": r2,
    }


def _ranking_metrics(scores, exact_cost, physical_cost, margin_frac):
    s = np.asarray(scores, dtype=np.float64)
    exact = np.asarray(exact_cost, dtype=np.float64)
    physical = np.asarray(physical_cost, dtype=np.float64)
    pair_acc, pair_count = _pairwise_accuracy(exact, s, margin_frac)
    regret, selected = _selection_regret(s, exact)
    return {
        "rho_exact_encoder": _spearman(s, exact),
        "rho_physical": _spearman(s, physical),
        "pairwise_exact_encoder": pair_acc,
        "pairwise_pairs": int(pair_count),
        "top10_exact_encoder_overlap": _topk_overlap(s, exact, 10),
        "exact_encoder_selection_regret": regret,
        "selected_index": selected,
        "selected_exact_encoder_cost": float(exact[selected]),
        "selected_physical_cost": float(physical[selected]),
    }


def _summary_metric(rows, key):
    return _summary(r[key] for r in rows)


def _aggregate_model(rows, prefix):
    keys = [
        "rho_exact_encoder",
        "rho_physical",
        "pairwise_exact_encoder",
        "top10_exact_encoder_overlap",
        "exact_encoder_selection_regret",
    ]
    out = {}
    for mode in ("full", "remove_parallel", "remove_orthogonal"):
        out[mode] = {
            key: _summary_metric(rows, f"{prefix}_{mode}_{key}")
            for key in keys
        }
    out["intervention_delta"] = {
        "rho_exact_remove_parallel_minus_full": _summary_metric(
            rows, f"{prefix}_delta_rho_exact_remove_parallel_minus_full"
        ),
        "rho_exact_remove_orthogonal_minus_full": _summary_metric(
            rows, f"{prefix}_delta_rho_exact_remove_orthogonal_minus_full"
        ),
        "pairwise_remove_parallel_minus_full": _summary_metric(
            rows, f"{prefix}_delta_pairwise_remove_parallel_minus_full"
        ),
        "pairwise_remove_orthogonal_minus_full": _summary_metric(
            rows, f"{prefix}_delta_pairwise_remove_orthogonal_minus_full"
        ),
        "top10_remove_parallel_minus_full": _summary_metric(
            rows, f"{prefix}_delta_top10_remove_parallel_minus_full"
        ),
        "top10_remove_orthogonal_minus_full": _summary_metric(
            rows, f"{prefix}_delta_top10_remove_orthogonal_minus_full"
        ),
    }
    out["error_geometry"] = {
        "parallel_error_energy_fraction": _summary_metric(
            rows, f"{prefix}_parallel_error_energy_fraction"
        ),
        "goal_coupled_term_fraction": _summary_metric(
            rows, f"{prefix}_goal_coupled_term_fraction"
        ),
        "mean_abs_cost_error": _summary_metric(
            rows, f"{prefix}_mean_abs_cost_error"
        ),
    }
    return out


def _group_summary(rows):
    return {
        "anchors": len(rows),
        "teacher": _aggregate_model(rows, "teacher"),
        "student": _aggregate_model(rows, "student"),
        "exact_encoder_physical_rho": _summary_metric(
            rows, "rho_exact_encoder_physical"
        ),
    }


def _load_reference_anchor_rows(path):
    if path is None:
        return None
    out = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["anchor_index"])] = row
    return out


def main():
    args = parse_args()
    if int(args.train_observation_history) != 3:
        raise ValueError("Formal MH-ALD train observation history is 3")
    if int(args.planner_observation_history) != 1:
        raise ValueError("Formal PushT planner observation history is 1")
    if int(args.predictor_max_history) != 3:
        raise ValueError("Formal LeWM predictor max history is 3")
    if int(args.horizon) != 5 or int(args.action_block) != 5:
        raise ValueError("Formal PushT expects horizon=5 and action_block=5")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    if int(cfg.world.history_size) != 1:
        raise RuntimeError("Official PushT eval config must have world.history_size=1")

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
    ep_col, anchors = _select_anchor_rows(
        dataset,
        args.num_anchors,
        args.seed,
        train_history_raw,
        future_raw,
    )
    if args.reference_context_summary:
        ref = json.loads(Path(args.reference_context_summary).read_text())
        if list(map(int, ref.get("selected_rows", []))) != anchors.tolist():
            raise RuntimeError(
                "Problem-2 anchor rows differ from the completed context-gap oracle"
            )

    reference_anchor_rows = _load_reference_anchor_rows(
        args.reference_context_anchor_csv
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
    anchor_rows = []
    candidate_rows = []
    frame_max_abs = 0.0
    semantic_check = {"teacher": None, "student": None}
    max_cost_identity_error = 0.0
    max_orthogonality_error = 0.0
    max_energy_identity_error = 0.0
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
            frame_diff = float(torch.max(torch.abs(hist_s - hist_t)).cpu())
            frame_max_abs = max(frame_max_abs, frame_diff)
            if frame_diff > 2e-5:
                raise RuntimeError(
                    f"Teacher/student frozen visual frames differ: {frame_diff:.3e}"
                )
            goal_z = _encode(
                student, transform, [goal_image], device, args.model_batch_size
            )[0]

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
            future_packed = np.stack([
                _pack_and_normalize(c, scaler, args.action_block)
                for c in candidates_raw
            ])

            teacher_roll = _planner_warmup_rollout(
                teacher,
                hist_t[-1].detach().cpu().numpy(),
                future_packed,
                args.predictor_max_history,
                device,
            )
            student_roll = _planner_warmup_rollout(
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
                    semantic_check[name] = _assert_planner_semantics(
                        model,
                        current_image,
                        transform,
                        emb,
                        future_packed[0],
                        args.predictor_max_history,
                        device,
                        args.semantic_check_atol,
                    )
                    print(
                        f"planner semantic check [{name}] "
                        f"max_abs={semantic_check[name]:.3e}",
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
            ).reshape(len(candidates_raw), int(args.horizon), -1)
            final_states = np.stack(final_states)
            replay_error = float(
                np.linalg.norm(
                    _state_factor(center_states[-1]) - _state_factor(goal_state)
                )
            )
            replay_good = replay_error <= float(args.replay_good_threshold)
            physical_cost = _physical_cost(final_states, goal_state)[0]
            exact_cost = (
                torch.sum((oracle_z[:, -1] - goal_z[None]) ** 2, dim=-1)
                .detach().cpu().numpy().astype(np.float64)
            )

            row_out = {
                "anchor_index": int(ai),
                "dataset_row": int(row),
                "episode_idx": int(episodes[row]),
                "step_idx": int(steps[row]),
                "num_candidates": int(len(candidates_raw)),
                "center_contact": bool(center_contact),
                "replay_endpoint_factor_error": replay_error,
                "replay_good": bool(replay_good),
                "rho_exact_encoder_physical": _spearman(exact_cost, physical_cost),
            }

            for prefix, rollout in (
                ("teacher", teacher_roll),
                ("student", student_roll),
            ):
                dec = _decompose_endpoint_error(
                    rollout[:, -1], oracle_z[:, -1], goal_z
                )
                values = {
                    k: v.detach().float().cpu().numpy().astype(np.float64)
                    for k, v in dec.items()
                    if k not in ("residual", "error", "e_parallel", "e_perp")
                }
                identity_error = np.max(np.abs(
                    (values["full_cost"] - values["exact_cost"])
                    - (values["cross_term"] + values["error_sq"])
                ))
                energy_error = np.max(np.abs(
                    values["error_sq"]
                    - (values["parallel_sq"] + values["perp_sq"])
                ))
                ortho = dec["e_perp"] * dec["residual"]
                ortho_error = float(
                    torch.max(torch.abs(torch.sum(ortho, dim=-1))).detach().cpu()
                )
                max_cost_identity_error = max(max_cost_identity_error, float(identity_error))
                max_energy_identity_error = max(max_energy_identity_error, float(energy_error))
                max_orthogonality_error = max(max_orthogonality_error, ortho_error)

                for mode, score_key in (
                    ("full", "full_cost"),
                    ("remove_parallel", "remove_parallel_cost"),
                    ("remove_orthogonal", "remove_orthogonal_cost"),
                ):
                    m = _ranking_metrics(
                        values[score_key], exact_cost, physical_cost,
                        args.pair_margin_frac,
                    )
                    for key, val in m.items():
                        row_out[f"{prefix}_{mode}_{key}"] = val

                row_out.update({
                    f"{prefix}_delta_rho_exact_remove_parallel_minus_full": (
                        row_out[f"{prefix}_remove_parallel_rho_exact_encoder"]
                        - row_out[f"{prefix}_full_rho_exact_encoder"]
                    ),
                    f"{prefix}_delta_rho_exact_remove_orthogonal_minus_full": (
                        row_out[f"{prefix}_remove_orthogonal_rho_exact_encoder"]
                        - row_out[f"{prefix}_full_rho_exact_encoder"]
                    ),
                    f"{prefix}_delta_pairwise_remove_parallel_minus_full": (
                        row_out[f"{prefix}_remove_parallel_pairwise_exact_encoder"]
                        - row_out[f"{prefix}_full_pairwise_exact_encoder"]
                    ),
                    f"{prefix}_delta_pairwise_remove_orthogonal_minus_full": (
                        row_out[f"{prefix}_remove_orthogonal_pairwise_exact_encoder"]
                        - row_out[f"{prefix}_full_pairwise_exact_encoder"]
                    ),
                    f"{prefix}_delta_top10_remove_parallel_minus_full": (
                        row_out[f"{prefix}_remove_parallel_top10_exact_encoder_overlap"]
                        - row_out[f"{prefix}_full_top10_exact_encoder_overlap"]
                    ),
                    f"{prefix}_delta_top10_remove_orthogonal_minus_full": (
                        row_out[f"{prefix}_remove_orthogonal_top10_exact_encoder_overlap"]
                        - row_out[f"{prefix}_full_top10_exact_encoder_overlap"]
                    ),
                    f"{prefix}_parallel_error_energy_fraction": float(np.mean(
                        values["parallel_sq"]
                        / np.maximum(values["error_sq"], 1e-12)
                    )),
                    f"{prefix}_goal_coupled_term_fraction": float(np.mean(
                        np.abs(values["cross_term"])
                        / np.maximum(
                            np.abs(values["cross_term"]) + values["error_sq"],
                            1e-12,
                        )
                    )),
                    f"{prefix}_mean_abs_cost_error": float(np.mean(np.abs(
                        values["full_cost"] - values["exact_cost"]
                    ))),
                })

                for ci in range(len(candidates_raw)):
                    meta_ci = meta[ci]
                    candidate_rows.append({
                        "anchor_index": int(ai),
                        "dataset_row": int(row),
                        "replay_good": bool(replay_good),
                        "center_contact": bool(center_contact),
                        "candidate_index": int(ci),
                        "probe_position": int(meta_ci["position"]),
                        "probe_direction": int(meta_ci["direction"]),
                        "probe_sign": int(meta_ci["sign"]),
                        "candidate_contact": bool(candidate_contacts[ci]),
                        "model": prefix,
                        "exact_encoder_cost": float(exact_cost[ci]),
                        "physical_cost": float(physical_cost[ci]),
                        "full_predicted_cost": float(values["full_cost"][ci]),
                        "remove_parallel_cost": float(values["remove_parallel_cost"][ci]),
                        "remove_orthogonal_cost": float(values["remove_orthogonal_cost"][ci]),
                        "cost_error": float(
                            values["full_cost"][ci] - values["exact_cost"][ci]
                        ),
                        "goal_cross_term": float(values["cross_term"][ci]),
                        "prediction_error_sq": float(values["error_sq"][ci]),
                        "parallel_error_sq": float(values["parallel_sq"][ci]),
                        "orthogonal_error_sq": float(values["perp_sq"][ci]),
                        "goal_residual_sq": float(values["residual_sq"][ci]),
                    })

            if reference_anchor_rows is not None:
                ref = reference_anchor_rows.get(int(ai))
                if ref is None:
                    raise RuntimeError(f"Missing reference context anchor {ai}")
                checks = {
                    "teacher": float(ref["rho_planner_teacher_exact_encoder"]),
                    "student": float(ref["rho_planner_student_exact_encoder"]),
                }
                for prefix, expected in checks.items():
                    actual = row_out[f"{prefix}_full_rho_exact_encoder"]
                    if not np.isclose(actual, expected, rtol=0.0, atol=1e-8):
                        raise RuntimeError(
                            f"Problem-2 full-ranking mismatch vs context oracle at "
                            f"anchor {ai}, {prefix}: {actual} vs {expected}"
                        )

            anchor_rows.append(row_out)
            elapsed = time.time() - t0
            eta = elapsed / (ai + 1) * (len(anchors) - ai - 1)
            print(
                f"anchor {ai+1:3d}/{len(anchors)} row={row} "
                f"good={int(replay_good)} contact={int(center_contact)} "
                f"student rho={row_out['student_full_rho_exact_encoder']:.3f} "
                f"-parallel={row_out['student_remove_parallel_rho_exact_encoder']:.3f} "
                f"-orth={row_out['student_remove_orthogonal_rho_exact_encoder']:.3f} "
                f"ETA={eta/60:.1f}m",
                flush=True,
            )
    finally:
        env.close()

    if max_cost_identity_error > float(args.decomposition_atol):
        raise RuntimeError(
            f"Cost identity check failed: {max_cost_identity_error:.3e}"
        )
    if max_energy_identity_error > float(args.decomposition_atol):
        raise RuntimeError(
            f"Error-energy identity check failed: {max_energy_identity_error:.3e}"
        )
    if max_orthogonality_error > float(args.decomposition_atol):
        raise RuntimeError(
            f"Projection orthogonality check failed: {max_orthogonality_error:.3e}"
        )

    anchor_csv = outdir / "goal_projection_anchor_metrics.csv"
    candidate_csv = outdir / "goal_projection_candidate_metrics.csv"
    _write_csv(anchor_csv, anchor_rows)
    _write_csv(candidate_csv, candidate_rows)

    good = [r for r in anchor_rows if r["replay_good"]]
    contact = [r for r in good if r["center_contact"]]
    no_contact = [r for r in good if not r["center_contact"]]
    summary = {
        "question": (
            "Does endpoint prediction error parallel to the exact frozen-encoder "
            "goal residual cause more planner ranking damage than orthogonal error?"
        ),
        "config": vars(args),
        "teacher_policy": args.teacher_policy,
        "student_policy": args.student_policy,
        "selected_rows": anchors.tolist(),
        "latent_frame_max_abs_teacher_vs_student": frame_max_abs,
        "planner_semantic_check_max_abs": semantic_check,
        "decomposition_checks": {
            "max_cost_identity_abs": max_cost_identity_error,
            "max_error_energy_identity_abs": max_energy_identity_error,
            "max_projection_orthogonality_abs": max_orthogonality_error,
        },
        "all": _group_summary(anchor_rows),
        "replay_good": _group_summary(good),
        "replay_good_contact": _group_summary(contact),
        "replay_good_no_contact": _group_summary(no_contact),
        "protocol_notes": [
            "Official planner rollout semantics: one observed latent, then self-generated history capped at 3.",
            "Same 40-anchor/probe selection as the completed context-gap oracle when seed/count/config match.",
            "Exact encoder and real simulator are oracle-only and never alter CEM or training.",
            "remove_parallel and remove_orthogonal are offline cost interventions on the same fixed candidates.",
            "Formal benchmark success and checkpoint terminal latent cost remain unchanged.",
        ],
        "elapsed_seconds": float(time.time() - t0),
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    print("\n===== GOAL-PROJECTION ORACLE: REPLAY-GOOD =====")
    print(json.dumps(_jsonable(summary["replay_good"]), indent=2))
    print("\n===== REPLAY-GOOD + CONTACT =====")
    print(json.dumps(_jsonable(summary["replay_good_contact"]), indent=2))
    print("decomposition_checks=" + json.dumps(summary["decomposition_checks"], sort_keys=True))
    print(f"Saved: {anchor_csv}")
    print(f"Saved: {candidate_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
