"""Pure Anchored Local Dynamics (Pure-ALD) objective.

This is the decisive loss-composition ablation of ALD. It keeps exactly the
same pretrained student initialization, frozen visual latent frame, frozen
teacher, H-step rollout semantics, bounded symmetric probes, and trainable
predictor-side modules as the existing ALD experiment, but optimizes ONLY

    L = L_ALD
      = 1/2 [ ||S(U+) - Y+||^2 + ||S(U-) - Y-||^2 ],

where

    Y+ = z_H + sg[T(U+) - T(U0)]
    Y- = z_H + sg[T(U-) - T(U0)].

No teacher-forcing loss, no demonstrated-rollout loss, no SIGReg, no APB,
no JVP/Gram/odd-even auxiliary loss, and no privileged simulator state.

This isolates whether the anchoring principle itself is sufficient:
observed future supplies the absolute anchor, while the frozen teacher
supplies local action-conditioned displacement.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from anchored_local_dynamics import (
    _frozen_visual_encode,
    _paired_local_predictions,
)
from stage1_bias_calibration import _autoregressive_rollout


def pure_ald_forward(self, batch, stage, cfg, action_mean, action_std):
    if not hasattr(self, "teacher_model"):
        raise RuntimeError(
            "Pure-ALD module has no frozen teacher_model. Use train_pure_ald.py."
        )

    student = self.model
    teacher = self.teacher_model
    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    horizon = int(cfg.pure_ald.rollout_horizon)

    if n_preds != 1:
        raise ValueError(f"Pure-ALD assumes wm.num_preds=1, got {n_preds}.")
    if horizon < 1:
        raise ValueError(
            f"pure_ald.rollout_horizon must be >= 1, got {horizon}."
        )

    # Match the planner/inference semantics used by the established ALD
    # experiment: frozen BN statistics and dropout off, while gradients remain
    # enabled through the student predictor side.
    student.eval()
    teacher.eval()
    student.encoder.eval()
    student.projector.eval()

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    if batch["pixels"].shape[1] < ctx_len + horizon:
        raise ValueError(
            "Pure-ALD sequence too short: need at least "
            f"{ctx_len + horizon} states, got {batch['pixels'].shape[1]}."
        )

    emb = _frozen_visual_encode(student, batch)
    z_goal = emb[:, ctx_len + horizon - 1].detach()

    expected_action_dim = student.action_encoder.patch_embed.in_channels
    if batch["action"].shape[-1] != expected_action_dim:
        raise RuntimeError(
            "Packed action dimension mismatch: "
            f"batch={batch['action'].shape[-1]} student={expected_action_dim}."
        )

    # Nominal teacher endpoint T(U0): used only as the zero-order reference for
    # the anchored targets and diagnostics. It is not optimized directly.
    with torch.no_grad():
        teacher_rollout = _autoregressive_rollout(
            teacher,
            emb,
            batch["action"],
            history_size=ctx_len,
            horizon=horizon,
        )
        teacher_center = teacher_rollout[:, -1].detach()

    (
        student_plus,
        student_minus,
        teacher_plus,
        teacher_minus,
        effective_radius,
    ) = _paired_local_predictions(
        student=student,
        teacher=teacher,
        emb=emb,
        normalized_actions=batch["action"],
        action_mean=action_mean,
        action_std=action_std,
        history_size=ctx_len,
        horizon=horizon,
        radius=float(cfg.pure_ald.perturb_radius),
        first_only=bool(cfg.pure_ald.perturb_first_only),
        num_directions=int(cfg.pure_ald.num_directions),
        directions_per_chunk=int(cfg.pure_ald.probe_directions_per_chunk),
    )

    goal = z_goal[:, None, :]
    teacher_center_expanded = teacher_center[:, None, :]
    target_plus = goal + (teacher_plus - teacher_center_expanded).detach()
    target_minus = goal + (teacher_minus - teacher_center_expanded).detach()

    plus_err = student_plus - target_plus
    minus_err = student_minus - target_minus

    # --------------------------- ONLY OPTIMIZATION LOSS ---------------------------
    pure_ald_loss = 0.5 * (
        plus_err.pow(2).mean() + minus_err.pow(2).mean()
    )
    # ------------------------------------------------------------------------------

    # Diagnostics only; none of these are added to the objective.
    student_mid = 0.5 * (student_plus + student_minus)
    target_mid = 0.5 * (target_plus + target_minus)
    student_half_response = 0.5 * (student_plus - student_minus)
    target_half_response = 0.5 * (target_plus - target_minus)

    midpoint_loss = (student_mid - target_mid).pow(2).mean()
    half_response_loss = (
        student_half_response - target_half_response
    ).pow(2).mean()
    decomposition_error = (
        pure_ald_loss.detach()
        - (midpoint_loss + half_response_loss).detach()
    ).abs()

    # Evaluate the demonstrated nominal endpoint only as a diagnostic.
    student_rollout = _autoregressive_rollout(
        student,
        emb,
        batch["action"],
        history_size=ctx_len,
        horizon=horizon,
    )
    student_center = student_rollout[:, -1]
    student_endpoint_mse = (student_center - z_goal).pow(2).mean()
    teacher_endpoint_mse = (teacher_center - z_goal).pow(2).mean()

    denom = (2.0 * effective_radius).clamp_min(1e-8)[..., None]
    student_response = (student_plus - student_minus) / denom
    teacher_response = (teacher_plus - teacher_minus) / denom

    student_response_norm = torch.linalg.vector_norm(
        student_response, dim=-1
    )
    teacher_response_norm = torch.linalg.vector_norm(
        teacher_response, dim=-1
    )
    response_cosine = F.cosine_similarity(
        student_response.detach(),
        teacher_response.detach(),
        dim=-1,
        eps=1e-8,
    )
    response_gain = (
        student_response_norm.detach()
        / teacher_response_norm.detach().clamp_min(1e-8)
    )

    init_equivalence_ratio = (
        pure_ald_loss.detach()
        / teacher_endpoint_mse.detach().clamp_min(1e-12)
    )

    output = {
        "loss": pure_ald_loss,
        "pure_ald_loss": pure_ald_loss,
    }

    self.log_dict(
        {
            f"{stage}/pure_ald_loss": pure_ald_loss.detach(),
            f"{stage}/pure_ald_midpoint_loss": midpoint_loss.detach(),
            f"{stage}/pure_ald_half_response_loss": half_response_loss.detach(),
            f"{stage}/pure_ald_decomposition_error": decomposition_error,
            f"{stage}/pure_ald_student_endpoint_mse": student_endpoint_mse.detach(),
            f"{stage}/pure_ald_teacher_endpoint_mse": teacher_endpoint_mse.detach(),
            f"{stage}/pure_ald_init_equivalence_ratio": init_equivalence_ratio,
            f"{stage}/pure_ald_student_response_norm": student_response_norm.mean().detach(),
            f"{stage}/pure_ald_teacher_response_norm": teacher_response_norm.mean().detach(),
            f"{stage}/pure_ald_response_cosine": response_cosine.mean().detach(),
            f"{stage}/pure_ald_response_gain": response_gain.mean().detach(),
            f"{stage}/pure_ald_effective_radius": effective_radius.mean().detach(),
        },
        on_step=True,
        sync_dist=True,
    )
    return output
