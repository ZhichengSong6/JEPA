"""Planner-context, goal-coupled Multi-Horizon ALD.

PC-GC-MH keeps the parts of PC-GP-MH that the completed diagnostics validated:

1) official planner-context warm-up: one observed latent, then 1->2->3 history;
2) frozen visual latent frame and frozen LeWM teacher;
3) anchored relative teacher responses for multi-horizon supervision;
4) finite probe radius selected outside this module (formal config uses 0.30).

The only scientific change from PC-GP-MH is the terminal anisotropic metric.
Instead of normalizing every candidate by its own goal-residual magnitude,
PC-GC-MH penalizes

    (d^T e)^2

where d is the anchored teacher target's terminal goal residual and e is the
student-to-target terminal error.  A single detached batch/probe normalization
scale keeps units comparable to latent MSE without cancelling candidate-wise
||d||^2 sensitivity.

Thus larger goal-coupled residuals receive proportionally larger gradients, as
suggested by the exact planner cost first-order term 2 d^T e.

No simulator state, counterfactual real rollout, learned reward/readout, rank
label, or planner modification is used by this training objective.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from anchored_local_dynamics import (
    _frozen_visual_encode,
    _mh_causal_mask,
    _mh_masked_mean,
    _mh_masked_mse,
)
from planner_context_goal_mh import (
    _paired_planner_context_predictions,
    planner_context_rollout,
)


def goal_coupled_mse(
    error: torch.Tensor,
    goal_residual: torch.Tensor,
    eps: float = 1e-8,
):
    """Residual-weighted goal-coupled terminal error.

    Args:
        error: [..., D] student-to-anchored-target terminal error.
        goal_residual: [..., D] anchored target minus goal latent.
        eps: numerical floor for the shared normalization scale.

    Returns:
        loss: scalar with the same latent-squared scale as ordinary MSE.
        valid_fraction: fraction of candidates with non-degenerate residual.
        direction_energy_scale: detached mean ||d||^2 over valid candidates.

    The numerator is E[(d^T e)^2].  The denominator is

        D * sg(E[||d||^2]),

    where the expectation is shared across the batch/probe set.  This makes
    the loss invariant to a global rescaling of d while preserving *relative*
    candidate weights proportional to ||d_i||^2.  That is the critical
    difference from per-candidate projection normalization.

    Dot products are evaluated in FP32 so bf16 autocast does not coarsen the
    decision-sensitive squared inner product.
    """
    if error.shape != goal_residual.shape or error.ndim < 2:
        raise ValueError(
            f"Expected matched [...,D] tensors, got {error.shape=} "
            f"{goal_residual.shape=}."
        )

    err32 = error.float()
    direction32 = goal_residual.detach().float()
    norm_sq = direction32.pow(2).sum(dim=-1)
    valid = norm_sq > float(eps)
    valid_f = valid.float()
    valid_count = valid_f.sum()

    dot = (err32 * direction32).sum(dim=-1)
    dot_sq = dot.pow(2)

    safe_count = valid_count.clamp_min(1.0)
    numerator = (dot_sq * valid_f).sum() / safe_count
    direction_energy = (norm_sq * valid_f).sum() / safe_count

    latent_dim = max(int(error.shape[-1]), 1)
    denom = (
        float(latent_dim)
        * direction_energy.detach().clamp_min(float(eps))
    )
    loss = numerator / denom

    # If no candidate has a meaningful direction, define the term as zero.
    loss = torch.where(valid_count > 0, loss, torch.zeros_like(loss))
    valid_fraction = valid_f.mean().detach()
    return loss, valid_fraction, direction_energy.detach()


def combine_goal_coupled_metric(
    base_mh_loss: torch.Tensor,
    terminal_goal_coupled_loss: torch.Tensor,
    goal_coupled_weight: float,
) -> torch.Tensor:
    """Add the residual-weighted decision-sensitive term to base MH loss."""
    weight = float(goal_coupled_weight)
    if weight < 0.0:
        raise ValueError(f"goal_coupled.weight must be >=0, got {weight}.")
    return base_mh_loss + weight * terminal_goal_coupled_loss


def pc_gc_mh_ald_forward(self, batch, stage, cfg, action_mean, action_std):
    """TF + planner-context rollout + residual-weighted planner-context MH-ALD."""
    if not hasattr(self, "teacher_model"):
        raise RuntimeError("PC-GC-MH module has no frozen teacher_model.")

    student = self.model
    teacher = self.teacher_model
    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    horizon = int(cfg.mh_ald.rollout_horizon)
    pc_cfg = cfg.mh_ald.planner_context
    gc_cfg = cfg.mh_ald.goal_coupled
    max_history = int(pc_cfg.predictor_max_history)

    if not bool(pc_cfg.enabled):
        raise ValueError("PC-GC-MH requires mh_ald.planner_context.enabled=True.")
    if not bool(gc_cfg.enabled):
        raise ValueError("PC-GC-MH requires mh_ald.goal_coupled.enabled=True.")
    if int(pc_cfg.observation_history) != 1:
        raise ValueError("Formal planner-context experiment requires observation_history=1.")
    if max_history != ctx_len:
        raise ValueError(
            "Formal experiment keeps predictor max history equal to original "
            f"wm.history_size: {max_history=} vs {ctx_len=}."
        )
    if n_preds != 1 or horizon < 1:
        raise ValueError(f"Unexpected {n_preds=} or {horizon=}.")

    student.eval()
    teacher.eval()
    student.encoder.eval()
    student.projector.eval()
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    if batch["pixels"].shape[1] < ctx_len + horizon:
        raise ValueError(
            "PC-GC-MH sequence too short: need at least "
            f"{ctx_len + horizon}, got {batch['pixels'].shape[1]}."
        )

    emb = _frozen_visual_encode(student, batch)
    rollout_target = emb[:, ctx_len : ctx_len + horizon].detach()
    terminal_goal = rollout_target[:, -1].detach()
    current_emb = emb[:, ctx_len - 1].detach()
    plan_start = ctx_len - 1
    plan_stop = plan_start + horizon
    demo_plan = batch["action"][:, plan_start:plan_stop]
    if demo_plan.shape[1] != horizon:
        raise RuntimeError(
            f"Planner-context demo plan has {demo_plan.shape[1]} blocks, expected {horizon}."
        )

    expected_action_dim = student.action_encoder.patch_embed.in_channels
    if batch["action"].shape[-1] != expected_action_dim:
        raise RuntimeError(
            "Packed action dimension mismatch: "
            f"batch={batch['action'].shape[-1]} student={expected_action_dim}."
        )

    # Preserve the original one-step teacher-forcing regularizer.
    ctx_emb = emb[:, :ctx_len]
    ctx_action = batch["action"][:, :ctx_len]
    ctx_act_emb = student.action_encoder(ctx_action)
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]
    pred_emb = student.predict(ctx_emb, ctx_act_emb)
    tf_loss = (pred_emb - tgt_emb).pow(2).mean()

    # Absolute rollout calibration uses exact official planner warm-up semantics.
    student_center_rollout = planner_context_rollout(
        student, current_emb, demo_plan, predictor_max_history=max_history
    )
    rollout_loss = (student_center_rollout - rollout_target).pow(2).mean()
    with torch.no_grad():
        teacher_center_rollout = planner_context_rollout(
            teacher, current_emb, demo_plan, predictor_max_history=max_history
        ).detach()

    student_endpoint_mse = (
        student_center_rollout[:, -1] - terminal_goal
    ).pow(2).mean()
    teacher_endpoint_mse = (
        teacher_center_rollout[:, -1] - terminal_goal
    ).pow(2).mean()

    (
        student_plus,
        student_minus,
        teacher_plus,
        teacher_minus,
        effective_radius,
        positions,
    ) = _paired_planner_context_predictions(
        student=student,
        teacher=teacher,
        current_emb=current_emb,
        normalized_actions=batch["action"],
        action_mean=action_mean,
        action_std=action_std,
        history_size=ctx_len,
        horizon=horizon,
        predictor_max_history=max_history,
        radius=float(cfg.mh_ald.perturb_radius),
        directions_per_position=int(cfg.mh_ald.directions_per_position),
        probes_per_chunk=int(cfg.mh_ald.probes_per_chunk),
    )

    # Anchored ALD targets: absolute future anchor + frozen teacher relative response.
    anchors = rollout_target[:, None, :, :]
    teacher_center = teacher_center_rollout[:, None, :, :]
    target_plus = anchors + (teacher_plus - teacher_center).detach()
    target_minus = anchors + (teacher_minus - teacher_center).detach()

    mask = _mh_causal_mask(
        positions,
        horizon,
        dtype=student_plus.dtype,
        device=student_plus.device,
    )
    plus_err = student_plus - target_plus
    minus_err = student_minus - target_minus
    base_mh_loss = 0.5 * (
        _mh_masked_mse(plus_err, mask)
        + _mh_masked_mse(minus_err, mask)
    )

    # Goal-coupled terminal term.
    #
    # Since the demonstrated terminal anchor equals terminal_goal,
    # target_{+/-}^H - goal = teacher_{+/-}^H - teacher_center^H.
    # This is an anchored teacher-response proxy for the decision-sensitive
    # goal residual; it uses no simulator counterfactual or physical state.
    goal = terminal_goal[:, None, :]
    plus_terminal_direction = target_plus[:, :, -1] - goal
    minus_terminal_direction = target_minus[:, :, -1] - goal
    terminal_error = torch.cat(
        [plus_err[:, :, -1], minus_err[:, :, -1]], dim=1
    )
    terminal_direction = torch.cat(
        [plus_terminal_direction, minus_terminal_direction], dim=1
    )
    (
        terminal_goal_coupled_loss,
        goal_direction_valid_fraction,
        goal_direction_energy_scale,
    ) = goal_coupled_mse(
        terminal_error,
        terminal_direction,
        eps=float(gc_cfg.normalization_eps),
    )
    calibration_loss = combine_goal_coupled_metric(
        base_mh_loss,
        terminal_goal_coupled_loss,
        goal_coupled_weight=float(gc_cfg.weight),
    )

    # Standard MH diagnostics remain useful under the matched planner context.
    student_mid = 0.5 * (student_plus + student_minus)
    target_mid = 0.5 * (target_plus + target_minus)
    student_half_response = 0.5 * (student_plus - student_minus)
    target_half_response = 0.5 * (target_plus - target_minus)
    midpoint_loss = _mh_masked_mse(student_mid - target_mid, mask)
    half_response_loss = _mh_masked_mse(
        student_half_response - target_half_response, mask
    )
    base_decomposition_error = (
        base_mh_loss.detach()
        - (midpoint_loss + half_response_loss).detach()
    ).abs()

    center_error = teacher_center - anchors
    teacher_center_masked_mse = _mh_masked_mse(
        center_error.expand(-1, positions.numel(), -1, -1), mask
    )
    init_equivalence_ratio = (
        base_mh_loss.detach()
        / teacher_center_masked_mse.detach().clamp_min(1e-12)
    )

    denom = (2.0 * effective_radius).clamp_min(1e-8)[:, :, None, None]
    student_response = (student_plus - student_minus) / denom
    teacher_response = (teacher_plus - teacher_minus) / denom
    student_norm = torch.linalg.vector_norm(student_response, dim=-1)
    teacher_norm = torch.linalg.vector_norm(teacher_response, dim=-1)
    response_cosine_cells = F.cosine_similarity(
        student_response.detach(), teacher_response.detach(), dim=-1, eps=1e-8
    )
    response_gain_cells = (
        student_norm.detach() / teacher_norm.detach().clamp_min(1e-8)
    )
    mask_bph = mask[..., 0]
    response_cosine = _mh_masked_mean(response_cosine_cells, mask_bph)
    response_gain = _mh_masked_mean(response_gain_cells, mask_bph)

    total_loss = (
        float(cfg.mh_ald.tf_weight) * tf_loss
        + float(cfg.mh_ald.rollout_weight) * rollout_loss
        + float(cfg.mh_ald.weight) * calibration_loss
    )

    output = {
        "loss": total_loss,
        "tf_loss": tf_loss,
        "pc_gc_rollout_loss": rollout_loss,
        "pc_gc_base_mh_loss": base_mh_loss,
        "pc_gc_terminal_goal_coupled_loss": terminal_goal_coupled_loss,
        "pc_gc_calibration_loss": calibration_loss,
        "pc_gc_midpoint_loss": midpoint_loss,
        "pc_gc_half_response_loss": half_response_loss,
    }
    losses = {
        f"{stage}/{key}": value.detach()
        for key, value in output.items()
        if "loss" in key
    }
    diagnostics = {
        f"{stage}/pc_gc_student_endpoint_mse": student_endpoint_mse.detach(),
        f"{stage}/pc_gc_teacher_endpoint_mse": teacher_endpoint_mse.detach(),
        f"{stage}/pc_gc_teacher_center_masked_mse": teacher_center_masked_mse.detach(),
        f"{stage}/pc_gc_base_decomposition_error": base_decomposition_error,
        f"{stage}/pc_gc_init_equivalence_ratio": init_equivalence_ratio,
        f"{stage}/pc_gc_response_cosine": response_cosine.detach(),
        f"{stage}/pc_gc_response_gain": response_gain.detach(),
        f"{stage}/pc_gc_effective_radius": effective_radius.mean().detach(),
        f"{stage}/pc_gc_goal_direction_valid_fraction": goal_direction_valid_fraction,
        f"{stage}/pc_gc_goal_direction_energy_scale": goal_direction_energy_scale,
        f"{stage}/pc_gc_goal_coupled_weight": torch.tensor(
            float(gc_cfg.weight),
            device=base_mh_loss.device,
            dtype=base_mh_loss.dtype,
        ),
        f"{stage}/pc_gc_goal_coupled_to_base_ratio": (
            terminal_goal_coupled_loss.detach()
            / base_mh_loss.detach().float().clamp_min(1e-12)
        ),
        f"{stage}/pc_gc_probe_count": torch.tensor(
            float(positions.numel()), device=base_mh_loss.device
        ),
    }
    for h in range(horizon):
        hmask = mask[:, :, h : h + 1, :]
        h_plus = _mh_masked_mse(plus_err[:, :, h : h + 1], hmask)
        h_minus = _mh_masked_mse(minus_err[:, :, h : h + 1], hmask)
        diagnostics[f"{stage}/pc_gc_h{h + 1}_loss"] = (
            0.5 * (h_plus + h_minus)
        ).detach()

    self.log_dict(losses, on_step=True, sync_dist=True)
    self.log_dict(diagnostics, on_step=True, sync_dist=True)
    return output
