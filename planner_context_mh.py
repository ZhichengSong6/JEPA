"""Pure Planner-Context Multi-Horizon ALD (PC-MH-ALD).

This module is a strict single-variable continuation of formal MH-ALD.
Everything is kept identical to formal MH-ALD except the autoregressive rollout
context used by the demonstrated rollout and synthetic MH probes:

    formal MH-ALD : starts from three real visual latents
    PC-MH-ALD     : starts from one observed latent, then warms 1 -> 2 -> 3

The predictor maximum history remains three.  The one-step teacher-forcing
branch remains unchanged and still uses the original three-real-latent context.
The official CEM planner, planning cost, visual latent frame, teacher, probe
radius, optimizer, and all loss weights are unchanged.

No simulator state, counterfactual real rollout, reward/readout, rank label, or
planner modification is used by this objective.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from anchored_local_dynamics import (
    _frozen_visual_encode,
    _mh_causal_mask,
    _mh_masked_mean,
    _mh_masked_mse,
    _repeat_for_candidates,
    _sample_blockwise_probe_actions,
)


def planner_context_rollout(
    model,
    current_emb: torch.Tensor,
    plan_actions: torch.Tensor,
    predictor_max_history: int,
) -> torch.Tensor:
    """Differentiable rollout matching official one-observation planner warm-up.

    Args:
        current_emb: [B,D] current observed latent.
        plan_actions: [B,H,A] normalized coarse action blocks.
        predictor_max_history: maximum predictor context length (formal LeWM: 3).

    Returns:
        [B,H,D] predicted future latents.
    """
    if current_emb.ndim != 2 or plan_actions.ndim != 3:
        raise ValueError(
            "planner_context_rollout expects current_emb [B,D] and "
            f"plan_actions [B,H,A], got {current_emb.shape=} {plan_actions.shape=}."
        )
    if current_emb.shape[0] != plan_actions.shape[0]:
        raise ValueError("Planner-context latent/action batch sizes differ.")
    max_history = int(predictor_max_history)
    if max_history < 1:
        raise ValueError(f"predictor_max_history must be >=1, got {max_history}.")
    horizon = int(plan_actions.shape[1])
    if horizon < 1:
        raise ValueError("Planner-context rollout requires at least one action block.")

    latent_history = current_emb[:, None]
    predicted = []
    for step in range(horizon):
        action_history = plan_actions[:, : step + 1]
        action_emb = model.action_encoder(action_history)
        emb_window = latent_history[:, -max_history:]
        act_window = action_emb[:, -max_history:]
        if emb_window.shape[1] != act_window.shape[1]:
            raise RuntimeError(
                "Planner-context latent/action windows differ: "
                f"{emb_window.shape=} {act_window.shape=}."
            )
        next_emb = model.predict(emb_window, act_window)[:, -1:]
        predicted.append(next_emb)
        latent_history = torch.cat([latent_history, next_emb], dim=1)
    return torch.cat(predicted, dim=1)


def _paired_planner_context_predictions(
    student,
    teacher,
    current_emb,
    normalized_actions,
    action_mean,
    action_std,
    history_size,
    horizon,
    predictor_max_history,
    radius,
    directions_per_position,
    probes_per_chunk,
):
    """Matched student/teacher MH probes under official planner context."""
    if probes_per_chunk < 1:
        raise ValueError(
            f"mh_ald.probes_per_chunk must be >=1, got {probes_per_chunk}."
        )

    plus_actions, minus_actions, effective_radius, positions = (
        _sample_blockwise_probe_actions(
            normalized_actions=normalized_actions,
            action_mean=action_mean,
            action_std=action_std,
            history_size=history_size,
            horizon=horizon,
            radius=radius,
            directions_per_position=directions_per_position,
        )
    )

    # Dataset/action layout is unchanged from formal MH-ALD.  The active plan
    # begins at action block history_size-1, corresponding to the current frame.
    plan_start = int(history_size) - 1
    plan_stop = plan_start + int(horizon)
    if normalized_actions.shape[1] < plan_stop:
        raise ValueError(
            f"Need plan action blocks [{plan_start}:{plan_stop}], got "
            f"{normalized_actions.shape[1]} blocks."
        )

    b, d = current_emb.shape
    probe_count = plus_actions.shape[1]
    student_plus_all, student_minus_all = [], []
    teacher_plus_all, teacher_minus_all = [], []

    student.eval()
    teacher.eval()

    for start in range(0, int(probe_count), int(probes_per_chunk)):
        stop = min(start + int(probes_per_chunk), int(probe_count))
        count = stop - start
        candidate_count = 2 * count
        action_chunk = torch.cat(
            [plus_actions[:, start:stop], minus_actions[:, start:stop]], dim=1
        )
        plan_chunk = action_chunk[:, :, plan_start:plan_stop]
        flat_plan = plan_chunk.reshape(
            b * candidate_count, int(horizon), plan_chunk.shape[-1]
        )
        flat_current = _repeat_for_candidates(current_emb.detach(), candidate_count)

        student_rollout = planner_context_rollout(
            student,
            flat_current,
            flat_plan,
            predictor_max_history=predictor_max_history,
        ).reshape(b, candidate_count, int(horizon), d)
        student_plus_all.append(student_rollout[:, :count])
        student_minus_all.append(student_rollout[:, count:])

        with torch.no_grad():
            teacher_rollout = planner_context_rollout(
                teacher,
                flat_current,
                flat_plan,
                predictor_max_history=predictor_max_history,
            ).reshape(b, candidate_count, int(horizon), d)
            teacher_plus_all.append(teacher_rollout[:, :count].detach())
            teacher_minus_all.append(teacher_rollout[:, count:].detach())

    return (
        torch.cat(student_plus_all, dim=1),
        torch.cat(student_minus_all, dim=1),
        torch.cat(teacher_plus_all, dim=1),
        torch.cat(teacher_minus_all, dim=1),
        effective_radius,
        positions,
    )


def pc_mh_ald_forward(self, batch, stage, cfg, action_mean, action_std):
    """TF + planner-context rollout + pure causal MH-ALD."""
    if not hasattr(self, "teacher_model"):
        raise RuntimeError("PC-MH module has no frozen teacher_model.")

    student = self.model
    teacher = self.teacher_model
    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    horizon = int(cfg.mh_ald.rollout_horizon)
    pc_cfg = cfg.mh_ald.planner_context
    max_history = int(pc_cfg.predictor_max_history)

    if not bool(pc_cfg.enabled):
        raise ValueError("PC-MH requires mh_ald.planner_context.enabled=True.")
    if int(pc_cfg.observation_history) != 1:
        raise ValueError("Formal PushT planner observation history must be 1.")
    if max_history != ctx_len:
        raise ValueError(
            "Controlled PC-MH experiment keeps predictor max history equal to "
            f"wm.history_size: {max_history=} vs {ctx_len=}."
        )
    if n_preds != 1:
        raise ValueError(f"PC-MH assumes wm.num_preds=1, got {n_preds}.")
    if horizon < 1:
        raise ValueError(f"mh_ald.rollout_horizon must be >=1, got {horizon}.")

    student.eval()
    teacher.eval()
    student.encoder.eval()
    student.projector.eval()
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    if batch["pixels"].shape[1] < ctx_len + horizon:
        raise ValueError(
            "PC-MH sequence too short: need at least "
            f"{ctx_len + horizon}, got {batch['pixels'].shape[1]}."
        )

    emb = _frozen_visual_encode(student, batch)
    rollout_target = emb[:, ctx_len : ctx_len + horizon].detach()
    current_emb = emb[:, ctx_len - 1].detach()

    plan_start = ctx_len - 1
    plan_stop = plan_start + horizon
    demo_plan = batch["action"][:, plan_start:plan_stop]
    if demo_plan.shape[1] != horizon:
        raise RuntimeError(
            f"Planner-context demo plan has {demo_plan.shape[1]} blocks, "
            f"expected {horizon}."
        )

    expected_action_dim = student.action_encoder.patch_embed.in_channels
    if batch["action"].shape[-1] != expected_action_dim:
        raise RuntimeError(
            "Packed action dimension mismatch: "
            f"batch={batch['action'].shape[-1]} student={expected_action_dim}."
        )

    # One-step teacher forcing is intentionally IDENTICAL to formal MH-ALD.
    ctx_emb = emb[:, :ctx_len]
    ctx_action = batch["action"][:, :ctx_len]
    ctx_act_emb = student.action_encoder(ctx_action)
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]
    pred_emb = student.predict(ctx_emb, ctx_act_emb)
    tf_loss = (pred_emb - tgt_emb).pow(2).mean()

    # Only this rollout recursion changes relative to formal MH-ALD.
    student_center_rollout = planner_context_rollout(
        student,
        current_emb,
        demo_plan,
        predictor_max_history=max_history,
    )
    rollout_loss = (student_center_rollout - rollout_target).pow(2).mean()

    with torch.no_grad():
        teacher_center_rollout = planner_context_rollout(
            teacher,
            current_emb,
            demo_plan,
            predictor_max_history=max_history,
        ).detach()

    student_endpoint_mse = (
        student_center_rollout[:, -1] - rollout_target[:, -1]
    ).pow(2).mean()
    teacher_endpoint_mse = (
        teacher_center_rollout[:, -1] - rollout_target[:, -1]
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
    mh_loss = 0.5 * (
        _mh_masked_mse(plus_err, mask)
        + _mh_masked_mse(minus_err, mask)
    )

    # Same exact midpoint / response decomposition as formal MH-ALD.
    student_mid = 0.5 * (student_plus + student_minus)
    target_mid = 0.5 * (target_plus + target_minus)
    student_half_response = 0.5 * (student_plus - student_minus)
    target_half_response = 0.5 * (target_plus - target_minus)
    midpoint_loss = _mh_masked_mse(student_mid - target_mid, mask)
    half_response_loss = _mh_masked_mse(
        student_half_response - target_half_response, mask
    )
    decomposition_error = (
        mh_loss.detach() - (midpoint_loss + half_response_loss).detach()
    ).abs()

    center_error = teacher_center - anchors
    teacher_center_masked_mse = _mh_masked_mse(
        center_error.expand(-1, positions.numel(), -1, -1), mask
    )
    init_equivalence_ratio = (
        mh_loss.detach()
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
        + float(cfg.mh_ald.weight) * mh_loss
    )

    output = {
        "loss": total_loss,
        "tf_loss": tf_loss,
        "pc_mh_rollout_loss": rollout_loss,
        "pc_mh_loss": mh_loss,
        "pc_mh_midpoint_loss": midpoint_loss,
        "pc_mh_half_response_loss": half_response_loss,
    }
    losses = {
        f"{stage}/{key}": value.detach()
        for key, value in output.items()
        if "loss" in key
    }
    diagnostics = {
        f"{stage}/pc_mh_student_endpoint_mse": student_endpoint_mse.detach(),
        f"{stage}/pc_mh_teacher_endpoint_mse": teacher_endpoint_mse.detach(),
        f"{stage}/pc_mh_teacher_center_masked_mse": (
            teacher_center_masked_mse.detach()
        ),
        f"{stage}/pc_mh_decomposition_error": decomposition_error,
        f"{stage}/pc_mh_init_equivalence_ratio": init_equivalence_ratio,
        f"{stage}/pc_mh_response_cosine": response_cosine.detach(),
        f"{stage}/pc_mh_response_gain": response_gain.detach(),
        f"{stage}/pc_mh_effective_radius": effective_radius.mean().detach(),
        f"{stage}/pc_mh_probe_count": torch.tensor(
            float(positions.numel()), device=mh_loss.device
        ),
    }

    for h in range(horizon):
        hmask = mask[:, :, h : h + 1, :]
        h_plus = _mh_masked_mse(plus_err[:, :, h : h + 1], hmask)
        h_minus = _mh_masked_mse(minus_err[:, :, h : h + 1], hmask)
        diagnostics[f"{stage}/pc_mh_h{h + 1}_loss"] = (
            0.5 * (h_plus + h_minus)
        ).detach()

    self.log_dict(losses, on_step=True, sync_dist=True)
    self.log_dict(diagnostics, on_step=True, sync_dist=True)
    return output
