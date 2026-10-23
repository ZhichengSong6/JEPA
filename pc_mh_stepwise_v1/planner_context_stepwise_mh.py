"""Stepwise observation-anchored PC-MH targets (experimental, training only).

Base: ZhichengSong6/JEPA, 851b60c06ee0aef6927d59f85763cb03a3aede34.
Only the construction of the synthetic teacher targets is changed.  The
student TF/rollout branches, probe sampler, causal mask and weights are reused.

For a factual visual trajectory z[0:H+1] and factual actions u[0:H],
  q[k]       = T(z[max(0,k+1-L):k+1], u[max(0,k+1-L):k+1])[-1]
  y[0]       = z[0]
  y[k+1]     = z[k+1] + T(y-history, u'-history)[-1] - q[k].
Both histories warm 1->2->3.  Only target generation sees the factual future.
The student always rolls out from a SINGLE initial observation, with no resets.
Targets are model-derived local approximations, NOT simulator ground truth.

Do not expect the old PC-MH initial response-loss=0 invariant: stepwise
anchoring deliberately changes the multi-step target response.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_trajectory(observed, demo_actions, max_history):
    if observed.ndim != 3 or demo_actions.ndim != 3:
        raise ValueError("Expected observed [B,H+1,D] and demo_actions [B,H,A].")
    if observed.shape[0] != demo_actions.shape[0]:
        raise ValueError("Observation/action batch dimensions differ.")
    if demo_actions.shape[1] < 1 or observed.shape[1] != demo_actions.shape[1] + 1:
        raise ValueError("Exactly H+1 factual observations and H actions are required.")
    if int(max_history) < 1:
        raise ValueError("max_history must be positive.")
    if observed.device != demo_actions.device:
        raise ValueError("Observations/actions must be on the same device.")


@torch.no_grad()
def factual_teacher_predictions(teacher, observed, demo_actions, max_history=3):
    """Predict each factual next frame using 1->2->3 TRUE-past context.

    Returns [B,H,D]; at step k, z[k+1] is not a teacher input.  Its observed
    value is only used later as the anchor.  No pre-start frames are supplied.
    """
    _validate_trajectory(observed, demo_actions, max_history)
    teacher.eval()
    observed, demo_actions = observed.detach(), demo_actions.detach()
    reference = []
    for k in range(demo_actions.shape[1]):
        left = max(0, k + 1 - int(max_history))
        action_emb = teacher.action_encoder(demo_actions[:, : k + 1])
        prediction = teacher.predict(
            observed[:, left : k + 1], action_emb[:, left : k + 1]
        )[:, -1:]
        if prediction.shape != observed[:, k + 1 : k + 2].shape:
            raise RuntimeError("Teacher output and observation latent shapes differ.")
        reference.append(prediction)
    return torch.cat(reference, dim=1).detach()


@torch.no_grad()
def stepwise_observation_targets(
    teacher, observed, demo_actions, probe_actions, max_history=3,
    reference_predictions=None,
):
    """Return [B,H,D] recursively anchored targets, without building a graph.

    reference_predictions can be precomputed once per batch and repeated for
    multiple probes.  They must belong to the SAME factual trajectory.
    There is no state-delta clipping, horizon reweighting, teacher updating,
    goal cost, inverse head, or use of the student in target construction.
    """
    _validate_trajectory(observed, demo_actions, max_history)
    if probe_actions.shape != demo_actions.shape:
        raise ValueError("probe_actions must have the same [B,H,A] shape as demo_actions.")
    if probe_actions.device != observed.device:
        raise ValueError("Probe actions must share the observation device.")
    teacher.eval()
    observed = observed.detach()
    demo_actions, probe_actions = demo_actions.detach(), probe_actions.detach()
    if reference_predictions is None:
        reference_predictions = factual_teacher_predictions(
            teacher, observed, demo_actions, max_history
        )
    if reference_predictions.shape != observed[:, 1:].shape:
        raise ValueError("reference_predictions must have shape [B,H,D].")
    if reference_predictions.device != observed.device:
        raise ValueError("Reference predictions must share the observation device.")
    reference_predictions = reference_predictions.detach()
    history = observed[:, :1]
    targets = []
    for k in range(probe_actions.shape[1]):
        left = max(0, k + 1 - int(max_history))
        action_emb = teacher.action_encoder(probe_actions[:, : k + 1])
        probe_prediction = teacher.predict(
            history[:, left : k + 1], action_emb[:, left : k + 1]
        )[:, -1:]
        # Subtract predictions before adding the anchor, as in original ALD.
        target = observed[:, k + 1 : k + 2] + (
            probe_prediction - reference_predictions[:, k : k + 1]
        )
        targets.append(target)
        history = torch.cat([history, target], dim=1)
    return torch.cat(targets, dim=1).detach()


def _paired_stepwise_predictions(
    student, teacher, observed, normalized_actions, action_mean, action_std,
    history_size, horizon, predictor_max_history, radius,
    directions_per_position, probes_per_chunk,
):
    # Reuse the ORIGINAL sampler and student recursion, without reimplementing
    # normalization, raw-action bounds, perturbation radius or candidate order.
    from anchored_local_dynamics import (
        _repeat_for_candidates, _sample_blockwise_probe_actions,
    )
    from planner_context_mh import planner_context_rollout

    if int(probes_per_chunk) < 1:
        raise ValueError("probes_per_chunk must be positive.")
    start = int(history_size) - 1
    stop = start + int(horizon)
    demo_plan = normalized_actions[:, start:stop]
    _validate_trajectory(observed, demo_plan, predictor_max_history)
    if demo_plan.shape[1] != int(horizon):
        raise ValueError("Insufficient action blocks for the requested horizon.")
    plus, minus, effective_radius, positions = _sample_blockwise_probe_actions(
        normalized_actions=normalized_actions,
        action_mean=action_mean, action_std=action_std,
        history_size=history_size, horizon=horizon, radius=radius,
        directions_per_position=directions_per_position,
    )
    student.eval()
    teacher.eval()
    references = factual_teacher_predictions(
        teacher, observed, demo_plan, predictor_max_history
    )
    batch_size, _, latent_dim = observed.shape
    sp, sm, yp, ym = [], [], [], []
    for p in range(0, plus.shape[1], int(probes_per_chunk)):
        end = min(p + int(probes_per_chunk), plus.shape[1])
        count = end - p
        nc = 2 * count
        actions = torch.cat([plus[:, p:end], minus[:, p:end]], dim=1)
        flat_plan = actions[:, :, start:stop].reshape(
            batch_size * nc, int(horizon), demo_plan.shape[-1]
        )
        flat_initial = _repeat_for_candidates(observed[:, 0].detach(), nc)
        student_rollout = planner_context_rollout(
            student, flat_initial, flat_plan, predictor_max_history
        ).reshape(batch_size, nc, int(horizon), latent_dim)
        sp.append(student_rollout[:, :count])
        sm.append(student_rollout[:, count:])
        target = stepwise_observation_targets(
            teacher,
            _repeat_for_candidates(observed.detach(), nc),
            _repeat_for_candidates(demo_plan.detach(), nc),
            flat_plan,
            max_history=predictor_max_history,
            reference_predictions=_repeat_for_candidates(references, nc),
        ).reshape(batch_size, nc, int(horizon), latent_dim)
        yp.append(target[:, :count])
        ym.append(target[:, count:])
    return (
        torch.cat(sp, 1), torch.cat(sm, 1), torch.cat(yp, 1), torch.cat(ym, 1),
        effective_radius, positions, references,
    )


def pc_stepwise_mh_forward(self, batch, stage, cfg, action_mean, action_std):
    """Original TF + original PC demo rollout + NEW stepwise-target MH loss."""
    from anchored_local_dynamics import (
        _frozen_visual_encode, _mh_causal_mask, _mh_masked_mean, _mh_masked_mse,
    )
    from planner_context_mh import planner_context_rollout

    if not hasattr(self, "teacher_model"):
        raise RuntimeError("Missing frozen teacher_model.")
    student, teacher = self.model, self.teacher_model
    ctx = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    horizon = int(cfg.mh_ald.rollout_horizon)
    pc = cfg.mh_ald.planner_context
    max_history = int(pc.predictor_max_history)
    if not bool(pc.enabled) or int(pc.observation_history) != 1:
        raise ValueError("This experiment requires a single observed initial latent.")
    if max_history != ctx or n_preds != 1 or horizon < 1:
        raise ValueError("Expected max_history=wm.history_size, num_preds=1, H>=1.")
    student.eval()
    teacher.eval()
    student.encoder.eval()
    student.projector.eval()
    # Keep the baseline's invalid-action handling for this controlled run.
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    if batch["pixels"].shape[1] < ctx + horizon:
        raise ValueError("Not enough observation frames for the requested horizon.")
    emb = _frozen_visual_encode(student, batch)
    observed = emb[:, ctx - 1 : ctx + horizon].detach()
    rollout_target = emb[:, ctx : ctx + horizon].detach()
    current_emb = emb[:, ctx - 1].detach()
    plan_start = ctx - 1
    demo_plan = batch["action"][:, plan_start : plan_start + horizon]
    if demo_plan.shape[1] != horizon:
        raise RuntimeError("Insufficient demonstrated action blocks.")
    if batch["action"].shape[-1] != student.action_encoder.patch_embed.in_channels:
        raise RuntimeError("Packed action dimension mismatch.")

    # These two branches are IDENTICAL to planner_context_mh.pc_mh_ald_forward.
    ctx_emb = emb[:, :ctx]
    ctx_action = batch["action"][:, :ctx]
    ctx_act_emb = student.action_encoder(ctx_action)
    tgt_emb = emb[:, n_preds : n_preds + ctx]
    pred_emb = student.predict(ctx_emb, ctx_act_emb)
    tf_loss = (pred_emb - tgt_emb).pow(2).mean()
    center = planner_context_rollout(student, current_emb, demo_plan, max_history)
    rollout_loss = (center - rollout_target).pow(2).mean()

    # Raw teacher center is a diagnostic only, never a stepwise target input.
    with torch.no_grad():
        teacher_center = planner_context_rollout(
            teacher, current_emb, demo_plan, max_history
        ).detach()
    sp, sm, yp, ym, radii, positions, references = _paired_stepwise_predictions(
        student, teacher, observed, batch["action"], action_mean, action_std,
        ctx, horizon, max_history, float(cfg.mh_ald.perturb_radius),
        int(cfg.mh_ald.directions_per_position), int(cfg.mh_ald.probes_per_chunk),
    )
    mask = _mh_causal_mask(positions, horizon, dtype=sp.dtype, device=sp.device)
    ep, em = sp - yp, sm - ym
    mh_loss = 0.5 * (_mh_masked_mse(ep, mask) + _mh_masked_mse(em, mask))
    midpoint_loss = _mh_masked_mse(0.5 * (sp + sm) - 0.5 * (yp + ym), mask)
    half_response_loss = _mh_masked_mse(0.5 * (sp - sm) - 0.5 * (yp - ym), mask)
    total = (
        float(cfg.mh_ald.tf_weight) * tf_loss
        + float(cfg.mh_ald.rollout_weight) * rollout_loss
        + float(cfg.mh_ald.weight) * mh_loss
    )
    output = {
        "loss": total, "tf_loss": tf_loss,
        "pc_stepwise_rollout_loss": rollout_loss,
        "pc_stepwise_mh_loss": mh_loss,
        "pc_stepwise_midpoint_loss": midpoint_loss,
        "pc_stepwise_half_response_loss": half_response_loss,
    }
    # Compute diagnostics without retaining any student graph. No new loss term.
    with torch.no_grad():
        denom = (2.0 * radii).clamp_min(1e-8)[:, :, None, None]
        sr = (sp.detach() - sm.detach()) / denom
        yr = (yp - ym) / denom
        mask3 = mask[..., 0]
        cosine = _mh_masked_mean(F.cosine_similarity(sr, yr, dim=-1, eps=1e-8), mask3)
        gain = _mh_masked_mean(
            torch.linalg.vector_norm(sr, dim=-1)
            / torch.linalg.vector_norm(yr, dim=-1).clamp_min(1e-8), mask3,
        )
        logs = {
            f"{stage}/pc_stepwise_student_endpoint_mse": (center[:, -1] - rollout_target[:, -1]).pow(2).mean(),
            f"{stage}/pc_stepwise_raw_teacher_endpoint_mse": (teacher_center[:, -1] - rollout_target[:, -1]).pow(2).mean(),
            f"{stage}/pc_stepwise_factual_teacher_mse": (references - rollout_target).pow(2).mean(),
            f"{stage}/pc_stepwise_decomposition_error": (mh_loss - midpoint_loss - half_response_loss).abs(),
            f"{stage}/pc_stepwise_response_cosine": cosine,
            f"{stage}/pc_stepwise_response_gain": gain,
            f"{stage}/pc_stepwise_effective_radius": radii.mean(),
            f"{stage}/pc_stepwise_probe_count": total.new_tensor(float(positions.numel())),
        }
        for h in range(horizon):
            hm = mask[:, :, h : h + 1]
            logs[f"{stage}/pc_stepwise_h{h+1}_loss"] = 0.5 * (
                _mh_masked_mse(ep[:, :, h : h + 1], hm)
                + _mh_masked_mse(em[:, :, h : h + 1], hm)
            )
    self.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items()},
                  on_step=True, sync_dist=True)
    self.log_dict({k: v.detach() for k, v in logs.items()}, on_step=True, sync_dist=True)
    return output
