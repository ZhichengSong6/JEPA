"""Stage II v2: bias suppression with direct action-response preservation.

This objective is the controlled follow-up to the failed odd+Gram Stage-II
experiment.  The previous objective could reduce odd cost by shrinking the
student's action-conditioned response, while its Gram loss had vanishing
restoring gradient as the student response approached zero.

Stage II v2 therefore uses

    L = L_LeWM + L_rollout + L_APB + L_JVP

where

* APB penalizes terminal rollout bias projected onto DETACHED UNIT student
  action-response directions.  Shrinking response magnitude does not directly
  reduce this term.
* JVP directly matches the student's local action-response vector to a frozen
  official LeWM teacher.  At zero student response its gradient remains
  non-zero whenever the teacher response is non-zero.

No simulator state, counterfactual ground-truth rollout, physical factor, or
planning-time oracle is used.  +/- action probes are synthetic predictor probes
around demonstrated action sequences.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from distributed_training import global_batch_sigreg
from stage1_bias_calibration import (
    _autoregressive_rollout,
    _denormalize_actions,
    _normalize_raw_actions,
    _sample_raw_action_perturbation,
)


def _repeat_for_candidates(x: torch.Tensor, count: int) -> torch.Tensor:
    """Repeat [B,...] along a candidate axis and flatten to [B*C,...]."""
    if count < 1:
        raise ValueError(f"candidate count must be positive, got {count}")
    b = x.shape[0]
    expanded = x[:, None].expand(b, count, *x.shape[1:])
    return expanded.reshape(b * count, *x.shape[1:])


def _sample_probe_actions(
    normalized_actions,
    action_mean,
    action_std,
    history_size,
    horizon,
    radius,
    first_only,
    num_directions,
):
    """Sample K exact bounded symmetric perturbations in raw PushT action space."""
    if radius <= 0:
        raise ValueError(f"stage2_v2.perturb_radius must be > 0, got {radius}.")
    if num_directions < 1:
        raise ValueError(
            f"stage2_v2.num_directions must be >= 1, got {num_directions}."
        )

    raw_actions = _denormalize_actions(
        normalized_actions.detach(), action_mean, action_std
    )

    plus_actions = []
    minus_actions = []
    radii = []
    for _ in range(int(num_directions)):
        plus_raw, minus_raw, effective_radius = _sample_raw_action_perturbation(
            raw_actions,
            history_size=history_size,
            horizon=horizon,
            radius=radius,
            first_only=first_only,
        )
        plus_actions.append(
            _normalize_raw_actions(plus_raw, action_mean, action_std)
        )
        minus_actions.append(
            _normalize_raw_actions(minus_raw, action_mean, action_std)
        )
        radii.append(effective_radius)

    return (
        torch.stack(plus_actions, dim=1),   # [B,K,T,A]
        torch.stack(minus_actions, dim=1),  # [B,K,T,A]
        torch.stack(radii, dim=1),          # [B,K]
    )


def _student_teacher_jvps(
    student,
    teacher,
    student_emb,
    teacher_emb,
    normalized_actions,
    action_mean,
    action_std,
    history_size,
    horizon,
    radius,
    first_only,
    num_directions,
    directions_per_chunk,
):
    """Compute paired student/teacher finite-difference local responses.

    Important implementation detail:
    synthetic student probes run with ``student.eval()`` so BatchNorm running
    statistics and dropout are not changed by counterfactual +/- candidate
    batches.  Gradients are still enabled for the student probe path, so JVP
    matching directly trains the predictor/action-conditioned mapping.

    Initial student embeddings are detached for this probe path.  The base
    one-step and demonstrated rollout losses remain responsible for learning
    the encoder representation; JVP preservation acts directly on the local
    action-conditioned predictor response.
    """
    if directions_per_chunk < 1:
        raise ValueError(
            "stage2_v2.probe_directions_per_chunk must be >= 1, got "
            f"{directions_per_chunk}."
        )

    plus_actions, minus_actions, effective_radius = _sample_probe_actions(
        normalized_actions=normalized_actions,
        action_mean=action_mean,
        action_std=action_std,
        history_size=history_size,
        horizon=horizon,
        radius=radius,
        first_only=first_only,
        num_directions=num_directions,
    )

    batch_size = student_emb.shape[0]
    latent_dim = student_emb.shape[-1]
    student_responses = []
    teacher_responses = []

    student_was_training = student.training
    student.eval()
    teacher.eval()

    try:
        for start in range(0, int(num_directions), int(directions_per_chunk)):
            stop = min(start + int(directions_per_chunk), int(num_directions))
            c = stop - start
            candidate_count = 2 * c

            # Candidate order: all plus directions, then all minus directions.
            action_chunk = torch.cat(
                [plus_actions[:, start:stop], minus_actions[:, start:stop]], dim=1
            )
            flat_actions = action_chunk.reshape(
                batch_size * candidate_count,
                *action_chunk.shape[2:],
            )

            flat_student_emb = _repeat_for_candidates(
                student_emb.detach(), candidate_count
            )
            student_endpoint = _autoregressive_rollout(
                student,
                flat_student_emb,
                flat_actions,
                history_size,
                horizon,
            )[:, -1]
            student_endpoint = student_endpoint.reshape(
                batch_size, candidate_count, latent_dim
            )
            student_plus = student_endpoint[:, :c]
            student_minus = student_endpoint[:, c:]

            radius_chunk = effective_radius[:, start:stop]
            denom = (2.0 * radius_chunk).clamp_min(1e-8)[..., None]
            student_response = (student_plus - student_minus) / denom

            with torch.no_grad():
                flat_teacher_emb = _repeat_for_candidates(
                    teacher_emb.detach(), candidate_count
                )
                teacher_endpoint = _autoregressive_rollout(
                    teacher,
                    flat_teacher_emb,
                    flat_actions,
                    history_size,
                    horizon,
                )[:, -1]
                teacher_endpoint = teacher_endpoint.reshape(
                    batch_size, candidate_count, latent_dim
                )
                teacher_plus = teacher_endpoint[:, :c]
                teacher_minus = teacher_endpoint[:, c:]
                teacher_response = (teacher_plus - teacher_minus) / denom

            student_responses.append(student_response)
            teacher_responses.append(teacher_response.detach())
    finally:
        student.train(student_was_training)
        teacher.eval()

    student_response = torch.cat(student_responses, dim=1)
    teacher_response = torch.cat(teacher_responses, dim=1)

    if student_response.shape[1] != int(num_directions):
        raise RuntimeError(
            "Stage-II-v2 probe bookkeeping error: "
            f"expected K={num_directions}, got {student_response.shape[1]}."
        )

    return student_response, teacher_response, effective_radius


def _relative_jvp_loss(student_response, teacher_response, eps: float):
    """Direct relative response-vector matching with nonzero anti-collapse gradient."""
    teacher_response = teacher_response.detach()
    diff_energy = (student_response - teacher_response).pow(2).mean(dim=-1)
    teacher_energy = teacher_response.pow(2).mean(dim=-1).clamp_min(float(eps))
    return (diff_energy / teacher_energy).mean()


def stage2_v2_forward(self, batch, stage, cfg, action_mean, action_std):
    """Official LeWM + H-step rollout + normalized APB + direct teacher JVP."""
    if not hasattr(self, "teacher_model"):
        raise RuntimeError(
            "Stage-II-v2 module has no frozen teacher_model. Use train_stage2_v2.py."
        )

    student = self.model
    teacher = self.teacher_model
    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    horizon = int(cfg.stage2_v2.rollout_horizon)
    sigreg_weight = float(cfg.loss.sigreg.weight)

    if n_preds != 1:
        raise ValueError(
            "Stage-II-v2 assumes official LeWM wm.num_preds=1, "
            f"got {n_preds}."
        )
    if horizon < 1:
        raise ValueError(
            f"stage2_v2.rollout_horizon must be >= 1, got {horizon}."
        )

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # ------------------------------------------------------------------
    # 1) Unchanged official one-step LeWM objective.
    # ------------------------------------------------------------------
    output = student.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    expected_action_dim = student.action_encoder.patch_embed.in_channels
    if batch["action"].shape[-1] != expected_action_dim:
        raise RuntimeError(
            "Packed action dimension mismatch: "
            f"batch={batch['action'].shape[-1]} student={expected_action_dim}."
        )
    if emb.shape[1] < ctx_len + horizon:
        raise ValueError(
            "Stage-II-v2 sequence too short: need at least "
            f"{ctx_len + horizon} states, got {emb.shape[1]}. "
            "Use data=pusht_stage2_v2."
        )

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]
    pred_emb = student.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_emb = emb[:, : ctx_len + n_preds]
    output["sigreg_loss"] = global_batch_sigreg(
        self.sigreg,
        sigreg_emb.transpose(0, 1),
        enabled=bool(cfg.loss.sigreg.get("global_batch_ddp", False)),
    )
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]

    # ------------------------------------------------------------------
    # 2) Demonstrated H-step autoregressive rollout supervision.
    # ------------------------------------------------------------------
    rollout_pred = _autoregressive_rollout(
        student,
        emb,
        batch["action"],
        history_size=ctx_len,
        horizon=horizon,
    )
    rollout_target = emb[:, ctx_len : ctx_len + horizon].detach()
    rollout_error = rollout_pred - rollout_target
    rollout_loss = rollout_error.pow(2).mean()
    endpoint_error = rollout_error[:, -1]

    output["stage2_v2_rollout_loss"] = rollout_loss
    output["stage2_v2_endpoint_mse"] = endpoint_error.pow(2).mean().detach()

    # ------------------------------------------------------------------
    # 3) Paired synthetic local responses from student and frozen teacher.
    # ------------------------------------------------------------------
    teacher.eval()
    with torch.no_grad():
        teacher_output = teacher.encode(batch)
        teacher_emb = teacher_output["emb"].detach()

    student_response, teacher_response, effective_radius = _student_teacher_jvps(
        student=student,
        teacher=teacher,
        student_emb=emb,
        teacher_emb=teacher_emb,
        normalized_actions=batch["action"],
        action_mean=action_mean,
        action_std=action_std,
        history_size=ctx_len,
        horizon=horizon,
        radius=float(cfg.stage2_v2.perturb_radius),
        first_only=bool(cfg.stage2_v2.perturb_first_only),
        num_directions=int(cfg.stage2_v2.num_directions),
        directions_per_chunk=int(cfg.stage2_v2.probe_directions_per_chunk),
    )

    # ------------------------------------------------------------------
    # 4) Scale-invariant APB: only the UNIT response direction is used, and
    #    that direction is detached.  Shrinking response magnitude cannot
    #    directly reduce this term.
    # ------------------------------------------------------------------
    response_dir = F.normalize(
        student_response.detach(), dim=-1, eps=float(cfg.stage2_v2.response_eps)
    )
    projected_bias = torch.einsum("bd,bkd->bk", endpoint_error, response_dir)
    raw_apb = projected_bias.pow(2).mean()
    latent_dim = endpoint_error.shape[-1]
    apb_loss = raw_apb / float(latent_dim)

    output["stage2_v2_apb_raw"] = raw_apb.detach()
    output["stage2_v2_apb_loss"] = apb_loss
    output["stage2_v2_projected_bias_abs"] = projected_bias.abs().mean().detach()

    # ------------------------------------------------------------------
    # 5) Direct JVP preservation.  Unlike Gram matching, at student_response=0
    #    this objective still has a nonzero gradient toward teacher_response.
    # ------------------------------------------------------------------
    jvp_loss = _relative_jvp_loss(
        student_response,
        teacher_response,
        eps=float(cfg.stage2_v2.jvp_normalization_eps),
    )
    output["stage2_v2_jvp_loss"] = jvp_loss

    student_norm = torch.linalg.vector_norm(student_response, dim=-1)
    teacher_norm = torch.linalg.vector_norm(teacher_response, dim=-1)
    response_cos = F.cosine_similarity(
        student_response.detach(), teacher_response.detach(), dim=-1, eps=1e-8
    )
    response_gain = student_norm.detach() / teacher_norm.detach().clamp_min(1e-8)

    output["loss"] = (
        output["loss"]
        + float(cfg.stage2_v2.rollout_weight) * rollout_loss
        + float(cfg.stage2_v2.apb_weight) * apb_loss
        + float(cfg.stage2_v2.jvp_weight) * jvp_loss
    )

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    diagnostics_dict = {
        f"{stage}/stage2_v2_endpoint_mse": output["stage2_v2_endpoint_mse"],
        f"{stage}/stage2_v2_apb_raw": output["stage2_v2_apb_raw"],
        f"{stage}/stage2_v2_projected_bias_abs": output[
            "stage2_v2_projected_bias_abs"
        ],
        f"{stage}/stage2_v2_student_response_norm": student_norm.mean().detach(),
        f"{stage}/stage2_v2_teacher_response_norm": teacher_norm.mean().detach(),
        f"{stage}/stage2_v2_response_cosine": response_cos.mean().detach(),
        f"{stage}/stage2_v2_response_gain": response_gain.mean().detach(),
        f"{stage}/stage2_v2_effective_radius": effective_radius.mean().detach(),
    }

    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    self.log_dict(diagnostics_dict, on_step=True, sync_dist=True)
    return output
