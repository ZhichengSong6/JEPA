"""Stage II: goal-relative landscape-faithful JEPA objective.

The mechanism established by the PushT diagnostics is:

* around a nominal action sequence whose real endpoint is the target endpoint,
  the true local goal-cost landscape becomes approximately even / curvature
  dominated;
* rollout endpoint bias injects a spurious odd (first-order) component into the
  predicted latent goal cost;
* Stage-I APB suppresses this component, but can weaken useful action
  curvature.

Stage II therefore optimizes two planner-facing quantities directly:

1) odd-symmetry calibration around an observed future endpoint target;
2) preservation of the pretrained LeWM action-response Gram geometry.

No simulator state, counterfactual ground-truth rollout, physical factor, or
planning-time oracle is used. The +/- actions are synthetic predictor probes;
the only target is the observed offline future image embedding.
"""

from __future__ import annotations

import torch

from distributed_training import global_batch_sigreg
from stage1_bias_calibration import (
    _autoregressive_rollout,
    _denormalize_actions,
    _normalize_raw_actions,
    _sample_raw_action_perturbation,
)


def _relative_gram_loss(student_response, teacher_response, eps: float):
    """Relative sampled-Gram error, preserving shape and response scale.

    Response tensors are [B, K, D]. Dividing inner products by D only keeps the
    numerical scale independent of latent width; it cancels in the relative
    loss.
    """
    latent_dim = float(student_response.shape[-1])
    gs = torch.einsum("bkd,bjd->bkj", student_response, student_response) / latent_dim
    gt = torch.einsum("bkd,bjd->bkj", teacher_response, teacher_response) / latent_dim
    gt = gt.detach()

    num = (gs - gt).pow(2).sum(dim=(-2, -1))
    den = gt.pow(2).sum(dim=(-2, -1)).clamp_min(float(eps))
    return (num / den).mean(), gs, gt


def _repeat_for_candidates(x: torch.Tensor, count: int) -> torch.Tensor:
    """Repeat a batch tensor along a candidate axis, then flatten B*candidate."""
    if count < 1:
        raise ValueError(f"candidate count must be positive, got {count}")
    b = x.shape[0]
    expanded = x[:, None].expand(b, count, *x.shape[1:])
    return expanded.reshape(b * count, *x.shape[1:])


def _stage2_probe_losses(
    student,
    teacher,
    student_emb,
    teacher_emb,
    normalized_actions,
    student_goal,
    action_mean,
    action_std,
    history_size,
    horizon,
    radius,
    first_only,
    num_directions,
    directions_per_chunk,
    odd_eps,
    gram_eps,
):
    """Compute differentiable odd loss and teacher-Gram preservation.

    All K bounded raw-action perturbations are sampled first. Probe directions
    are then processed in vectorized chunks: for C directions, + and - probes
    are concatenated into a single 2*C candidate axis and rolled out in one
    student call and one frozen-teacher call. This preserves the exact Stage-II
    objective while reducing Python/model-call overhead and keeping peak memory
    bounded.
    """
    if radius <= 0:
        raise ValueError(f"stage2.perturb_radius must be > 0, got {radius}.")
    if num_directions < 2:
        raise ValueError(
            "stage2.num_directions must be >= 2 so sampled curvature has "
            f"cross-direction structure, got {num_directions}."
        )
    if directions_per_chunk < 1:
        raise ValueError(
            "stage2.probe_directions_per_chunk must be >=1, got "
            f"{directions_per_chunk}."
        )

    raw_actions = _denormalize_actions(
        normalized_actions.detach(), action_mean, action_std
    )

    # Sample all directions before model forward passes. Shape after stacking:
    # [B,K,T,A] for actions and [B,K] for exact effective radii.
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

    plus_actions = torch.stack(plus_actions, dim=1)
    minus_actions = torch.stack(minus_actions, dim=1)
    effective_radius = torch.stack(radii, dim=1)

    student_responses = []
    teacher_responses = []
    odd_increments = []
    teacher_quadratic = []

    teacher.eval()
    batch_size = student_emb.shape[0]
    latent_dim = student_goal.shape[-1]

    for start in range(0, int(num_directions), int(directions_per_chunk)):
        stop = min(start + int(directions_per_chunk), int(num_directions))
        c = stop - start

        # Candidate order is [all plus directions, all minus directions].
        action_chunk = torch.cat(
            [plus_actions[:, start:stop], minus_actions[:, start:stop]], dim=1
        )
        candidate_count = 2 * c
        flat_actions = action_chunk.reshape(
            batch_size * candidate_count,
            *action_chunk.shape[2:],
        )

        flat_student_emb = _repeat_for_candidates(student_emb, candidate_count)
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

        goal = student_goal[:, None, :]
        c_plus = (student_plus - goal).pow(2).mean(dim=-1)
        c_minus = (student_minus - goal).pow(2).mean(dim=-1)
        odd = 0.5 * (c_plus - c_minus)

        with torch.no_grad():
            flat_teacher_emb = _repeat_for_candidates(teacher_emb, candidate_count)
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

            # r^2 ||Jv||^2 / D: frozen teacher quadratic cost scale at the
            # actual perturbation radius along each sampled direction.
            teacher_q = (
                radius_chunk.pow(2)
                * teacher_response.pow(2).mean(dim=-1)
            )

        student_responses.append(student_response)
        teacher_responses.append(teacher_response)
        odd_increments.append(odd)
        teacher_quadratic.append(teacher_q)

    student_response = torch.cat(student_responses, dim=1)
    teacher_response = torch.cat(teacher_responses, dim=1)
    odd = torch.cat(odd_increments, dim=1)
    teacher_q = torch.cat(teacher_quadratic, dim=1)

    if student_response.shape[1] != int(num_directions):
        raise RuntimeError(
            "Vectorized Stage-II probe bookkeeping error: "
            f"expected K={num_directions}, got {student_response.shape[1]}."
        )

    # Dimensionless odd energy normalized by the FROZEN teacher's useful local
    # curvature energy. The denominator is detached, so the student cannot
    # lower L_odd by collapsing its own response magnitude.
    odd_energy = odd.pow(2).mean(dim=1)
    teacher_curv_energy = teacher_q.pow(2).mean(dim=1).detach()
    odd_loss = (odd_energy / teacher_curv_energy.clamp_min(float(odd_eps))).mean()

    curvature_loss, gs, gt = _relative_gram_loss(
        student_response, teacher_response, eps=gram_eps
    )

    diagnostics = {
        "odd_abs": odd.abs().mean().detach(),
        "odd_rms": odd.pow(2).mean().sqrt().detach(),
        "teacher_quadratic_rms": teacher_q.pow(2).mean().sqrt().detach(),
        "effective_radius": effective_radius.mean().detach(),
        "student_response_norm": torch.linalg.vector_norm(
            student_response, dim=-1
        ).mean().detach(),
        "teacher_response_norm": torch.linalg.vector_norm(
            teacher_response, dim=-1
        ).mean().detach(),
        "gram_student_fro": torch.linalg.matrix_norm(
            gs, ord="fro", dim=(-2, -1)
        ).mean().detach(),
        "gram_teacher_fro": torch.linalg.matrix_norm(
            gt, ord="fro", dim=(-2, -1)
        ).mean().detach(),
    }
    return odd_loss, curvature_loss, diagnostics


def stage2_forward(self, batch, stage, cfg, action_mean, action_std):
    """Official LeWM objective + rollout + odd symmetry + curvature retention."""
    if cfg.factor.enabled:
        raise ValueError(
            "Stage II intentionally excludes privileged-state factor supervision; "
            "set factor.enabled=False."
        )
    if cfg.stage1.enabled:
        raise ValueError(
            "Stage II first experiment must not mix the Stage-I APB objective. "
            "Set stage1.enabled=False."
        )
    if not hasattr(self, "teacher_model"):
        raise RuntimeError(
            "Stage-II Lightning module has no frozen teacher_model. "
            "Use train_stage2.py."
        )

    student = self.model
    teacher = self.teacher_model
    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    horizon = int(cfg.stage2.rollout_horizon)
    sigreg_weight = float(cfg.loss.sigreg.weight)

    if n_preds != 1:
        raise ValueError(
            "Stage-II implementation assumes official LeWM wm.num_preds=1, "
            f"got {n_preds}."
        )
    if horizon < 1:
        raise ValueError(f"stage2.rollout_horizon must be >=1, got {horizon}.")

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # ------------------------------------------------------------------
    # Student encoding and unchanged official one-step LeWM prediction loss.
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
            "Stage-II sequence too short: need at least "
            f"{ctx_len + horizon} states, got {emb.shape[1]}. "
            "Use data=pusht_stage2."
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
    # H-step autoregressive supervision.
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
    output["stage2_rollout_loss"] = rollout_loss
    output["stage2_endpoint_mse"] = rollout_error[:, -1].pow(2).mean().detach()

    # The observed H-step future endpoint is the LOCAL GOAL for symmetry.
    student_goal = emb[:, ctx_len + horizon - 1].detach()

    teacher.eval()
    with torch.no_grad():
        teacher_output = teacher.encode(batch)
        teacher_emb = teacher_output["emb"].detach()

    odd_loss, curvature_loss, diag = _stage2_probe_losses(
        student=student,
        teacher=teacher,
        student_emb=emb,
        teacher_emb=teacher_emb,
        normalized_actions=batch["action"],
        student_goal=student_goal,
        action_mean=action_mean,
        action_std=action_std,
        history_size=ctx_len,
        horizon=horizon,
        radius=float(cfg.stage2.perturb_radius),
        first_only=bool(cfg.stage2.perturb_first_only),
        num_directions=int(cfg.stage2.num_directions),
        directions_per_chunk=int(cfg.stage2.probe_directions_per_chunk),
        odd_eps=float(cfg.stage2.odd_normalization_eps),
        gram_eps=float(cfg.stage2.gram_normalization_eps),
    )

    output["stage2_odd_loss"] = odd_loss
    output["stage2_curvature_loss"] = curvature_loss
    output["loss"] = (
        output["loss"]
        + float(cfg.stage2.rollout_weight) * rollout_loss
        + float(cfg.stage2.odd_weight) * odd_loss
        + float(cfg.stage2.curvature_weight) * curvature_loss
    )

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    diagnostics_dict = {
        f"{stage}/stage2_endpoint_mse": output["stage2_endpoint_mse"],
        f"{stage}/stage2_odd_abs": diag["odd_abs"],
        f"{stage}/stage2_odd_rms": diag["odd_rms"],
        f"{stage}/stage2_teacher_quadratic_rms": diag["teacher_quadratic_rms"],
        f"{stage}/stage2_effective_radius": diag["effective_radius"],
        f"{stage}/stage2_student_response_norm": diag["student_response_norm"],
        f"{stage}/stage2_teacher_response_norm": diag["teacher_response_norm"],
        f"{stage}/stage2_gram_student_fro": diag["gram_student_fro"],
        f"{stage}/stage2_gram_teacher_fro": diag["gram_teacher_fro"],
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    self.log_dict(diagnostics_dict, on_step=True, sync_dist=True)
    return output
