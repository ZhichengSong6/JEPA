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
planning-time oracle is used.  The +/- actions are synthetic predictor probes;
the only target is the observed offline future image embedding.
"""

from __future__ import annotations

import torch

from stage1_bias_calibration import (
    _autoregressive_rollout,
    _denormalize_actions,
    _normalize_raw_actions,
    _sample_raw_action_perturbation,
)


def _relative_gram_loss(student_response, teacher_response, eps: float):
    """Relative sampled-Gram error, preserving both shape and response scale.

    response tensors are [B, K, D].  Dividing inner products by D only keeps
    the numerical scale independent of latent width; it cancels in the
    relative loss.
    """
    latent_dim = float(student_response.shape[-1])
    gs = torch.einsum("bkd,bjd->bkj", student_response, student_response) / latent_dim
    gt = torch.einsum("bkd,bjd->bkj", teacher_response, teacher_response) / latent_dim
    gt = gt.detach()

    num = (gs - gt).pow(2).sum(dim=(-2, -1))
    den = gt.pow(2).sum(dim=(-2, -1)).clamp_min(float(eps))
    return (num / den).mean(), gs, gt


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
    odd_eps,
    gram_eps,
):
    """Compute differentiable student odd loss and teacher-Gram preservation.

    The SAME bounded raw-action +/- probes are sent through student and frozen
    teacher.  Student probe rollouts retain gradients; teacher rollouts run
    under no_grad().
    """
    if radius <= 0:
        raise ValueError(f"stage2.perturb_radius must be > 0, got {radius}.")
    if num_directions < 2:
        raise ValueError(
            "stage2.num_directions must be >= 2 so sampled curvature has "
            f"cross-direction structure, got {num_directions}."
        )

    raw_actions = _denormalize_actions(
        normalized_actions.detach(), action_mean, action_std
    )

    student_responses = []
    teacher_responses = []
    odd_increments = []
    teacher_quadratic = []
    effective_radii = []

    # The parent Lightning module may recursively put teacher into train mode.
    # Reassert eval semantics here; parameters are also requires_grad=False.
    teacher.eval()

    for _ in range(int(num_directions)):
        plus_raw, minus_raw, effective_radius = _sample_raw_action_perturbation(
            raw_actions,
            history_size=history_size,
            horizon=horizon,
            radius=radius,
            first_only=first_only,
        )
        plus = _normalize_raw_actions(plus_raw, action_mean, action_std)
        minus = _normalize_raw_actions(minus_raw, action_mean, action_std)

        # Student: gradients flow through both perturbed rollouts.  This is the
        # key difference from Stage-I's detached response-direction probe.
        student_plus = _autoregressive_rollout(
            student, student_emb, plus, history_size, horizon
        )[:, -1]
        student_minus = _autoregressive_rollout(
            student, student_emb, minus, history_size, horizon
        )[:, -1]

        denom = (2.0 * effective_radius).clamp_min(1e-8)[:, None]
        student_response = (student_plus - student_minus) / denom

        # Planner-facing latent costs use per-coordinate MSE here so the loss
        # scale is independent of latent width.  Their odd part is exactly zero
        # for a locally symmetric endpoint-target landscape.
        c_plus = (student_plus - student_goal).pow(2).mean(dim=-1)
        c_minus = (student_minus - student_goal).pow(2).mean(dim=-1)
        odd = 0.5 * (c_plus - c_minus)

        with torch.no_grad():
            teacher_plus = _autoregressive_rollout(
                teacher, teacher_emb, plus, history_size, horizon
            )[:, -1]
            teacher_minus = _autoregressive_rollout(
                teacher, teacher_emb, minus, history_size, horizon
            )[:, -1]
            teacher_response = (teacher_plus - teacher_minus) / denom

            # At this fixed perturbation radius, r^2 ||Jv||^2 / D is the
            # teacher's local quadratic goal-cost scale along this probe.
            teacher_q = (
                effective_radius.pow(2)
                * teacher_response.pow(2).mean(dim=-1)
            )

        student_responses.append(student_response)
        teacher_responses.append(teacher_response)
        odd_increments.append(odd)
        teacher_quadratic.append(teacher_q)
        effective_radii.append(effective_radius)

    student_response = torch.stack(student_responses, dim=1)   # [B,K,D]
    teacher_response = torch.stack(teacher_responses, dim=1)   # [B,K,D]
    odd = torch.stack(odd_increments, dim=1)                   # [B,K]
    teacher_q = torch.stack(teacher_quadratic, dim=1)          # [B,K]
    effective_radius = torch.stack(effective_radii, dim=1)     # [B,K]

    # Dimensionless per-sample odd energy normalized by the FROZEN teacher's
    # useful local curvature energy.  The denominator is detached by
    # construction, so the student cannot lower L_odd by collapsing curvature.
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
    # Student encoding and the unchanged official one-step LeWM objective.
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
    output["sigreg_loss"] = self.sigreg(sigreg_emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]

    # ------------------------------------------------------------------
    # H-step autoregressive supervision.  Same Stage-I rollout term, but the
    # student starts from the official pretrained LeWM checkpoint.
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

    # The observed H-step future endpoint is the LOCAL GOAL for the symmetry
    # objective.  This does NOT assert that arbitrary task goals are symmetric;
    # it only says that around an action sequence observed to reach this future
    # endpoint, the local goal cost should not contain a model-bias-induced odd
    # tilt.
    student_goal = emb[:, ctx_len + horizon - 1].detach()

    # Frozen teacher uses its own latent coordinates.  Gram supervision is
    # invariant to a common orthogonal change of latent basis and does not force
    # exact student/teacher response-vector equality.
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
