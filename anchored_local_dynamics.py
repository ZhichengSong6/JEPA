"""Anchored Local Dynamics (ALD) calibration for full PushT LeWM.

The pretrained LeWM teacher is empirically useful in *relative* local
response to action perturbations, while its multi-step absolute endpoint can
be biased.  ALD therefore separates these two roles:

    observed future        -> trusted absolute anchor
    frozen LeWM difference -> trusted local action-conditioned displacement

For a demonstrated action sequence U0 and symmetric bounded probes U+/U-,

    z_g = E(o_{t+H})
    Y+  = z_g + sg[T(U+) - T(U0)]
    Y-  = z_g + sg[T(U-) - T(U0)]

and the student is trained with

    L_ALD = 1/2 ( ||S(U+) - Y+||^2 + ||S(U-) - Y-||^2 ).

The visual encoder/projector are frozen.  No simulator state, reward,
counterfactual ground-truth rollout, physical factor, or planner oracle is
used.  Synthetic U+/U- are only queried through student/teacher predictors.

In the local affine regime this gives

    E_v[L_ALD] = ||e_S||^2 + delta^2/m ||J_S - J_T||_F^2,

so zero-order rollout bias is corrected while the useful teacher local
Jacobian is preserved.  Unlike Gram/quadratic matching, J_S=0 is not a
stationary point when J_T != 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from stage1_bias_calibration import (
    _autoregressive_rollout,
    _bounded_delta_batch,
    _denormalize_actions,
    _normalize_raw_actions,
    _sample_raw_action_perturbation,
)


def _repeat_for_candidates(x: torch.Tensor, count: int) -> torch.Tensor:
    """Repeat [B,...] along a candidate axis and flatten to [B*C,...]."""
    if count < 1:
        raise ValueError(f"candidate count must be positive, got {count}")
    b = x.shape[0]
    return x[:, None].expand(b, count, *x.shape[1:]).reshape(
        b * count, *x.shape[1:]
    )


def _frozen_visual_encode(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode observed pixels in a fixed, inference-consistent latent frame.

    ``requires_grad_(False)`` is set in the training entrypoint, but that alone
    does not freeze BatchNorm running statistics.  We therefore explicitly keep
    the frozen visual encoder/projector in eval mode and encode without grad.
    """
    model.encoder.eval()
    model.projector.eval()
    with torch.no_grad():
        visual_batch = {"pixels": batch["pixels"]}
        emb = model.encode(visual_batch)["emb"]
    return emb.detach()


def _sample_probe_actions(
    normalized_actions: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    history_size: int,
    horizon: int,
    radius: float,
    first_only: bool,
    num_directions: int,
):
    """Sample exact bounded symmetric perturbations in raw PushT action space."""
    if radius <= 0:
        raise ValueError(f"ald.perturb_radius must be > 0, got {radius}.")
    if num_directions < 1:
        raise ValueError(
            f"ald.num_directions must be >= 1, got {num_directions}."
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
        torch.stack(plus_actions, dim=1),
        torch.stack(minus_actions, dim=1),
        torch.stack(radii, dim=1),
    )


def _paired_local_predictions(
    student,
    teacher,
    emb: torch.Tensor,
    normalized_actions: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    history_size: int,
    horizon: int,
    radius: float,
    first_only: bool,
    num_directions: int,
    directions_per_chunk: int,
):
    """Return student/teacher endpoints for matched U+/U- probes."""
    if directions_per_chunk < 1:
        raise ValueError(
            "ald.probe_directions_per_chunk must be >= 1, got "
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

    b = emb.shape[0]
    d = emb.shape[-1]
    student_plus_all = []
    student_minus_all = []
    teacher_plus_all = []
    teacher_minus_all = []

    # ALD is a post-training calibration objective.  Run both student and
    # teacher in inference semantics (dropout off, BN running stats fixed),
    # while gradients remain enabled through the student predictor path.
    student.eval()
    teacher.eval()

    for start in range(0, int(num_directions), int(directions_per_chunk)):
        stop = min(start + int(directions_per_chunk), int(num_directions))
        c = stop - start
        candidate_count = 2 * c

        action_chunk = torch.cat(
            [plus_actions[:, start:stop], minus_actions[:, start:stop]], dim=1
        )
        flat_actions = action_chunk.reshape(
            b * candidate_count, *action_chunk.shape[2:]
        )
        flat_emb = _repeat_for_candidates(emb.detach(), candidate_count)

        student_endpoint = _autoregressive_rollout(
            student,
            flat_emb,
            flat_actions,
            history_size=history_size,
            horizon=horizon,
        )[:, -1]
        student_endpoint = student_endpoint.reshape(b, candidate_count, d)
        student_plus_all.append(student_endpoint[:, :c])
        student_minus_all.append(student_endpoint[:, c:])

        with torch.no_grad():
            teacher_endpoint = _autoregressive_rollout(
                teacher,
                flat_emb,
                flat_actions,
                history_size=history_size,
                horizon=horizon,
            )[:, -1]
            teacher_endpoint = teacher_endpoint.reshape(b, candidate_count, d)
            teacher_plus_all.append(teacher_endpoint[:, :c].detach())
            teacher_minus_all.append(teacher_endpoint[:, c:].detach())

    return (
        torch.cat(student_plus_all, dim=1),
        torch.cat(student_minus_all, dim=1),
        torch.cat(teacher_plus_all, dim=1),
        torch.cat(teacher_minus_all, dim=1),
        effective_radius,
    )


def ald_forward(self, batch, stage, cfg, action_mean, action_std):
    """Frozen-latent-frame TF + H-step rollout + Anchored Local Dynamics."""
    if not hasattr(self, "teacher_model"):
        raise RuntimeError(
            "ALD module has no frozen teacher_model. Use train_ald.py."
        )

    student = self.model
    teacher = self.teacher_model
    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    horizon = int(cfg.ald.rollout_horizon)

    if n_preds != 1:
        raise ValueError(f"ALD assumes wm.num_preds=1, got {n_preds}.")
    if horizon < 1:
        raise ValueError(f"ald.rollout_horizon must be >= 1, got {horizon}.")

    # Lightning may put the full module back in train mode before forward.
    # ALD deliberately calibrates under planner/inference semantics.  This also
    # keeps all BN running statistics fixed, including pred_proj.
    student.eval()
    teacher.eval()
    student.encoder.eval()
    student.projector.eval()

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    if batch["pixels"].shape[1] < ctx_len + horizon:
        raise ValueError(
            "ALD sequence too short: need at least "
            f"{ctx_len + horizon} states, got {batch['pixels'].shape[1]}."
        )

    # ------------------------------------------------------------------
    # 1) Frozen observation latent frame.
    # ------------------------------------------------------------------
    emb = _frozen_visual_encode(student, batch)
    z_goal = emb[:, ctx_len + horizon - 1].detach()

    expected_action_dim = student.action_encoder.patch_embed.in_channels
    if batch["action"].shape[-1] != expected_action_dim:
        raise RuntimeError(
            "Packed action dimension mismatch: "
            f"batch={batch['action'].shape[-1]} student={expected_action_dim}."
        )

    # ------------------------------------------------------------------
    # 2) One-step teacher-forcing loss.  SIGReg is intentionally absent:
    #    the visual representation is frozen, so SIGReg would be a constant
    #    with zero optimization effect.
    # ------------------------------------------------------------------
    ctx_emb = emb[:, :ctx_len]
    ctx_action = batch["action"][:, :ctx_len]
    ctx_act_emb = student.action_encoder(ctx_action)
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]
    pred_emb = student.predict(ctx_emb, ctx_act_emb)
    tf_loss = (pred_emb - tgt_emb).pow(2).mean()

    # ------------------------------------------------------------------
    # 3) Demonstrated H-step autoregressive rollout anchor.
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
    student_center = rollout_pred[:, -1]
    student_endpoint_mse = (student_center - z_goal).pow(2).mean()

    with torch.no_grad():
        teacher_rollout = _autoregressive_rollout(
            teacher,
            emb,
            batch["action"],
            history_size=ctx_len,
            horizon=horizon,
        )
        teacher_center = teacher_rollout[:, -1].detach()
        teacher_center_mse = (teacher_center - z_goal).pow(2).mean()

    # ------------------------------------------------------------------
    # 4) Synthetic symmetric local probes.  Teacher provides only RELATIVE
    #    displacement; the observed future z_goal provides absolute location.
    # ------------------------------------------------------------------
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
        radius=float(cfg.ald.perturb_radius),
        first_only=bool(cfg.ald.perturb_first_only),
        num_directions=int(cfg.ald.num_directions),
        directions_per_chunk=int(cfg.ald.probe_directions_per_chunk),
    )

    goal_expanded = z_goal[:, None, :]
    teacher_center_expanded = teacher_center[:, None, :]
    target_plus = (
        goal_expanded + (teacher_plus - teacher_center_expanded).detach()
    )
    target_minus = (
        goal_expanded + (teacher_minus - teacher_center_expanded).detach()
    )

    plus_err = student_plus - target_plus
    minus_err = student_minus - target_minus
    ald_loss = 0.5 * (
        plus_err.pow(2).mean() + minus_err.pow(2).mean()
    )

    # Exact midpoint/response decomposition, used only for diagnosis:
    # L_ALD == L_midpoint + L_half_response (up to floating-point error).
    student_mid = 0.5 * (student_plus + student_minus)
    target_mid = 0.5 * (target_plus + target_minus)
    student_half_response = 0.5 * (student_plus - student_minus)
    target_half_response = 0.5 * (target_plus - target_minus)
    midpoint_loss = (student_mid - target_mid).pow(2).mean()
    half_response_loss = (
        student_half_response - target_half_response
    ).pow(2).mean()
    decomposition_error = (
        ald_loss.detach() - (midpoint_loss + half_response_loss).detach()
    ).abs()

    # Finite-difference response diagnostics in the established raw-action scale.
    denom = (2.0 * effective_radius).clamp_min(1e-8)[..., None]
    student_response = (student_plus - student_minus) / denom
    teacher_response = (teacher_plus - teacher_minus) / denom
    student_response_norm = torch.linalg.vector_norm(student_response, dim=-1)
    teacher_response_norm = torch.linalg.vector_norm(teacher_response, dim=-1)
    response_cosine = F.cosine_similarity(
        student_response.detach(), teacher_response.detach(), dim=-1, eps=1e-8
    )
    response_gain = (
        student_response_norm.detach()
        / teacher_response_norm.detach().clamp_min(1e-8)
    )

    teacher_even_shift = (
        0.5 * (teacher_plus + teacher_minus) - teacher_center_expanded
    )
    target_displacement = torch.cat(
        [
            target_plus - goal_expanded,
            target_minus - goal_expanded,
        ],
        dim=1,
    )

    # At exact student=teacher initialization, ALD must equal teacher center
    # endpoint MSE and half-response loss must be zero.  This is our smoke-test
    # invariant and catches latent-frame / mode inconsistencies immediately.
    init_equivalence_ratio = (
        ald_loss.detach() / teacher_center_mse.detach().clamp_min(1e-12)
    )

    total_loss = (
        float(cfg.ald.tf_weight) * tf_loss
        + float(cfg.ald.rollout_weight) * rollout_loss
        + float(cfg.ald.weight) * ald_loss
    )

    output = {
        "loss": total_loss,
        "tf_loss": tf_loss,
        "ald_rollout_loss": rollout_loss,
        "ald_loss": ald_loss,
        "ald_midpoint_loss": midpoint_loss,
        "ald_half_response_loss": half_response_loss,
    }

    losses = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    diagnostics = {
        f"{stage}/ald_student_endpoint_mse": student_endpoint_mse.detach(),
        f"{stage}/ald_teacher_center_mse": teacher_center_mse.detach(),
        f"{stage}/ald_decomposition_error": decomposition_error,
        f"{stage}/ald_init_equivalence_ratio": init_equivalence_ratio,
        f"{stage}/ald_student_response_norm": student_response_norm.mean().detach(),
        f"{stage}/ald_teacher_response_norm": teacher_response_norm.mean().detach(),
        f"{stage}/ald_response_cosine": response_cosine.mean().detach(),
        f"{stage}/ald_response_gain": response_gain.mean().detach(),
        f"{stage}/ald_effective_radius": effective_radius.mean().detach(),
        f"{stage}/ald_teacher_even_shift_norm": torch.linalg.vector_norm(
            teacher_even_shift, dim=-1
        ).mean().detach(),
        f"{stage}/ald_target_displacement_norm": torch.linalg.vector_norm(
            target_displacement, dim=-1
        ).mean().detach(),
    }

    self.log_dict(losses, on_step=True, sync_dist=True)
    self.log_dict(diagnostics, on_step=True, sync_dist=True)
    return output


# ============================================================================
# Multi-Horizon ALD
# ============================================================================
# Terminal ALD calibrates the response of the H-step endpoint to perturbations
# of the first planner action block.  MH-ALD extends the same anchored-relative
# construction over the whole planner horizon.  Each coarse action block p is
# perturbed separately and its response is calibrated at every causally
# affected future step h >= p.  The visual latent frame and planning-time cost
# remain unchanged.


def _sample_blockwise_probe_actions(
    normalized_actions,
    action_mean,
    action_std,
    history_size,
    horizon,
    radius,
    directions_per_position,
):
    """Sample equal-radius symmetric probes at every planner action block."""
    if radius <= 0:
        raise ValueError(f"mh_ald.perturb_radius must be > 0, got {radius}.")
    if directions_per_position < 1:
        raise ValueError(
            "mh_ald.directions_per_position must be >= 1, got "
            f"{directions_per_position}."
        )

    raw_actions = _denormalize_actions(
        normalized_actions.detach(), action_mean, action_std
    )
    plus_actions = []
    minus_actions = []
    radii = []
    positions = []

    for position in range(int(horizon)):
        block_idx = int(history_size) - 1 + position
        if block_idx >= raw_actions.shape[1]:
            raise ValueError(
                f"Need action block index {block_idx}, but sequence has "
                f"{raw_actions.shape[1]} blocks."
            )

        base = raw_actions[:, block_idx : block_idx + 1]
        base = torch.nan_to_num(base, nan=0.0, posinf=1.0, neginf=-1.0)
        base = base.clamp(-1.0, 1.0)
        slack = (1.0 - base.abs()).clamp_min(0.0)

        for _ in range(int(directions_per_position)):
            flat_delta, effective_radius = _bounded_delta_batch(slack, radius)
            delta = flat_delta.view_as(base)

            plus_raw = raw_actions.clone()
            minus_raw = raw_actions.clone()
            plus_raw[:, block_idx : block_idx + 1] = base + delta
            minus_raw[:, block_idx : block_idx + 1] = base - delta

            plus_actions.append(
                _normalize_raw_actions(plus_raw, action_mean, action_std)
            )
            minus_actions.append(
                _normalize_raw_actions(minus_raw, action_mean, action_std)
            )
            radii.append(effective_radius)
            positions.append(position)

    return (
        torch.stack(plus_actions, dim=1),
        torch.stack(minus_actions, dim=1),
        torch.stack(radii, dim=1),
        torch.tensor(
            positions, device=normalized_actions.device, dtype=torch.long
        ),
    )


def _paired_multihorizon_predictions(
    student,
    teacher,
    emb,
    normalized_actions,
    action_mean,
    action_std,
    history_size,
    horizon,
    radius,
    directions_per_position,
    probes_per_chunk,
):
    """Return complete student and teacher rollout trajectories for probes."""
    if probes_per_chunk < 1:
        raise ValueError(
            f"mh_ald.probes_per_chunk must be >= 1, got {probes_per_chunk}."
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

    b = emb.shape[0]
    d = emb.shape[-1]
    probe_count = plus_actions.shape[1]
    student_plus_all = []
    student_minus_all = []
    teacher_plus_all = []
    teacher_minus_all = []

    student.eval()
    teacher.eval()

    for start in range(0, int(probe_count), int(probes_per_chunk)):
        stop = min(start + int(probes_per_chunk), int(probe_count))
        count = stop - start
        candidate_count = 2 * count

        action_chunk = torch.cat(
            [plus_actions[:, start:stop], minus_actions[:, start:stop]], dim=1
        )
        flat_actions = action_chunk.reshape(
            b * candidate_count, *action_chunk.shape[2:]
        )
        flat_emb = _repeat_for_candidates(emb.detach(), candidate_count)

        student_rollout = _autoregressive_rollout(
            student,
            flat_emb,
            flat_actions,
            history_size=history_size,
            horizon=horizon,
        ).reshape(b, candidate_count, horizon, d)
        student_plus_all.append(student_rollout[:, :count])
        student_minus_all.append(student_rollout[:, count:])

        with torch.no_grad():
            teacher_rollout = _autoregressive_rollout(
                teacher,
                flat_emb,
                flat_actions,
                history_size=history_size,
                horizon=horizon,
            ).reshape(b, candidate_count, horizon, d)
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


def _mh_causal_mask(positions, horizon, dtype, device):
    """Mask [1,P,H,1]: a perturbation at block p can affect steps h >= p."""
    h = torch.arange(int(horizon), device=device)
    valid = h[None, :] >= positions[:, None]
    return valid.to(dtype=dtype)[None, :, :, None]


def _mh_masked_mse(error, mask):
    """Mean squared error over causally valid probe/horizon cells."""
    if error.ndim != 4 or mask.ndim != 4:
        raise ValueError(
            f"Expected rank-4 error/mask, got {error.shape=} {mask.shape=}."
        )
    per_cell = error.pow(2).mean(dim=-1, keepdim=True)
    expanded = mask.expand(error.shape[0], -1, -1, -1)
    return (per_cell * expanded).sum() / expanded.sum().clamp_min(1.0)


def _mh_masked_mean(values, mask):
    """Mean [B,P,H] values under a broadcastable [1,P,H] mask."""
    mask = mask.to(device=values.device, dtype=values.dtype)
    if mask.shape[0] == 1 and values.shape[0] != 1:
        mask = mask.expand(values.shape[0], -1, -1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def mh_ald_forward(self, batch, stage, cfg, action_mean, action_std):
    """TF + H-step rollout + causal multi-horizon anchored response loss."""
    if not hasattr(self, "teacher_model"):
        raise RuntimeError(
            "MH-ALD module has no frozen teacher_model. Use train_mh_ald.py."
        )

    student = self.model
    teacher = self.teacher_model
    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    horizon = int(cfg.mh_ald.rollout_horizon)

    if n_preds != 1:
        raise ValueError(f"MH-ALD assumes wm.num_preds=1, got {n_preds}.")
    if horizon < 1:
        raise ValueError(f"mh_ald.rollout_horizon must be >= 1, got {horizon}.")

    student.eval()
    teacher.eval()
    student.encoder.eval()
    student.projector.eval()
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    if batch["pixels"].shape[1] < ctx_len + horizon:
        raise ValueError(
            "MH-ALD sequence too short: need at least "
            f"{ctx_len + horizon} states, got {batch['pixels'].shape[1]}."
        )

    emb = _frozen_visual_encode(student, batch)
    rollout_target = emb[:, ctx_len : ctx_len + horizon].detach()

    expected_action_dim = student.action_encoder.patch_embed.in_channels
    if batch["action"].shape[-1] != expected_action_dim:
        raise RuntimeError(
            "Packed action dimension mismatch: "
            f"batch={batch['action'].shape[-1]} student={expected_action_dim}."
        )

    # Original one-step teacher-forcing term.
    ctx_emb = emb[:, :ctx_len]
    ctx_action = batch["action"][:, :ctx_len]
    ctx_act_emb = student.action_encoder(ctx_action)
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]
    pred_emb = student.predict(ctx_emb, ctx_act_emb)
    tf_loss = (pred_emb - tgt_emb).pow(2).mean()

    # Demonstrated autoregressive rollout anchor, unchanged from ALD.
    student_center_rollout = _autoregressive_rollout(
        student,
        emb,
        batch["action"],
        history_size=ctx_len,
        horizon=horizon,
    )
    rollout_loss = (
        student_center_rollout - rollout_target
    ).pow(2).mean()

    with torch.no_grad():
        teacher_center_rollout = _autoregressive_rollout(
            teacher,
            emb,
            batch["action"],
            history_size=ctx_len,
            horizon=horizon,
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
    ) = _paired_multihorizon_predictions(
        student=student,
        teacher=teacher,
        emb=emb,
        normalized_actions=batch["action"],
        action_mean=action_mean,
        action_std=action_std,
        history_size=ctx_len,
        horizon=horizon,
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
    mh_ald_loss = 0.5 * (
        _mh_masked_mse(plus_err, mask)
        + _mh_masked_mse(minus_err, mask)
    )

    # Exact midpoint/half-response decomposition on the same causal cells.
    student_mid = 0.5 * (student_plus + student_minus)
    target_mid = 0.5 * (target_plus + target_minus)
    student_half_response = 0.5 * (student_plus - student_minus)
    target_half_response = 0.5 * (target_plus - target_minus)
    midpoint_loss = _mh_masked_mse(student_mid - target_mid, mask)
    half_response_loss = _mh_masked_mse(
        student_half_response - target_half_response, mask
    )
    decomposition_error = (
        mh_ald_loss.detach()
        - (midpoint_loss + half_response_loss).detach()
    ).abs()

    # At exact student=teacher initialization, probe errors reduce to the
    # center-rollout bias at each causally valid horizon.
    center_error = teacher_center - anchors
    teacher_center_masked_mse = _mh_masked_mse(
        center_error.expand(-1, positions.numel(), -1, -1), mask
    )
    init_equivalence_ratio = (
        mh_ald_loss.detach()
        / teacher_center_masked_mse.detach().clamp_min(1e-12)
    )

    denom = (
        (2.0 * effective_radius)
        .clamp_min(1e-8)[:, :, None, None]
    )
    student_response = (student_plus - student_minus) / denom
    teacher_response = (teacher_plus - teacher_minus) / denom
    student_norm = torch.linalg.vector_norm(student_response, dim=-1)
    teacher_norm = torch.linalg.vector_norm(teacher_response, dim=-1)
    response_cosine_cells = F.cosine_similarity(
        student_response.detach(),
        teacher_response.detach(),
        dim=-1,
        eps=1e-8,
    )
    response_gain_cells = (
        student_norm.detach()
        / teacher_norm.detach().clamp_min(1e-8)
    )
    mask_bph = mask[..., 0]
    response_cosine = _mh_masked_mean(
        response_cosine_cells, mask_bph
    )
    response_gain = _mh_masked_mean(
        response_gain_cells, mask_bph
    )

    total_loss = (
        float(cfg.mh_ald.tf_weight) * tf_loss
        + float(cfg.mh_ald.rollout_weight) * rollout_loss
        + float(cfg.mh_ald.weight) * mh_ald_loss
    )

    output = {
        "loss": total_loss,
        "tf_loss": tf_loss,
        "mh_ald_rollout_loss": rollout_loss,
        "mh_ald_loss": mh_ald_loss,
        "mh_ald_midpoint_loss": midpoint_loss,
        "mh_ald_half_response_loss": half_response_loss,
    }

    losses = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    diagnostics = {
        f"{stage}/mh_ald_student_endpoint_mse": student_endpoint_mse.detach(),
        f"{stage}/mh_ald_teacher_endpoint_mse": teacher_endpoint_mse.detach(),
        f"{stage}/mh_ald_teacher_center_masked_mse": (
            teacher_center_masked_mse.detach()
        ),
        f"{stage}/mh_ald_decomposition_error": decomposition_error,
        f"{stage}/mh_ald_init_equivalence_ratio": init_equivalence_ratio,
        f"{stage}/mh_ald_response_cosine": response_cosine.detach(),
        f"{stage}/mh_ald_response_gain": response_gain.detach(),
        f"{stage}/mh_ald_effective_radius": effective_radius.mean().detach(),
        f"{stage}/mh_ald_probe_count": torch.tensor(
            float(positions.numel()), device=mh_ald_loss.device
        ),
    }

    # Per-horizon diagnostics test the actual hypothesis: whether response
    # calibration remains healthy as action effects propagate through rollout.
    for h in range(horizon):
        hmask = mask[:, :, h : h + 1, :]
        h_plus = _mh_masked_mse(
            plus_err[:, :, h : h + 1], hmask
        )
        h_minus = _mh_masked_mse(
            minus_err[:, :, h : h + 1], hmask
        )
        diagnostics[f"{stage}/mh_ald_h{h + 1}_loss"] = (
            0.5 * (h_plus + h_minus)
        ).detach()

    self.log_dict(losses, on_step=True, sync_dist=True)
    self.log_dict(diagnostics, on_step=True, sync_dist=True)
    return output
