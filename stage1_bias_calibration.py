"""Stage I: bias-calibrated action-conditioned JEPA training objective.

This module intentionally leaves the official LeWM training path untouched.
It adds two auxiliary objectives for the Stage-I experiment:

1. multi-step autoregressive rollout supervision, which reduces zero-order
   endpoint/rollout bias under the same autoregressive execution used at
   planning time;
2. action-projected bias calibration, which penalizes the component of the
   terminal rollout error that lies along locally action-responsive latent
   directions.

The second term targets the harmful local-landscape interaction

    2 * e_H^T J_H delta_U,

without requiring simulator state, counterfactual ground-truth rollouts,
expert-distance ranking labels, or a planning-time verifier.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _autoregressive_rollout(model, initial_emb, actions, history_size, horizon):
    """Roll the predictor forward for ``horizon`` coarse PushT steps."""
    if initial_emb.ndim != 3 or actions.ndim != 3:
        raise ValueError(
            f"Expected emb/actions with rank 3, got {initial_emb.shape=} {actions.shape=}."
        )
    if initial_emb.shape[0] != actions.shape[0]:
        raise ValueError("Embedding/action batch sizes do not match.")
    if initial_emb.shape[1] < history_size + horizon:
        raise ValueError(
            "Stage-I dataset sequence is too short: "
            f"need at least {history_size + horizon} states, got {initial_emb.shape[1]}."
        )

    min_action_steps = history_size + horizon - 1
    if actions.shape[1] < min_action_steps:
        raise ValueError(
            "Stage-I dataset action sequence is too short: "
            f"need at least {min_action_steps}, got {actions.shape[1]}."
        )

    latent_history = initial_emb[:, :history_size]
    predicted = []

    for step in range(horizon):
        action_end = history_size + step
        action_start = action_end - history_size

        emb_window = latent_history[:, -history_size:]
        action_window = actions[:, action_start:action_end]
        action_emb = model.action_encoder(action_window)

        next_emb = model.predict(emb_window, action_emb)[:, -1:]
        predicted.append(next_emb)
        latent_history = torch.cat([latent_history, next_emb], dim=1)

    return torch.cat(predicted, dim=1)


def _expand_action_stat(stat, target_dim, device, dtype):
    """Repeat raw 2-D action statistics across a packed coarse action block."""
    stat = torch.as_tensor(stat, device=device, dtype=dtype).reshape(-1)
    if stat.numel() == target_dim:
        return stat
    if target_dim % stat.numel() != 0:
        raise ValueError(
            f"Cannot broadcast action statistic dim {stat.numel()} to coarse dim {target_dim}."
        )
    return stat.repeat(target_dim // stat.numel())


def _normalize_raw_actions(raw_actions, action_mean, action_std):
    mean = _expand_action_stat(
        action_mean, raw_actions.shape[-1], raw_actions.device, raw_actions.dtype
    )
    std = _expand_action_stat(
        action_std, raw_actions.shape[-1], raw_actions.device, raw_actions.dtype
    )
    return (raw_actions - mean) / std.clamp_min(1e-8)


def _denormalize_actions(actions, action_mean, action_std):
    """Invert official normalization on packed frameskip coarse actions."""
    mean = _expand_action_stat(
        action_mean, actions.shape[-1], actions.device, actions.dtype
    )
    std = _expand_action_stat(
        action_std, actions.shape[-1], actions.device, actions.dtype
    )
    return actions * std + mean


def _bounded_delta_batch(slack, requested_radius):
    """Torch version of the evaluator's bounded equal-norm perturbation."""
    slack = slack.flatten(start_dim=1).clamp_min(0.0)
    feasible = torch.linalg.vector_norm(slack, dim=-1)
    requested = torch.full_like(feasible, float(requested_radius))
    radius = torch.minimum(requested, feasible * 0.999)

    g = torch.randn_like(slack).abs().clamp_min(1e-12)
    signs = torch.where(
        torch.rand_like(slack) < 0.5,
        -torch.ones_like(slack),
        torch.ones_like(slack),
    )

    def norm_at(alpha):
        mag = torch.minimum(alpha[:, None] * g, slack)
        return torch.linalg.vector_norm(mag, dim=-1)

    lo = torch.zeros_like(radius)
    hi = torch.ones_like(radius)
    for _ in range(32):
        need_more = norm_at(hi) < radius
        if not bool(need_more.any()):
            break
        hi = torch.where(need_more, hi * 2.0, hi)

    for _ in range(48):
        mid = 0.5 * (lo + hi)
        below = norm_at(mid) < radius
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)

    delta = signs * torch.minimum(hi[:, None] * g, slack)
    return delta, radius


def _sample_raw_action_perturbation(
    raw_actions, history_size, horizon, radius, first_only
):
    """Bounded symmetric perturbation in packed raw PushT action coordinates."""
    active_start = history_size - 1
    active_count = 1 if first_only else horizon
    active_end = active_start + active_count

    if raw_actions.shape[1] < active_end:
        raise ValueError(
            f"Need {active_end} action blocks for Stage-I perturbation, "
            f"got {raw_actions.shape[1]}."
        )

    active = raw_actions[:, active_start:active_end]
    active = torch.nan_to_num(active, nan=0.0, posinf=1.0, neginf=-1.0)
    active = active.clamp(-1.0, 1.0)

    # One coarse block is 5 raw 2-D PushT actions = 10 scalars. Flattening the
    # active block therefore exactly matches the fixed-action evaluator's 10-D
    # perturbation when first_only=True.
    slack = (1.0 - active.abs()).clamp_min(0.0)
    flat_delta, effective_radius = _bounded_delta_batch(slack, radius)
    delta = flat_delta.view_as(active)

    plus = raw_actions.clone()
    minus = raw_actions.clone()
    plus[:, active_start:active_end] = active + delta
    minus[:, active_start:active_end] = active - delta
    return plus, minus, effective_radius


def _detached_response_directions(
    model,
    emb,
    normalized_actions,
    action_mean,
    action_std,
    history_size,
    horizon,
    radius,
    first_only,
    num_directions,
):
    """Estimate terminal action-response directions from bounded raw probes.

    ``normalized_actions`` is the official packed LeWM tensor with last
    dimension frameskip * action_dim (=10 for PushT). We invert the official
    z-score normalization, perturb in bounded raw [-1,1] coordinates, then
    apply the same normalization before calling the action encoder.
    """
    if radius <= 0:
        raise ValueError(f"stage1.perturb_radius must be > 0, got {radius}.")
    if num_directions < 1:
        raise ValueError(
            f"stage1.num_directions must be >= 1, got {num_directions}."
        )

    was_training = model.training
    model.eval()
    response_dirs = []
    response_norms = []
    effective_radii = []

    try:
        with torch.no_grad():
            probe_emb = emb.detach()
            probe_actions = normalized_actions.detach()
            probe_raw_actions = _denormalize_actions(
                probe_actions, action_mean, action_std
            )

            if probe_raw_actions.shape[-1] != model.action_encoder.patch_embed.in_channels:
                raise RuntimeError(
                    "Stage-I coarse action dimension mismatch after de-normalization: "
                    f"got {probe_raw_actions.shape[-1]}, action encoder expects "
                    f"{model.action_encoder.patch_embed.in_channels}."
                )

            for _ in range(int(num_directions)):
                plus_raw, minus_raw, effective_radius = _sample_raw_action_perturbation(
                    probe_raw_actions,
                    history_size=history_size,
                    horizon=horizon,
                    radius=radius,
                    first_only=first_only,
                )
                plus = _normalize_raw_actions(plus_raw, action_mean, action_std)
                minus = _normalize_raw_actions(minus_raw, action_mean, action_std)

                pred_plus = _autoregressive_rollout(
                    model, probe_emb, plus, history_size, horizon
                )[:, -1]
                pred_minus = _autoregressive_rollout(
                    model, probe_emb, minus, history_size, horizon
                )[:, -1]

                denom = (2.0 * effective_radius).clamp_min(1e-8)[:, None]
                response = (pred_plus - pred_minus) / denom
                response_norm = torch.linalg.vector_norm(response, dim=-1)
                response_dir = F.normalize(response, dim=-1, eps=1e-8)

                response_dirs.append(response_dir)
                response_norms.append(response_norm)
                effective_radii.append(effective_radius)
    finally:
        if was_training:
            model.train()

    return (
        torch.stack(response_dirs, dim=1),
        torch.stack(response_norms, dim=1),
        torch.stack(effective_radii, dim=1),
    )


def stage1_forward(self, batch, stage, cfg, action_mean, action_std):
    """LeWM objective + Stage-I rollout and action-projected bias losses."""
    if cfg.factor.enabled:
        raise ValueError(
            "Stage I is intentionally evaluated without privileged-state factor supervision. "
            "Set factor.enabled=False."
        )

    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    horizon = int(cfg.stage1.rollout_horizon)
    lambd = float(cfg.loss.sigreg.weight)

    if n_preds != 1:
        raise ValueError(
            "The Stage-I implementation currently assumes the official LeWM one-step "
            f"offset wm.num_preds=1, got {n_preds}."
        )
    if horizon < 1:
        raise ValueError(f"stage1.rollout_horizon must be >= 1, got {horizon}.")

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    expected_action_dim = self.model.action_encoder.patch_embed.in_channels
    if batch["action"].shape[-1] != expected_action_dim:
        raise RuntimeError(
            "Official packed action tensor has unexpected dimension: "
            f"batch action dim={batch['action'].shape[-1]}, encoder expects "
            f"{expected_action_dim}."
        )

    if emb.shape[1] < ctx_len + horizon:
        raise ValueError(
            "Stage-I training needs a longer offline sequence than official one-step LeWM: "
            f"need {ctx_len + horizon} encoded states, got {emb.shape[1]}. "
            "Use data=pusht_stage1."
        )

    # 1) Official one-step LeWM objective on the same initial 4-state window.
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_emb = emb[:, : ctx_len + n_preds]
    output["sigreg_loss"] = self.sigreg(sigreg_emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # 2) Autoregressive rollout supervision: reduce zero-order H-step bias.
    rollout_pred = _autoregressive_rollout(
        self.model,
        emb,
        batch["action"],
        history_size=ctx_len,
        horizon=horizon,
    )
    rollout_target = emb[:, ctx_len : ctx_len + horizon].detach()
    rollout_error = rollout_pred - rollout_target
    output["stage1_rollout_loss"] = rollout_error.pow(2).mean()
    output["stage1_endpoint_mse"] = rollout_error[:, -1].pow(2).mean().detach()

    # 3) Action-projected bias calibration.
    response_dirs, response_norms, effective_radii = _detached_response_directions(
        self.model,
        emb,
        batch["action"],
        action_mean=action_mean,
        action_std=action_std,
        history_size=ctx_len,
        horizon=horizon,
        radius=float(cfg.stage1.perturb_radius),
        first_only=bool(cfg.stage1.perturb_first_only),
        num_directions=int(cfg.stage1.num_directions),
    )

    endpoint_error = rollout_error[:, -1]
    projected_bias = torch.einsum("bd,bkd->bk", endpoint_error, response_dirs)

    # projected_bias is a scalar projection onto a unit latent response direction,
    # whereas rollout/endpoint losses are mean-squared errors per latent coordinate.
    # Normalize by latent dimensionality so APB is on the same per-dimension scale
    # and cannot dominate merely because one aligned direction can contain O(D)
    # times the per-coordinate error energy.
    latent_dim = endpoint_error.shape[-1]
    raw_apb = projected_bias.pow(2).mean()
    output["stage1_action_projected_bias_raw"] = raw_apb.detach()
    output["stage1_action_projected_bias_loss"] = raw_apb / float(latent_dim)
    output["stage1_projected_bias_abs"] = projected_bias.abs().mean().detach()
    output["stage1_response_norm"] = response_norms.mean().detach()
    output["stage1_effective_radius"] = effective_radii.mean().detach()

    output["loss"] = (
        output["loss"]
        + float(cfg.stage1.rollout_weight) * output["stage1_rollout_loss"]
        + float(cfg.stage1.action_projected_bias_weight)
        * output["stage1_action_projected_bias_loss"]
    )

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    diagnostics_dict = {
        f"{stage}/stage1_endpoint_mse": output["stage1_endpoint_mse"],
        f"{stage}/stage1_action_projected_bias_raw": output[
            "stage1_action_projected_bias_raw"
        ],
        f"{stage}/stage1_projected_bias_abs": output["stage1_projected_bias_abs"],
        f"{stage}/stage1_response_norm": output["stage1_response_norm"],
        f"{stage}/stage1_effective_radius": output["stage1_effective_radius"],
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    self.log_dict(diagnostics_dict, on_step=True, sync_dist=True)
    return output
