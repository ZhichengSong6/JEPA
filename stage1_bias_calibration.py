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
    """Roll the predictor forward for ``horizon`` coarse PushT steps.

    Args:
        model: JEPA model.
        initial_emb: encoded trajectory, shape (B, T, D). The first
            ``history_size`` embeddings initialize the rollout.
        actions: normalized coarse action blocks, shape (B, T, A).
        history_size: predictor context length.
        horizon: number of newly predicted latent states.

    Returns:
        Predicted future embeddings with shape (B, horizon, D), corresponding
        to encoder targets at indices
        ``history_size : history_size + horizon``.
    """
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
    # Predicting H states from z_{history_size-1} needs actions through
    # index history_size + H - 2. The dataset normally contains one extra
    # action entry aligned with the final state, so this check is conservative.
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


def _sample_action_perturbation(actions, history_size, horizon, radius, first_only):
    """Sample one unit-norm perturbation in the locally controllable action blocks.

    Past actions before ``history_size - 1`` are history, not decision variables.
    The active action sequence therefore starts with the action that moves the
    current state to the first predicted state.
    """
    active_start = history_size - 1
    active_count = 1 if first_only else horizon
    active_end = active_start + active_count

    if actions.shape[1] < active_end:
        raise ValueError(
            f"Need {active_end} action blocks for Stage-I perturbation, got {actions.shape[1]}."
        )

    active = actions[:, active_start:active_end]
    direction = torch.randn_like(active)
    flat = direction.flatten(start_dim=1)
    flat = F.normalize(flat, dim=-1, eps=1e-8)
    direction = flat.view_as(active)

    delta = float(radius) * direction
    plus = actions.clone()
    minus = actions.clone()
    plus[:, active_start:active_end] = active + delta
    minus[:, active_start:active_end] = active - delta
    return plus, minus


def _detached_response_directions(
    model,
    emb,
    actions,
    history_size,
    horizon,
    radius,
    first_only,
    num_directions,
):
    """Estimate terminal action-response directions with deterministic model mode.

    The response probe is deliberately detached. Therefore the bias-calibration
    loss cannot reduce itself by shrinking action sensitivity; it can only move
    the nominal endpoint error relative to the currently exposed action-response
    directions.
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

    try:
        with torch.no_grad():
            # The encoder targets are held fixed. Only the action sequence is
            # perturbed, exactly as in the local CEM-facing diagnostic.
            probe_emb = emb.detach()
            probe_actions = actions.detach()

            for _ in range(int(num_directions)):
                plus, minus = _sample_action_perturbation(
                    probe_actions,
                    history_size=history_size,
                    horizon=horizon,
                    radius=radius,
                    first_only=first_only,
                )
                pred_plus = _autoregressive_rollout(
                    model, probe_emb, plus, history_size, horizon
                )[:, -1]
                pred_minus = _autoregressive_rollout(
                    model, probe_emb, minus, history_size, horizon
                )[:, -1]

                response = (pred_plus - pred_minus) / (2.0 * float(radius))
                response_norm = torch.linalg.vector_norm(response, dim=-1)
                response_dir = F.normalize(response, dim=-1, eps=1e-8)

                response_dirs.append(response_dir)
                response_norms.append(response_norm)
    finally:
        if was_training:
            model.train()

    return torch.stack(response_dirs, dim=1), torch.stack(response_norms, dim=1)


def stage1_forward(self, batch, stage, cfg):
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

    # NaNs occur at sequence boundaries in the official dataset.
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    if emb.shape[1] < ctx_len + horizon:
        raise ValueError(
            "Stage-I training needs a longer offline sequence than official one-step LeWM: "
            f"need {ctx_len + horizon} encoded states, got {emb.shape[1]}. "
            "Use data=pusht_stage1."
        )

    # ------------------------------------------------------------------
    # 1) Exact official LeWM loss on the initial history window.
    #    For horizon=1 this is numerically the same alignment as train.py.
    # ------------------------------------------------------------------
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # ------------------------------------------------------------------
    # 2) Autoregressive rollout supervision: reduce zero-order H-step bias.
    #    Targets are detached so this auxiliary loss trains the imagined
    #    predictor toward the representation instead of moving the encoder
    #    target to follow the rollout error.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3) Action-projected bias calibration.
    #
    #    e_H = zhat_H - stopgrad(z_H)
    #    r(v) ~= Jhat_H v
    #    L_APB = E_v[(e_H^T stopgrad(r(v)/||r(v)||))^2]
    #
    #    This is a Monte-Carlo penalty on the component of endpoint bias that
    #    produces the harmful first-order local-cost term Jhat_H^T e_H.
    # ------------------------------------------------------------------
    response_dirs, response_norms = _detached_response_directions(
        self.model,
        emb,
        batch["action"],
        history_size=ctx_len,
        horizon=horizon,
        radius=float(cfg.stage1.perturb_radius),
        first_only=bool(cfg.stage1.perturb_first_only),
        num_directions=int(cfg.stage1.num_directions),
    )

    endpoint_error = rollout_error[:, -1]
    projected_bias = torch.einsum("bd,bkd->bk", endpoint_error, response_dirs)
    output["stage1_action_projected_bias_loss"] = projected_bias.pow(2).mean()
    output["stage1_projected_bias_abs"] = projected_bias.abs().mean().detach()
    output["stage1_response_norm"] = response_norms.mean().detach()

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
        f"{stage}/stage1_projected_bias_abs": output["stage1_projected_bias_abs"],
        f"{stage}/stage1_response_norm": output["stage1_response_norm"],
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    self.log_dict(diagnostics_dict, on_step=True, sync_dist=True)
    return output
