"""Bias-only latent calibration for full PushT LeWM.

This experiment isolates the zero-order endpoint-bias mechanism while keeping
LeWM's action-conditioned predictor geometry exactly fixed.

Training
--------
For an observed current frame z_t, demonstrated H-step action sequence U_0,
frozen LeWM endpoint T(U_0), and observed future endpoint z_H,

    b* = z_H - T(U_0)
    b_phi = B_phi([z_t, T(U_0)])
    L_bias = ||T(U_0) + b_phi - z_H||^2.

The entire pretrained LeWM is frozen. Only B_phi is optimized.

Planning
--------
At every CEM refinement iteration with center mu_i,

    b_i = B_phi([z_t, T(mu_i)])

is computed once and shared by every sampled candidate U_j in that iteration:

    T_corr(U_j) = T(U_j) + b_i.

Therefore, within one CEM population,

    d T_corr / dU = d T / dU

exactly. The experiment can improve zero-order placement of the predicted
rollout manifold but has no architectural freedom to change local action
response geometry.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn


class LatentBiasCalibrator(nn.Module):
    """Small zero-initialized residual calibrator in the frozen latent frame."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(2 * self.latent_dim),
            nn.Linear(2 * self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        # Start as the exact frozen LeWM baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        current_latent: torch.Tensor,
        center_endpoint: torch.Tensor,
    ) -> torch.Tensor:
        if current_latent.shape != center_endpoint.shape:
            raise ValueError(
                "Bias calibrator inputs must have identical shapes, got "
                f"{current_latent.shape} and {center_endpoint.shape}."
            )
        if current_latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {current_latent.shape}."
            )
        return self.net(torch.cat([current_latent, center_endpoint], dim=-1))


def planner_style_rollout_from_single(
    model: nn.Module,
    current_latent: torch.Tensor,
    future_actions: torch.Tensor,
    *,
    history_size: int,
    horizon: int,
) -> torch.Tensor:
    """Match JEPA.rollout() when planning starts from one observed frame.

    Args:
        current_latent: [B, D] frozen encoded current observation.
        future_actions: [B, H, A] normalized packed LeWM action blocks.
    Returns:
        [B, H, D] predicted endpoints z_{t+1:t+H}.
    """
    if current_latent.ndim != 2 or future_actions.ndim != 3:
        raise ValueError(
            f"Expected [B,D] and [B,H,A], got {current_latent.shape}, "
            f"{future_actions.shape}."
        )
    if future_actions.shape[1] < int(horizon):
        raise ValueError(
            f"Need {horizon} future action blocks, got {future_actions.shape[1]}."
        )

    latent_history = current_latent[:, None, :]
    predicted = []
    for step in range(int(horizon)):
        # This reproduces JEPA.rollout(): history grows from 1 to history_size,
        # and action history is truncated to the same causal window.
        start = max(0, step - int(history_size) + 1)
        emb_window = latent_history[:, -int(history_size) :]
        action_window = future_actions[:, start : step + 1]
        act_emb = model.action_encoder(action_window)
        next_emb = model.predict(emb_window, act_emb)[:, -1:]
        predicted.append(next_emb)
        latent_history = torch.cat([latent_history, next_emb], dim=1)
    return torch.cat(predicted, dim=1)


def _clone_info_value(value: Any):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _device_copy(info_dict: dict, device: torch.device) -> dict:
    out = {}
    for k, v in info_dict.items():
        v = _clone_info_value(v)
        if torch.is_tensor(v):
            v = v.to(device)
        out[k] = v
    return out


def _first_candidate_info(info_dict: dict) -> dict:
    """Keep the solver's [B,S,...] structure but reduce S to one."""
    out = {}
    for k, v in info_dict.items():
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            if v.ndim >= 2:
                out[k] = v[:, :1]
            else:
                out[k] = v
        else:
            out[k] = v
    return out


class BiasOnlyWorldModel(nn.Module):
    """Planner-compatible wrapper around a frozen LeWM and a tiny bias head."""

    def __init__(
        self,
        base_model: nn.Module,
        calibrator: LatentBiasCalibrator,
        history_size: int = 3,
    ):
        super().__init__()
        self.base_model = base_model
        self.calibrator = calibrator
        self.history_size = int(history_size)

        self.base_model.requires_grad_(False)
        self.base_model.eval()

    # Expose the original representation/dynamics modules for diagnostics.
    @property
    def encoder(self):
        return self.base_model.encoder

    @property
    def projector(self):
        return self.base_model.projector

    @property
    def action_encoder(self):
        return self.base_model.action_encoder

    @property
    def predictor(self):
        return self.base_model.predictor

    @property
    def pred_proj(self):
        return self.base_model.pred_proj

    @property
    def factor_heads(self):
        return getattr(self.base_model, "factor_heads", None)

    def encode(self, info):
        return self.base_model.encode(info)

    def predict(self, emb, act_emb):
        return self.base_model.predict(emb, act_emb)

    def rollout(self, info, action_sequence, history_size: int = 3):
        # Raw rollout is deliberately unchanged. Bias-aware planning must use
        # get_cost_with_center(), where a single common CEM-center bias exists.
        return self.base_model.rollout(
            info, action_sequence, history_size=history_size
        )

    def bias(
        self,
        current_latent: torch.Tensor,
        center_endpoint: torch.Tensor,
    ) -> torch.Tensor:
        return self.calibrator(current_latent, center_endpoint)

    def _goal_embedding(self, info_dict: dict, device: torch.device):
        """Mirror JEPA.get_cost() goal preparation in the frozen latent frame."""
        goal = {
            k: v[:, 0]
            for k, v in info_dict.items()
            if torch.is_tensor(v)
        }
        if "goal" not in goal:
            raise KeyError("BiasOnlyWorldModel requires goal pixels in info_dict.")
        goal["pixels"] = goal["goal"]

        for k in list(info_dict.keys()):
            if k.startswith("goal_") and k in goal:
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action", None)
        for k in list(goal.keys()):
            if torch.is_tensor(goal[k]):
                goal[k] = goal[k].to(device)

        self.base_model.eval()
        self.base_model.interpolate_pos_encoding = True
        return self.base_model.encode(goal)["emb"]

    @torch.inference_mode()
    def get_cost_with_center(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
        center_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Bias-correct candidate costs using one shared CEM-center offset.

        Args:
            info_dict: solver-expanded [B,S,...] observation dictionary.
            action_candidates: [B,S,H,A].
            center_actions: [B,H,A], exact CEM mean before this update.
        Returns:
            [B,S] corrected terminal latent squared distances.
        """
        device = next(self.calibrator.parameters()).device
        self.base_model.eval()
        self.calibrator.eval()
        self.base_model.interpolate_pos_encoding = True

        info = _device_copy(info_dict, device)
        candidates = action_candidates.to(device)
        center = center_actions.to(device)

        goal_emb = self._goal_embedding(info, device)
        goal_endpoint = goal_emb[:, -1]

        candidate_info = _device_copy(info, device)
        rolled = self.base_model.rollout(
            candidate_info,
            candidates,
            history_size=self.history_size,
        )
        candidate_endpoint = rolled["predicted_emb"][..., -1, :]
        current_latent = rolled["emb"][:, 0, -1, :]

        center_info = _first_candidate_info(_device_copy(info, device))
        center_rolled = self.base_model.rollout(
            center_info,
            center[:, None],
            history_size=self.history_size,
        )
        center_endpoint = center_rolled["predicted_emb"][:, 0, -1, :]

        common_bias = self.calibrator(current_latent, center_endpoint)
        corrected_endpoint = candidate_endpoint + common_bias[:, None, :]
        return (corrected_endpoint - goal_endpoint[:, None, :]).pow(2).sum(dim=-1)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        raise RuntimeError(
            "BiasOnlyWorldModel requires the exact current CEM center. "
            "Use solver=bias_cem (or another solver that calls "
            "get_cost_with_center)."
        )


def frozen_visual_encode(
    base_model: nn.Module,
    pixels: torch.Tensor,
) -> torch.Tensor:
    base_model.eval()
    base_model.encoder.eval()
    base_model.projector.eval()
    with torch.no_grad():
        return base_model.encode({"pixels": pixels})["emb"].detach()


def bias_only_forward(self, batch, stage, cfg):
    """Single-objective Bias-Only training forward."""
    wrapper: BiasOnlyWorldModel = self.model
    base = wrapper.base_model
    base.eval()
    wrapper.calibrator.train(stage == "train")

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    horizon = int(cfg.bias_only.rollout_horizon)
    ctx_len = int(cfg.wm.history_size)
    current_index = ctx_len - 1
    goal_index = current_index + horizon

    if batch["pixels"].shape[1] <= goal_index:
        raise ValueError(
            "Bias-Only sequence too short: need observed indices through "
            f"{goal_index}, got T={batch['pixels'].shape[1]}."
        )
    if batch["action"].shape[1] < goal_index:
        raise ValueError(
            "Bias-Only action sequence too short for planner-style H-step "
            f"rollout: need >= {goal_index}, got {batch['action'].shape[1]}."
        )

    emb = frozen_visual_encode(base, batch["pixels"])
    current_latent = emb[:, current_index].detach()
    goal_latent = emb[:, goal_index].detach()
    future_actions = batch["action"][
        :, current_index : current_index + horizon
    ]

    with torch.no_grad():
        teacher_rollout = planner_style_rollout_from_single(
            base,
            current_latent,
            future_actions,
            history_size=ctx_len,
            horizon=horizon,
        )
        teacher_endpoint = teacher_rollout[:, -1].detach()

    pred_bias = wrapper.calibrator(current_latent, teacher_endpoint)
    corrected_endpoint = teacher_endpoint + pred_bias

    # The ONLY optimization objective.
    loss = (corrected_endpoint - goal_latent).pow(2).mean()

    target_bias = goal_latent - teacher_endpoint
    teacher_endpoint_mse = target_bias.pow(2).mean()
    residual = corrected_endpoint - goal_latent

    pred_norm = torch.linalg.vector_norm(pred_bias, dim=-1)
    target_norm = torch.linalg.vector_norm(target_bias, dim=-1)
    bias_cos = torch.nn.functional.cosine_similarity(
        pred_bias.detach(), target_bias.detach(), dim=-1, eps=1e-8
    )

    output = {
        "loss": loss,
        "bias_only_loss": loss,
    }
    self.log_dict(
        {
            f"{stage}/bias_only_loss": loss.detach(),
            f"{stage}/teacher_endpoint_mse": teacher_endpoint_mse.detach(),
            f"{stage}/corrected_endpoint_mse": residual.pow(2).mean().detach(),
            f"{stage}/pred_bias_norm": pred_norm.mean().detach(),
            f"{stage}/target_bias_norm": target_norm.mean().detach(),
            f"{stage}/bias_cosine": bias_cos.mean().detach(),
        },
        on_step=True,
        sync_dist=True,
    )
    return output
