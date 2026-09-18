"""Training-only inverse-dynamics supervision for LeWM/JEPAs.

The auxiliary head predicts the normalized action block that connects two
consecutive observation latents:

    a_hat_t = h_inv(z_t, z_{t+1})

and is optimized with mean-squared error.  Gradients intentionally flow through
both latents so the encoder/projector geometry becomes action-identifiable.
The head is never consulted by JEPA.rollout(), criterion(), or get_cost().
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class InverseDynamicsHead(nn.Module):
    """Two-hidden-layer MLP used only during training.

    This matches the inverse-only DA-LeWM ablation architecture: two hidden
    layers of width 256 by default, followed by a linear action prediction.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
    ):
        super().__init__()
        if latent_dim < 1 or action_dim < 1 or hidden_dim < 1:
            raise ValueError("latent_dim, action_dim, and hidden_dim must be positive.")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")

        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        layers: list[nn.Module] = []
        in_dim = 2 * self.latent_dim
        for _ in range(self.depth):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.GELU(),
                ]
            )
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        if z_t.shape != z_tp1.shape:
            raise ValueError(
                f"z_t and z_tp1 must have identical shapes, got "
                f"{tuple(z_t.shape)} and {tuple(z_tp1.shape)}."
            )
        if z_t.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {tuple(z_t.shape)}."
            )
        return self.net(torch.cat([z_t, z_tp1], dim=-1))

    def configuration(self) -> dict[str, int]:
        return {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
        }


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Stable scalar Pearson correlation for diagnostics only."""
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    if x.numel() < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().mean() * y.square().mean()).clamp_min(eps)
    return (x * y).mean() / denom


def inverse_dynamics_objective(
    head: nn.Module,
    emb: torch.Tensor,
    normalized_actions: torch.Tensor,
    *,
    max_transitions: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute inverse-action loss on aligned adjacent latent transitions.

    Args:
        head: maps (z_t, z_{t+1}) to the packed normalized action block a_t.
        emb: observation latents shaped (B, T, D).
        normalized_actions: packed actions shaped (B, T, A).  NaN action rows
            are treated as invalid sequence-boundary transitions and masked.
        max_transitions: optional prefix transition count.  When omitted, all
            available adjacent transitions are used.

    The alignment is exactly the one used by LeWM next-state prediction:
    action[:, t] connects emb[:, t] to emb[:, t+1].
    """
    if emb.ndim != 3:
        raise ValueError(f"Expected emb=(B,T,D), got {tuple(emb.shape)}.")
    if normalized_actions.ndim != 3:
        raise ValueError(
            "Expected normalized_actions=(B,T,A), got "
            f"{tuple(normalized_actions.shape)}."
        )
    if emb.shape[0] != normalized_actions.shape[0]:
        raise ValueError("Embedding/action batch sizes differ.")

    available = min(int(emb.shape[1]) - 1, int(normalized_actions.shape[1]))
    if max_transitions is not None:
        available = min(available, int(max_transitions))
    if available < 1:
        raise ValueError("Inverse dynamics requires at least one transition.")

    z_t = emb[:, :available]
    z_tp1 = emb[:, 1 : available + 1]
    action_target_raw = normalized_actions[:, :available]
    valid = torch.isfinite(action_target_raw).all(dim=-1)
    valid_count = valid.sum()
    if int(valid_count.detach().cpu()) == 0:
        raise ValueError("Inverse dynamics batch contains no valid action transitions.")

    action_target = torch.nan_to_num(
        action_target_raw, nan=0.0, posinf=0.0, neginf=0.0
    ).float()
    action_pred = head(z_t, z_tp1).float()
    if action_pred.shape != action_target.shape:
        raise ValueError(
            f"Inverse head output {tuple(action_pred.shape)} does not match "
            f"action target {tuple(action_target.shape)}."
        )

    per_transition_mse = (action_pred - action_target).square().mean(dim=-1)
    valid_f = valid.to(dtype=per_transition_mse.dtype)
    loss = (per_transition_mse * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    with torch.no_grad():
        pred_valid = action_pred[valid]
        target_valid = action_target[valid]
        latent_delta_valid = (z_tp1 - z_t)[valid]

        mae = (pred_valid - target_valid).abs().mean()
        cosine = F.cosine_similarity(
            pred_valid,
            target_valid,
            dim=-1,
            eps=1e-8,
        ).mean()
        latent_delta_norm = torch.linalg.vector_norm(latent_delta_valid.float(), dim=-1)
        action_norm = torch.linalg.vector_norm(target_valid.float(), dim=-1)
        norm_corr = _pearson_corr(latent_delta_norm, action_norm)

    return {
        "loss": loss,
        "mae": mae,
        "cosine": cosine,
        "latent_action_norm_corr": norm_corr,
        "valid_fraction": valid_f.mean().detach(),
        "valid_count": valid_count.detach().to(dtype=torch.float32),
    }
