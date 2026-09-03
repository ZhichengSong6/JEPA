"""Coordinate-system hypothesis test for planning-compatible JEPA latents.

Stage-I keeps the pretrained LeWM world model fixed and learns only a global
invertible SPD coordinate transform

    y = A z,    A = exp(1/2 S),    S = S^T.

Raw Euclidean distance in y is therefore the Mahalanobis geometry
exp(S) in the original z coordinates.  The adapter is intentionally part of
the model's latent coordinate system, not an external reward/readout head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SPDCoordinateAdapter(nn.Module):
    """Invertible near-identity latent coordinate transform."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.log_metric_raw = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.register_buffer(
            "_cached_matrix", torch.eye(self.dim), persistent=True
        )
        self.register_buffer(
            "_cached_inverse", torch.eye(self.dim), persistent=True
        )
        self.register_buffer(
            "_cache_valid", torch.tensor(False), persistent=True
        )

    def symmetric_log_metric(self) -> torch.Tensor:
        x = self.log_metric_raw
        return 0.5 * (x + x.transpose(-1, -2))

    def _fresh_matrix(self) -> torch.Tensor:
        return torch.matrix_exp(0.5 * self.symmetric_log_metric())

    def _fresh_inverse(self) -> torch.Tensor:
        return torch.matrix_exp(-0.5 * self.symmetric_log_metric())

    def matrix(self) -> torch.Tensor:
        if not self.training and bool(self._cache_valid.item()):
            return self._cached_matrix
        return self._fresh_matrix()

    def inverse_matrix(self) -> torch.Tensor:
        if not self.training and bool(self._cache_valid.item()):
            return self._cached_inverse
        return self._fresh_inverse()

    @torch.no_grad()
    def refresh_cache(self) -> None:
        self._cached_matrix.copy_(self._fresh_matrix())
        self._cached_inverse.copy_(self._fresh_inverse())
        self._cache_valid.fill_(True)

    def train(self, mode: bool = True):
        if mode:
            self._cache_valid.fill_(False)
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear(x, A) == x @ A^T.
        return F.linear(x.float(), self.matrix().float())

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return F.linear(y.float(), self.inverse_matrix().float())

    def identity_regularizer(self) -> torch.Tensor:
        # Normalize by latent dimension rather than D^2: changing a small
        # number of latent modes should still incur a visible penalty.
        s = self.symmetric_log_metric()
        return s.square().sum() / float(self.dim)

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        s = torch.linalg.eigvalsh(self.symmetric_log_metric().float())
        metric_eigs = torch.exp(s)
        coord_scales = torch.exp(0.5 * s)
        return {
            "metric_eig_min": float(metric_eigs.min()),
            "metric_eig_max": float(metric_eigs.max()),
            "metric_condition": float(
                metric_eigs.max() / metric_eigs.min().clamp_min(1e-12)
            ),
            "coord_scale_min": float(coord_scales.min()),
            "coord_scale_max": float(coord_scales.max()),
            "log_metric_fro": float(
                torch.linalg.vector_norm(self.symmetric_log_metric().float())
            ),
        }
