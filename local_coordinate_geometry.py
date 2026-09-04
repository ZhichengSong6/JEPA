"""Invertible nonlinear latent coordinates for local planning geometry.

The adapter is an exactly invertible composition of additive coupling blocks:

    y = Phi(z),     z = Phi^{-1}(y).

Each block is initialized to identity.  Unlike the global SPD Stage-I adapter,
Phi is nonlinear, so the Euclidean metric in y induces a state-dependent local
metric in z:

    G(z) = J_Phi(z)^T J_Phi(z).

The adapter is part of the latent coordinate system itself, not a deployed
reward or metric head.  jepa.JEPA.predict() conjugates the frozen predictor
through adapter.inverse()/adapter(), so the underlying pretrained dynamics are
unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


class _ShiftMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, max_shift: float):
        super().__init__()
        self.max_shift = float(max_shift)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        # Exact identity initialization for the whole coupling block.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # Bound coordinate displacement while leaving the Jacobian free to be
        # state dependent.  This is a stability device, not a geometry target.
        return self.max_shift * torch.tanh(self.net(x))


class AdditiveCoupling(nn.Module):
    """Exactly invertible additive coupling with alternating conditioned half."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        flip: bool,
        max_shift: float,
    ):
        super().__init__()
        if dim % 2:
            raise ValueError(f"AdditiveCoupling needs even dim, got {dim}.")
        self.dim = int(dim)
        self.half = self.dim // 2
        self.flip = bool(flip)
        self.shift = _ShiftMLP(self.half, int(hidden_dim), float(max_shift))

    def _split(self, x):
        a, b = x.split(self.half, dim=-1)
        return (b, a) if self.flip else (a, b)

    def _merge(self, a, b):
        return torch.cat((b, a), dim=-1) if self.flip else torch.cat((a, b), dim=-1)

    def forward(self, x):
        a, b = self._split(x)
        b = b + self.shift(a)
        return self._merge(a, b)

    def inverse(self, y):
        a, b = self._split(y)
        b = b - self.shift(a)
        return self._merge(a, b)


class LocalCoordinateAdapter(nn.Module):
    """Volume-preserving nonlinear coordinate map with exact inverse."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        max_shift: float = 1.0,
    ):
        super().__init__()
        if dim % 2:
            raise ValueError(f"LocalCoordinateAdapter needs even dim, got {dim}.")
        if num_blocks < 2:
            raise ValueError("Use at least two coupling blocks.")
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.max_shift = float(max_shift)
        self.blocks = nn.ModuleList(
            [
                AdditiveCoupling(
                    self.dim,
                    self.hidden_dim,
                    flip=bool(i % 2),
                    max_shift=self.max_shift,
                )
                for i in range(self.num_blocks)
            ]
        )

    def forward(self, z):
        y = z
        for block in self.blocks:
            y = block(y)
        return y

    def inverse(self, y):
        z = y
        for block in reversed(self.blocks):
            z = block.inverse(z)
        return z

    def identity_regularizer(self, z):
        y = self(z)
        return (y - z).square().mean()

    @torch.no_grad()
    def diagnostics(self, sample=None):
        params = torch.cat(
            [p.detach().float().reshape(-1) for p in self.parameters()]
        )
        out = {
            "parameter_rms": float(params.square().mean().sqrt()),
            "parameter_abs_max": float(params.abs().max()),
            "num_blocks": self.num_blocks,
            "hidden_dim": self.hidden_dim,
            "max_shift": self.max_shift,
        }
        if sample is not None:
            x = sample.detach().float()
            y = self(x)
            xr = self.inverse(y)
            disp = y - x
            out.update(
                {
                    "sample_displacement_rms": float(
                        disp.square().mean().sqrt()
                    ),
                    "sample_displacement_abs_max": float(disp.abs().max()),
                    "sample_roundtrip_abs_max": float((xr - x).abs().max()),
                }
            )
        return out
