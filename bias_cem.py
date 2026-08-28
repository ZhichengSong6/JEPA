"""CEM solver that exposes the exact refinement center to BiasOnlyWorldModel."""
from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box


class BiasCEMSolver:
    def __init__(
        self,
        model,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1.0,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cuda",
        seed: int = 42,
        verbose: bool = True,
    ):
        self.model = model
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.var_scale = float(var_scale)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(self.seed)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any):
        if not isinstance(action_space, Box):
            raise TypeError(
                f"BiasCEMSolver requires Box action space, got {type(action_space)}"
            )
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        self._raw_action_dim = int(np.prod(action_space.shape[1:]))

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._raw_action_dim * int(self._config.action_block)

    @property
    def horizon(self):
        return int(self._config.horizon)

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def init_action_distrib(self, actions=None):
        var = self.var_scale * torch.ones(
            [self.n_envs, self.horizon, self.action_dim], dtype=torch.float32
        )
        mean = (
            torch.zeros(
                [self.n_envs, 0, self.action_dim], dtype=torch.float32
            )
            if actions is None
            else actions
        )
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            mean = torch.cat(
                [
                    mean,
                    torch.zeros(
                        [self.n_envs, remaining, self.action_dim],
                        dtype=mean.dtype,
                        device=mean.device,
                    ),
                ],
                dim=1,
            )
        return mean, var

    def _expand_info(self, info_dict, start_idx, end_idx):
        current_bs = end_idx - start_idx
        expanded = {}
        for k, v in info_dict.items():
            v_batch = v[start_idx:end_idx]
            if torch.is_tensor(v):
                v_batch = v_batch.to(self.device).unsqueeze(1)
                v_batch = v_batch.expand(
                    current_bs,
                    self.num_samples,
                    *v_batch.shape[2:],
                )
            elif isinstance(v, np.ndarray):
                v_batch = np.repeat(
                    v_batch[:, None, ...],
                    self.num_samples,
                    axis=1,
                )
            expanded[k] = v_batch
        return expanded

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None):
        if not hasattr(self.model, "get_cost_with_center"):
            raise TypeError(
                "BiasCEMSolver requires a model implementing get_cost_with_center()."
            )

        start_time = time.time()
        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        outputs = {"costs": [], "mean": [], "var": []}

        for start_idx in range(0, self.n_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            expanded_infos = self._expand_info(
                info_dict, start_idx, end_idx
            )
            final_batch_cost = None

            for _step in range(self.n_steps):
                noise = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=batch_mean.dtype,
                )
                candidates = (
                    batch_mean.unsqueeze(1)
                    + noise * batch_var.unsqueeze(1)
                )

                costs = self.model.get_cost_with_center(
                    expanded_infos.copy(),
                    candidates,
                    batch_mean,
                )
                if costs.shape != (current_bs, self.num_samples):
                    raise RuntimeError(
                        f"Expected cost {(current_bs, self.num_samples)}, "
                        f"got {tuple(costs.shape)}"
                    )

                topk_vals, topk_inds = torch.topk(
                    costs,
                    k=self.topk,
                    dim=1,
                    largest=False,
                )
                bidx = torch.arange(
                    current_bs, device=self.device
                )[:, None]
                elites = candidates[bidx, topk_inds]
                batch_mean = elites.mean(dim=1)
                batch_var = elites.std(dim=1)
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        if self.verbose:
            print(
                f"BiasCEM solve time: {time.time()-start_time:.4f}s | "
                f"steps={self.n_steps} samples={self.num_samples}"
            )
        return outputs
