"""Traceable CEM solver for planner-visited landscape diagnostics.

This is numerically the same CEM update used by stable-worldmodel's standard
solver (Gaussian samples -> lowest-cost top-k -> mean/std update), but records
EVERY refinement center so we can later evaluate the learned and physical
landscape around the exact plans visited by CEM.

Trace files are diagnostic-only.  They are not consumed by the planner.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class TracedCEMSolver:
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
        trace_dir: str | None = None,
        save_candidates: bool = False,
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
        self.torch_gen = torch.Generator(device=self.device).manual_seed(self.seed)
        self.trace_dir = Path(
            trace_dir
            or os.environ.get("JEPA_CEM_TRACE_DIR", "outputs/pusht_cem_trace")
        )
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.save_candidates = bool(save_candidates)
        self.verbose = bool(verbose)
        self._solve_index = 0

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any):
        if not isinstance(action_space, Box):
            raise TypeError(f"TracedCEMSolver requires Box action space, got {type(action_space)}")
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
            torch.zeros([self.n_envs, 0, self.action_dim], dtype=torch.float32)
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
                v_batch = v_batch.expand(current_bs, self.num_samples, *v_batch.shape[2:])
            elif isinstance(v, np.ndarray):
                v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
            expanded[k] = v_batch
        return expanded

    def _capture_info(self, info_dict, env_idx):
        out = {}
        # Keep only compact state-like arrays needed to replay the physical center.
        for k in ("state", "goal_state", "proprio", "goal_proprio"):
            if k not in info_dict:
                continue
            try:
                arr = _np(info_dict[k][env_idx])
                if arr.size <= 512:
                    out[k] = arr
            except Exception:
                pass
        return out

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None):
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
            expanded_infos = self._expand_info(info_dict, start_idx, end_idx)

            mean_hist, var_hist = [], []
            elite_cost_mean_hist, elite_cost_min_hist = [], []
            candidate_hist = []
            candidate_cost_hist = []
            topk_index_hist = []
            final_batch_cost = None

            # Center BEFORE each CEM update: mu_0, ..., mu_{I-1}.
            for _step in range(self.n_steps):
                mean_hist.append(batch_mean.detach().cpu().numpy())
                var_hist.append(batch_var.detach().cpu().numpy())

                noise = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=batch_mean.dtype,
                )
                candidates = batch_mean.unsqueeze(1) + noise * batch_var.unsqueeze(1)
                costs = self.model.get_cost(expanded_infos.copy(), candidates)
                if costs.shape != (current_bs, self.num_samples):
                    raise RuntimeError(
                        f"Expected cost {(current_bs, self.num_samples)}, got {tuple(costs.shape)}"
                    )

                topk_vals, topk_inds = torch.topk(
                    costs, k=self.topk, dim=1, largest=False
                )
                bidx = torch.arange(current_bs, device=self.device)[:, None]
                elites = candidates[bidx, topk_inds]
                batch_mean = elites.mean(dim=1)
                batch_var = elites.std(dim=1)

                elite_cost_mean_hist.append(topk_vals.mean(dim=1).detach().cpu().numpy())
                elite_cost_min_hist.append(topk_vals.min(dim=1).values.detach().cpu().numpy())
                if self.save_candidates:
                    candidate_hist.append(candidates.detach().cpu().numpy())
                    candidate_cost_hist.append(costs.detach().cpu().numpy())
                    topk_index_hist.append(topk_inds.detach().cpu().numpy())
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            # Also save mu_I after the final update.
            mean_hist.append(batch_mean.detach().cpu().numpy())
            var_hist.append(batch_var.detach().cpu().numpy())

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

            for local in range(current_bs):
                self._solve_index += 1
                payload = {
                    "mean": np.stack([x[local] for x in mean_hist], axis=0),
                    "var": np.stack([x[local] for x in var_hist], axis=0),
                    "elite_cost_mean": np.asarray(
                        [x[local] for x in elite_cost_mean_hist], dtype=np.float32
                    ),
                    "elite_cost_min": np.asarray(
                        [x[local] for x in elite_cost_min_hist], dtype=np.float32
                    ),
                    "solve_index": np.asarray(self._solve_index, dtype=np.int64),
                    "env_index": np.asarray(start_idx + local, dtype=np.int64),
                    "n_steps": np.asarray(self.n_steps, dtype=np.int64),
                    "num_samples": np.asarray(self.num_samples, dtype=np.int64),
                    "topk": np.asarray(self.topk, dtype=np.int64),
                    "horizon": np.asarray(self.horizon, dtype=np.int64),
                    "action_block": np.asarray(int(self._config.action_block), dtype=np.int64),
                    "trace_value_space": np.asarray("normalized_planner_action"),
                }
                for k, arr in self._capture_info(info_dict, start_idx + local).items():
                    payload[f"info_{k}"] = arr
                if self.save_candidates:
                    payload["candidates"] = np.stack(
                        [x[local] for x in candidate_hist], axis=0
                    )
                    payload["candidate_costs"] = np.stack(
                        [x[local] for x in candidate_cost_hist], axis=0
                    )
                    payload["topk_indices"] = np.stack(
                        [x[local] for x in topk_index_hist], axis=0
                    )
                path = self.trace_dir / f"solve_{self._solve_index:06d}.npz"
                np.savez_compressed(path, **payload)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        if self.verbose:
            print(
                f"TracedCEM solve time: {time.time()-start_time:.4f}s | "
                f"traces={self.trace_dir} total_solves={self._solve_index}"
            )
        return outputs
