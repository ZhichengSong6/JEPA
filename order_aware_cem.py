"""Symmetry-aware CEM diagnostic for JEPA planner-landscape order mismatch.

This module intentionally targets a very narrow question:

    Can a planner-side rule avoid continuing to optimize with a potentially
    spurious first-order predicted cost once the local landscape becomes
    curvature/even dominated?

A naive "fully evenized CEM" is NOT a valid optimizer: if paired candidates
U+ = mu + delta and U- = mu - delta receive the same averaged cost, symmetric
elites average exactly back to mu and the CEM mean cannot move.

We therefore use symmetric candidate pairs only to *diagnose local landscape
order*.  While the predicted landscape is odd/slope dominated, CEM performs
its ordinary raw-cost update.  Once the paired local landscape becomes
curvature/even dominated, the solver stops updating the mean and returns the
current plan instead of allowing later CEM iterations to chase a potentially
bias-induced first-order tilt.

Two modes are provided:

  paired_raw  : symmetric sampling, otherwise ordinary raw-cost CEM.
                This is the sampling-control experiment.

  order_stop  : same symmetric sampling and raw-cost CEM updates, but stop
                when RMS(odd) / RMS(even increment) <= transition_ratio.

The implementation matches the stable-worldmodel 0.0.x CEM interface used by
this repository (constructor takes ``model`` and WorldModelPolicy calls
``configure`` + ``solve``).
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box


class OrderAwareCEMSolver:
    """Paired CEM with an optional landscape-order-aware stopping rule."""

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
        mode: str = "order_stop",
        transition_ratio: float = 1.0,
        min_steps: int = 5,
        eps: float = 1.0e-8,
        verbose: bool = True,
    ) -> None:
        if num_samples % 2 != 0:
            raise ValueError(
                "OrderAwareCEMSolver requires an even num_samples so every "
                f"candidate has a symmetric partner; got {num_samples}."
            )
        if num_samples < 4:
            raise ValueError("num_samples must be >= 4.")
        if topk < 1 or topk > num_samples:
            raise ValueError(f"Invalid topk={topk} for num_samples={num_samples}.")
        if mode not in {"paired_raw", "order_stop"}:
            raise ValueError(
                f"mode must be 'paired_raw' or 'order_stop', got {mode!r}."
            )
        if transition_ratio <= 0:
            raise ValueError("transition_ratio must be positive.")
        if min_steps < 1:
            raise ValueError("min_steps must be >= 1.")

        self.model = model
        self.batch_size = int(batch_size)
        self.var_scale = float(var_scale)
        self.num_samples = int(num_samples)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.device = torch.device(device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(int(seed))

        self.mode = mode
        self.transition_ratio = float(transition_ratio)
        self.min_steps = int(min_steps)
        self.eps = float(eps)
        self.verbose = bool(verbose)

        # Counts environment-level planning problems, not outer batched calls.
        self._solve_count = 0
        self._stopped_count = 0
        self._stop_steps: list[int] = []
        self._final_ratios: list[float] = []

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            print(
                "WARNING: OrderAwareCEMSolver was designed for continuous Box "
                f"actions, got {type(action_space)}."
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * int(self._config.action_block)

    @property
    def horizon(self) -> int:
        return int(self._config.horizon)

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def init_action_distrib(
        self, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Match stable-worldmodel 0.0.x CEM semantics: `var` is used as the
        # Gaussian scale directly, despite the historical variable name.
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
            new_mean = torch.zeros(
                [self.n_envs, remaining, self.action_dim],
                dtype=mean.dtype,
                device=mean.device,
            )
            mean = torch.cat([mean, new_mean], dim=1)

        return mean, var

    def _make_paired_candidates(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        current_bs: int,
    ) -> torch.Tensor:
        """Return [B,N,H,D] candidates arranged as (+,-) pairs.

        Pair 0 is exactly (mean, mean).  The remaining N/2-1 pairs use exact
        antithetic Gaussian noise.  Therefore the total model-evaluation budget
        remains exactly `num_samples` per CEM iteration.
        """
        n_pairs = self.num_samples // 2
        noise = torch.randn(
            current_bs,
            n_pairs,
            self.horizon,
            self.action_dim,
            generator=self.torch_gen,
            device=self.device,
            dtype=batch_mean.dtype,
        )
        noise[:, 0] = 0.0

        delta = noise * batch_var.unsqueeze(1)
        plus = batch_mean.unsqueeze(1) + delta
        minus = batch_mean.unsqueeze(1) - delta

        # Interleaved layout: [pair0+, pair0-, pair1+, pair1-, ...].
        return torch.stack([plus, minus], dim=2).reshape(
            current_bs,
            self.num_samples,
            self.horizon,
            self.action_dim,
        )

    def _order_ratio(self, costs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compute model-only local odd/even order statistics.

        For pair i:
            O_i = (C_i+ - C_i-) / 2
            E_i = (C_i+ + C_i-) / 2

        Pair 0 is (mean,mean), hence E_0 = C(mean).  The local even increment is
            Q_i = E_i - E_0.

        We report
            R = RMS(O_i) / (RMS(Q_i) + eps),  i > 0.

        R > 1 : first-order/odd variation dominates sampled even curvature.
        R <= 1: even/curvature variation is at least as large as odd variation.

        This is deliberately a *model-only* diagnostic; no simulator state or
        future oracle is used by the stopping decision.
        """
        pair_costs = costs.reshape(costs.shape[0], self.num_samples // 2, 2)
        c_plus = pair_costs[..., 0]
        c_minus = pair_costs[..., 1]

        odd = 0.5 * (c_plus - c_minus)
        even = 0.5 * (c_plus + c_minus)
        center = even[:, :1]

        # Exclude pair 0 because it is the duplicated center and contributes
        # exact zeros to both terms.
        odd_nonzero = odd[:, 1:]
        even_increment = even[:, 1:] - center

        odd_rms = odd_nonzero.pow(2).mean(dim=1).sqrt()
        even_rms = even_increment.pow(2).mean(dim=1).sqrt()
        ratio = odd_rms / even_rms.clamp_min(self.eps)
        return ratio, odd_rms, even_rms, center.squeeze(1)

    def diagnostics_summary(self) -> dict[str, Any]:
        ratios = np.asarray(self._final_ratios, dtype=np.float64)
        stops = np.asarray(self._stop_steps, dtype=np.float64)
        return {
            "mode": self.mode,
            "solve_count": int(self._solve_count),
            "stopped_count": int(self._stopped_count),
            "stop_fraction": (
                float(self._stopped_count / self._solve_count)
                if self._solve_count
                else 0.0
            ),
            "mean_stop_step": float(stops.mean()) if stops.size else None,
            "median_stop_step": float(np.median(stops)) if stops.size else None,
            "mean_final_order_ratio": float(ratios.mean()) if ratios.size else None,
            "median_final_order_ratio": (
                float(np.median(ratios)) if ratios.size else None
            ),
        }

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        start_time = time.time()
        outputs = {"costs": [], "mean": [], "var": []}

        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        total_envs = self.n_envs

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            self._solve_count += current_bs
            solve_id = self._solve_count

            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded_infos = {}
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
                        v_batch[:, None, ...], self.num_samples, axis=1
                    )
                expanded_infos[k] = v_batch

            final_batch_cost = None
            last_ratio = torch.full(
                (current_bs,), float("nan"), device=self.device
            )
            stop_step = None

            for step in range(self.n_steps):
                candidates = self._make_paired_candidates(
                    batch_mean, batch_var, current_bs
                )

                costs = self.model.get_cost(expanded_infos.copy(), candidates)
                if not isinstance(costs, torch.Tensor):
                    raise TypeError(
                        f"Expected model cost Tensor, got {type(costs)}."
                    )
                if costs.shape != (current_bs, self.num_samples):
                    raise RuntimeError(
                        "Expected cost shape "
                        f"({current_bs},{self.num_samples}), got {tuple(costs.shape)}."
                    )

                ratio, odd_rms, even_rms, center_cost = self._order_ratio(costs)
                last_ratio = ratio

                # The official PushT evaluator uses solver batch_size=1.  We
                # support larger current batches conservatively: only stop the
                # whole batch when every environment has entered the even regime.
                should_stop = (
                    self.mode == "order_stop"
                    and (step + 1) >= self.min_steps
                    and bool(torch.all(ratio <= self.transition_ratio))
                )

                if should_stop:
                    stop_step = step + 1
                    final_batch_cost = center_cost.cpu().tolist()
                    self._stopped_count += current_bs
                    self._stop_steps.extend([stop_step] * current_bs)
                    if self.verbose:
                        print(
                            f"[order-aware CEM] solve={solve_id} STOP step={stop_step}/{self.n_steps} "
                            f"ratio={ratio.mean().item():.4f} "
                            f"odd_rms={odd_rms.mean().item():.4g} "
                            f"even_rms={even_rms.mean().item():.4g}"
                        )
                    break

                # Ordinary raw-cost CEM update.  Pairing changes only the Monte
                # Carlo sampling pattern; no cost is corrected here.
                topk_vals, topk_inds = torch.topk(
                    costs, k=self.topk, dim=1, largest=False
                )
                batch_indices = (
                    torch.arange(current_bs, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, self.topk)
                )
                topk_candidates = candidates[batch_indices, topk_inds]
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            if final_batch_cost is None:
                raise RuntimeError("CEM produced no final cost.")

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

            finite_ratios = last_ratio[torch.isfinite(last_ratio)]
            self._final_ratios.extend(finite_ratios.detach().cpu().tolist())

            if self.verbose and stop_step is None:
                print(
                    f"[order-aware CEM] solve={solve_id} END step={self.n_steps}/{self.n_steps} "
                    f"ratio={last_ratio.mean().item():.4f} mode={self.mode}"
                )

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        if self.verbose:
            print(
                f"OrderAwareCEM solve time: {time.time() - start_time:.4f}s | "
                f"summary={self.diagnostics_summary()}"
            )

        return outputs
