#!/usr/bin/env python3
"""
Raw-latent gradient planner for the official/full PushT LeWM evaluation.

This is intentionally a planner diagnostic, not a new model or a new cost.
It keeps the official evaluation path and changes only the action optimizer:

    CEM sampling  -->  Adam on the SAME terminal raw-latent LeWM cost.

The optimized variable is unconstrained v, while normalized model actions are
obtained through an affine tanh map whose endpoints correspond exactly to the
raw PushT action-space bounds after the official StandardScaler transform.
Thus the environment action always remains inside its Box bounds.

No factor head, GT state, readout, or privileged physical quantity is used by
the planner. GT state is read only AFTER/ALONGSIDE execution for evaluation.

The script can evaluate multiple checkpoints and multiple restart counts on the
same dataset starts. Start 0 is always the zero-action plan; additional starts
are random perturbations in unconstrained v-space.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from omegaconf import OmegaConf
from sklearn import preprocessing

from eval import get_dataset, get_episodes_length, img_transform


# -----------------------------------------------------------------------------
# CLI / utilities
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Raw-latent Adam planner for full PushT LeWM checkpoints."
    )
    p.add_argument(
        "--policies",
        nargs="+",
        required=True,
        help=(
            "Checkpoint names/paths relative to STABLEWM_HOME, without "
            "_object.ckpt, exactly as accepted by AutoCostModel."
        ),
    )
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--config", default="config/eval/pusht.yaml")
    p.add_argument("--num-eval", type=int, default=20)
    p.add_argument(
        "--num-starts",
        nargs="+",
        type=int,
        default=[1],
        help="Gradient restarts. Start 0 is always the zero-action plan.",
    )
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument(
        "--init-std",
        type=float,
        default=0.5,
        help="Std of random starts in unconstrained tanh-variable space.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of parallel environments optimized in each gradient batch.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--strict-theta-deg", type=float, default=10.0)
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: $STABLEWM_HOME/pusht_gradient_planner",
    )
    p.add_argument(
        "--save-video",
        action="store_true",
        help="Save videos through the official world evaluator.",
    )
    return p.parse_args()


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _latest_state(state_array):
    x = _numpy(state_array)
    if x.ndim == 3:
        x = x[:, -1]
    if x.ndim != 2 or x.shape[-1] < 5:
        raise RuntimeError(f"Unexpected PushT state shape: {x.shape}")
    return np.asarray(x, dtype=np.float64)


def _angle_error_rad(a, b):
    d = np.asarray(a) - np.asarray(b)
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def _error_components(states, goals):
    states = np.asarray(states, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    return {
        "pusher_xy": np.linalg.norm(states[:, :2] - goals[:, :2], axis=1),
        "block_xy": np.linalg.norm(states[:, 2:4] - goals[:, 2:4], axis=1),
        "joint_pos": np.linalg.norm(states[:, :4] - goals[:, :4], axis=1),
        "theta_rad": _angle_error_rad(states[:, 4], goals[:, 4]),
    }


def _summary(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }


def _safe_relative_reduction(initial, final):
    initial = float(initial)
    final = float(final)
    if not np.isfinite(initial) or not np.isfinite(final) or abs(initial) <= 1e-12:
        return float("nan")
    return float((initial - final) / abs(initial))


def _label(policy, labels, i):
    if labels is not None:
        return labels[i]
    return Path(policy).name.replace("_epoch_10", "").replace("lewm_", "")


# -----------------------------------------------------------------------------
# Raw-latent gradient solver
# -----------------------------------------------------------------------------


class RawLatentGradientSolver:
    """Adam planner using the model's existing get_cost() without modification.

    The solver obeys the old stable-worldmodel Solver protocol used by this
    LeWM checkout. `action_mean` and `action_scale` are the exact statistics of
    the official StandardScaler fitted in evaluation.
    """

    def __init__(
        self,
        model,
        action_mean,
        action_scale,
        *,
        n_steps=30,
        num_starts=1,
        lr=0.1,
        init_std=0.5,
        batch_size=4,
        device="cuda:0",
        seed=42,
    ):
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if num_starts <= 0:
            raise ValueError("num_starts must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model = model
        self.n_steps = int(n_steps)
        self.num_starts = int(num_starts)
        self.lr = float(lr)
        self.init_std = float(init_std)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.action_mean = np.asarray(action_mean, dtype=np.float32).reshape(-1)
        self.action_scale = np.asarray(action_scale, dtype=np.float32).reshape(-1)
        self.action_scale = np.maximum(self.action_scale, 1e-8)

        self.torch_gen = torch.Generator(device=self.device).manual_seed(self.seed)
        self.records = []
        self.solve_summaries = []
        self._solve_index = 0
        self._configured = False

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        if not isinstance(action_space, Box):
            raise TypeError(
                f"RawLatentGradientSolver requires a continuous Box action space, got {type(action_space)}"
            )

        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config

        # The vectorized World action space is [n_envs, raw_action_dim].
        # Match stable-worldmodel's solver convention exactly.
        self._raw_action_dim = int(np.prod(action_space.shape[1:]))
        if self._raw_action_dim != self.action_mean.size:
            raise RuntimeError(
                "Action-stat dimension does not match environment action dimension: "
                f"stats={self.action_mean.size}, env={self._raw_action_dim}, "
                f"action_space.shape={action_space.shape}"
            )

        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        if low.ndim >= 2 and low.shape[0] == self._n_envs:
            low = low[0].reshape(-1)
            high = high[0].reshape(-1)
        else:
            low = low.reshape(-1)
            high = high.reshape(-1)

        low = low[: self._raw_action_dim]
        high = high[: self._raw_action_dim]
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise RuntimeError("Gradient planner currently requires finite Box action bounds.")

        # WorldModelPolicy asks the solver to optimize *normalized* actions and
        # inverse-transforms them afterwards. Convert raw Box endpoints through
        # that exact affine StandardScaler here.
        norm_low = (low - self.action_mean) / self.action_scale
        norm_high = (high - self.action_mean) / self.action_scale

        # One coarse LeWM action concatenates action_block raw actions.
        norm_low = np.tile(norm_low, int(config.action_block))
        norm_high = np.tile(norm_high, int(config.action_block))

        self._norm_low = torch.as_tensor(norm_low, device=self.device)
        self._norm_high = torch.as_tensor(norm_high, device=self.device)
        self._norm_mid = 0.5 * (self._norm_low + self._norm_high)
        self._norm_half = 0.5 * (self._norm_high - self._norm_low)
        self._configured = True

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

    def _v_to_action(self, v):
        """Unconstrained v -> bounded normalized LeWM action."""
        return self._norm_mid + self._norm_half * torch.tanh(v)

    def _action_to_v(self, action):
        """Bounded normalized action -> unconstrained v for warm starts."""
        x = (action - self._norm_mid) / torch.clamp(self._norm_half, min=1e-8)
        x = torch.clamp(x, -0.999, 0.999)
        return torch.atanh(x)

    def _init_v(self, env_slice, init_action):
        current_bs = env_slice.stop - env_slice.start
        base_v = torch.zeros(
            current_bs,
            self.horizon,
            self.action_dim,
            device=self.device,
            dtype=torch.float32,
        )

        if init_action is not None and init_action.numel() > 0:
            warm = init_action[env_slice].to(self.device).float()
            keep = min(int(warm.shape[1]), self.horizon)
            if keep > 0:
                base_v[:, :keep] = self._action_to_v(warm[:, :keep])

        v = base_v.unsqueeze(1).repeat(1, self.num_starts, 1, 1)
        if self.num_starts > 1 and self.init_std > 0:
            noise = torch.randn(
                current_bs,
                self.num_starts - 1,
                self.horizon,
                self.action_dim,
                generator=self.torch_gen,
                device=self.device,
            )
            v[:, 1:] += self.init_std * noise

        # start 0 remains exact warm/zero initialization.
        return v

    def _expand_info(self, info_dict, start_idx, end_idx):
        current_bs = end_idx - start_idx
        out = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                x = v[start_idx:end_idx]
                x = x.unsqueeze(1).expand(current_bs, self.num_starts, *x.shape[1:])
                out[k] = x
            elif isinstance(v, np.ndarray):
                x = v[start_idx:end_idx]
                out[k] = np.repeat(x[:, None, ...], self.num_starts, axis=1)
            else:
                out[k] = v
        return out

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        if not self._configured:
            raise RuntimeError("Solver must be configured by WorldModelPolicy.set_env first.")

        solve_index = self._solve_index
        self._solve_index += 1
        solve_start = time.time()

        top_actions = []
        solve_record_indices = []

        for start_idx in range(0, self.n_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_envs)
            env_slice = slice(start_idx, end_idx)
            current_bs = end_idx - start_idx

            v = self._init_v(env_slice, init_action).detach()
            v.requires_grad_(True)
            optim = torch.optim.Adam([v], lr=self.lr)
            expanded_infos = self._expand_info(info_dict, start_idx, end_idx)

            best_cost_curve = [[] for _ in range(current_bs)]
            zero_cost_curve = [[] for _ in range(current_bs)]
            zero_grad_curve = [[] for _ in range(current_bs)]
            mean_grad_curve = [[] for _ in range(current_bs)]
            best_initial_cost = None
            zero_initial_cost = None

            for step in range(self.n_steps):
                optim.zero_grad(set_to_none=True)
                actions = self._v_to_action(v)
                costs = self.model.get_cost(expanded_infos.copy(), actions)

                if costs.shape != (current_bs, self.num_starts):
                    raise RuntimeError(
                        f"Expected costs {(current_bs, self.num_starts)}, got {tuple(costs.shape)}"
                    )
                if not torch.isfinite(costs).all():
                    raise RuntimeError(
                        f"Non-finite latent planning cost at solve={solve_index}, step={step}."
                    )
                if not costs.requires_grad:
                    raise RuntimeError("LeWM cost does not require grad with respect to actions.")

                if step == 0:
                    best_initial_cost = costs.detach().min(dim=1).values.cpu().numpy()
                    zero_initial_cost = costs.detach()[:, 0].cpu().numpy()

                for bi in range(current_bs):
                    best_cost_curve[bi].append(float(costs[bi].min().detach().cpu()))
                    zero_cost_curve[bi].append(float(costs[bi, 0].detach().cpu()))

                loss = costs.sum()
                loss.backward()

                if v.grad is None or not torch.isfinite(v.grad).all():
                    raise RuntimeError(
                        f"Non-finite/missing action gradient at solve={solve_index}, step={step}."
                    )

                grad_norm = torch.linalg.vector_norm(
                    v.grad.reshape(current_bs, self.num_starts, -1), dim=-1
                )
                for bi in range(current_bs):
                    zero_grad_curve[bi].append(float(grad_norm[bi, 0].detach().cpu()))
                    mean_grad_curve[bi].append(float(grad_norm[bi].mean().detach().cpu()))

                optim.step()

            # Evaluate AFTER the final Adam update.
            with torch.no_grad():
                final_actions_all = self._v_to_action(v)
                final_costs = self.model.get_cost(
                    expanded_infos.copy(), final_actions_all
                )
                best_idx = final_costs.argmin(dim=1)

                for bi in range(current_bs):
                    best_cost_curve[bi].append(float(final_costs[bi].min().cpu()))
                    zero_cost_curve[bi].append(float(final_costs[bi, 0].cpu()))

                batch_ids = torch.arange(current_bs, device=self.device)
                selected_actions = final_actions_all[batch_ids, best_idx]
                selected_tanh = torch.tanh(v[batch_ids, best_idx])
                top_actions.append(selected_actions.detach().cpu())

            for bi in range(current_bs):
                env_i = start_idx + bi
                final_best = float(final_costs[bi, best_idx[bi]].detach().cpu())
                initial_zero = float(zero_initial_cost[bi])
                initial_best = float(best_initial_cost[bi])
                initial_grad_zero = float(zero_grad_curve[bi][0])
                initial_rel_grad_zero = initial_grad_zero / max(abs(initial_zero), 1e-12)
                saturation = float(
                    (selected_tanh[bi].abs() > 0.95).float().mean().detach().cpu()
                )

                rec = {
                    "solve_index": int(solve_index),
                    "env_index": int(env_i),
                    "num_starts": int(self.num_starts),
                    "n_steps": int(self.n_steps),
                    "lr": float(self.lr),
                    "zero_initial_cost": initial_zero,
                    "best_initial_cost": initial_best,
                    "final_best_cost": final_best,
                    "zero_to_final_relative_reduction": _safe_relative_reduction(
                        initial_zero, final_best
                    ),
                    "best_to_final_relative_reduction": _safe_relative_reduction(
                        initial_best, final_best
                    ),
                    "selected_start": int(best_idx[bi].detach().cpu()),
                    "initial_zero_grad_norm": initial_grad_zero,
                    "initial_zero_grad_over_cost": float(initial_rel_grad_zero),
                    "final_action_saturation_fraction": saturation,
                    "best_cost_curve": best_cost_curve[bi],
                    "zero_cost_curve": zero_cost_curve[bi],
                    "zero_grad_norm_curve": zero_grad_curve[bi],
                    "mean_grad_norm_curve": mean_grad_curve[bi],
                }
                self.records.append(rec)
                solve_record_indices.append(len(self.records) - 1)

        elapsed = time.time() - solve_start
        self.solve_summaries.append(
            {
                "solve_index": int(solve_index),
                "num_envs": int(self.n_envs),
                "num_starts": int(self.num_starts),
                "elapsed_seconds": float(elapsed),
                "record_indices": solve_record_indices,
            }
        )
        print(
            f"Gradient solve {solve_index}: starts={self.num_starts}, "
            f"steps={self.n_steps}, time={elapsed:.3f}s"
        )

        return {"actions": torch.cat(top_actions, dim=0)}


# -----------------------------------------------------------------------------
# Official eval setup shared across every model/restart count
# -----------------------------------------------------------------------------


def _build_process(cfg, dataset):
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = np.asarray(dataset.get_col_data(col))
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    if "action" not in process:
        raise RuntimeError("Official PushT eval requires an action StandardScaler.")
    return process


def _select_eval_rows(cfg, dataset, num_eval, seed):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - int(cfg.eval.goal_offset_steps) - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.asarray(
        [max_start_idx_dict[ep] for ep in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    if num_eval > len(valid_indices) - 1:
        raise ValueError(f"Requested {num_eval} eval starts, only {len(valid_indices)-1} available.")

    # Match eval.py exactly, including the historical len(valid_indices)-1 choice.
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    rows = np.sort(valid_indices[sampled])
    data = dataset.get_row_data(rows)
    return (
        col_name,
        rows,
        np.asarray(data[col_name]),
        np.asarray(data["step_idx"]),
    )


def _load_start_goal_states(dataset, episodes, start_steps, goal_offset):
    starts = np.asarray(start_steps)
    ends = starts + int(goal_offset)
    chunks = dataset.load_chunk(np.asarray(episodes), starts, ends)
    start_states, goal_states = [], []
    for ep in chunks:
        s = _numpy(ep["state"])
        start_states.append(np.asarray(s[0], dtype=np.float64))
        goal_states.append(np.asarray(s[-1], dtype=np.float64))
    return np.stack(start_states), np.stack(goal_states)


def _solver_aggregate(records, solve_summaries):
    def vals(key):
        return [r[key] for r in records]

    return {
        "num_planning_records": len(records),
        "num_solver_calls": len(solve_summaries),
        "total_solver_time_seconds": float(
            sum(x["elapsed_seconds"] for x in solve_summaries)
        ),
        "zero_initial_cost": _summary(vals("zero_initial_cost")),
        "best_initial_cost": _summary(vals("best_initial_cost")),
        "final_best_cost": _summary(vals("final_best_cost")),
        "zero_to_final_relative_reduction": _summary(
            vals("zero_to_final_relative_reduction")
        ),
        "best_to_final_relative_reduction": _summary(
            vals("best_to_final_relative_reduction")
        ),
        "initial_zero_grad_norm": _summary(vals("initial_zero_grad_norm")),
        "initial_zero_grad_over_cost": _summary(
            vals("initial_zero_grad_over_cost")
        ),
        "final_action_saturation_fraction": _summary(
            vals("final_action_saturation_fraction")
        ),
        "selected_nonzero_start_fraction": float(
            np.mean([r["selected_start"] != 0 for r in records])
        ),
    }


def _run_one(
    cfg,
    dataset,
    eval_rows,
    eval_episodes,
    eval_start_steps,
    start_states,
    goal_states,
    *,
    policy_name,
    label,
    num_starts,
    args,
    output_root,
):
    device = torch.device(args.device)

    # A fresh World/policy for every model+restart setting prevents state carryover.
    cfg.world.num_envs = int(args.num_eval)
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    world = swm.World(**cfg.world, image_shape=(224, 224))

    process = _build_process(cfg, dataset)
    model = swm.policy.AutoCostModel(policy_name).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    solver = RawLatentGradientSolver(
        model,
        action_mean=process["action"].mean_,
        action_scale=process["action"].scale_,
        n_steps=args.n_steps,
        num_starts=num_starts,
        lr=args.lr,
        init_std=args.init_std,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    plan_config = swm.PlanConfig(**cfg.plan_config)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )
    world.set_policy(policy)

    n = len(eval_episodes)
    first_official_step = np.full(n, -1, dtype=np.int64)
    first_strict_step = np.full(n, -1, dtype=np.int64)
    first_official_state = np.full_like(goal_states, np.nan, dtype=np.float64)
    first_strict_state = np.full_like(goal_states, np.nan, dtype=np.float64)

    original_step = world.step
    step_counter = 0
    strict_theta = np.deg2rad(float(args.strict_theta_deg))

    def recording_step():
        nonlocal step_counter
        original_step()
        step_counter += 1
        states = _latest_state(world.infos["state"])

        official_now = np.asarray(world.terminateds, dtype=bool)
        new_official = official_now & (first_official_step < 0)
        if np.any(new_official):
            first_official_step[new_official] = step_counter
            first_official_state[new_official] = states[new_official]

        err = _error_components(states, goal_states)
        strict_now = (err["joint_pos"] < 20.0) & (err["theta_rad"] < strict_theta)
        new_strict = strict_now & (first_strict_step < 0)
        if np.any(new_strict):
            first_strict_step[new_strict] = step_counter
            first_strict_state[new_strict] = states[new_strict]

    world.step = recording_step

    run_name = f"{label}_starts{num_starts}"
    run_dir = output_root / f"starts_{num_starts}" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / "videos"

    t0 = time.time()
    try:
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_start_steps.tolist(),
            goal_offset_steps=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            save_video=bool(args.save_video),
            video_path=video_path,
        )
    finally:
        world.step = original_step
    elapsed = time.time() - t0

    final_states = _latest_state(world.infos["state"])
    final_err = _error_components(final_states, goal_states)
    theta_deg = np.rad2deg(final_err["theta_rad"])

    official_success = np.asarray(metrics["episode_successes"], dtype=bool)
    strict_success = first_strict_step >= 0
    final_official = (final_err["joint_pos"] < 20.0) & (
        final_err["theta_rad"] < np.pi / 9
    )
    final_strict = (final_err["joint_pos"] < 20.0) & (
        final_err["theta_rad"] < strict_theta
    )

    # Solver is called once per MPC replan. Attach solve records to episode ids.
    planning_by_env = {i: [] for i in range(n)}
    for rec in solver.records:
        planning_by_env[int(rec["env_index"])].append(rec)

    episode_rows = []
    for i in range(n):
        recs = sorted(planning_by_env[i], key=lambda x: x["solve_index"])
        episode_rows.append(
            {
                "eval_index": i + 1,
                "dataset_row": int(eval_rows[i]),
                "episode_idx": int(eval_episodes[i]),
                "start_step": int(eval_start_steps[i]),
                "official_rollout_success": bool(official_success[i]),
                "strict_rollout_success": bool(strict_success[i]),
                "first_official_success_step": int(first_official_step[i]),
                "first_strict_success_step": int(first_strict_step[i]),
                "final_pusher_xy_error_px": float(final_err["pusher_xy"][i]),
                "final_block_xy_error_px": float(final_err["block_xy"][i]),
                "final_joint_position_error_px": float(final_err["joint_pos"][i]),
                "final_theta_error_deg": float(theta_deg[i]),
                "final_within_official_20deg": bool(final_official[i]),
                "final_within_strict_theta": bool(final_strict[i]),
                "num_replans": len(recs),
                "mean_zero_to_final_cost_reduction": float(
                    np.mean([r["zero_to_final_relative_reduction"] for r in recs])
                )
                if recs
                else float("nan"),
                "mean_final_best_latent_cost": float(
                    np.mean([r["final_best_cost"] for r in recs])
                )
                if recs
                else float("nan"),
                "mean_initial_zero_grad_norm": float(
                    np.mean([r["initial_zero_grad_norm"] for r in recs])
                )
                if recs
                else float("nan"),
                "mean_action_saturation_fraction": float(
                    np.mean([r["final_action_saturation_fraction"] for r in recs])
                )
                if recs
                else float("nan"),
                "start_pusher_x": float(start_states[i, 0]),
                "start_pusher_y": float(start_states[i, 1]),
                "start_block_x": float(start_states[i, 2]),
                "start_block_y": float(start_states[i, 3]),
                "start_theta_rad": float(start_states[i, 4]),
                "goal_pusher_x": float(goal_states[i, 0]),
                "goal_pusher_y": float(goal_states[i, 1]),
                "goal_block_x": float(goal_states[i, 2]),
                "goal_block_y": float(goal_states[i, 3]),
                "goal_theta_rad": float(goal_states[i, 4]),
                "final_pusher_x": float(final_states[i, 0]),
                "final_pusher_y": float(final_states[i, 1]),
                "final_block_x": float(final_states[i, 2]),
                "final_block_y": float(final_states[i, 3]),
                "final_theta_rad": float(final_states[i, 4]),
            }
        )

    solver_summary = _solver_aggregate(solver.records, solver.solve_summaries)
    summary = {
        "run_name": run_name,
        "policy": policy_name,
        "label": label,
        "num_eval": n,
        "num_starts": int(num_starts),
        "n_steps": int(args.n_steps),
        "lr": float(args.lr),
        "init_std": float(args.init_std),
        "official_rollout_success_rate": float(metrics["success_rate"]),
        "strict_rollout_success_rate": float(np.mean(strict_success) * 100.0),
        "strict_theta_deg": float(args.strict_theta_deg),
        "final_within_official_20deg_rate": float(np.mean(final_official) * 100.0),
        "final_within_strict_theta_rate": float(np.mean(final_strict) * 100.0),
        "final_pusher_xy_error_px": _summary(final_err["pusher_xy"]),
        "final_block_xy_error_px": _summary(final_err["block_xy"]),
        "final_joint_position_error_px": _summary(final_err["joint_pos"]),
        "final_theta_error_deg": _summary(theta_deg),
        "solver": solver_summary,
        "evaluation_time_seconds": float(elapsed),
        "planner_note": (
            "Planner optimizes the unchanged LeWM terminal raw-latent cost. "
            "No factor head or GT state enters the planner. Physical state is "
            "used only for post-hoc success/error diagnostics. Actions are "
            "bounded through an affine tanh map in normalized action space."
        ),
    }

    with (run_dir / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        w.writeheader()
        w.writerows(episode_rows)

    with (run_dir / "solver_records.json").open("w") as f:
        json.dump(
            _to_jsonable(
                {
                    "solve_summaries": solver.solve_summaries,
                    "records": solver.records,
                }
            ),
            f,
            indent=2,
        )

    with (run_dir / "results.json").open("w") as f:
        json.dump(
            _to_jsonable(
                {
                    "summary": summary,
                    "metrics": metrics,
                    "episodes": episode_rows,
                    "eval_rows": eval_rows,
                    "eval_episodes": eval_episodes,
                    "eval_start_steps": eval_start_steps,
                }
            ),
            f,
            indent=2,
        )

    print(f"\n===== {run_name} =====")
    print(json.dumps(_to_jsonable(summary), indent=2))
    print(f"Saved: {run_dir}")

    if hasattr(world, "close"):
        try:
            world.close()
        except Exception:
            pass

    return summary


def _print_comparison(payload):
    print("\n===== GRADIENT PLANNER COMPARISON =====")
    header = (
        f"{'run':<24} {'succ20':>7} {'succ10':>7} {'final10':>8} "
        f"{'costRed':>8} {'g/cost':>8} {'sat':>7} {'plan_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for run_name, s in payload["runs"].items():
        sol = s["solver"]
        print(
            f"{run_name:<24} "
            f"{s['official_rollout_success_rate']:7.1f} "
            f"{s['strict_rollout_success_rate']:7.1f} "
            f"{s['final_within_strict_theta_rate']:8.1f} "
            f"{sol['zero_to_final_relative_reduction']['median']:8.3f} "
            f"{sol['initial_zero_grad_over_cost']['median']:8.3f} "
            f"{sol['final_action_saturation_fraction']['mean']:7.3f} "
            f"{sol['total_solver_time_seconds']:8.1f}"
        )
    print(
        "succ20=official ever-success; succ10=strict-theta ever-success; "
        "final10=final-state strict success;\n"
        "costRed=median within-model latent-cost reduction from zero init; "
        "g/cost=median initial ||grad_v L||/L; sat=selected-action tanh saturation fraction."
    )


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.policies):
        raise ValueError("--labels must match --policies length")
    if any(x <= 0 for x in args.num_starts):
        raise ValueError("Every --num-starts value must be positive")

    cfg = OmegaConf.load(args.config)
    cfg.eval.num_eval = int(args.num_eval)
    cfg.seed = int(args.seed)
    cfg.world.num_envs = int(args.num_eval)
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)

    if int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block) > int(
        cfg.eval.eval_budget
    ):
        raise ValueError("Planning horizon must be <= eval budget")

    cache_root = Path(
        os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir())
    )
    output_root = (
        Path(args.output_dir)
        if args.output_dir is not None
        else cache_root / "pusht_gradient_planner"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(cfg, str(cfg.eval.dataset_name))
    col_name, eval_rows, eval_episodes, eval_start_steps = _select_eval_rows(
        cfg, dataset, args.num_eval, args.seed
    )
    del col_name
    print("Selected paired eval rows:")
    print(eval_rows)

    start_states, goal_states = _load_start_goal_states(
        dataset,
        eval_episodes,
        eval_start_steps,
        cfg.eval.goal_offset_steps,
    )

    payload = {
        "settings": {
            "config": args.config,
            "num_eval": int(args.num_eval),
            "num_starts": list(args.num_starts),
            "n_steps": int(args.n_steps),
            "lr": float(args.lr),
            "init_std": float(args.init_std),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "device": str(args.device),
            "strict_theta_deg": float(args.strict_theta_deg),
            "horizon": int(cfg.plan_config.horizon),
            "receding_horizon": int(cfg.plan_config.receding_horizon),
            "action_block": int(cfg.plan_config.action_block),
            "goal_offset_steps": int(cfg.eval.goal_offset_steps),
            "eval_budget": int(cfg.eval.eval_budget),
            "eval_rows": eval_rows,
            "eval_episodes": eval_episodes,
            "eval_start_steps": eval_start_steps,
        },
        "runs": {},
    }

    for num_starts in args.num_starts:
        for i, policy_name in enumerate(args.policies):
            label = _label(policy_name, args.labels, i)
            summary = _run_one(
                cfg,
                dataset,
                eval_rows,
                eval_episodes,
                eval_start_steps,
                start_states,
                goal_states,
                policy_name=policy_name,
                label=label,
                num_starts=int(num_starts),
                args=args,
                output_root=output_root,
            )
            payload["runs"][summary["run_name"]] = summary

    comparison_path = output_root / "comparison.json"
    with comparison_path.open("w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    _print_comparison(payload)
    print(f"\nSaved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
