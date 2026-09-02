#!/usr/bin/env python3
"""Low-budget CEM failure autopsy for official/full PushT.

This diagnostic reruns the *same* official CEM algorithm, but records its
candidate populations.  After closed-loop evaluation it automatically selects

  * every failed episode, and
  * difficulty-matched successful controls,

then physically replays selected CEM populations from the exact solve-start
simulator state.

Primary questions:
  1) Sampling ceiling: did the population already contain a physically good /
     successful candidate?
  2) Ranking failure: if so, did predicted latent cost fail to select / elite it?
  3) Refinement drift: at which solve/iteration do elite updates stop agreeing
     with physical oracle updates?

The physical oracle is diagnosis-only. It never changes CEM selection, actions,
closed-loop execution, model training, or reported official success.

Outputs:
  closed_loop_eval.json
  case_manifest.csv
  population_metrics.csv
  episode_summary.csv
  autopsy_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

from eval import get_dataset, get_episodes_length, img_transform


def _numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _latest_state(x):
    x = _numpy(x)
    if x.ndim == 3:
        x = x[:, -1]
    if x.ndim != 2 or x.shape[-1] < 5:
        raise RuntimeError(f"Unexpected state shape: {x.shape}")
    return np.asarray(x, dtype=np.float64)


def _angle_error_rad(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def _physical_cost(states, goal):
    s = np.asarray(states, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    joint = np.linalg.norm(s[..., :4] - g[:4], axis=-1)
    theta = _angle_error_rad(s[..., 4], g[4])
    cost = (joint / 20.0) ** 2 + (theta / (np.pi / 9.0)) ** 2
    success = (joint < 20.0) & (theta < np.pi / 9.0)
    return cost, joint, theta, success


def _rankdata_average(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = _rankdata_average(a[m])
    rb = _rankdata_average(b[m])
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(np.dot(ra, rb) / den) if den > 1e-12 else float("nan")


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def _selected_percentile(cost, idx):
    r = _rankdata_average(cost) - 1.0
    return float(r[int(idx)] / max(len(r) - 1, 1))


def _elite_overlap(pred_cost, phys_cost, k):
    ip = set(np.argsort(pred_cost)[:k].tolist())
    iph = set(np.argsort(phys_cost)[:k].tolist())
    return float(len(ip & iph) / k)


def _summary(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _prepare_eval_rows(cfg, dataset):
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start = episode_len - int(cfg.eval.goal_offset_steps) - 1
    max_by_ep = {ep: max_start[i] for i, ep in enumerate(ep_indices)}
    max_per_row = np.asarray([max_by_ep[e] for e in dataset.get_col_data(col)])
    valid = np.asarray(dataset.get_col_data("step_idx")) <= max_per_row
    valid_indices = np.nonzero(valid)[0]
    if int(cfg.eval.num_eval) > len(valid_indices) - 1:
        raise ValueError("Not enough valid official eval starts.")
    g = np.random.default_rng(int(cfg.seed))
    pick = g.choice(len(valid_indices) - 1, size=int(cfg.eval.num_eval), replace=False)
    rows = np.sort(valid_indices[pick])
    data = dataset.get_row_data(rows)
    return col, rows, np.asarray(data[col]), np.asarray(data["step_idx"])


def _load_start_goal_states(dataset, episodes, start_steps, goal_offset):
    starts = np.asarray(start_steps)
    chunks = dataset.load_chunk(
        np.asarray(episodes), starts, starts + int(goal_offset) + 1
    )
    start_states, goal_states = [], []
    for ep in chunks:
        s = _numpy(ep["state"])
        start_states.append(np.asarray(s[0], dtype=np.float64))
        goal_states.append(np.asarray(s[-1], dtype=np.float64))
    return np.stack(start_states), np.stack(goal_states)


def _build_process(cfg, dataset):
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        p = preprocessing.StandardScaler()
        x = np.asarray(dataset.get_col_data(col))
        x = x[~np.isnan(x).any(axis=1)]
        p.fit(x)
        process[col] = p
        if col != "action":
            process[f"goal_{col}"] = p
    return process


class TraceCEMSolver:
    """Numerically identical to the installed official CEM, plus tracing."""

    def __init__(
        self,
        model,
        batch_size=1,
        num_samples=30,
        var_scale=1.0,
        n_steps=10,
        topk=3,
        device="cuda",
        seed=42,
        state_scaler=None,
    ):
        self.model = model
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.var_scale = float(var_scale)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.device = torch.device(device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.trace: list[dict] = []
        self.state_scaler = state_scaler
        self._env_id_to_index = {}

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any):
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim * int(self._config.action_block)

    @property
    def horizon(self):
        return int(self._config.horizon)

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def init_action_distrib(self, n_envs, actions=None):
        # Exact stable-worldmodel==0.0.6 behavior: default float32 tensors.
        var = self.var_scale * torch.ones(
            [n_envs, self.horizon, self.action_dim]
        )
        mean = (
            torch.zeros([n_envs, 0, self.action_dim])
            if actions is None else actions
        )
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            new_mean = torch.zeros(
                [n_envs, remaining, self.action_dim]
            )
            mean = torch.cat([mean, new_mean], dim=1).to(mean.device)
        return mean, var

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        total_envs = len(next(iter(info_dict.values())))

        # Recover stable global env identity directly from the official policy's
        # processed info dict. EverythingToInfoWrapper supplies a persistent
        # per-episode numeric "id"; the first solver call contains all envs in
        # canonical order, and later subset replans keep those ids.
        ids = info_dict.get("id", None)
        if ids is None:
            raise RuntimeError("Expected numeric 'id' in solver info_dict.")
        ids_np = _numpy(ids)
        if ids_np.ndim > 1:
            ids_np = ids_np.reshape(total_envs, -1)[:, -1]
        ids_np = np.asarray(ids_np).reshape(-1)
        if len(self._env_id_to_index) == 0:
            if total_envs != self.n_envs:
                raise RuntimeError(
                    "First tracing solve did not contain every environment; "
                    f"got {total_envs}, expected {self.n_envs}."
                )
            self._env_id_to_index = {
                int(ids_np[i]): int(i) for i in range(total_envs)
            }
        try:
            global_idx = np.asarray(
                [self._env_id_to_index[int(x)] for x in ids_np],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Unknown environment id during replan: {exc}"
            ) from exc

        # The official WorldModelPolicy has already applied StandardScaler to
        # state before calling the solver. Invert it here, diagnosis-only, to
        # recover the exact solve-start PushT state without touching policy
        # internals or action buffers.
        st = info_dict.get("state", None)
        if st is None:
            raise RuntimeError("Expected 'state' in solver info_dict.")
        st_np = _numpy(st)
        if st_np.ndim >= 3:
            st_np = st_np[:, -1]
        st_np = np.asarray(st_np, dtype=np.float64).reshape(total_envs, -1)
        raw_states = (
            self.state_scaler.inverse_transform(st_np)
            if self.state_scaler is not None
            else st_np
        ).astype(np.float64)

        # Match the older stable-worldmodel CEM used by this LeWM checkout:
        # init_action_distrib() itself zero-pads a partial warm start to the
        # planning horizon.  Newer stable-worldmodel moved that behavior
        # behind solver.utils.prepare_init_action(), which is not available
        # in the server environment.
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean, var = mean.to(self.device), var.to(self.device)

        candidates_store = np.empty(
            (total_envs, self.n_steps, self.num_samples, self.horizon, self.action_dim),
            dtype=np.float32,
        )
        costs_store = np.empty(
            (total_envs, self.n_steps, self.num_samples), dtype=np.float32
        )
        prev_mean_store = np.empty(
            (total_envs, self.n_steps, self.horizon, self.action_dim), dtype=np.float32
        )
        mean_store = np.empty_like(prev_mean_store)
        var_store = np.empty_like(prev_mean_store)
        topk_store = np.empty(
            (total_envs, self.n_steps, self.topk), dtype=np.int16
        )

        final_cost = [None] * total_envs
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded = {}
            for k, v in info_dict.items():
                vb = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    # Exact 0.0.6 CEM behavior: do not pre-move/pre-cast
                    # info tensors here. LeWM.get_cost() moves its tensor
                    # entries to the model device.
                    vb = vb.unsqueeze(1)
                    vb = vb.expand(
                        current_bs, self.num_samples, *vb.shape[2:]
                    )
                elif isinstance(v, np.ndarray):
                    vb = np.repeat(vb[:, None, ...], self.num_samples, axis=1)
                expanded[k] = vb

            for step in range(self.n_steps):
                candidates = torch.randn(
                    current_bs, self.num_samples, self.horizon, self.action_dim,
                    generator=self.torch_gen, device=self.device,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean
                # IMPORTANT: exact 0.0.6 behavior. LeWM.get_cost() mutates
                # its input dict in-place by adding goal_emb/action/emb/
                # predicted_emb, so every CEM iteration must receive a fresh
                # shallow copy.
                current_info = expanded.copy()
                costs = self.model.get_cost(current_info, candidates)

                vals, inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                bi = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                elite = candidates[bi, inds]

                prev_mean = batch_mean
                batch_mean = elite.mean(dim=1)
                batch_var = elite.std(dim=1)

                sl = slice(start_idx, end_idx)
                candidates_store[sl, step] = candidates.detach().float().cpu().numpy()
                costs_store[sl, step] = costs.detach().float().cpu().numpy()
                prev_mean_store[sl, step] = prev_mean.detach().float().cpu().numpy()
                mean_store[sl, step] = batch_mean.detach().float().cpu().numpy()
                var_store[sl, step] = batch_var.detach().float().cpu().numpy()
                topk_store[sl, step] = inds.detach().cpu().numpy().astype(np.int16)
                for j in range(current_bs):
                    final_cost[start_idx + j] = float(vals[j].mean().detach().cpu())

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var

        self.trace.append({
            "solve_index": len(self.trace),
            "global_env_indices": global_idx.copy(),
            "solve_start_states": raw_states.copy(),
            "candidates": candidates_store,
            "predicted_costs": costs_store,
            "prev_mean": prev_mean_store,
            "mean_after": mean_store,
            "var_after": var_store,
            "topk_indices": topk_store,
        })

        return {
            "actions": mean.detach().cpu(),
            "costs": final_cost,
            "mean": [mean.detach().cpu()],
            "var": [var.detach().cpu()],
        }


def _reset_replay_env(env, state, goal, seed):
    try:
        env.reset(
            seed=int(seed),
            options={
                "state": np.asarray(state, dtype=np.float64),
                "goal_state": np.asarray(goal, dtype=np.float64),
            },
        )
    except Exception:
        env.reset(seed=int(seed))
        raw = env.unwrapped
        raw._set_goal_state(np.asarray(goal, dtype=np.float64))
        raw._set_state(np.asarray(state, dtype=np.float64))


def _normalized_plan_to_raw(plan, action_scaler, action_block):
    p = np.asarray(plan, dtype=np.float32)
    action_dim = int(action_scaler.mean_.shape[0])
    if p.shape[-1] != action_dim * int(action_block):
        raise RuntimeError(
            f"Packed action dim mismatch: {p.shape[-1]} vs "
            f"{action_dim}*{action_block}"
        )
    norm = p.reshape(-1, action_dim)
    return action_scaler.inverse_transform(norm).astype(np.float32)


def _replay_plan(env, start_state, goal_state, norm_plan, scaler, action_block, seed):
    raw_actions = _normalized_plan_to_raw(norm_plan, scaler, action_block)
    low = np.asarray(env.action_space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float64).reshape(-1)
    raw_oob = np.mean(
        (raw_actions < low[None, :]) | (raw_actions > high[None, :])
    )

    _reset_replay_env(env, start_state, goal_state, seed)
    raw = env.unwrapped
    ever_success = False
    final_state = np.asarray(start_state, dtype=np.float64)
    for a in raw_actions:
        obs, _, term, trunc, _ = raw.step(a)
        final_state = np.asarray(obs["state"], dtype=np.float64)
        ever_success = ever_success or bool(term)
        if term or trunc:
            break
    cost, joint, theta, endpoint_success = _physical_cost(
        final_state[None], goal_state
    )
    return {
        "cost": float(cost[0]),
        "joint_error_px": float(joint[0]),
        "theta_error_deg": float(np.degrees(theta[0])),
        "endpoint_success": bool(endpoint_success[0]),
        "ever_success": bool(ever_success),
        "raw_oob_fraction": float(raw_oob),
    }


def _match_success_controls(start_states, goal_states, success, failures, max_controls):
    if len(failures) == 0:
        return []
    success_idx = np.where(success)[0].tolist()
    if not success_idx:
        return []
    start_cost = np.asarray([
        _physical_cost(start_states[i:i+1], goal_states[i])[0][0]
        for i in range(len(start_states))
    ])
    unused = set(success_idx)
    controls = []
    for f in failures:
        if len(controls) >= int(max_controls) or not unused:
            break
        j = min(unused, key=lambda s: abs(float(start_cost[s] - start_cost[f])))
        controls.append(j)
        unused.remove(j)
    return controls


def _aggregate_episode(rows):
    if not rows:
        return {}
    oracle_available = np.asarray([r["oracle_has_success_candidate"] for r in rows], dtype=bool)
    selected_success = np.asarray([r["pred_selected_ever_success"] for r in rows], dtype=bool)
    ranking_miss = oracle_available & ~selected_success
    return {
        "num_population_snapshots": int(len(rows)),
        "oracle_success_available_fraction": float(np.mean(oracle_available)),
        "ranking_miss_given_oracle_count": int(ranking_miss.sum()),
        "ranking_miss_given_oracle_fraction_all": float(np.mean(ranking_miss)),
        "selected_success_fraction": float(np.mean(selected_success)),
        "rho_pred_phys": _summary(r["rho_pred_phys"] for r in rows),
        "elite_overlap_pred_phys": _summary(r["elite_overlap_pred_phys"] for r in rows),
        "selected_phys_percentile": _summary(r["pred_selected_phys_percentile"] for r in rows),
        "selection_regret": _summary(r["selection_regret"] for r in rows),
        "cem_update_cos_pred_phys": _summary(r["cem_update_cos_pred_phys"] for r in rows),
        "oracle_best_phys_cost": _summary(r["oracle_best_phys_cost"] for r in rows),
        "center_after_phys_cost": _summary(r["center_after_phys_cost"] for r in rows),
        "candidate_raw_oob_fraction": _summary(r["candidate_raw_oob_fraction"] for r in rows),
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    if cfg.policy == "random":
        raise ValueError("Autopsy requires a trained world-model policy.")

    acfg = cfg.get("autopsy", {})
    output_dir = Path(str(acfg.get(
        "output_dir", "outputs/lowbudget_failure_autopsy"
    )))
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_iterations = list(map(int, acfg.get(
        "replay_iterations", [0, 1, 3, 5, 9]
    )))
    num_controls = int(acfg.get("num_success_controls", 12))
    expected_success = acfg.get("expected_success", None)

    if int(cfg.solver.num_samples) != 30 or int(cfg.solver.n_steps) != 10:
        print(
            "WARNING: canonical autopsy target is N=30,I=10; current config is "
            f"N={cfg.solver.num_samples},I={cfg.solver.n_steps}."
        )
    if int(cfg.solver.topk) <= 0:
        raise ValueError("topk must be positive")
    for it in replay_iterations:
        if it < 0 or it >= int(cfg.solver.n_steps):
            raise ValueError(f"Invalid replay iteration {it}")

    # stable-worldmodel==0.0.6 requires the environment horizon to cover
    # both the closed-loop evaluation budget and the dataset goal offset.
    # This matters for smoke tests where eval_budget may be intentionally tiny.
    cfg.world.max_episode_steps = max(
        2 * int(cfg.eval.eval_budget),
        int(cfg.eval.goal_offset_steps) + 1,
    )
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col, eval_rows, eval_episodes, eval_start = _prepare_eval_rows(cfg, dataset)
    start_states, goal_states = _load_start_goal_states(
        dataset, eval_episodes, eval_start, cfg.eval.goal_offset_steps
    )
    process = _build_process(cfg, dataset)

    device = torch.device(str(cfg.solver.device))
    model = swm.policy.AutoCostModel(str(cfg.policy)).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    solver = TraceCEMSolver(
        model=model,
        batch_size=int(cfg.solver.batch_size),
        num_samples=int(cfg.solver.num_samples),
        var_scale=float(cfg.solver.var_scale),
        n_steps=int(cfg.solver.n_steps),
        topk=int(cfg.solver.topk),
        device=str(cfg.solver.device),
        seed=int(cfg.seed),
        state_scaler=process.get("state"),
    )
    plan_config = swm.PlanConfig(**cfg.plan_config)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    # Use the installed official stable-worldmodel==0.0.6 policy verbatim.
    # Only the solver is instrumented.
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_config, process=process, transform=transform
    )
    world.set_policy(policy)

    print("===== CLOSED-LOOP TRACE EVAL =====")
    try:
        import importlib.metadata as _ilm
        print("stable-worldmodel=" + _ilm.version("stable-worldmodel"))
    except Exception:
        pass
    print(f"policy={cfg.policy}")
    print(
        f"N={cfg.solver.num_samples} I={cfg.solver.n_steps} "
        f"K={cfg.solver.topk} eval={cfg.eval.num_eval}"
    )
    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start.tolist(),
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
    )
    closed_loop_seconds = time.time() - t0
    success = np.asarray(metrics["episode_successes"], dtype=bool)
    failures = np.where(~success)[0].tolist()
    controls = _match_success_controls(
        start_states, goal_states, success, failures, num_controls
    )
    selected = failures + controls

    print(
        f"Closed-loop success={metrics['success_rate']:.1f}% "
        f"failures={len(failures)} controls={len(controls)} "
        f"solver_calls={len(solver.trace)}"
    )
    if expected_success is not None:
        diff = abs(float(metrics["success_rate"]) - float(expected_success))
        if diff > 1e-6:
            print(
                f"WARNING expected success {expected_success}, got "
                f"{metrics['success_rate']}; keep result but audit reproducibility."
            )

    closed_loop_payload = {
        "metrics": metrics,
        "closed_loop_seconds": closed_loop_seconds,
        "policy": str(cfg.policy),
        "dataset_rows": eval_rows,
        "episode_idx": eval_episodes,
        "start_step": eval_start,
        "failure_eval_indices_zero_based": failures,
        "control_eval_indices_zero_based": controls,
        "solver_calls": len(solver.trace),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (output_dir / "closed_loop_eval.json").write_text(
        json.dumps(_jsonable(closed_loop_payload), indent=2)
    )

    manifest = []
    for i in selected:
        init_cost = float(_physical_cost(start_states[i:i+1], goal_states[i])[0][0])
        manifest.append({
            "eval_index": int(i),
            "case_type": "failure" if not success[i] else "success_control",
            "official_success": bool(success[i]),
            "dataset_row": int(eval_rows[i]),
            "episode_idx": int(eval_episodes[i]),
            "start_step": int(eval_start[i]),
            "initial_physical_cost": init_cost,
        })
    _write_csv(output_dir / "case_manifest.csv", manifest)

    print("===== PHYSICAL POPULATION REPLAY =====")
    replay_env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    pop_rows = []
    replay_start = time.time()
    try:
        for case_pos, env_i in enumerate(selected):
            ctype = "failure" if not success[env_i] else "success_control"
            print(
                f"case {case_pos+1}/{len(selected)} env={env_i} type={ctype}"
            )
            for solve in solver.trace:
                global_idx = solve["global_env_indices"]
                locs = np.where(global_idx == env_i)[0]
                if len(locs) == 0:
                    continue
                li = int(locs[0])
                solve_idx = int(solve["solve_index"])
                start_state = solve["solve_start_states"][li]
                goal = goal_states[env_i]

                start_phys = float(
                    _physical_cost(start_state[None], goal)[0][0]
                )

                for it in replay_iterations:
                    candidates = solve["candidates"][li, it]
                    pred_cost = solve["predicted_costs"][li, it].astype(np.float64)
                    prev_mean = solve["prev_mean"][li, it]
                    mean_after = solve["mean_after"][li, it]

                    phys_cost = np.empty(len(candidates), dtype=np.float64)
                    ever_success = np.zeros(len(candidates), dtype=bool)
                    endpoint_success = np.zeros(len(candidates), dtype=bool)
                    oob = np.empty(len(candidates), dtype=np.float64)

                    for ci, cand in enumerate(candidates):
                        rr = _replay_plan(
                            replay_env, start_state, goal, cand,
                            process["action"], int(cfg.plan_config.action_block),
                            seed=100000 + 1000 * env_i + 10 * solve_idx + ci,
                        )
                        phys_cost[ci] = rr["cost"]
                        ever_success[ci] = rr["ever_success"]
                        endpoint_success[ci] = rr["endpoint_success"]
                        oob[ci] = rr["raw_oob_fraction"]

                    center_rr = _replay_plan(
                        replay_env, start_state, goal, mean_after,
                        process["action"], int(cfg.plan_config.action_block),
                        seed=200000 + 1000 * env_i + 10 * solve_idx + it,
                    )

                    pred_sel = int(np.argmin(pred_cost))
                    oracle_sel = int(np.argmin(phys_cost))
                    k = int(cfg.solver.topk)
                    pred_elite = np.argsort(pred_cost)[:k]
                    phys_elite = np.argsort(phys_cost)[:k]
                    pred_update = mean_after - prev_mean
                    phys_elite_mean = candidates[phys_elite].mean(axis=0)
                    phys_update = phys_elite_mean - prev_mean
                    pred_ranks = _rankdata_average(pred_cost) - 1.0
                    oracle_pred_rank_pct = float(
                        pred_ranks[oracle_sel] / max(len(pred_cost) - 1, 1)
                    )

                    pop_rows.append({
                        "eval_index": int(env_i),
                        "case_type": ctype,
                        "dataset_row": int(eval_rows[env_i]),
                        "episode_idx": int(eval_episodes[env_i]),
                        "start_step": int(eval_start[env_i]),
                        "solve_index": solve_idx,
                        "cem_iteration": int(it),
                        "num_samples": int(cfg.solver.num_samples),
                        "topk": k,
                        "solve_start_phys_cost": start_phys,
                        "rho_pred_phys": _spearman(pred_cost, phys_cost),
                        "elite_overlap_pred_phys": _elite_overlap(pred_cost, phys_cost, k),
                        "cem_update_cos_pred_phys": _cosine(pred_update, phys_update),
                        "pred_selected_idx": pred_sel,
                        "oracle_best_idx": oracle_sel,
                        "pred_selected_phys_cost": float(phys_cost[pred_sel]),
                        "oracle_best_phys_cost": float(phys_cost[oracle_sel]),
                        "selection_regret": float(
                            phys_cost[pred_sel] - phys_cost[oracle_sel]
                        ),
                        "pred_selected_phys_percentile": _selected_percentile(
                            phys_cost, pred_sel
                        ),
                        "oracle_best_predicted_percentile": oracle_pred_rank_pct,
                        "oracle_has_success_candidate": bool(np.any(ever_success)),
                        "oracle_success_candidate_fraction": float(np.mean(ever_success)),
                        "pred_selected_ever_success": bool(ever_success[pred_sel]),
                        "pred_selected_endpoint_success": bool(endpoint_success[pred_sel]),
                        "pred_elite_success_fraction": float(
                            np.mean(ever_success[pred_elite])
                        ),
                        "oracle_elite_success_fraction": float(
                            np.mean(ever_success[phys_elite])
                        ),
                        "pred_elite_phys_mean_cost": float(
                            np.mean(phys_cost[pred_elite])
                        ),
                        "oracle_elite_phys_mean_cost": float(
                            np.mean(phys_cost[phys_elite])
                        ),
                        "center_after_phys_cost": float(center_rr["cost"]),
                        "center_after_ever_success": bool(center_rr["ever_success"]),
                        "center_after_raw_oob_fraction": float(
                            center_rr["raw_oob_fraction"]
                        ),
                        "candidate_raw_oob_fraction": float(np.mean(oob)),
                    })
    finally:
        replay_env.close()

    _write_csv(output_dir / "population_metrics.csv", pop_rows)

    episode_rows = []
    episode_payload = {}
    for i in selected:
        rr = [r for r in pop_rows if r["eval_index"] == i]
        agg = _aggregate_episode(rr)
        row = {
            "eval_index": int(i),
            "case_type": "failure" if not success[i] else "success_control",
            "official_success": bool(success[i]),
            "dataset_row": int(eval_rows[i]),
            "episode_idx": int(eval_episodes[i]),
            "start_step": int(eval_start[i]),
            "num_population_snapshots": agg.get("num_population_snapshots", 0),
            "oracle_success_available_fraction": agg.get(
                "oracle_success_available_fraction", float("nan")
            ),
            "ranking_miss_given_oracle_count": agg.get(
                "ranking_miss_given_oracle_count", 0
            ),
            "ranking_miss_given_oracle_fraction_all": agg.get(
                "ranking_miss_given_oracle_fraction_all", float("nan")
            ),
            "selected_success_fraction": agg.get(
                "selected_success_fraction", float("nan")
            ),
            "median_rho_pred_phys": agg.get("rho_pred_phys", {}).get(
                "median", float("nan")
            ),
            "median_elite_overlap": agg.get(
                "elite_overlap_pred_phys", {}
            ).get("median", float("nan")),
            "median_selected_phys_percentile": agg.get(
                "selected_phys_percentile", {}
            ).get("median", float("nan")),
            "median_selection_regret": agg.get(
                "selection_regret", {}
            ).get("median", float("nan")),
            "median_update_cos": agg.get(
                "cem_update_cos_pred_phys", {}
            ).get("median", float("nan")),
            "median_candidate_oob_fraction": agg.get(
                "candidate_raw_oob_fraction", {}
            ).get("median", float("nan")),
        }
        episode_rows.append(row)
        episode_payload[str(i)] = agg

    _write_csv(output_dir / "episode_summary.csv", episode_rows)

    def group_rows(case_type):
        return [r for r in pop_rows if r["case_type"] == case_type]

    group_summary = {}
    for gt in ["failure", "success_control"]:
        gr = group_rows(gt)
        group_summary[gt] = {
            "count_population_snapshots": len(gr),
            "rho_pred_phys": _summary(r["rho_pred_phys"] for r in gr),
            "elite_overlap_pred_phys": _summary(
                r["elite_overlap_pred_phys"] for r in gr
            ),
            "selected_phys_percentile": _summary(
                r["pred_selected_phys_percentile"] for r in gr
            ),
            "selection_regret": _summary(r["selection_regret"] for r in gr),
            "cem_update_cos_pred_phys": _summary(
                r["cem_update_cos_pred_phys"] for r in gr
            ),
            "oracle_success_available_fraction": float(
                np.mean([r["oracle_has_success_candidate"] for r in gr])
            ) if gr else None,
            "ranking_miss_when_oracle_available_fraction": float(
                np.mean([
                    r["oracle_has_success_candidate"]
                    and not r["pred_selected_ever_success"]
                    for r in gr
                ])
            ) if gr else None,
            "candidate_raw_oob_fraction": _summary(
                r["candidate_raw_oob_fraction"] for r in gr
            ),
        }

    summary = {
        "scientific_question": (
            "At low budget N=30,I=10, are remaining failures caused mainly by "
            "sampling ceiling, model ranking/elite errors, or later CEM drift?"
        ),
        "closed_loop_success_rate": float(metrics["success_rate"]),
        "num_failures": len(failures),
        "num_success_controls": len(controls),
        "failure_eval_indices_zero_based": failures,
        "control_eval_indices_zero_based": controls,
        "replay_iterations": replay_iterations,
        "group_summary": group_summary,
        "episode_summary": episode_payload,
        "timing": {
            "closed_loop_seconds": closed_loop_seconds,
            "physical_replay_seconds": float(time.time() - replay_start),
            "total_seconds": float(time.time() - t0),
        },
        "definitions": {
            "sampling_evidence": (
                "oracle_has_success_candidate means at least one sampled action "
                "sequence reaches the official PushT success condition during "
                "diagnosis-only physical replay."
            ),
            "ranking_miss": (
                "A success candidate exists in the sampled population but the "
                "candidate with lowest predicted latent cost does not reach success."
            ),
            "elite_overlap": (
                "Set overlap between predicted-cost top-K and physical-cost top-K."
            ),
            "center_after": (
                "Physical replay of the CEM mean after the current update; at the "
                "final iteration this is the plan returned by the official solver."
            ),
        },
    }
    (output_dir / "autopsy_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2)
    )

    print("===== AUTOPSY SUMMARY =====")
    print(json.dumps(_jsonable({
        "success_rate": summary["closed_loop_success_rate"],
        "num_failures": summary["num_failures"],
        "num_controls": summary["num_success_controls"],
        "group_summary": summary["group_summary"],
        "timing": summary["timing"],
    }), indent=2))
    print(f"Saved: {output_dir / 'closed_loop_eval.json'}")
    print(f"Saved: {output_dir / 'case_manifest.csv'}")
    print(f"Saved: {output_dir / 'population_metrics.csv'}")
    print(f"Saved: {output_dir / 'episode_summary.csv'}")
    print(f"Saved: {output_dir / 'autopsy_summary.json'}")
    print("=== DONE ===")


if __name__ == "__main__":
    run()
