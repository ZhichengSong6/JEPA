#!/usr/bin/env python3
"""B=1000 PushT planning ceiling diagnostic for MH-ALD.

Formal protocol
---------------
1) Run the current MH-ALD planner on 100 paired official starts with
   CEM N=100, I=10, K=10.
2) Select every MH-ALD failure plus a few difficulty-matched successful controls.
3) Re-run ONLY those selected starts with:
     encoder oracle: physically roll out each sampled candidate, encode its true
       terminal observation with the frozen visual encoder, and score raw latent L2
       to the true goal encoding. This removes predictor error while preserving the
       current encoder/metric and the same CEM budget/horizon.
     physical oracle: physically roll out each sampled candidate and score the
       official PushT terminal physical cost. This removes both predictor and latent
       metric error while preserving the same CEM budget/horizon.
4) Report optimistic success ceilings by adding rescued baseline failures to the
   baseline successes. Controls are sanity checks only.

All oracle information is diagnostic only.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from eval import get_dataset, img_transform
from eval_lowbudget_failure_autopsy import (
    _build_process,
    _jsonable,
    _load_start_goal_states,
    _physical_cost,
    _prepare_eval_rows,
)
from eval_b3000_paired_failure_analysis import _normalized_to_raw
from pusht_exact_replay import (
    LiveVariationCapturePolicy,
    VariationInjectedDataset,
    load_goal_images,
    reset_physical_exact,
)
from eval_pusht_horizon_directional import _encode


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _close_world(world):
    try:
        world.envs.close()
    except Exception:
        try:
            world.close()
        except Exception:
            pass


def _latest_2d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim >= 3:
        x = x[:, -1]
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    return np.asarray(x, dtype=np.float64)


def _matched_controls(start_states, goal_states, success, failures, max_controls):
    pool = [i for i, ok in enumerate(success) if bool(ok)]
    if not pool or int(max_controls) <= 0:
        return []
    init_cost = np.asarray([
        float(_physical_cost(start_states[i:i+1], goal_states[i])[0][0])
        for i in range(len(start_states))
    ])
    unused = set(pool)
    controls = []
    for target in failures:
        if len(controls) >= int(max_controls) or not unused:
            break
        j = min(unused, key=lambda x: abs(init_cost[x] - init_cost[target]))
        controls.append(int(j))
        unused.remove(j)
    return controls


def _run_mh_baseline(cfg, dataset, process, policy_name, eval_episodes, eval_start):
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = int(len(eval_episodes))
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    world = swm.World(**world_cfg, image_shape=(224, 224))

    model = swm.policy.AutoCostModel(str(policy_name)).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    solver_cfg = copy.deepcopy(cfg.solver)
    solver = hydra.utils.instantiate(solver_cfg, model=model)
    plan_config = swm.PlanConfig(**cfg.plan_config)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    policy = LiveVariationCapturePolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )
    world.set_policy(policy)

    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=np.asarray(eval_start).tolist(),
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=np.asarray(eval_episodes).tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
    )
    elapsed = time.time() - t0
    live_reset_contexts = policy.live_reset_contexts
    _close_world(world)
    if live_reset_contexts is None:
        raise RuntimeError(
            "Failed to capture live baseline variation snapshots."
        )

    return {
        "metrics": metrics,
        "success": np.asarray(metrics["episode_successes"], dtype=bool),
        "elapsed_seconds": elapsed,
        "live_reset_contexts": live_reset_contexts,
    }


class OracleCEMSolver:
    """Same Gaussian CEM, with diagnostic ground-truth candidate scoring."""

    def __init__(
        self,
        mode,
        encoder_model,
        transform,
        action_scaler,
        state_scaler,
        env_name,
        num_samples=100,
        var_scale=1.0,
        n_steps=10,
        topk=10,
        device="cuda",
        seed=42,
        model_batch_size=64,
        reset_contexts=None,
        goal_images=None,
    ):
        if mode not in {"encoder", "physical"}:
            raise ValueError(mode)
        self.mode = mode
        self.model = encoder_model
        self.encoder_model = encoder_model
        self.transform = transform
        self.action_scaler = action_scaler
        self.state_scaler = state_scaler
        self.env_name = str(env_name)
        self.num_samples = int(num_samples)
        self.var_scale = float(var_scale)
        self.n_steps = int(n_steps)
        self.topk = int(topk)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.model_batch_size = int(model_batch_size)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(self.seed)
        self.env = gym.make(self.env_name, render_mode="rgb_array")
        self.goal_cache = {}
        self.solve_count = 0
        self.diagnostic_rows = []
        self.reset_contexts = list(reset_contexts or [])
        self.goal_images = list(goal_images or [])
        self._env_id_to_context_index = {}

    def configure(self, *, action_space: gym.Space, n_envs: int, config):
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

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def _init_action_distrib(self, total_envs, actions=None):
        var = self.var_scale * torch.ones(
            [total_envs, self.horizon, self.action_dim], dtype=torch.float32
        )
        mean = (
            torch.zeros([total_envs, 0, self.action_dim], dtype=torch.float32)
            if actions is None
            else actions
        )
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            mean = torch.cat(
                [
                    mean,
                    torch.zeros(
                        [total_envs, remaining, self.action_dim],
                        dtype=mean.dtype,
                        device=mean.device,
                    ),
                ],
                dim=1,
            )
        return mean.to(self.device), var.to(self.device)

    def _env_id(self, info_dict, env_i):
        x = info_dict.get("id", None)
        if x is None:
            return int(env_i)
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        v = x[env_i]
        if np.asarray(v).ndim:
            v = np.asarray(v).reshape(-1)[-1]
        return int(v)

    def _raw_states(self, info_dict):
        st = _latest_2d(info_dict["state"])
        gt = _latest_2d(info_dict["goal_state"])
        if self.state_scaler is not None:
            st = self.state_scaler.inverse_transform(st)
            gt = self.state_scaler.inverse_transform(gt)
        return st.astype(np.float64), gt.astype(np.float64)

    def _context_index(self, info_dict, env_i):
        env_id = self._env_id(info_dict, env_i)
        if not self._env_id_to_context_index:
            total_envs = len(next(iter(info_dict.values())))
            if len(self.reset_contexts) != total_envs:
                raise RuntimeError(
                    "Reset-context count does not match first oracle solve: "
                    f"{len(self.reset_contexts)} vs {total_envs}"
                )
            for j in range(total_envs):
                self._env_id_to_context_index[
                    self._env_id(info_dict, j)
                ] = j
        if env_id not in self._env_id_to_context_index:
            raise RuntimeError(f"Unknown oracle env id {env_id}")
        return self._env_id_to_context_index[env_id]

    def _goal_latent(self, context_index):
        if context_index in self.goal_cache:
            return self.goal_cache[context_index]
        if context_index >= len(self.goal_images):
            raise RuntimeError(
                f"Missing exact dataset goal image for context {context_index}"
            )
        z = _encode(
            self.encoder_model,
            self.transform,
            [np.asarray(self.goal_images[context_index])],
            self.device,
            self.model_batch_size,
        )[0].detach()
        self.goal_cache[context_index] = z
        return z

    def _score_population(self, info_dict, env_i, candidates, raw_state, goal_state):
        context_index = self._context_index(info_dict, env_i)
        reset_context = self.reset_contexts[context_index]
        raw_candidates = _normalized_to_raw(
            candidates.detach().cpu().numpy(),
            self.action_scaler,
            int(self._config.action_block),
        )

        final_states = np.empty(
            (len(raw_candidates), len(raw_state)), dtype=np.float64
        )
        final_images = [] if self.mode == "encoder" else None

        env_id = self._env_id(info_dict, env_i)
        for ci, acts in enumerate(raw_candidates):
            reset_physical_exact(
                self.env,
                raw_state,
                goal_state,
                reset_context,
            )
            raw = self.env.unwrapped
            obs = None
            for action in acts:
                obs, _, _, _, _ = raw.step(action)
            final_states[ci] = np.asarray(obs["state"], dtype=np.float64)
            if final_images is not None:
                final_images.append(np.asarray(raw.render()))

        phys_cost, _, _, phys_success = _physical_cost(final_states, goal_state)

        if self.mode == "physical":
            score = np.asarray(phys_cost, dtype=np.float64)
        else:
            zg = self._goal_latent(context_index)
            zr = _encode(
                self.encoder_model,
                self.transform,
                final_images,
                self.device,
                self.model_batch_size,
            )
            score = (
                torch.sum((zr - zg[None]) ** 2, dim=-1)
                .detach().cpu().numpy().astype(np.float64)
            )

        return score, np.asarray(phys_cost), np.asarray(phys_success)

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        total_envs = len(next(iter(info_dict.values())))
        raw_states, goal_states = self._raw_states(info_dict)
        mean, var = self._init_action_distrib(total_envs, init_action)
        outputs = {"costs": [], "mean": [], "var": []}
        self.solve_count += 1

        last_topk = [None] * total_envs
        for it in range(self.n_steps):
            noise = torch.randn(
                total_envs,
                self.num_samples,
                self.horizon,
                self.action_dim,
                generator=self.torch_gen,
                device=self.device,
                dtype=mean.dtype,
            )
            candidates = mean[:, None] + noise * var[:, None]
            all_cost = torch.empty(
                total_envs, self.num_samples, device=self.device, dtype=torch.float32
            )

            for env_i in range(total_envs):
                score, phys_cost, phys_success = self._score_population(
                    info_dict,
                    env_i,
                    candidates[env_i],
                    raw_states[env_i],
                    goal_states[env_i],
                )
                all_cost[env_i] = torch.as_tensor(
                    score, device=self.device, dtype=torch.float32
                )
                selected = int(np.argmin(score))
                oracle_best = int(np.argmin(phys_cost))
                self.diagnostic_rows.append({
                    "mode": self.mode,
                    "solve_count": int(self.solve_count),
                    "env_local_index": int(env_i),
                    "env_id": int(self._env_id(info_dict, env_i)),
                    "cem_iteration": int(it),
                    "oracle_success_candidate_fraction": float(np.mean(phys_success)),
                    "selected_phys_success": bool(phys_success[selected]),
                    "selected_phys_cost": float(phys_cost[selected]),
                    "oracle_best_phys_cost": float(phys_cost[oracle_best]),
                    "dataset_seed": self.reset_contexts[
                        self._context_index(info_dict, env_i)
                    ].get("seed", None),
                    "variation_count": int(self.reset_contexts[
                        self._context_index(info_dict, env_i)
                    ].get("variation_count", 0)),
                })

            vals, inds = torch.topk(
                all_cost, k=self.topk, dim=1, largest=False
            )
            bidx = torch.arange(total_envs, device=self.device)[:, None]
            elites = candidates[bidx, inds]
            mean = elites.mean(dim=1)
            var = elites.std(dim=1)
            last_topk = vals.mean(dim=1).detach().cpu().tolist()

        outputs["costs"] = last_topk
        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        return outputs


def _run_oracle(
    cfg,
    dataset,
    process,
    mode,
    encoder_policy,
    eval_episodes,
    eval_start,
    output_dir,
    reset_contexts,
    goal_images,
):
    device = torch.device(str(cfg.solver.device))
    model = swm.policy.AutoCostModel(str(encoder_policy)).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    solver = OracleCEMSolver(
        mode=mode,
        encoder_model=model,
        transform=img_transform(cfg),
        action_scaler=process["action"],
        state_scaler=process.get("state"),
        env_name=str(cfg.world.env_name),
        num_samples=int(cfg.solver.num_samples),
        var_scale=float(cfg.solver.var_scale),
        n_steps=int(cfg.solver.n_steps),
        topk=int(cfg.solver.topk),
        device=str(cfg.solver.device),
        seed=int(cfg.seed),
        model_batch_size=int(cfg.get("ceiling", {}).get("model_batch_size", 64)),
        reset_contexts=reset_contexts,
        goal_images=goal_images,
    )

    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = int(len(eval_episodes))
    world_cfg["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    injected_dataset = VariationInjectedDataset(
        dataset, reset_contexts
    )
    world = swm.World(**world_cfg, image_shape=(224, 224))
    plan_config = swm.PlanConfig(**cfg.plan_config)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )
    world.set_policy(policy)

    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        injected_dataset,
        start_steps=np.asarray(eval_start).tolist(),
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=np.asarray(eval_episodes).tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
    )
    elapsed = time.time() - t0
    success = np.asarray(metrics["episode_successes"], dtype=bool)

    _close_world(world)
    solver.close()
    _write_csv(Path(output_dir) / f"{mode}_solver_diagnostics.csv", solver.diagnostic_rows)

    return {
        "mode": mode,
        "success_rate_selected_subset": float(metrics["success_rate"]),
        "episode_successes": success.tolist(),
        "elapsed_seconds": float(elapsed),
        "solver_calls": int(solver.solve_count),
    }


def _load_json(path):
    return json.loads(Path(path).read_text())


def _save_json(path, payload):
    Path(path).write_text(json.dumps(_jsonable(payload), indent=2))


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    ccfg = cfg.get("ceiling", {})
    phase = str(ccfg.get("phase", "baseline"))
    outdir = Path(str(ccfg.get("output_dir", "outputs/mh_ald_b1000_ceiling")))
    outdir.mkdir(parents=True, exist_ok=True)

    mh_policy = str(ccfg.get(
        "mh_policy",
        "pusht_mh_ald_h5_seed3072_ep10_ddp4/lewm_mh_ald_h5_ddp4_epoch_10",
    ))
    max_controls = int(ccfg.get("max_success_controls", 4))
    expected_baseline = ccfg.get("expected_baseline_success", None)

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    _, eval_rows, eval_episodes, eval_start = _prepare_eval_rows(cfg, dataset)
    start_states, goal_states = _load_start_goal_states(
        dataset, eval_episodes, eval_start, cfg.eval.goal_offset_steps
    )
    process = _build_process(cfg, dataset)
    exact_goal_images = load_goal_images(
        dataset,
        eval_episodes,
        eval_start,
        cfg.eval.goal_offset_steps,
    )
    baseline_path = outdir / "baseline.json"

    if phase == "baseline":
        print("============================================================")
        print("B=1000 ceiling baseline")
        print(f"MH policy: {mh_policy}")
        print(
            f"CEM N={cfg.solver.num_samples} I={cfg.solver.n_steps} "
            f"K={cfg.solver.topk} B={int(cfg.solver.num_samples)*int(cfg.solver.n_steps)}"
        )
        print(f"eval episodes={cfg.eval.num_eval} seed={cfg.seed}")
        print("============================================================")

        base = _run_mh_baseline(
            cfg, dataset, process, mh_policy, eval_episodes, eval_start
        )
        success = base["success"]
        failures = np.nonzero(~success)[0].astype(int).tolist()
        controls = _matched_controls(
            start_states, goal_states, success, failures, max_controls
        )
        selected = failures + [x for x in controls if x not in failures]

        if not selected:
            selected = list(range(min(2, len(success))))

        payload = {
            "mh_policy": mh_policy,
            "num_eval": int(cfg.eval.num_eval),
            "seed": int(cfg.seed),
            "num_samples": int(cfg.solver.num_samples),
            "iterations": int(cfg.solver.n_steps),
            "topk": int(cfg.solver.topk),
            "budget_B": int(cfg.solver.num_samples) * int(cfg.solver.n_steps),
            "eval_rows": eval_rows.tolist(),
            "eval_episodes": eval_episodes.tolist(),
            "eval_start": eval_start.tolist(),
            "episode_successes": success.tolist(),
            "success_rate": float(base["metrics"]["success_rate"]),
            "failure_eval_indices": failures,
            "control_eval_indices": controls,
            "selected_eval_indices": selected,
            "elapsed_seconds": float(base["elapsed_seconds"]),
            "live_reset_contexts": base["live_reset_contexts"],
        }
        if expected_baseline is not None and abs(
            payload["success_rate"] - float(expected_baseline)
        ) > 1e-6:
            print(
                f"WARNING: expected baseline {expected_baseline}%, "
                f"got {payload['success_rate']}%"
            )

        _save_json(baseline_path, payload)
        print(json.dumps(payload, indent=2))
        print(f"Saved: {baseline_path}")
        return

    if not baseline_path.exists():
        raise FileNotFoundError(f"Run baseline phase first: {baseline_path}")
    baseline = _load_json(baseline_path)
    selected = np.asarray(baseline["selected_eval_indices"], dtype=np.int64)

    if phase in {"encoder", "physical"}:
        subset_ep = np.asarray(eval_episodes)[selected]
        subset_start = np.asarray(eval_start)[selected]
        print("============================================================")
        print(f"B=1000 {phase.upper()} ORACLE")
        print(f"selected baseline cases={selected.tolist()}")
        print("Oracle information is diagnostic only.")
        print("============================================================")

        baseline_contexts = baseline.get("live_reset_contexts", None)
        if baseline_contexts is None:
            raise RuntimeError(
                "Baseline JSON predates live variation capture. "
                "Re-run ceiling baseline/formal before oracle phases."
            )
        subset_contexts = [
            baseline_contexts[int(i)] for i in selected
        ]
        subset_goal_images = [exact_goal_images[int(i)] for i in selected]
        result = _run_oracle(
            cfg,
            dataset,
            process,
            phase,
            mh_policy,
            subset_ep,
            subset_start,
            outdir,
            subset_contexts,
            subset_goal_images,
        )
        result["selected_eval_indices"] = selected.tolist()
        path = outdir / f"{phase}_oracle.json"
        _save_json(path, result)
        print(json.dumps(result, indent=2))
        print(f"Saved: {path}")
        return

    if phase != "summary":
        raise ValueError(f"Unknown ceiling.phase={phase}")

    enc = _load_json(outdir / "encoder_oracle.json")
    phy = _load_json(outdir / "physical_oracle.json")
    selected_list = list(map(int, baseline["selected_eval_indices"]))
    enc_map = {
        idx: bool(ok)
        for idx, ok in zip(selected_list, enc["episode_successes"])
    }
    phy_map = {
        idx: bool(ok)
        for idx, ok in zip(selected_list, phy["episode_successes"])
    }

    failures = set(map(int, baseline["failure_eval_indices"]))
    controls = set(map(int, baseline["control_eval_indices"]))
    base_success = np.asarray(baseline["episode_successes"], dtype=bool)

    manifest = []
    enc_rescues = 0
    phy_rescues = 0
    for idx in selected_list:
        is_failure = idx in failures
        enc_ok = enc_map[idx]
        phy_ok = phy_map[idx]
        if is_failure:
            enc_rescues += int(enc_ok)
            phy_rescues += int(phy_ok)
            if enc_ok:
                category = "predictor_limited_rescued_by_encoder_oracle"
            elif phy_ok:
                category = "encoder_metric_limited_physical_only"
            else:
                category = "search_coverage_horizon_limited_even_physical"
        else:
            category = "success_control"

        manifest.append({
            "eval_index": int(idx),
            "dataset_row": int(baseline["eval_rows"][idx]),
            "baseline_success": bool(base_success[idx]),
            "is_baseline_failure": bool(is_failure),
            "is_control": bool(idx in controls),
            "encoder_oracle_success": bool(enc_ok),
            "physical_oracle_success": bool(phy_ok),
            "category": category,
        })

    n = int(baseline["num_eval"])
    base_count = int(base_success.sum())
    enc_ceiling = 100.0 * (base_count + enc_rescues) / n
    phy_ceiling = 100.0 * (base_count + phy_rescues) / n
    target = float(ccfg.get("target_success", 95.6))

    summary = {
        "scientific_question": (
            "At fixed B=1000, how much closed-loop success headroom remains "
            "after removing predictor error only, and after removing both "
            "predictor and latent-metric error?"
        ),
        "baseline": {
            "success_rate": float(baseline["success_rate"]),
            "success_count": base_count,
            "failure_count": len(failures),
        },
        "encoder_oracle": {
            "rescued_baseline_failures": int(enc_rescues),
            "failure_count": len(failures),
            "optimistic_ceiling_success_rate": float(enc_ceiling),
            "meaning": (
                "Perfect terminal prediction under the current frozen encoder "
                "and raw latent L2, with the same CEM budget/horizon."
            ),
        },
        "physical_oracle": {
            "rescued_baseline_failures": int(phy_rescues),
            "failure_count": len(failures),
            "optimistic_ceiling_success_rate": float(phy_ceiling),
            "meaning": (
                "Perfect physical terminal ranking with the same CEM budget/horizon."
            ),
        },
        "target_success_rate": target,
        "exact_live_reset_context": True,
        "reset_context_protocol": (
            "Baseline official World snapshots every live sub-env's current "
            "variation_space values after reset. Oracle Worlds are reset via "
            "an injected dataset view to those SAME captured variations, and "
            "every oracle candidate replay reuses the same case snapshot. "
            "Encoder-oracle goal embedding uses the exact raw dataset goal frame."
        ),
        "encoder_oracle_reaches_target": bool(enc_ceiling >= target),
        "physical_oracle_reaches_target": bool(phy_ceiling >= target),
        "control_retention": {
            "count": len(controls),
            "encoder_successes": int(sum(enc_map[i] for i in controls)),
            "physical_successes": int(sum(phy_map[i] for i in controls)),
        },
        "interpretation": {
            "encoder_reaches_target": (
                "If true, 95.6@B1000 is in principle reachable without changing "
                "the encoder/metric; predictor/search-aligned dynamics still have enough headroom."
            ),
            "encoder_below_target_physical_above": (
                "If true, improving predictor alone cannot reach the target; "
                "encoder/raw-latent metric is a necessary next target."
            ),
            "physical_below_target": (
                "If true, even perfect terminal ranking cannot reach the target "
                "with this CEM budget/horizon on these starts; sampling/search structure also limits performance."
            ),
        },
    }

    _write_csv(outdir / "ceiling_case_manifest.csv", manifest)
    _save_json(outdir / "ceiling_summary.json", summary)

    print("===== B=1000 CEILING SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {outdir / 'ceiling_case_manifest.csv'}")
    print(f"Saved: {outdir / 'ceiling_summary.json'}")
    print("=== CEILING DONE ===")


if __name__ == "__main__":
    run()
