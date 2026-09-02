#!/usr/bin/env python3
"""Paired B=3000 LeWM vs ALD+TF failure/rescue analysis for full PushT.

Step-1 diagnostic only. CEM is frozen.

Both models are evaluated on the exact same 100 official dataset starts with
    N=300, I=10, K=30, B=N*I=3000
in the formal configuration.

Episodes are partitioned into:
    both_success
    lewm_fail_ald_success
    both_fail
    lewm_success_ald_fail

For all non-both-success episodes plus difficulty-matched both-success controls,
selected CEM populations from BOTH models' trajectories are physically replayed.
On each *same candidate population* we score both LeWM and ALD+TF, so sampling
trajectory and model ranking are disentangled.

Primary outputs:
    paired_manifest.csv
    cross_population_metrics.csv
    case_summary.csv
    paired_summary.json
    cross_candidate_metrics.npz
    closed_loop_results.json

Physical simulator information is diagnosis-only and never changes CEM.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from eval import get_dataset, img_transform
from eval_lowbudget_failure_autopsy import (
    TraceCEMSolver,
    _build_process,
    _jsonable,
    _load_start_goal_states,
    _physical_cost,
    _prepare_eval_rows,
    _rankdata_average,
    _spearman,
)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _clone_value(v):
    if torch.is_tensor(v):
        return v.detach().cpu().clone()
    if isinstance(v, np.ndarray):
        return v.copy()
    return copy.deepcopy(v)


def _clone_info(info):
    return {k: _clone_value(v) for k, v in info.items()}


class CrossTraceCEMSolver(TraceCEMSolver):
    """TraceCEMSolver plus the exact processed solver inputs for cross-scoring."""

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        snap = _clone_info(info_dict)
        out = super().solve(info_dict, init_action=init_action)
        self.trace[-1]["solver_info"] = snap
        return out


def _close_world(world):
    try:
        world.envs.close()
    except Exception:
        try:
            world.close()
        except Exception:
            pass


def _run_closed_loop(
    cfg,
    dataset,
    process,
    policy_name,
    label,
    eval_episodes,
    eval_start,
):
    device = torch.device(str(cfg.solver.device))
    model = swm.policy.AutoCostModel(str(policy_name)).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    solver = CrossTraceCEMSolver(
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
    world = swm.World(**cfg.world, image_shape=(224, 224))
    plan_config = swm.PlanConfig(**cfg.plan_config)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_config, process=process, transform=transform
    )
    world.set_policy(policy)

    print(f"===== CLOSED LOOP [{label}] =====")
    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start.tolist(),
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
    )
    elapsed = time.time() - t0
    success = np.asarray(metrics["episode_successes"], dtype=bool)
    print(
        f"[{label}] success={float(metrics['success_rate']):.1f}% "
        f"failures={int((~success).sum())} solver_calls={len(solver.trace)} "
        f"time={elapsed:.1f}s"
    )
    _close_world(world)
    return {
        "label": label,
        "policy": str(policy_name),
        "model": model,
        "solver": solver,
        "metrics": metrics,
        "success": success,
        "elapsed": elapsed,
    }


def _slice_info(info: dict, local_idx: int):
    out = {}
    for k, v in info.items():
        if torch.is_tensor(v):
            out[k] = v[local_idx:local_idx + 1]
        elif isinstance(v, np.ndarray):
            out[k] = v[local_idx:local_idx + 1]
        elif isinstance(v, list):
            out[k] = [v[local_idx]]
        else:
            out[k] = v
    return out


@torch.inference_mode()
def _cross_model_cost(model, info_one: dict, candidates, device):
    """Run model.get_cost on one exact solver state and a fixed population."""
    x = torch.as_tensor(
        candidates, dtype=torch.float32, device=device
    ).unsqueeze(0)
    n = x.shape[1]

    expanded = {}
    for k, v in info_one.items():
        if torch.is_tensor(v):
            vb = v
            vb = vb.unsqueeze(1)
            vb = vb.expand(1, n, *vb.shape[2:])
            expanded[k] = vb
        elif isinstance(v, np.ndarray):
            expanded[k] = np.repeat(v[:, None, ...], n, axis=1)
        elif isinstance(v, list):
            expanded[k] = v * n
        else:
            expanded[k] = v

    current = expanded.copy()
    cost = model.get_cost(current, x)
    return np.asarray(cost.detach().float().cpu().numpy()[0], dtype=np.float64)


def _normalized_to_raw(candidates, action_scaler, action_block):
    c = np.asarray(candidates, dtype=np.float32)
    n, h, packed = c.shape
    raw_dim = int(action_scaler.mean_.shape[0])
    if packed != int(action_block) * raw_dim:
        raise RuntimeError(
            f"Packed action mismatch: {packed} != {action_block}*{raw_dim}"
        )
    flat = c.reshape(n * h * int(action_block), raw_dim)
    raw = action_scaler.inverse_transform(flat).astype(np.float32)
    return raw.reshape(n, h * int(action_block), raw_dim)


def _extract_variations(info_one):
    names = []
    vals = {}
    for k, v in info_one.items():
        if not str(k).startswith("variation."):
            continue
        name = str(k)[len("variation."):]
        x = v
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.shape[0] == 1:
            x = x[0]
        names.append(name)
        vals[name] = x
    return names, vals


def _reset_physical(env, state, goal, variations, seed):
    names, vals = variations
    options = {}
    if names:
        options["variation"] = names
        options["variation_values"] = vals
    env.reset(seed=int(seed), options=options if options else None)
    raw = env.unwrapped
    raw._set_goal_state(np.asarray(goal, dtype=np.float64))
    raw._set_state(np.asarray(state, dtype=np.float64))


def _replay_population(
    env,
    start_state,
    goal_state,
    raw_candidates,
    variations,
    seed_base,
):
    n = len(raw_candidates)
    phys = np.empty(n, dtype=np.float64)
    ever_success = np.zeros(n, dtype=bool)
    endpoint_success = np.zeros(n, dtype=bool)
    final_states = np.empty((n, len(start_state)), dtype=np.float64)
    for ci, acts in enumerate(raw_candidates):
        _reset_physical(
            env, start_state, goal_state, variations, seed_base + ci
        )
        raw = env.unwrapped
        ever = False
        obs = None
        for a in acts:
            obs, _, term, _, _ = raw.step(a)
            ever = ever or bool(term)
        fs = np.asarray(obs["state"], dtype=np.float64)
        final_states[ci] = fs
        pc, _, _, suc = _physical_cost(fs[None], goal_state)
        phys[ci] = float(pc[0])
        ever_success[ci] = ever
        endpoint_success[ci] = bool(suc[0])
    return phys, ever_success, endpoint_success, final_states


def _rank_percentile(cost, idx):
    ranks = _rankdata_average(np.asarray(cost, dtype=np.float64)) - 1.0
    return float(ranks[int(idx)] / max(len(ranks) - 1, 1))


def _topk_recall(model_cost, phys_cost, k):
    a = set(np.argsort(model_cost)[:k].tolist())
    b = set(np.argsort(phys_cost)[:k].tolist())
    return float(len(a & b) / max(k, 1))


def _selection_stats(model_cost, phys_cost, ever_success):
    sel = int(np.argmin(model_cost))
    oracle = int(np.argmin(phys_cost))
    return {
        "selected_idx": sel,
        "oracle_best_idx": oracle,
        "selected_phys_cost": float(phys_cost[sel]),
        "oracle_best_phys_cost": float(phys_cost[oracle]),
        "selection_regret": float(phys_cost[sel] - phys_cost[oracle]),
        "selected_phys_percentile": _rank_percentile(phys_cost, sel),
        "oracle_best_rank_percentile": _rank_percentile(model_cost, oracle),
        "selected_ever_success": bool(ever_success[sel]),
    }


def _numeric_summary(values):
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "count": 0, "mean": None, "median": None,
            "p10": None, "p90": None,
        }
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def _case_type(lewm_success, ald_success):
    if lewm_success and ald_success:
        return "both_success"
    if (not lewm_success) and ald_success:
        return "lewm_fail_ald_success"
    if (not lewm_success) and (not ald_success):
        return "both_fail"
    return "lewm_success_ald_fail"


def _matched_controls(
    start_states,
    goal_states,
    case_types,
    target_indices,
    max_controls,
):
    controls_pool = [
        i for i, t in enumerate(case_types) if t == "both_success"
    ]
    if not controls_pool or int(max_controls) <= 0:
        return []
    init_cost = np.asarray([
        _physical_cost(start_states[i:i+1], goal_states[i])[0][0]
        for i in range(len(start_states))
    ])
    unused = set(controls_pool)
    out = []
    for t in target_indices:
        if len(out) >= int(max_controls) or not unused:
            break
        j = min(unused, key=lambda x: abs(init_cost[x] - init_cost[t]))
        out.append(int(j))
        unused.remove(j)
    return out


def _audit_native(native, recomputed):
    native = np.asarray(native, dtype=np.float64)
    recomputed = np.asarray(recomputed, dtype=np.float64)
    max_abs = float(np.max(np.abs(native - recomputed)))
    scale = float(max(1.0, np.max(np.abs(native))))
    rho = _spearman(native, recomputed)
    ok = bool(max_abs <= 2e-4 * scale and (np.isnan(rho) or rho > 0.999999))
    return ok, max_abs, scale, rho


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    pcfg = cfg.get("paired", {})
    outdir = Path(str(pcfg.get(
        "output_dir", "outputs/b3000_paired_failure_analysis"
    )))
    outdir.mkdir(parents=True, exist_ok=True)

    lewm_policy = str(pcfg.get("lewm_policy", "lewm_epoch_10"))
    ald_policy = str(pcfg.get(
        "ald_policy",
        "pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10",
    ))
    replay_iterations = list(map(int, pcfg.get(
        "replay_iterations", [0, 3, 9]
    )))
    max_controls = int(pcfg.get("max_success_controls", 12))
    expected_lewm = pcfg.get("expected_lewm_success", None)
    expected_ald = pcfg.get("expected_ald_success", None)

    for it in replay_iterations:
        if it < 0 or it >= int(cfg.solver.n_steps):
            raise ValueError(f"Invalid replay iteration {it}")

    cfg.world.max_episode_steps = max(
        2 * int(cfg.eval.eval_budget),
        int(cfg.eval.goal_offset_steps) + 1,
    )

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    _, eval_rows, eval_episodes, eval_start = _prepare_eval_rows(cfg, dataset)
    start_states, goal_states = _load_start_goal_states(
        dataset, eval_episodes, eval_start, cfg.eval.goal_offset_steps
    )
    process = _build_process(cfg, dataset)

    print("============================================================")
    print("Paired B=3000 JEPA failure/rescue analysis")
    print(
        f"N={cfg.solver.num_samples} I={cfg.solver.n_steps} "
        f"K={cfg.solver.topk} B={int(cfg.solver.num_samples)*int(cfg.solver.n_steps)}"
    )
    print(f"eval episodes={cfg.eval.num_eval}")
    print(f"LeWM  : {lewm_policy}")
    print(f"ALD+TF: {ald_policy}")
    print(f"replay iterations={replay_iterations}")
    print("CEM is frozen; physical oracle is diagnosis-only.")
    print("============================================================")

    lewm = _run_closed_loop(
        cfg, dataset, process, lewm_policy, "lewm",
        eval_episodes, eval_start,
    )
    ald = _run_closed_loop(
        cfg, dataset, process, ald_policy, "ald_tf",
        eval_episodes, eval_start,
    )

    if expected_lewm is not None and abs(
        float(lewm["metrics"]["success_rate"]) - float(expected_lewm)
    ) > 1e-6:
        print(
            f"WARNING: expected LeWM {expected_lewm}%, got "
            f"{lewm['metrics']['success_rate']}%"
        )
    if expected_ald is not None and abs(
        float(ald["metrics"]["success_rate"]) - float(expected_ald)
    ) > 1e-6:
        print(
            f"WARNING: expected ALD+TF {expected_ald}%, got "
            f"{ald['metrics']['success_rate']}%"
        )

    case_types = [
        _case_type(bool(lewm["success"][i]), bool(ald["success"][i]))
        for i in range(len(eval_rows))
    ]

    critical = [
        i for i, t in enumerate(case_types) if t != "both_success"
    ]
    controls = _matched_controls(
        start_states, goal_states, case_types, critical, max_controls
    )
    selected = critical + [i for i in controls if i not in critical]

    manifest = []
    for i, ctype in enumerate(case_types):
        init_cost = float(_physical_cost(
            start_states[i:i+1], goal_states[i]
        )[0][0])
        manifest.append({
            "eval_index": i,
            "dataset_row": int(eval_rows[i]),
            "episode_idx": int(eval_episodes[i]),
            "start_step": int(eval_start[i]),
            "lewm_success": bool(lewm["success"][i]),
            "ald_tf_success": bool(ald["success"][i]),
            "case_type": ctype,
            "selected_for_cross_eval": bool(i in selected),
            "is_matched_control": bool(i in controls),
            "initial_physical_cost": init_cost,
        })
    _write_csv(outdir / "paired_manifest.csv", manifest)

    counts = {
        t: int(sum(x == t for x in case_types))
        for t in [
            "both_success", "lewm_fail_ald_success",
            "both_fail", "lewm_success_ald_fail",
        ]
    }
    print("===== PAIRED OUTCOME COUNTS =====")
    print(json.dumps(counts, indent=2))
    print(f"critical cases={critical}")
    print(f"matched controls={controls}")

    replay_env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    device = torch.device(str(cfg.solver.device))
    cross_rows = []
    cand_phys = []
    cand_lewm = []
    cand_ald = []
    cand_success = []
    pop_keys = []
    pop_eval_idx = []
    pop_source = []
    pop_solve = []
    pop_iter = []
    replay_t0 = time.time()

    runs = {"lewm": lewm, "ald_tf": ald}
    try:
        for case_no, env_i in enumerate(selected):
            ctype = case_types[env_i]
            print(
                f"cross case {case_no+1}/{len(selected)} "
                f"env={env_i} type={ctype}"
            )
            for source_label, source in runs.items():
                for trace in source["solver"].trace:
                    gidx = np.asarray(trace["global_env_indices"], dtype=np.int64)
                    loc = np.where(gidx == env_i)[0]
                    if len(loc) == 0:
                        continue
                    li = int(loc[0])
                    solve_idx = int(trace["solve_index"])
                    info_one = _slice_info(trace["solver_info"], li)
                    variations = _extract_variations(info_one)
                    start_state = np.asarray(
                        trace["solve_start_states"][li], dtype=np.float64
                    )
                    goal_state = np.asarray(goal_states[env_i], dtype=np.float64)

                    for it in replay_iterations:
                        candidates = np.asarray(
                            trace["candidates"][li, it], dtype=np.float32
                        )
                        native = np.asarray(
                            trace["predicted_costs"][li, it], dtype=np.float64
                        )

                        lewm_cost = _cross_model_cost(
                            lewm["model"], info_one, candidates, device
                        )
                        ald_cost = _cross_model_cost(
                            ald["model"], info_one, candidates, device
                        )

                        recomputed_source = (
                            lewm_cost if source_label == "lewm" else ald_cost
                        )
                        audit_ok, audit_abs, audit_scale, audit_rho = _audit_native(
                            native, recomputed_source
                        )
                        if not audit_ok:
                            raise RuntimeError(
                                "Cross-score audit failed for "
                                f"case={env_i} source={source_label} "
                                f"solve={solve_idx} iter={it}: "
                                f"max_abs={audit_abs:.6g}, scale={audit_scale:.6g}, "
                                f"rho={audit_rho}"
                            )

                        raw_candidates = _normalized_to_raw(
                            candidates, process["action"],
                            int(cfg.plan_config.action_block),
                        )
                        phys, ever, endpoint, _ = _replay_population(
                            replay_env, start_state, goal_state,
                            raw_candidates, variations,
                            seed_base=(
                                1_000_000
                                + 100_000 * env_i
                                + 10_000 * solve_idx
                                + 1_000 * it
                                + (0 if source_label == "lewm" else 500)
                            ),
                        )

                        k = int(cfg.solver.topk)
                        ls = _selection_stats(lewm_cost, phys, ever)
                        as_ = _selection_stats(ald_cost, phys, ever)
                        raw_low = np.asarray(
                            replay_env.action_space.low, dtype=np.float64
                        ).reshape(-1)
                        raw_high = np.asarray(
                            replay_env.action_space.high, dtype=np.float64
                        ).reshape(-1)
                        oob = float(np.mean(
                            (raw_candidates < raw_low[None, None, :])
                            | (raw_candidates > raw_high[None, None, :])
                        ))

                        row = {
                            "eval_index": int(env_i),
                            "case_type": ctype,
                            "is_matched_control": bool(env_i in controls),
                            "source_population": source_label,
                            "solve_index": solve_idx,
                            "cem_iteration": int(it),
                            "num_samples": len(candidates),
                            "topk": k,
                            "rho_lewm_phys": _spearman(lewm_cost, phys),
                            "rho_ald_phys": _spearman(ald_cost, phys),
                            "rho_lewm_ald": _spearman(lewm_cost, ald_cost),
                            "top10_recall_lewm": _topk_recall(
                                lewm_cost, phys, k
                            ),
                            "top10_recall_ald": _topk_recall(
                                ald_cost, phys, k
                            ),
                            "oracle_best_rank_pct_lewm":
                                ls["oracle_best_rank_percentile"],
                            "oracle_best_rank_pct_ald":
                                as_["oracle_best_rank_percentile"],
                            "selected_phys_percentile_lewm":
                                ls["selected_phys_percentile"],
                            "selected_phys_percentile_ald":
                                as_["selected_phys_percentile"],
                            "selection_regret_lewm": ls["selection_regret"],
                            "selection_regret_ald": as_["selection_regret"],
                            "selected_phys_cost_lewm": ls["selected_phys_cost"],
                            "selected_phys_cost_ald": as_["selected_phys_cost"],
                            "oracle_best_phys_cost": ls["oracle_best_phys_cost"],
                            "selected_success_lewm": ls["selected_ever_success"],
                            "selected_success_ald": as_["selected_ever_success"],
                            "oracle_success_available": bool(np.any(ever)),
                            "oracle_success_fraction": float(np.mean(ever)),
                            "candidate_raw_oob_fraction": oob,
                            "native_recompute_max_abs": audit_abs,
                            "native_recompute_rho": audit_rho,
                            "delta_rho_ald_minus_lewm":
                                _spearman(ald_cost, phys)
                                - _spearman(lewm_cost, phys),
                            "delta_top10_recall_ald_minus_lewm":
                                _topk_recall(ald_cost, phys, k)
                                - _topk_recall(lewm_cost, phys, k),
                            "delta_oracle_rank_pct_ald_minus_lewm":
                                as_["oracle_best_rank_percentile"]
                                - ls["oracle_best_rank_percentile"],
                            "delta_selected_pct_ald_minus_lewm":
                                as_["selected_phys_percentile"]
                                - ls["selected_phys_percentile"],
                            "delta_regret_ald_minus_lewm":
                                as_["selection_regret"]
                                - ls["selection_regret"],
                        }
                        cross_rows.append(row)

                        key = (
                            f"e{env_i:03d}_{ctype}_{source_label}_"
                            f"s{solve_idx:02d}_i{it:02d}"
                        )
                        pop_keys.append(key)
                        pop_eval_idx.append(env_i)
                        pop_source.append(source_label)
                        pop_solve.append(solve_idx)
                        pop_iter.append(it)
                        cand_phys.append(phys.astype(np.float32))
                        cand_lewm.append(lewm_cost.astype(np.float32))
                        cand_ald.append(ald_cost.astype(np.float32))
                        cand_success.append(ever.astype(np.bool_))
    finally:
        replay_env.close()

    _write_csv(outdir / "cross_population_metrics.csv", cross_rows)

    if cand_phys:
        np.savez_compressed(
            outdir / "cross_candidate_metrics.npz",
            population_key=np.asarray(pop_keys),
            eval_index=np.asarray(pop_eval_idx, dtype=np.int32),
            source_population=np.asarray(pop_source),
            solve_index=np.asarray(pop_solve, dtype=np.int32),
            cem_iteration=np.asarray(pop_iter, dtype=np.int32),
            physical_cost=np.stack(cand_phys),
            lewm_cost=np.stack(cand_lewm),
            ald_tf_cost=np.stack(cand_ald),
            ever_success=np.stack(cand_success),
        )
    else:
        np.savez_compressed(
            outdir / "cross_candidate_metrics.npz",
            population_key=np.asarray([], dtype=str),
        )

    summary_keys = [
        "rho_lewm_phys", "rho_ald_phys",
        "top10_recall_lewm", "top10_recall_ald",
        "oracle_best_rank_pct_lewm", "oracle_best_rank_pct_ald",
        "selected_phys_percentile_lewm", "selected_phys_percentile_ald",
        "selection_regret_lewm", "selection_regret_ald",
        "delta_rho_ald_minus_lewm",
        "delta_top10_recall_ald_minus_lewm",
        "delta_oracle_rank_pct_ald_minus_lewm",
        "delta_selected_pct_ald_minus_lewm",
        "delta_regret_ald_minus_lewm",
    ]

    case_summary_rows = []
    grouped = {}
    group_names = [
        "lewm_fail_ald_success", "both_fail",
        "lewm_success_ald_fail", "both_success_control",
    ]
    for group in group_names:
        if group == "both_success_control":
            rr = [r for r in cross_rows if r["is_matched_control"]]
        else:
            rr = [r for r in cross_rows if r["case_type"] == group]
        grouped[group] = {"count_populations": len(rr)}
        for k in summary_keys:
            grouped[group][k] = _numeric_summary(r[k] for r in rr)

        for source_label in ["lewm", "ald_tf"]:
            rs = [r for r in rr if r["source_population"] == source_label]
            row = {
                "case_group": group,
                "source_population": source_label,
                "count_populations": len(rs),
            }
            for k in summary_keys:
                s = _numeric_summary(r[k] for r in rs)
                row[f"{k}_mean"] = s["mean"]
                row[f"{k}_median"] = s["median"]
            case_summary_rows.append(row)

    _write_csv(outdir / "case_summary.csv", case_summary_rows)

    closed = {
        "lewm": {
            "policy": lewm_policy,
            "success_rate": float(lewm["metrics"]["success_rate"]),
            "episode_successes": lewm["success"],
            "solver_calls": len(lewm["solver"].trace),
            "elapsed_seconds": lewm["elapsed"],
        },
        "ald_tf": {
            "policy": ald_policy,
            "success_rate": float(ald["metrics"]["success_rate"]),
            "episode_successes": ald["success"],
            "solver_calls": len(ald["solver"].trace),
            "elapsed_seconds": ald["elapsed"],
        },
        "paired_counts": counts,
        "eval_rows": eval_rows,
        "episode_idx": eval_episodes,
        "start_step": eval_start,
    }
    (outdir / "closed_loop_results.json").write_text(
        json.dumps(_jsonable(closed), indent=2)
    )

    summary = {
        "scientific_question": (
            "At B=3000 with frozen CEM, which exact PushT episodes are "
            "rescued by ALD+TF, and on the same candidate populations does "
            "ALD+TF improve planner-facing physical ranking relative to LeWM?"
        ),
        "config": {
            "num_samples": int(cfg.solver.num_samples),
            "cem_iterations": int(cfg.solver.n_steps),
            "topk": int(cfg.solver.topk),
            "budget_B": int(cfg.solver.num_samples) * int(cfg.solver.n_steps),
            "num_eval": int(cfg.eval.num_eval),
            "eval_budget": int(cfg.eval.eval_budget),
            "goal_offset_steps": int(cfg.eval.goal_offset_steps),
            "replay_iterations": replay_iterations,
            "lewm_policy": lewm_policy,
            "ald_tf_policy": ald_policy,
            "cem_modified": False,
            "physical_oracle_used_for_planning": False,
        },
        "closed_loop": {
            "lewm_success_rate": float(lewm["metrics"]["success_rate"]),
            "ald_tf_success_rate": float(ald["metrics"]["success_rate"]),
            "paired_counts": counts,
            "critical_eval_indices": critical,
            "matched_control_indices": controls,
        },
        "cross_population_groups": grouped,
        "num_cross_populations": len(cross_rows),
        "physical_replay_seconds": float(time.time() - replay_t0),
        "metric_direction": {
            "rho_*": "higher is better",
            "top10_recall_*": "higher is better; top10 means CEM K/N=10%",
            "oracle_best_rank_pct_*": "lower is better",
            "selected_phys_percentile_*": "lower is better",
            "selection_regret_*": "lower is better",
            "delta_*_ald_minus_lewm": (
                "For rho/recall positive favors ALD; for rank/percentile/regret "
                "negative favors ALD."
            ),
        },
    }
    (outdir / "paired_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2)
    )

    print("===== FINAL PAIRED SUMMARY =====")
    print(json.dumps(_jsonable({
        "lewm_success_rate": summary["closed_loop"]["lewm_success_rate"],
        "ald_tf_success_rate": summary["closed_loop"]["ald_tf_success_rate"],
        "paired_counts": counts,
        "critical_eval_indices": critical,
        "matched_control_indices": controls,
        "num_cross_populations": len(cross_rows),
        "rescue_group": grouped["lewm_fail_ald_success"],
        "both_fail_group": grouped["both_fail"],
    }), indent=2))
    print(f"Saved: {outdir / 'closed_loop_results.json'}")
    print(f"Saved: {outdir / 'paired_manifest.csv'}")
    print(f"Saved: {outdir / 'cross_population_metrics.csv'}")
    print(f"Saved: {outdir / 'case_summary.csv'}")
    print(f"Saved: {outdir / 'cross_candidate_metrics.npz'}")
    print(f"Saved: {outdir / 'paired_summary.json'}")
    print("=== DONE ===")


if __name__ == "__main__":
    run()
