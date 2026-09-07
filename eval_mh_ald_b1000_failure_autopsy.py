#!/usr/bin/env python3
"""Focused B=1000 MH-ALD failure-case autopsy.

Formal target:
  current MH-ALD, CEM N=100 I=10 K=10, seed=42, 100 official PushT starts.

The script traces the exact closed-loop MH-ALD CEM populations and analyzes the
baseline failures identified by the latest B=1000 ceiling run. For every target
case, solve, and selected CEM iteration it compares on the SAME candidate set:

  1) MH predictor terminal latent cost used by the planner.
  2) Exact frozen-encoder terminal latent cost after physical replay.
  3) Official physical PushT terminal cost / success.

Unlike the older low-budget autopsy, physical replay restores variation fields
captured from the exact solver info whenever available. This avoids silently
changing block/agent shape, scale, color, etc. between the official run and the
diagnostic replay.

Primary questions:
  - Did a successful candidate exist in the sampled population?
  - If yes, did predictor ranking miss it?
  - Did the exact encoder metric also miss it?
  - Does CEM refinement improve or destroy physical candidate quality?
  - Which cases are metric-limited versus sampling/refinement-limited?

Oracle information is diagnosis-only and never changes planning.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
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
    _cosine,
    _elite_overlap,
    _jsonable,
    _load_start_goal_states,
    _physical_cost,
    _prepare_eval_rows,
    _rankdata_average,
    _selected_percentile,
    _spearman,
    _summary,
)
from eval_b3000_paired_failure_analysis import (
    CrossTraceCEMSolver,
    _close_world,
    _normalized_to_raw,
)
from pusht_exact_replay import (
    LiveVariationCapturePolicy,
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


def _load_ceiling_manifest(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _bool(x):
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def _case_group_from_ceiling(row):
    if row is None:
        return "unclassified"
    enc = _bool(row.get("encoder_oracle_success", False))
    phy = _bool(row.get("physical_oracle_success", False))
    if enc:
        return "predictor_limited_encoder_oracle_rescue"
    if phy:
        return "encoder_metric_candidate"
    return "search_or_horizon_candidate"


def _run_closed_loop(
    cfg,
    dataset,
    process,
    policy_name,
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

    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = int(len(eval_episodes))
    world_cfg["max_episode_steps"] = max(
        2 * int(cfg.eval.eval_budget),
        int(cfg.eval.goal_offset_steps) + 1,
    )
    world = swm.World(**world_cfg, image_shape=(224, 224))
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
    success = np.asarray(metrics["episode_successes"], dtype=bool)
    live_reset_contexts = policy.live_reset_contexts
    _close_world(world)

    if live_reset_contexts is None:
        raise RuntimeError(
            "Failed to capture live official-environment variations."
        )

    return model, solver, metrics, success, elapsed, live_reset_contexts


def _replay_population(
    env,
    start_state,
    goal_state,
    normalized_candidates,
    action_scaler,
    action_block,
    reset_context,
    need_images,
):
    raw_candidates = _normalized_to_raw(
        normalized_candidates, action_scaler, int(action_block)
    )
    n = len(raw_candidates)
    states = np.empty((n, len(start_state)), dtype=np.float64)
    images = [] if need_images else None
    ever = np.zeros(n, dtype=bool)
    endpoint = np.zeros(n, dtype=bool)
    joint = np.zeros(n, dtype=np.float64)
    theta_deg = np.zeros(n, dtype=np.float64)

    for ci, acts in enumerate(raw_candidates):
        reset_physical_exact(
            env,
            start_state,
            goal_state,
            reset_context,
        )
        raw = env.unwrapped
        obs = None
        ever_ci = False
        for action in acts:
            obs, _, term, trunc, _ = raw.step(action)
            ever_ci = ever_ci or bool(term)
            if term or trunc:
                break
        fs = np.asarray(obs["state"], dtype=np.float64)
        states[ci] = fs
        pc, je, te, suc = _physical_cost(fs[None], goal_state)
        ever[ci] = ever_ci
        endpoint[ci] = bool(suc[0])
        joint[ci] = float(je[0])
        theta_deg[ci] = float(np.degrees(te[0]))
        if images is not None:
            images.append(np.asarray(raw.render()))

    phys_cost, _, _, _ = _physical_cost(states, goal_state)
    return {
        "states": states,
        "images": images,
        "phys_cost": np.asarray(phys_cost, dtype=np.float64),
        "ever_success": ever,
        "endpoint_success": endpoint,
        "joint_error_px": joint,
        "theta_error_deg": theta_deg,
    }


def _score_stats(score, phys_cost, ever_success, k):
    score = np.asarray(score, dtype=np.float64)
    phys = np.asarray(phys_cost, dtype=np.float64)
    selected = int(np.argmin(score))
    oracle = int(np.argmin(phys))
    elite = np.argsort(score)[:k]
    phys_elite = np.argsort(phys)[:k]
    ranks = _rankdata_average(score) - 1.0
    return {
        "rho_phys": _spearman(score, phys),
        "elite_overlap_phys": _elite_overlap(score, phys, k),
        "selected_idx": selected,
        "selected_phys_cost": float(phys[selected]),
        "selected_phys_percentile": _selected_percentile(phys, selected),
        "selected_success": bool(ever_success[selected]),
        "elite_success_fraction": float(np.mean(ever_success[elite])),
        "oracle_best_rank_percentile": float(
            ranks[oracle] / max(len(score) - 1, 1)
        ),
        "selection_regret": float(phys[selected] - phys[oracle]),
        "phys_elite_success_fraction": float(np.mean(ever_success[phys_elite])),
    }


def _candidate_category(row):
    if row["oracle_has_success_candidate"]:
        if row["encoder_selected_success"]:
            return "success_candidate_encoder_can_select"
        if row["pred_selected_success"]:
            return "success_candidate_predictor_can_select"
        return "success_candidate_present_both_miss"
    return "no_success_candidate_in_population"


def _group_summary(rows):
    if not rows:
        return {}
    return {
        "population_count": len(rows),
        "oracle_success_available_fraction": float(
            np.mean([r["oracle_has_success_candidate"] for r in rows])
        ),
        "pred_ranking_miss_fraction": float(np.mean([
            r["oracle_has_success_candidate"] and not r["pred_selected_success"]
            for r in rows
        ])),
        "encoder_ranking_miss_fraction": float(np.mean([
            r["oracle_has_success_candidate"] and not r["encoder_selected_success"]
            for r in rows
        ])),
        "rho_pred_phys": _summary(r["rho_pred_phys"] for r in rows),
        "rho_encoder_phys": _summary(r["rho_encoder_phys"] for r in rows),
        "rho_pred_encoder": _summary(r["rho_pred_encoder"] for r in rows),
        "pred_elite_overlap_phys": _summary(
            r["pred_elite_overlap_phys"] for r in rows
        ),
        "encoder_elite_overlap_phys": _summary(
            r["encoder_elite_overlap_phys"] for r in rows
        ),
        "pred_selection_regret": _summary(
            r["pred_selection_regret"] for r in rows
        ),
        "encoder_selection_regret": _summary(
            r["encoder_selection_regret"] for r in rows
        ),
        "pred_selected_phys_percentile": _summary(
            r["pred_selected_phys_percentile"] for r in rows
        ),
        "encoder_selected_phys_percentile": _summary(
            r["encoder_selected_phys_percentile"] for r in rows
        ),
        "center_after_phys_cost": _summary(
            r["center_after_phys_cost"] for r in rows
        ),
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    acfg = cfg.get("failure_autopsy", {})
    outdir = Path(str(acfg.get(
        "output_dir", "outputs/mh_ald_b1000_failure_autopsy"
    )))
    outdir.mkdir(parents=True, exist_ok=True)

    replay_iterations = list(map(int, acfg.get(
        "replay_iterations", [0, 1, 3, 5, 9]
    )))
    max_solves_per_case = int(acfg.get("max_solves_per_case", 0))
    model_batch_size = int(acfg.get("model_batch_size", 64))
    ceiling_manifest_value = str(acfg.get(
        "ceiling_manifest",
        "outputs/mh_ald_b1000_ceiling_latest/ceiling_case_manifest.csv",
    )).strip()
    ceiling_manifest_path = (
        Path(ceiling_manifest_value) if ceiling_manifest_value else None
    )
    expected_success = acfg.get("expected_success", None)
    mh_policy = str(acfg.get(
        "mh_policy",
        "pusht_mh_ald_h5_seed3072_ep10_ddp4/lewm_mh_ald_h5_ddp4_epoch_10",
    ))

    for it in replay_iterations:
        if it < 0 or it >= int(cfg.solver.n_steps):
            raise ValueError(f"Invalid replay iteration {it}")

    # Resolve the mandatory world horizon exactly as in the validated autopsy
    # scripts before constructing any World/OmegaConf container.
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
    exact_goal_images = load_goal_images(
        dataset,
        eval_episodes,
        eval_start,
        cfg.eval.goal_offset_steps,
    )
    print("============================================================")
    print("MH-ALD B=1000 FAILURE AUTOPSY")
    print(f"policy={mh_policy}")
    print(
        f"N={cfg.solver.num_samples} I={cfg.solver.n_steps} "
        f"K={cfg.solver.topk} B={int(cfg.solver.num_samples)*int(cfg.solver.n_steps)}"
    )
    print(f"eval={cfg.eval.num_eval} seed={cfg.seed}")
    print(
        "ceiling_manifest="
        + (str(ceiling_manifest_path) if ceiling_manifest_path is not None else "<disabled>")
    )
    print("============================================================")

    t0 = time.time()
    (
        model,
        solver,
        metrics,
        success,
        closed_loop_seconds,
        reset_contexts,
    ) = _run_closed_loop(
        cfg, dataset, process, mh_policy, eval_episodes, eval_start
    )
    variation_counts = sorted(set(
        int(x.get("variation_count", 0)) for x in reset_contexts
    ))
    sources = sorted(set(str(x.get("source", "")) for x in reset_contexts))
    print(
        f"[exact-replay] live variation snapshots: {len(reset_contexts)}; "
        f"variation counts per case: {variation_counts}; sources={sources}"
    )

    if expected_success is not None and abs(
        float(metrics["success_rate"]) - float(expected_success)
    ) > 1e-6:
        print(
            f"WARNING: expected success={expected_success}, "
            f"got {metrics['success_rate']}"
        )

    failures = np.nonzero(~success)[0].astype(int).tolist()
    ceiling_rows = (
        _load_ceiling_manifest(ceiling_manifest_path)
        if ceiling_manifest_path is not None
        else []
    )
    ceiling_by_idx = {
        int(r["eval_index"]): r
        for r in ceiling_rows
        if _bool(r.get("is_baseline_failure", False))
    }

    if ceiling_rows:
        target = sorted(ceiling_by_idx)
        mismatch = sorted(set(target) ^ set(failures))
        if mismatch:
            raise RuntimeError(
                "Current traced run does not reproduce the ceiling failure "
                f"partition. symmetric_difference={mismatch}; "
                f"current_failures={failures}; ceiling_failures={target}"
            )
    else:
        target = failures

    print(f"Target failures: {target}")

    device = torch.device(str(cfg.solver.device))
    transform = img_transform(cfg)
    replay_env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    pop_rows = []
    candidate_payload = defaultdict(list)

    try:
        for case_pos, env_i in enumerate(target):
            case_group = _case_group_from_ceiling(ceiling_by_idx.get(env_i))
            print(
                f"case {case_pos+1}/{len(target)} eval={env_i} group={case_group}",
                flush=True,
            )

            case_solves = []
            for solve in solver.trace:
                locs = np.where(solve["global_env_indices"] == env_i)[0]
                if len(locs):
                    case_solves.append((solve, int(locs[0])))
            if max_solves_per_case > 0:
                case_solves = case_solves[:max_solves_per_case]

            for solve, li in case_solves:
                solve_idx = int(solve["solve_index"])
                start_state = np.asarray(
                    solve["solve_start_states"][li], dtype=np.float64
                )
                goal_state = np.asarray(goal_states[env_i], dtype=np.float64)

                reset_context = reset_contexts[env_i]
                variation_names = list(
                    reset_context.get("variation_names", [])
                )
                goal_image = np.asarray(exact_goal_images[env_i])
                zg = _encode(
                    model,
                    transform,
                    [goal_image],
                    device,
                    model_batch_size,
                )[0]

                for it in replay_iterations:
                    candidates = np.asarray(
                        solve["candidates"][li, it], dtype=np.float32
                    )
                    pred_cost = np.asarray(
                        solve["predicted_costs"][li, it], dtype=np.float64
                    )
                    prev_mean = np.asarray(
                        solve["prev_mean"][li, it], dtype=np.float32
                    )
                    mean_after = np.asarray(
                        solve["mean_after"][li, it], dtype=np.float32
                    )

                    replay = _replay_population(
                        replay_env,
                        start_state,
                        goal_state,
                        candidates,
                        process["action"],
                        int(cfg.plan_config.action_block),
                        reset_context,
                        need_images=True,
                    )
                    phys_cost = replay["phys_cost"]
                    ever_success = replay["ever_success"]

                    zr = _encode(
                        model,
                        transform,
                        replay["images"],
                        device,
                        model_batch_size,
                    )
                    enc_cost = (
                        torch.sum((zr - zg[None]) ** 2, dim=-1)
                        .detach().cpu().numpy().astype(np.float64)
                    )

                    pred = _score_stats(
                        pred_cost, phys_cost, ever_success, int(cfg.solver.topk)
                    )
                    enc = _score_stats(
                        enc_cost, phys_cost, ever_success, int(cfg.solver.topk)
                    )

                    phys_elite = np.argsort(phys_cost)[: int(cfg.solver.topk)]
                    phys_update = (
                        candidates[phys_elite].mean(axis=0) - prev_mean
                    )
                    pred_elite = np.argsort(pred_cost)[: int(cfg.solver.topk)]
                    enc_elite = np.argsort(enc_cost)[: int(cfg.solver.topk)]
                    pred_update = candidates[pred_elite].mean(axis=0) - prev_mean
                    enc_update = candidates[enc_elite].mean(axis=0) - prev_mean

                    center_replay = _replay_population(
                        replay_env,
                        start_state,
                        goal_state,
                        mean_after[None],
                        process["action"],
                        int(cfg.plan_config.action_block),
                        reset_context,
                        need_images=False,
                    )

                    oracle_idx = int(np.argmin(phys_cost))
                    pred_idx = int(pred["selected_idx"])
                    enc_idx = int(enc["selected_idx"])

                    row = {
                        "eval_index": int(env_i),
                        "dataset_row": int(eval_rows[env_i]),
                        "episode_idx": int(eval_episodes[env_i]),
                        "start_step": int(eval_start[env_i]),
                        "case_group": case_group,
                        "solve_index": solve_idx,
                        "cem_iteration": int(it),
                        "num_samples": int(len(candidates)),
                        "topk": int(cfg.solver.topk),
                        "dataset_seed": reset_context.get("seed", None),
                        "variation_names": "|".join(variation_names),
                        "variation_count": int(
                            reset_context.get("variation_count", 0)
                        ),
                        "solve_start_phys_cost": float(
                            _physical_cost(
                                start_state[None], goal_state
                            )[0][0]
                        ),
                        "oracle_has_success_candidate": bool(
                            np.any(ever_success)
                        ),
                        "oracle_success_candidate_fraction": float(
                            np.mean(ever_success)
                        ),
                        "oracle_best_idx": oracle_idx,
                        "oracle_best_phys_cost": float(phys_cost[oracle_idx]),
                        "oracle_best_joint_error_px": float(
                            replay["joint_error_px"][oracle_idx]
                        ),
                        "oracle_best_theta_error_deg": float(
                            replay["theta_error_deg"][oracle_idx]
                        ),
                        "rho_pred_phys": float(pred["rho_phys"]),
                        "rho_encoder_phys": float(enc["rho_phys"]),
                        "rho_pred_encoder": float(
                            _spearman(pred_cost, enc_cost)
                        ),
                        "pred_elite_overlap_phys": float(
                            pred["elite_overlap_phys"]
                        ),
                        "encoder_elite_overlap_phys": float(
                            enc["elite_overlap_phys"]
                        ),
                        "pred_update_cos_phys": float(
                            _cosine(pred_update, phys_update)
                        ),
                        "encoder_update_cos_phys": float(
                            _cosine(enc_update, phys_update)
                        ),
                        "pred_selected_idx": pred_idx,
                        "encoder_selected_idx": enc_idx,
                        "pred_selected_success": bool(
                            pred["selected_success"]
                        ),
                        "encoder_selected_success": bool(
                            enc["selected_success"]
                        ),
                        "pred_selected_phys_cost": float(
                            pred["selected_phys_cost"]
                        ),
                        "encoder_selected_phys_cost": float(
                            enc["selected_phys_cost"]
                        ),
                        "pred_selected_phys_percentile": float(
                            pred["selected_phys_percentile"]
                        ),
                        "encoder_selected_phys_percentile": float(
                            enc["selected_phys_percentile"]
                        ),
                        "pred_selection_regret": float(
                            pred["selection_regret"]
                        ),
                        "encoder_selection_regret": float(
                            enc["selection_regret"]
                        ),
                        "pred_oracle_best_rank_percentile": float(
                            pred["oracle_best_rank_percentile"]
                        ),
                        "encoder_oracle_best_rank_percentile": float(
                            enc["oracle_best_rank_percentile"]
                        ),
                        "pred_elite_success_fraction": float(
                            pred["elite_success_fraction"]
                        ),
                        "encoder_elite_success_fraction": float(
                            enc["elite_success_fraction"]
                        ),
                        "center_after_phys_cost": float(
                            center_replay["phys_cost"][0]
                        ),
                        "center_after_success": bool(
                            center_replay["ever_success"][0]
                        ),
                    }
                    row["population_category"] = _candidate_category(row)
                    pop_rows.append(row)

                    key = f"case{env_i}_solve{solve_idx}_iter{it}"
                    candidate_payload["key"].append(key)
                    candidate_payload["eval_index"].append(int(env_i))
                    candidate_payload["solve_index"].append(solve_idx)
                    candidate_payload["cem_iteration"].append(int(it))
                    candidate_payload["pred_cost"].append(
                        pred_cost.astype(np.float32)
                    )
                    candidate_payload["encoder_cost"].append(
                        enc_cost.astype(np.float32)
                    )
                    candidate_payload["physical_cost"].append(
                        phys_cost.astype(np.float32)
                    )
                    candidate_payload["ever_success"].append(
                        ever_success.astype(np.uint8)
                    )

                    print(
                        f"  solve={solve_idx:2d} it={it:2d} "
                        f"succCand={int(np.any(ever_success))} "
                        f"rhoP={pred['rho_phys']:.3f} "
                        f"rhoE={enc['rho_phys']:.3f} "
                        f"regP={pred['selection_regret']:.3f} "
                        f"regE={enc['selection_regret']:.3f}",
                        flush=True,
                    )
    finally:
        replay_env.close()

    _write_csv(outdir / "population_autopsy.csv", pop_rows)

    if candidate_payload["key"]:
        np.savez_compressed(
            outdir / "candidate_autopsy.npz",
            key=np.asarray(candidate_payload["key"]),
            eval_index=np.asarray(candidate_payload["eval_index"], dtype=np.int32),
            solve_index=np.asarray(candidate_payload["solve_index"], dtype=np.int32),
            cem_iteration=np.asarray(
                candidate_payload["cem_iteration"], dtype=np.int32
            ),
            pred_cost=np.stack(candidate_payload["pred_cost"]),
            encoder_cost=np.stack(candidate_payload["encoder_cost"]),
            physical_cost=np.stack(candidate_payload["physical_cost"]),
            ever_success=np.stack(candidate_payload["ever_success"]),
        )

    case_rows = []
    for env_i in target:
        rr = [r for r in pop_rows if r["eval_index"] == env_i]
        if not rr:
            continue
        oracle_available = [
            r for r in rr if r["oracle_has_success_candidate"]
        ]
        both_miss = [
            r for r in oracle_available
            if not r["pred_selected_success"]
            and not r["encoder_selected_success"]
        ]
        pred_only_miss = [
            r for r in oracle_available
            if not r["pred_selected_success"]
            and r["encoder_selected_success"]
        ]
        enc_only_miss = [
            r for r in oracle_available
            if r["pred_selected_success"]
            and not r["encoder_selected_success"]
        ]

        first_success_pop = min(
            (
                (r["solve_index"], r["cem_iteration"])
                for r in rr
                if r["oracle_has_success_candidate"]
            ),
            default=None,
        )
        case_rows.append({
            "eval_index": int(env_i),
            "dataset_row": int(eval_rows[env_i]),
            "case_group": rr[0]["case_group"],
            "num_population_snapshots": len(rr),
            "success_candidate_population_fraction": float(
                np.mean([r["oracle_has_success_candidate"] for r in rr])
            ),
            "num_populations_with_success_candidate": len(oracle_available),
            "num_both_predictor_encoder_miss": len(both_miss),
            "num_predictor_miss_encoder_selects": len(pred_only_miss),
            "num_encoder_miss_predictor_selects": len(enc_only_miss),
            "first_success_candidate_solve": (
                first_success_pop[0] if first_success_pop else -1
            ),
            "first_success_candidate_iteration": (
                first_success_pop[1] if first_success_pop else -1
            ),
            "median_rho_pred_phys": float(np.nanmedian([
                r["rho_pred_phys"] for r in rr
            ])),
            "median_rho_encoder_phys": float(np.nanmedian([
                r["rho_encoder_phys"] for r in rr
            ])),
            "median_pred_regret": float(np.nanmedian([
                r["pred_selection_regret"] for r in rr
            ])),
            "median_encoder_regret": float(np.nanmedian([
                r["encoder_selection_regret"] for r in rr
            ])),
            "final_snapshot_center_phys_cost": float(
                sorted(
                    rr,
                    key=lambda r: (r["solve_index"], r["cem_iteration"])
                )[-1]["center_after_phys_cost"]
            ),
        })

    _write_csv(outdir / "case_summary.csv", case_rows)

    iteration_rows = []
    for group in sorted(set(r["case_group"] for r in pop_rows)):
        for it in replay_iterations:
            rr = [
                r for r in pop_rows
                if r["case_group"] == group
                and r["cem_iteration"] == it
            ]
            if not rr:
                continue
            gs = _group_summary(rr)
            iteration_rows.append({
                "case_group": group,
                "cem_iteration": int(it),
                "population_count": gs["population_count"],
                "oracle_success_available_fraction":
                    gs["oracle_success_available_fraction"],
                "pred_ranking_miss_fraction":
                    gs["pred_ranking_miss_fraction"],
                "encoder_ranking_miss_fraction":
                    gs["encoder_ranking_miss_fraction"],
                "median_rho_pred_phys":
                    gs["rho_pred_phys"]["median"],
                "median_rho_encoder_phys":
                    gs["rho_encoder_phys"]["median"],
                "median_rho_pred_encoder":
                    gs["rho_pred_encoder"]["median"],
                "median_pred_selection_regret":
                    gs["pred_selection_regret"]["median"],
                "median_encoder_selection_regret":
                    gs["encoder_selection_regret"]["median"],
                "median_center_after_phys_cost":
                    gs["center_after_phys_cost"]["median"],
            })
    _write_csv(outdir / "iteration_group_summary.csv", iteration_rows)

    grouped = {}
    for group in sorted(set(r["case_group"] for r in pop_rows)):
        grouped[group] = _group_summary([
            r for r in pop_rows if r["case_group"] == group
        ])

    summary = {
        "scientific_question": (
            "For the exact B=1000 MH-ALD failures, do sampled successful "
            "candidates exist, and are they rejected by predictor ranking, "
            "the frozen encoder metric, or both? If no successful candidate "
            "exists, is the case sampling/refinement limited?"
        ),
        "config": {
            "policy": mh_policy,
            "seed": int(cfg.seed),
            "num_eval": int(cfg.eval.num_eval),
            "num_samples": int(cfg.solver.num_samples),
            "cem_iterations": int(cfg.solver.n_steps),
            "topk": int(cfg.solver.topk),
            "budget_B": int(cfg.solver.num_samples)
                        * int(cfg.solver.n_steps),
            "replay_iterations": replay_iterations,
            "max_solves_per_case": max_solves_per_case,
            "ceiling_manifest": (
                str(ceiling_manifest_path)
                if ceiling_manifest_path is not None
                else None
            ),
            "exact_live_reset_context": True,
            "reset_context_protocol": (
                "Official closed-loop World snapshots every live sub-env's "
                "current variation_space values after reset. Every diagnostic "
                "candidate in that case reuses the SAME captured variation "
                "snapshot, then solve-start state/goal are restored. Goal "
                "embedding uses the exact raw dataset goal frame."
            ),
            "physical_oracle_used_for_planning": False,
        },
        "closed_loop": {
            "success_rate": float(metrics["success_rate"]),
            "failure_eval_indices": failures,
            "target_eval_indices": target,
            "closed_loop_seconds": float(closed_loop_seconds),
        },
        "case_groups": {
            str(i): _case_group_from_ceiling(ceiling_by_idx.get(i))
            for i in target
        },
        "group_summary": grouped,
        "interpretation_rules": {
            "success_candidate_present_both_miss": (
                "Sampling was sufficient at this population, but both current "
                "MH predictor and exact frozen-encoder raw-L2 ranking failed."
            ),
            "predictor_miss_encoder_selects": (
                "A direct predictor-side ranking failure: exact encoder metric "
                "would select a successful candidate from the same population."
            ),
            "no_success_candidate_in_population": (
                "This population cannot be fixed by reranking alone; coverage, "
                "earlier elite updates, finite horizon, or action distribution "
                "must change."
            ),
            "encoder_metric_candidate": (
                "Ceiling experiment was rescued by physical oracle but not "
                "encoder oracle; use this autopsy to see where raw latent-L2 "
                "rejects physically useful candidates."
            ),
        },
        "timing": {
            "total_seconds": float(time.time() - t0),
        },
    }
    (outdir / "failure_autopsy_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2)
    )

    print("===== FAILURE AUTOPSY SUMMARY =====")
    print(json.dumps(_jsonable({
        "closed_loop": summary["closed_loop"],
        "case_groups": summary["case_groups"],
        "group_summary": summary["group_summary"],
    }), indent=2))
    print(f"Saved: {outdir / 'population_autopsy.csv'}")
    print(f"Saved: {outdir / 'candidate_autopsy.npz'}")
    print(f"Saved: {outdir / 'case_summary.csv'}")
    print(f"Saved: {outdir / 'iteration_group_summary.csv'}")
    print(f"Saved: {outdir / 'failure_autopsy_summary.json'}")
    print("=== FAILURE AUTOPSY DONE ===")


if __name__ == "__main__":
    run()
