#!/usr/bin/env python3
"""Hard-state encoder geometry autopsy for official PushT.

Motivation
----------
Step-2 showed that on ALD+TF rescue states the real-future encoder cost can
track physical task cost well, while the two remaining both-fail cases
(eval 27 and 53) have a much weaker encoder ceiling.

This diagnostic asks *what physical factor breaks the encoder geometry?*

It keeps CEM frozen, reruns the official ALD+TF B=3000 closed-loop trajectory
only to recover the exact solve-1 states/populations, and physically replays
selected candidate populations.  It then decomposes physical goal error into:

  pusher_xy   : pusher position error
  block_xy    : T-block position error
  theta       : T-block orientation error
  joint_xy    : pusher_xy + block_xy (official position term)
  object_task : block_xy + theta (diagnostic: removes pusher nuisance)
  official    : pusher_xy + block_xy + theta

All costs use the same normalization as the existing PushT diagnostics:
20 px for XY and pi/9 for theta.

Primary cases:
  27, 53  : both-fail
  23      : ALD regression control from Step 1
  3 hardest rescue controls, selected automatically by largest ALD solve-1
            physical cost from the previous mean_plan_causal_chain.csv.

The encoder is frozen/shared by LeWM and ALD+TF, so a single encoder is enough.

Outputs
-------
selected_cases.csv
population_component_metrics.csv
case_component_summary.csv
candidate_components.npz
summary.json

No physical signal is used by CEM.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from eval import get_dataset, img_transform
from eval_lowbudget_failure_autopsy import (
    _angle_error_rad,
    _build_process,
    _jsonable,
    _load_start_goal_states,
    _prepare_eval_rows,
    _spearman,
)
from eval_b3000_paired_failure_analysis import (
    _extract_variations,
    _normalized_to_raw,
    _rank_percentile,
    _slice_info,
    _topk_recall,
)
from eval_b3000_critical_jepa_mechanism import (
    _find_trace_for_env,
    _goal_embedding_from_solver,
    _run_closed_loop_recording,
)
from eval_pusht_action_ranking import _encode_images, _rollout_candidate


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _summary(x):
    x = np.asarray(list(x), dtype=np.float64)
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


def _rankdata(x):
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


def _partial_spearman(y, x, controls):
    """Partial Spearman via residual correlation of rank-transformed variables."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    mask = np.isfinite(y) & np.isfinite(x)
    mask &= np.all(np.isfinite(controls), axis=1)
    if mask.sum() < max(5, controls.shape[1] + 3):
        return float("nan")

    yr = _rankdata(y[mask])
    xr = _rankdata(x[mask])
    cr = np.stack(
        [_rankdata(controls[mask, j]) for j in range(controls.shape[1])],
        axis=1,
    )
    design = np.concatenate(
        [np.ones((len(cr), 1), dtype=np.float64), cr], axis=1
    )
    by, *_ = np.linalg.lstsq(design, yr, rcond=None)
    bx, *_ = np.linalg.lstsq(design, xr, rcond=None)
    ry = yr - design @ by
    rx = xr - design @ bx
    den = np.linalg.norm(ry) * np.linalg.norm(rx)
    return float(np.dot(ry, rx) / den) if den > 1e-12 else float("nan")


def _matched_pair_accuracy(score, target, nuisance, match_frac=0.10, diff_frac=0.10):
    """Does score order target correctly when nuisance is approximately matched?"""
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    nuisance = np.asarray(nuisance, dtype=np.float64)

    tq25, tq75 = np.percentile(target, [25, 75])
    nq25, nq75 = np.percentile(nuisance, [25, 75])
    tmargin = float(diff_frac) * max(float(tq75 - tq25), 1e-12)
    nmargin = float(match_frac) * max(float(nq75 - nq25), 1e-12)

    correct = 0.0
    total = 0
    for i in range(len(score)):
        for j in range(i + 1, len(score)):
            if abs(nuisance[i] - nuisance[j]) > nmargin:
                continue
            dt = target[i] - target[j]
            if abs(dt) <= tmargin:
                continue
            ds = score[i] - score[j]
            total += 1
            if abs(ds) <= 1e-12:
                correct += 0.5
            elif np.sign(ds) == np.sign(dt):
                correct += 1.0
    return (
        float(correct / total) if total else float("nan"),
        int(total),
    )


def _component_costs(states, goal_state):
    s = np.asarray(states, dtype=np.float64)
    g = np.asarray(goal_state, dtype=np.float64)
    pusher = np.linalg.norm(s[:, 0:2] - g[0:2], axis=1)
    block = np.linalg.norm(s[:, 2:4] - g[2:4], axis=1)
    theta = _angle_error_rad(s[:, 4], g[4])

    pusher_c = (pusher / 20.0) ** 2
    block_c = (block / 20.0) ** 2
    theta_c = (theta / (np.pi / 9.0)) ** 2
    joint_xy_c = pusher_c + block_c
    object_task_c = block_c + theta_c
    official_c = joint_xy_c + theta_c
    return {
        "pusher_xy_error_px": pusher,
        "block_xy_error_px": block,
        "theta_error_rad": theta,
        "theta_error_deg": np.degrees(theta),
        "pusher_cost": pusher_c,
        "block_cost": block_c,
        "theta_cost": theta_c,
        "joint_xy_cost": joint_xy_c,
        "object_task_cost": object_task_c,
        "official_cost": official_c,
    }


def _selection_metrics(score, target, k):
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    sel = int(np.argmin(score))
    oracle = int(np.argmin(target))
    return {
        "rho": _spearman(score, target),
        "top10_recall": _topk_recall(score, target, int(k)),
        "oracle_best_rank_pct": _rank_percentile(score, oracle),
        "selected_target_percentile": _rank_percentile(target, sel),
        "selection_regret": float(target[sel] - target[oracle]),
    }


def _read_reference_manifest(path: Path):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "eval_index": int(r["eval_index"]),
                "case_type": r["case_type"],
            })
    return rows


def _choose_rescue_controls(mean_chain_path: Path, n_controls: int):
    """Choose hardest ALD rescue states by previous solve-1 physical cost."""
    by_case = {}
    with mean_chain_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("case_type") != "lewm_fail_ald_success":
                continue
            if r.get("source_trajectory") != "ald_tf":
                continue
            # physical fields are repeated for each scoring_model; de-duplicate.
            i = int(r["eval_index"])
            cost = float(r["next_solve_phys_cost"])
            by_case[i] = cost
    ordered = sorted(by_case.items(), key=lambda kv: kv[1], reverse=True)
    return [i for i, _ in ordered[: int(n_controls)]], dict(ordered)


def _group_summary(rows, group_key):
    groups = {}
    for r in rows:
        groups.setdefault(str(r[group_key]), []).append(r)

    metric_keys = [
        k for k in rows[0]
        if (
            k.startswith("rho_enc_")
            or k.startswith("partial_rho_")
            or k.startswith("top10_")
            or k.startswith("oracle_best_")
            or k.startswith("selected_")
            or k.startswith("regret_")
            or k.startswith("matched_")
            or k.endswith("_iqr")
            or k in {"contact_fraction", "block_motion_mean_px"}
        )
    ]
    out = {}
    flat_rows = []
    for name, rr in groups.items():
        payload = {"num_populations": len(rr)}
        flat = {group_key: name, "num_populations": len(rr)}
        for k in metric_keys:
            s = _summary(x.get(k, np.nan) for x in rr)
            payload[k] = s
            flat[f"{k}_mean"] = s["mean"]
            flat[f"{k}_median"] = s["median"]
        out[name] = payload
        flat_rows.append(flat)
    return out, flat_rows


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    gcfg = cfg.get("geometry", {})
    outdir = Path(str(gcfg.get(
        "output_dir", "outputs/b3000_hardstate_encoder_geometry"
    )))
    outdir.mkdir(parents=True, exist_ok=True)

    policy_name = str(gcfg.get(
        "ald_policy",
        "pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10",
    ))
    reference_manifest = Path(str(gcfg.get("reference_manifest", "")))
    reference_mean_chain = Path(str(gcfg.get("reference_mean_chain", "")))
    hard_cases = list(map(int, gcfg.get("hard_cases", [27, 53])))
    regression_cases = list(map(int, gcfg.get("regression_cases", [23])))
    n_rescue_controls = int(gcfg.get("num_rescue_controls", 3))
    manual_cases = list(map(int, gcfg.get("manual_cases", [])))
    replay_iterations = list(map(int, gcfg.get("replay_iterations", [0, 3, 9])))
    solve_index = int(gcfg.get("solve_index", 1))
    model_batch = int(gcfg.get("model_batch_size", 64))

    cfg.world.max_episode_steps = max(
        2 * int(cfg.eval.eval_budget),
        int(cfg.eval.goal_offset_steps) + 1,
    )

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    _, eval_rows, eval_episodes, eval_start = _prepare_eval_rows(cfg, dataset)
    _, goal_states = _load_start_goal_states(
        dataset, eval_episodes, eval_start, cfg.eval.goal_offset_steps
    )
    process = _build_process(cfg, dataset)

    if manual_cases:
        selected = manual_cases
        case_reason = {i: "manual" for i in selected}
        reference = None
        rescue_difficulty = {}
    else:
        if not reference_manifest.is_file():
            raise FileNotFoundError(
                f"Missing Step-1 reference manifest: {reference_manifest}"
            )
        if not reference_mean_chain.is_file():
            raise FileNotFoundError(
                f"Missing Step-2 mean chain: {reference_mean_chain}"
            )
        reference = _read_reference_manifest(reference_manifest)
        label_by_i = {r["eval_index"]: r["case_type"] for r in reference}
        controls, rescue_difficulty = _choose_rescue_controls(
            reference_mean_chain, n_rescue_controls
        )
        selected = list(dict.fromkeys(hard_cases + regression_cases + controls))
        case_reason = {}
        for i in hard_cases:
            case_reason[i] = "both_fail_primary"
        for i in regression_cases:
            case_reason[i] = "regression_control"
        for i in controls:
            case_reason[i] = "hard_rescue_control"
        for i in selected:
            if i not in label_by_i:
                raise RuntimeError(f"Selected eval index {i} missing from manifest.")

    print("============================================================")
    print("Hard-state encoder geometry autopsy")
    print(
        f"Frozen CEM: N={cfg.solver.num_samples} I={cfg.solver.n_steps} "
        f"K={cfg.solver.topk} B={int(cfg.solver.num_samples)*int(cfg.solver.n_steps)}"
    )
    print(f"source trajectory: ALD+TF")
    print(f"solve={solve_index}, iterations={replay_iterations}")
    print(f"selected eval indices={selected}")
    print("No planner modification. Physical state is diagnosis-only.")
    print("============================================================")

    run_result = _run_closed_loop_recording(
        cfg, dataset, process, policy_name, "ald_tf",
        eval_episodes, eval_start, selected,
    )

    if reference is not None:
        label_by_i = {r["eval_index"]: r["case_type"] for r in reference}
        bad = []
        for i in selected:
            current_success = bool(run_result["success"][i])
            ctype = label_by_i[i]
            expected_success = ctype in {
                "both_success", "lewm_fail_ald_success"
            }
            if current_success != expected_success:
                bad.append((i, ctype, expected_success, current_success))
        if bad:
            raise RuntimeError(
                "Selected ALD closed-loop outcomes drifted from reference: "
                + repr(bad)
            )
    else:
        label_by_i = {i: "manual" for i in selected}

    selection_rows = []
    for i in selected:
        selection_rows.append({
            "eval_index": i,
            "case_type": label_by_i[i],
            "selection_reason": case_reason[i],
            "episode_idx": int(eval_episodes[i]),
            "start_step": int(eval_start[i]),
            "previous_ald_solve1_phys_cost":
                rescue_difficulty.get(i, np.nan),
            "current_ald_success": bool(run_result["success"][i]),
        })
    _write_csv(outdir / "selected_cases.csv", selection_rows)

    model = run_result["model"]
    device = torch.device(str(cfg.solver.device))
    transform = img_transform(cfg)
    action_block = int(cfg.plan_config.action_block)
    k = int(cfg.solver.topk)

    env = gym.make(str(cfg.world.env_name), render_mode="rgb_array")
    pop_rows = []
    meta = []
    arrays = {
        "enc_cost": [],
        "official_cost": [],
        "object_task_cost": [],
        "pusher_cost": [],
        "block_cost": [],
        "theta_cost": [],
        "pusher_xy_error_px": [],
        "block_xy_error_px": [],
        "theta_error_deg": [],
        "had_contact": [],
        "contact_steps": [],
        "final_state": [],
    }

    t0 = time.time()
    try:
        for ci, env_i in enumerate(selected):
            print(
                f"case {ci+1}/{len(selected)} env={env_i} "
                f"type={label_by_i[env_i]} reason={case_reason[env_i]}"
            )
            tr, li = _find_trace_for_env(
                run_result["solver"].trace, env_i, solve_index
            )
            if tr is None or li is None:
                raise RuntimeError(
                    f"Missing solve={solve_index} trace for eval {env_i}."
                )

            info_one = _slice_info(tr["solver_info"], li)
            variations = _extract_variations(info_one)
            start_state = np.asarray(
                tr["solve_start_states"][li], dtype=np.float64
            )
            goal_state = np.asarray(goal_states[env_i], dtype=np.float64)
            zg = _goal_embedding_from_solver(model, info_one, device)

            for it in replay_iterations:
                candidates = np.asarray(
                    tr["candidates"][li, it], dtype=np.float32
                )
                raw_candidates = _normalized_to_raw(
                    candidates, process["action"], action_block
                )

                final_states = []
                final_images = []
                had_contact = []
                contact_steps = []
                for j, raw_actions in enumerate(raw_candidates):
                    fs, fi, hc, cs = _rollout_candidate(
                        env,
                        start_state,
                        goal_state,
                        raw_actions,
                        seed=(
                            8_000_000
                            + 100_000 * env_i
                            + 1_000 * it
                            + j
                        ),
                    )
                    final_states.append(fs)
                    final_images.append(fi)
                    had_contact.append(hc)
                    contact_steps.append(cs)

                final_states = np.asarray(final_states, dtype=np.float64)
                had_contact = np.asarray(had_contact, dtype=bool)
                contact_steps = np.asarray(contact_steps, dtype=np.int16)

                zr = _encode_images(
                    model, transform, final_images, device, model_batch
                )
                enc = torch.sum(
                    (zr - zg[None]) ** 2, dim=-1
                ).cpu().numpy().astype(np.float64)

                comp = _component_costs(final_states, goal_state)
                p = comp["pusher_cost"]
                b = comp["block_cost"]
                th = comp["theta_cost"]
                obj = comp["object_task_cost"]
                off = comp["official_cost"]

                metrics = {}
                for name, target in [
                    ("official", off),
                    ("object_task", obj),
                    ("pusher", p),
                    ("block", b),
                    ("theta", th),
                ]:
                    m = _selection_metrics(enc, target, k)
                    metrics[f"rho_enc_{name}"] = m["rho"]
                    metrics[f"top10_enc_{name}"] = m["top10_recall"]
                    metrics[f"oracle_best_rank_pct_enc_{name}"] = (
                        m["oracle_best_rank_pct"]
                    )
                    metrics[f"selected_{name}_percentile"] = (
                        m["selected_target_percentile"]
                    )
                    metrics[f"regret_{name}"] = m["selection_regret"]

                # Direct factor association after controlling the other two.
                metrics["partial_rho_enc_pusher_given_block_theta"] = (
                    _partial_spearman(enc, p, np.stack([b, th], axis=1))
                )
                metrics["partial_rho_enc_block_given_pusher_theta"] = (
                    _partial_spearman(enc, b, np.stack([p, th], axis=1))
                )
                metrics["partial_rho_enc_theta_given_pusher_block"] = (
                    _partial_spearman(enc, th, np.stack([p, b], axis=1))
                )

                # Matched-pair tests isolate nuisance/task factors.
                acc, n = _matched_pair_accuracy(enc, p, obj)
                metrics["matched_object_pusher_acc"] = acc
                metrics["matched_object_pusher_pairs"] = n

                acc, n = _matched_pair_accuracy(enc, obj, p)
                metrics["matched_pusher_object_acc"] = acc
                metrics["matched_pusher_object_pairs"] = n

                acc, n = _matched_pair_accuracy(enc, th, b)
                metrics["matched_block_theta_acc"] = acc
                metrics["matched_block_theta_pairs"] = n

                acc, n = _matched_pair_accuracy(enc, b, th)
                metrics["matched_theta_block_acc"] = acc
                metrics["matched_theta_block_pairs"] = n

                for name, x in [
                    ("pusher", p), ("block", b), ("theta", th),
                    ("object_task", obj), ("official", off),
                ]:
                    q25, q75 = np.percentile(x, [25, 75])
                    metrics[f"{name}_iqr"] = float(q75 - q25)

                metrics["contact_fraction"] = float(np.mean(had_contact))
                metrics["mean_contact_steps"] = float(np.mean(contact_steps))
                metrics["block_motion_mean_px"] = float(np.mean(
                    np.linalg.norm(
                        final_states[:, 2:4] - start_state[None, 2:4],
                        axis=1,
                    )
                ))

                pop_rows.append({
                    "eval_index": int(env_i),
                    "case_type": label_by_i[env_i],
                    "selection_reason": case_reason[env_i],
                    "solve_index": solve_index,
                    "cem_iteration": int(it),
                    "num_samples": len(candidates),
                    **metrics,
                })

                meta.append((env_i, label_by_i[env_i], case_reason[env_i], it))
                arrays["enc_cost"].append(enc.astype(np.float32))
                for key in [
                    "official_cost", "object_task_cost", "pusher_cost",
                    "block_cost", "theta_cost", "pusher_xy_error_px",
                    "block_xy_error_px", "theta_error_deg",
                ]:
                    arrays[key].append(
                        np.asarray(comp[key], dtype=np.float32)
                    )
                arrays["had_contact"].append(had_contact)
                arrays["contact_steps"].append(contact_steps)
                arrays["final_state"].append(final_states.astype(np.float32))
    finally:
        env.close()

    _write_csv(outdir / "population_component_metrics.csv", pop_rows)

    np.savez_compressed(
        outdir / "candidate_components.npz",
        eval_index=np.asarray([x[0] for x in meta], dtype=np.int32),
        case_type=np.asarray([x[1] for x in meta]),
        selection_reason=np.asarray([x[2] for x in meta]),
        cem_iteration=np.asarray([x[3] for x in meta], dtype=np.int32),
        **{k: np.stack(v) for k, v in arrays.items()},
    )

    # Two useful aggregate views: scientific cohort and individual case.
    by_reason, flat_reason = _group_summary(pop_rows, "selection_reason")
    by_case, flat_case = _group_summary(pop_rows, "eval_index")
    _write_csv(
        outdir / "case_component_summary.csv",
        flat_reason + flat_case,
    )

    summary = {
        "scientific_question": (
            "Why does real-future encoder Euclidean goal geometry degrade in "
            "the remaining hard PushT failures: pusher nuisance, block XY, "
            "rotation theta, or their composition?"
        ),
        "config": {
            "num_samples": int(cfg.solver.num_samples),
            "cem_iterations": int(cfg.solver.n_steps),
            "topk": int(cfg.solver.topk),
            "budget_B": int(cfg.solver.num_samples) * int(cfg.solver.n_steps),
            "solve_index": solve_index,
            "replay_iterations": replay_iterations,
            "source_trajectory": "ald_tf",
            "hard_cases": hard_cases,
            "regression_cases": regression_cases,
            "num_rescue_controls": n_rescue_controls,
            "selected_cases": selected,
            "encoder_shared_frozen": True,
            "cem_modified": False,
            "physical_oracle_used_for_planning": False,
        },
        "component_definitions": {
            "pusher_cost": "(pusher_xy_error_px / 20)^2",
            "block_cost": "(block_xy_error_px / 20)^2",
            "theta_cost": "(wrapped_theta_error_rad / (pi/9))^2",
            "joint_xy_cost": "pusher_cost + block_cost",
            "object_task_cost": "block_cost + theta_cost; diagnostic only",
            "official_cost": (
                "pusher_cost + block_cost + theta_cost; same existing "
                "diagnostic decomposition as official joint-position + theta"
            ),
        },
        "matched_pair_interpretation": {
            "matched_object_pusher_acc": (
                "With object_task approximately matched, does encoder prefer "
                "the candidate with smaller pusher error? High value indicates "
                "pusher/nuisance sensitivity."
            ),
            "matched_pusher_object_acc": (
                "With pusher approximately matched, does encoder prefer lower "
                "object-task error? This is desired task sensitivity."
            ),
            "matched_block_theta_acc": (
                "With block XY approximately matched, can encoder order theta?"
            ),
            "matched_theta_block_acc": (
                "With theta approximately matched, can encoder order block XY?"
            ),
        },
        "by_reason": by_reason,
        "by_eval_index": by_case,
        "ald_success_rate_current": float(run_result["metrics"]["success_rate"]),
        "elapsed_component_seconds": float(time.time() - t0),
    }
    (outdir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2)
    )

    print("===== HARD-STATE ENCODER GEOMETRY SUMMARY =====")
    print(json.dumps(_jsonable({
        "selected_cases": selected,
        "selection_rows": selection_rows,
        "by_reason": by_reason,
        "elapsed_component_seconds": summary["elapsed_component_seconds"],
    }), indent=2))
    print(f"Saved: {outdir / 'selected_cases.csv'}")
    print(f"Saved: {outdir / 'population_component_metrics.csv'}")
    print(f"Saved: {outdir / 'case_component_summary.csv'}")
    print(f"Saved: {outdir / 'candidate_components.npz'}")
    print(f"Saved: {outdir / 'summary.json'}")
    print("=== DONE ===")


if __name__ == "__main__":
    run()
