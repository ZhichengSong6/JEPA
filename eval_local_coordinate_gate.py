#!/usr/bin/env python3
"""Planner-side gate for the nonlinear Local Coordinate model.

Scientific question
-------------------
Does the learned nonlinear, exactly invertible coordinate map

    y = Phi(z)

make the *raw Euclidean* JEPA cost more planning-compatible on the exact hard
populations where the original LeWM geometry failed?

This evaluator does NOT rerun CEM.  It reads the frozen populations from the
authoritative official diagnostic run, physically replays the same candidates
only to recover endpoint images, and scores every candidate with:

    1) original LeWM raw encoder/predictor cost;
    2) Local-Coordinate raw encoder/predictor cost.

The exported Local-Coordinate model conjugates the frozen predictor exactly,

    P_y(y,a) = Phi(P_z(Phi^{-1}(y), a)),

so a strict inverse-coordinate audit verifies that the underlying dynamics are
unchanged.  Any score/ranking change is therefore attributable to the learned
coordinate metric, up to numerical tolerance.

Primary cases:
    7, 27, 53, 77

Matched-success controls are taken from the official paired_manifest.csv.

No physical signal is used to generate or alter candidates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig

from eval import get_dataset, img_transform
from eval_lowbudget_failure_autopsy import _build_process, _spearman
from eval_b3000_paired_failure_analysis import (
    _extract_variations,
    _normalized_to_raw,
    _slice_info,
)
from eval_b3000_critical_jepa_mechanism import (
    _find_trace_for_env,
    _goal_embedding_from_solver,
    _predict_processed_context,
)
from eval_pusht_action_ranking import _encode_images
from eval_pusht_official_diagnostic import (
    load_model,
    load_recording,
    replay,
    score_metrics,
    write_csv,
    write_json,
)


POP_RE = re.compile(
    r"^eval(?P<case>\d+)_(?P<source>lewm|ald_tf)_solve(?P<solve>\d+)_iter(?P<it>\d+)\.npz$"
)


def _bool_csv(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def _read_manifest(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _base_weights_digest(model) -> str:
    """Hash all pretrained weights while excluding the coordinate adapter."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("coordinate_adapter."):
            continue
        h.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        h.update(raw.numpy().tobytes())
    return h.hexdigest()


def _safe_mean(values):
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def _score_row(meta, model_name, score_name, score, components, topk):
    score = np.asarray(score, dtype=np.float64)
    return {
        **meta,
        "model": model_name,
        "score": score_name,
        **score_metrics(score, components, int(topk)),
    }


def _pair_rows(metric_rows):
    """Pair Coordinate and LeWM rows for the exact same population/score."""
    keys = ("eval_index", "case_type", "is_control", "source", "solve",
            "cem_iteration", "score")
    by = defaultdict(dict)
    for row in metric_rows:
        k = tuple(row[x] for x in keys)
        by[k][row["model"]] = row

    out = []
    wanted = [
        "physical_rho",
        "object_rho",
        "theta_rho",
        "block_rho",
        "pusher_rho",
        "physical_selected_target_percentile",
        "object_selected_target_percentile",
        "physical_top10_recall",
        "object_top10_recall",
        "theta_matched_accuracy",
        "block_matched_accuracy",
        "pusher_matched_accuracy",
    ]
    for k, pair in by.items():
        if set(pair) != {"lewm", "coordinate"}:
            continue
        base = pair["lewm"]
        coord = pair["coordinate"]
        row = dict(zip(keys, k))
        for name in wanted:
            b = float(base.get(name, float("nan")))
            c = float(coord.get(name, float("nan")))
            row[f"lewm_{name}"] = b
            row[f"coordinate_{name}"] = c
            row[f"delta_{name}"] = c - b
        out.append(row)
    return out


def _aggregate(comparison_rows):
    groups = defaultdict(list)
    for row in comparison_rows:
        groups[("all", "all", row["score"])].append(row)
        groups[("case", str(row["eval_index"]), row["score"])].append(row)
        groups[
            ("cohort", "control" if row["is_control"] else "hard", row["score"])
        ].append(row)

    summary = []
    for (kind, name, score), rows in sorted(groups.items()):
        def vals(key):
            return [r.get(key, float("nan")) for r in rows]

        d_phys = np.asarray(vals("delta_physical_rho"), dtype=np.float64)
        d_obj = np.asarray(vals("delta_object_rho"), dtype=np.float64)
        d_sel = np.asarray(
            vals("delta_physical_selected_target_percentile"), dtype=np.float64
        )
        d_top = np.asarray(vals("delta_physical_top10_recall"), dtype=np.float64)

        finite_phys = np.isfinite(d_phys)
        finite_obj = np.isfinite(d_obj)
        finite_sel = np.isfinite(d_sel)
        finite_top = np.isfinite(d_top)

        summary.append({
            "group_kind": kind,
            "group": name,
            "score": score,
            "population_count": len(rows),
            "mean_delta_physical_rho": _safe_mean(d_phys),
            "mean_delta_object_rho": _safe_mean(d_obj),
            "mean_delta_theta_rho": _safe_mean(vals("delta_theta_rho")),
            "mean_delta_block_rho": _safe_mean(vals("delta_block_rho")),
            "mean_delta_selected_physical_percentile": _safe_mean(d_sel),
            "mean_delta_physical_top10_recall": _safe_mean(d_top),
            "physical_rho_better_fraction": (
                float(np.mean(d_phys[finite_phys] > 0)) if finite_phys.any()
                else None
            ),
            "object_rho_better_fraction": (
                float(np.mean(d_obj[finite_obj] > 0)) if finite_obj.any()
                else None
            ),
            "selected_physical_better_fraction": (
                float(np.mean(d_sel[finite_sel] < 0)) if finite_sel.any()
                else None
            ),
            "physical_top10_better_fraction": (
                float(np.mean(d_top[finite_top] > 0)) if finite_top.any()
                else None
            ),
        })
    return summary


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    dc = cfg.get("diagnostic", {})
    run_dir = Path(str(dc.get("run_dir", ""))).expanduser().resolve()
    if not (run_dir / "run_identity.json").is_file():
        raise ValueError(
            "Set +diagnostic.run_dir to the complete official formal diagnostic."
        )
    if not (run_dir / "summary.json").is_file():
        raise ValueError("Official diagnostic run is incomplete.")

    out = Path(
        str(dc.get("output_dir", "outputs/coordinate_stage1_gate"))
    ).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    coordinate_policy = str(
        dc.get(
            "coordinate_policy",
            "pusht_local_coordinate_seed3072/lewm_local_coordinate",
        )
    )
    primary_cases = [
        int(x) for x in dc.get("cases", [7, 27, 53, 77])
    ]
    sources = [str(x) for x in dc.get("sources", ["lewm", "ald_tf"])]
    solves = [int(x) for x in dc.get("solves", [0, 1])]
    iterations = [int(x) for x in dc.get("iterations", [0, 3, 9, 19, 29])]
    n_controls = int(dc.get("num_matched_controls", 2))
    batch = int(dc.get("model_batch_size", 64))
    replay_tol = float(dc.get("replay_state_tolerance", 2e-4))
    coord_audit_tol = float(dc.get("coordinate_audit_tolerance", 3e-4))

    identity = json.loads((run_dir / "run_identity.json").read_text())
    protocol = identity["protocol"]
    topk = int(protocol["topk"])
    action_block = int(protocol["action_block"])
    device = torch.device(str(cfg.solver.device))

    manifest = _read_manifest(run_dir / "paired_manifest.csv")
    by_case = {int(r["eval_index"]): r for r in manifest}
    missing = [x for x in primary_cases if x not in by_case]
    if missing:
        raise ValueError(f"Cases absent from official manifest: {missing}")

    controls = [
        int(r["eval_index"])
        for r in manifest
        if _bool_csv(r.get("is_matched_control", "false"))
        and r.get("case_type") == "both_success"
    ][:max(n_controls, 0)]
    selected_cases = list(dict.fromkeys(primary_cases + controls))

    base = load_model("lewm_epoch_10", device)
    coordinate = load_model(coordinate_policy, device)
    adapter = getattr(coordinate, "coordinate_adapter", None)
    if adapter is None:
        raise RuntimeError("Local-coordinate policy has no coordinate_adapter.")

    base_digest = _base_weights_digest(base)
    coord_digest = _base_weights_digest(coordinate)
    if base_digest != coord_digest:
        raise RuntimeError(
            "Local-coordinate training changed pretrained model weights. "
            f"base={base_digest} coordinate={coord_digest}"
        )

    runs = {
        src: load_recording(run_dir / "recordings" / f"{src}.pt", device)
        for src in sources
    }

    dataset = get_dataset(cfg, str(protocol["dataset_name"]))
    process = _build_process(cfg, dataset)
    transform = img_transform(cfg)

    population_dir = run_dir / "populations"
    available = {}
    for path in population_dir.glob("*.npz"):
        m = POP_RE.match(path.name)
        if not m:
            continue
        key = (
            int(m.group("case")),
            m.group("source"),
            int(m.group("solve")),
            int(m.group("it")),
        )
        available[key] = path

    requested = [
        (case, source, solve, it)
        for case in selected_cases
        for source in sources
        for solve in solves
        for it in iterations
    ]
    missing_pops = [k for k in requested if k not in available]
    if missing_pops:
        raise FileNotFoundError(
            "The authoritative run does not contain all requested populations. "
            f"First missing: {missing_pops[:10]}"
        )

    metric_rows = []
    audit_rows = []
    score_arrays = {}
    env = gym.make(str(protocol["env_name"]), render_mode="rgb_array")
    t0 = time.time()

    try:
        for idx, (case, source, solve, it) in enumerate(requested, 1):
            pop_path = available[(case, source, solve, it)]
            with np.load(pop_path, allow_pickle=False) as npz:
                candidates = np.asarray(npz["candidates_normalized"])
                saved_final = np.asarray(npz["final_state"], dtype=np.float64)
                goal = np.asarray(npz["goal_state"], dtype=np.float64)
                start = np.asarray(npz["start_state"], dtype=np.float64)
                components = {
                    name: np.asarray(npz[name], dtype=np.float64)
                    for name in (
                        "pusher_cost",
                        "block_cost",
                        "theta_cost",
                        "joint_xy_cost",
                        "object_task_cost",
                        "official_cost",
                    )
                }

            source_run = runs[source]
            tr, li = _find_trace_for_env(source_run["solver"].trace, case, solve)
            if li is None:
                raise RuntimeError(
                    f"Recording missing case={case} source={source} solve={solve}"
                )
            info = _slice_info(tr["solver_info"], li)
            variations = _extract_variations(info)
            px = torch.as_tensor(info["pixels"])[0, -1:]

            raw = _normalized_to_raw(
                candidates, process["action"], action_block
            )
            replay_final, images, _, _ = replay(
                env, start, goal, raw, variations, 42
            )
            replay_err = float(np.max(np.abs(replay_final - saved_final)))
            if replay_err > replay_tol:
                raise RuntimeError(
                    f"Replay audit failed for {pop_path.name}: "
                    f"max_abs_state_error={replay_err:.3e}"
                )

            models = {"lewm": base, "coordinate": coordinate}
            encoded = {}
            predicted = {}
            goals = {}
            for name, model in models.items():
                goals[name] = _goal_embedding_from_solver(
                    model, info, device
                ).float()
                encoded[name] = _encode_images(
                    model, transform, images, device, batch
                ).float()
                predicted[name] = _predict_processed_context(
                    model,
                    px,
                    np.empty((0, 2), dtype=np.float32),
                    candidates,
                    process["action"],
                    action_block,
                    device,
                    batch,
                ).float()

            with torch.no_grad():
                inv_enc = adapter.inverse(encoded["coordinate"])
                inv_pred = adapter.inverse(predicted["coordinate"])
                inv_goal = adapter.inverse(goals["coordinate"])

            enc_audit = float(
                torch.max(torch.abs(inv_enc - encoded["lewm"])).item()
            )
            pred_audit = float(
                torch.max(torch.abs(inv_pred - predicted["lewm"])).item()
            )
            goal_audit = float(
                torch.max(torch.abs(inv_goal - goals["lewm"])).item()
            )
            max_coord_audit = max(enc_audit, pred_audit, goal_audit)
            if max_coord_audit > coord_audit_tol:
                raise RuntimeError(
                    "Coordinate conjugacy audit failed: "
                    f"enc={enc_audit:.3e} pred={pred_audit:.3e} "
                    f"goal={goal_audit:.3e}"
                )

            case_type = by_case[case]["case_type"]
            is_control = case in controls
            meta = {
                "eval_index": case,
                "case_type": case_type,
                "is_control": is_control,
                "source": source,
                "solve": solve,
                "cem_iteration": it,
                "num_samples": len(candidates),
            }

            audit_rows.append({
                **meta,
                "population_file": pop_path.name,
                "replay_state_max_abs": replay_err,
                "inverse_encoder_max_abs": enc_audit,
                "inverse_predictor_max_abs": pred_audit,
                "inverse_goal_max_abs": goal_audit,
            })

            for name in ("lewm", "coordinate"):
                zg = goals[name]
                enc_cost = (
                    (encoded[name] - zg[None]).square().sum(-1)
                    .detach().cpu().numpy()
                )
                pred_cost = (
                    (predicted[name] - zg[None]).square().sum(-1)
                    .detach().cpu().numpy()
                )

                metric_rows.append(
                    _score_row(
                        meta, name, "encoder_raw",
                        enc_cost, components, topk
                    )
                )
                metric_rows.append(
                    _score_row(
                        meta, name, "predictor_raw",
                        pred_cost, components, topk
                    )
                )

                prefix = (
                    f"eval{case:03d}_{source}_solve{solve}_iter{it:02d}_{name}"
                )
                score_arrays[f"{prefix}_encoder_cost"] = enc_cost.astype(
                    np.float32
                )
                score_arrays[f"{prefix}_predictor_cost"] = pred_cost.astype(
                    np.float32
                )

            print(
                f"[{idx:03d}/{len(requested):03d}] "
                f"case={case} {case_type} source={source} "
                f"solve={solve} iter={it} "
                f"audit={max_coord_audit:.2e}",
                flush=True,
            )
    finally:
        env.close()

    comparison_rows = _pair_rows(metric_rows)
    aggregate_rows = _aggregate(comparison_rows)

    write_csv(out / "metric_rows.csv", metric_rows)
    write_csv(out / "paired_comparison.csv", comparison_rows)
    write_csv(out / "aggregate_summary.csv", aggregate_rows)
    write_csv(out / "audits.csv", audit_rows)
    np.savez_compressed(out / "candidate_scores.npz", **score_arrays)

    # Human-facing compact summary for the primary hard cases.
    primary_summary = [
        r for r in aggregate_rows
        if r["group_kind"] == "case"
        and int(r["group"]) in primary_cases
    ]
    control_summary = [
        r for r in aggregate_rows
        if r["group_kind"] == "cohort" and r["group"] == "control"
    ]
    hard_summary = [
        r for r in aggregate_rows
        if r["group_kind"] == "cohort" and r["group"] == "hard"
    ]

    summary = {
        "status": "complete",
        "scientific_question": (
            "Can a nonlinear exactly invertible latent coordinate map improve "
            "raw Euclidean CEM ordering on exact hard official populations "
            "without changing the pretrained LeWM dynamics?"
        ),
        "official_run": str(run_dir),
        "coordinate_policy": coordinate_policy,
        "base_policy": "lewm_epoch_10",
        "base_weights_sha256": base_digest,
        "coordinate_base_weights_sha256": coord_digest,
        "primary_cases": primary_cases,
        "matched_success_controls": controls,
        "sources": sources,
        "solves": solves,
        "iterations": iterations,
        "num_populations": len(requested),
        "elapsed_seconds": float(time.time() - t0),
        "max_replay_state_error": max(
            r["replay_state_max_abs"] for r in audit_rows
        ),
        "max_inverse_coordinate_audit_error": max(
            max(
                r["inverse_encoder_max_abs"],
                r["inverse_predictor_max_abs"],
                r["inverse_goal_max_abs"],
            )
            for r in audit_rows
        ),
        "primary_case_summary": primary_summary,
        "hard_cohort_summary": hard_summary,
        "control_cohort_summary": control_summary,
        "metric_direction": {
            "rho": "higher is better",
            "selected_target_percentile": "lower is better",
            "top10_recall": "higher is better",
            "delta": "coordinate minus LeWM",
        },
        "interpretation_rule": (
            "Local Coordinate is supported only if hard-case raw ordering "
            "and candidate selection improve meaningfully while matched-success "
            "controls are not materially degraded. Training alignment alone is "
            "not sufficient."
        ),
    }
    write_json(out / "summary.json", summary)

    print("\n===== LOCAL COORDINATE PLANNER-SIDE GATE =====")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {out}")


if __name__ == "__main__":
    run()
