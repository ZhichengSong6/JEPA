#!/usr/bin/env python3
"""Summarize all 20 official-CEM runs; refuse incomplete/unpaired results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

LABELS = ("mh", "cemmh")
SEEDS = tuple(range(42, 47))
BUDGETS = (500, 1000)
EPISODES = 100


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_overrides(record, path, seed, budget):
    """Do not trust the filename alone to identify the evaluation protocol."""
    overrides = record.get("hydra_overrides")
    if not isinstance(overrides, list):
        raise ValueError(f"Missing Hydra overrides: {path}")
    values = {}
    for item in overrides:
        if not isinstance(item, str):
            raise ValueError(f"Invalid Hydra override: {path}")
        if "=" in item:
            key, value = item.split("=", 1)
            values[key.lstrip("+")] = value
    expected = {
        "solver": "cem", "seed": str(seed),
        "solver.num_samples": "100", "solver.n_steps": str(budget // 100),
        "solver.topk": "10", "solver.batch_size": "1",
        "solver.var_scale": "1.0", "world.history_size": "1",
        "world.frame_skip": "1", "plan_config.horizon": "5",
        "plan_config.receding_horizon": "5", "plan_config.action_block": "5",
        "eval.num_eval": str(EPISODES), "eval.eval_budget": "50",
        "eval.goal_offset_steps": "25", "eval.img_size": "224",
        "eval.dataset_name": "pusht_expert_train",
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(
                f"Protocol mismatch in {path}: {key}={values.get(key)!r}; "
                f"expected {expected_value!r}"
            )
    policy = values.get("policy", "")
    if not policy or policy == "random":
        raise ValueError(f"Missing model policy: {path}")
    return policy


def load_result(path, seed, budget):
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation: {path}")
    record = json.loads(path.read_text())
    if record.get("status") != "complete":
        raise ValueError(f"Incomplete evaluation: {path}")
    record["policy"] = validate_overrides(record, path, seed, budget)
    metrics = record["metrics"]
    successes = metrics.get("episode_successes")
    if not isinstance(successes, list) or len(successes) != EPISODES:
        raise ValueError(f"Need {EPISODES} episode_successes in {path}")
    if any(not isinstance(x, (bool, int, float)) or x not in (0, 1)
           for x in successes):
        raise ValueError(f"Non-binary episode_successes in {path}")
    successes = [bool(x) for x in successes]
    # Derive percentages from per-episode outcomes, not a guessed rate unit.
    rate_pct = 100.0 * sum(successes) / EPISODES
    raw = float(metrics["success_rate"])
    if not (math.isclose(raw, rate_pct, abs_tol=1e-4)
            or math.isclose(raw * 100.0, rate_pct, abs_tol=1e-4)):
        raise ValueError(f"success_rate disagrees with episode outcomes: {path}")
    selection = record["selection"]
    for key in ("episodes_idx", "start_steps"):
        if not isinstance(selection.get(key), list) or len(selection[key]) != EPISODES:
            raise ValueError(f"Invalid {key} in {path}")
    if selection.get("goal_offset_steps") != 25 or selection.get("eval_budget") != 50:
        raise ValueError(f"Unexpected evaluation protocol in {path}")
    elapsed = record.get("world_evaluation_seconds")
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError(f"Invalid evaluation time in {path}")
    record["successes"] = successes
    record["success_rate_pct"] = rate_pct
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    indir, outdir = Path(args.input_dir), Path(args.output_dir)
    records, rows = {}, []
    for budget in BUDGETS:
        for seed in SEEDS:
            for label in LABELS:
                path = indir / f"{label}_seed{seed}_b{budget}.json"
                r = load_result(path, seed, budget)
                records[label, seed, budget] = r
                rows.append({
                    "label": label, "seed": seed, "budget": budget,
                    "policy": r["policy"],
                    "episodes": EPISODES, "successes": sum(r["successes"]),
                    "success_rate_pct": r["success_rate_pct"],
                    "world_evaluation_seconds": r["world_evaluation_seconds"],
                    "source_file": path.name,
                })
    policies = {}
    for label in LABELS:
        seen = {r["policy"] for (name, _, _), r in records.items() if name == label}
        if len(seen) != 1:
            raise ValueError(f"Mixed model policies for label {label}: {sorted(seen)}")
        policies[label] = seen.pop()
    if policies["mh"] == policies["cemmh"]:
        raise ValueError("MH and CEM-MH must be different policy paths")
    # Pair requested dataset episode/start IDs, including across budgets.
    # This is NOT a validation of unrecorded live simulator variations.
    for seed in SEEDS:
        reference = records["mh", seed, 500]["selection"]
        for label in LABELS:
            for budget in BUDGETS:
                if records[label, seed, budget]["selection"] != reference:
                    raise ValueError(f"Dataset start/goal mismatch: {label}, {seed}, B={budget}")

    aggregate, paired, episode_rows, lines = {}, {}, [], []
    lines.append("MH vs CEM-MH | official CEM | 5 evaluation seeds x 100 cases")
    lines.append("Rates: percent; deltas: percentage points; std: sample std over evaluation seeds.")
    lines.append("Pairing checks dataset starts only; live simulator variations are not captured.")
    lines.append("Recorded wall time includes environment/video work; it is NOT planning latency.")
    for budget in BUDGETS:
        aggregate[str(budget)] = {}
        for label in LABELS:
            vals = [records[label, s, budget]["success_rate_pct"] for s in SEEDS]
            aggregate[str(budget)][label] = {
                "mean_pct": mean(vals), "std_pct": stdev(vals), "values_pct": vals,
            }
        seed_rows = []
        for seed in SEEDS:
            a, b = records["mh", seed, budget], records["cemmh", seed, budget]
            rescued = sum(not x and y for x, y in zip(a["successes"], b["successes"]))
            regressed = sum(x and not y for x, y in zip(a["successes"], b["successes"]))
            seed_rows.append({
                "seed": seed, "mh_pct": a["success_rate_pct"],
                "cemmh_pct": b["success_rate_pct"],
                "delta_pp": b["success_rate_pct"] - a["success_rate_pct"],
                "rescued_cases": rescued, "regressed_cases": regressed,
            })
            for i, (x, y) in enumerate(zip(a["successes"], b["successes"])):
                episode_rows.append({
                    "budget": budget, "seed": seed, "eval_index": i,
                    "episode_idx": a["selection"]["episodes_idx"][i],
                    "start_step": a["selection"]["start_steps"][i],
                    "mh_success": int(x), "cemmh_success": int(y),
                    "delta": int(y) - int(x),
                })
        deltas = [r["delta_pp"] for r in seed_rows]
        p = paired[str(budget)] = {
            "mean_delta_pp": mean(deltas), "std_delta_pp": stdev(deltas),
            "seed_wins": sum(d > 0 for d in deltas),
            "seed_ties": sum(d == 0 for d in deltas),
            "seed_losses": sum(d < 0 for d in deltas),
            "rescued_case_occurrences": sum(r["rescued_cases"] for r in seed_rows),
            "regressed_case_occurrences": sum(r["regressed_cases"] for r in seed_rows),
            "per_seed": seed_rows,
        }
        a, b = aggregate[str(budget)]["mh"], aggregate[str(budget)]["cemmh"]
        lines.append(
            f"\nB={budget}: MH={a['mean_pct']:.2f} +/- {a['std_pct']:.2f}; "
            f"CEM-MH={b['mean_pct']:.2f} +/- {b['std_pct']:.2f}; "
            f"delta={p['mean_delta_pp']:+.2f} pp; "
            f"seed W/T/L={p['seed_wins']}/{p['seed_ties']}/{p['seed_losses']}"
        )
        for r in seed_rows:
            lines.append(
                f"  seed {r['seed']}: {r['mh_pct']:.1f} -> {r['cemmh_pct']:.1f} "
                f"({r['delta_pp']:+.1f} pp), rescued={r['rescued_cases']}, "
                f"regressed={r['regressed_cases']}"
            )
    payload = {
        "definition": {
            "mh": "Original formal MH-ALD epoch 10, rerun in this evaluation",
            "cemmh": "CEM-aligned MH-ALD epoch 10",
            "policies": policies,
            "seeds": list(SEEDS), "episodes_per_run": EPISODES,
            "budgets": {str(b): {"num_samples": 100, "iterations": b // 100, "topk": 10}
                        for b in BUDGETS},
            "pairing": "Requested dataset episode/start IDs verified; live reset variations not captured",
            "std": "Sample standard deviation over evaluation seeds (ddof=1), NOT training seeds",
            "time": "world.evaluate_from_dataset wall time, NOT planning latency",
            "case_counts": "Evaluation-case occurrences; underlying episodes may recur across seeds",
        },
        "aggregate": aggregate, "paired": paired,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "paired_seed_results.csv", rows)
    write_csv(outdir / "paired_episode_results.csv", episode_rows)
    (outdir / "paired_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {outdir}")


if __name__ == "__main__":
    main()
