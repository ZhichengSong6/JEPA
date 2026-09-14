#!/usr/bin/env python3
"""Summarize five diagnostic seeds, including blocked runs; archive without large traces."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cem_mh_diag_core import dump


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def csv_write(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        p.error("Root does not exist")
    seeds, cases, populations, lines = {}, [], [], []
    lines.extend(["CEM-MH diagnostic: NEW paired seeded-reset cohort, not a historical rerun.",
                  "Both models use the installed official CEM (N=100,I=10,K=10).",
                  "Repeat 0 includes passive cost recording; repeat 1 has no cost tap.",
                  "Oracle costs: fixed 25-step endpoint. Ever-success is recorded separately.",
                  "Mechanism cohorts are selected diagnostics, NOT an unbiased success benchmark."])
    for seed in range(42, 47):
        out = root / f"seed{seed}"
        manifest = read(out/"manifest.json", [])
        protocol = read(out/"protocol.json", {})
        gate = read(out/"repeatability.json")
        repeat_status = "not_run" if gate is None else "passed" if gate.get("passed") else "failed"
        gate = gate or {}
        preflight = read(root/"preflight"/f"seed{seed}"/"preflight.json")
        block = read(out/"BLOCKED.json") or read(root/"preflight"/f"seed{seed}"/"BLOCKED.json")
        mechanism = read(out/"mechanism.json", {})
        complete = gate.get("passed") and mechanism.get("status") == "complete" and not block
        seeds[str(seed)] = {"complete": bool(complete), "gate": gate, "blocked": block,
                            "repeatability_status": repeat_status, "preflight": preflight, "selected_cases": len(manifest),
                            "population_count": len(mechanism.get("populations", []))}
        runs = {l: read(out/f"{l}_repeat0.json") for l in ("mh", "cemmh")}
        lines.append(f"\nseed={seed}: complete={bool(complete)}; repeatability={repeat_status}; "
                     f"selected={len(manifest)}; populations={seeds[str(seed)]['population_count']}")
        if block:
            lines.append("  BLOCKED: " + block["error"])
        for l in ("mh", "cemmh"):
            if runs[l]:
                lines.append(f"  {l} diagnostic repeat0: {sum(runs[l]['successes'])}/100")
            if l in gate.get("models", {}):
                lines.append(f"  {l} repeat check: {gate['models'][l]}")
        if all(runs.values()):
            selected = {m["eval_index"]: m["historical_group"] for m in manifest}
            for i in range(100):
                h = protocol["historical_successes"]
                cases.append({"seed": seed, "eval_index": i, "selected": i in selected,
                              "historical_group": selected.get(i, "not_selected"),
                              "historical_mh": int(h["mh"][i]), "historical_cemmh": int(h["cemmh"][i]),
                              "diagnostic_mh": int(runs["mh"]["successes"][i]),
                              "diagnostic_cemmh": int(runs["cemmh"]["successes"][i]),
                              "repeatability_passed": bool(gate.get("passed"))})
        for row in mechanism.get("populations", []):
            flat = {k: row[k] for k in ("key", "source", "eval_index", "solve_no", "iteration", "historical_group", "prefix_steps")}
            flat["seed"] = seed
            for l in ("mh", "cemmh"):
                for k, v in row["metrics"][l].items():
                    flat[l+"_"+k] = v
            for l in ("mh", "cemmh", "encoder"):
                for k in ("ever_success", "endpoint_success", "first_success_step", "official_endpoint_distance"):
                    flat[l+"_elite_mean_"+k] = row["mean_replays"][l][k]
            populations.append(flat)
    complete = all(s["complete"] for s in seeds.values())
    lines.append("\nSTATUS: " + ("complete" if complete else "blocked_or_incomplete; inspect gate/logs before interpretation"))
    dump(root/"diagnostic_summary.json", {"status": "complete" if complete else "blocked_or_incomplete", "seeds": seeds})
    (root/"summary.txt").write_text("\n".join(lines)+"\n")
    csv_write(root/"case_outcomes.csv", cases)
    csv_write(root/"cross_population_metrics.csv", populations)
    print("\n".join(lines))
    archive = root/"results.tar.gz"
    # Keep locally-generated .pt traces on server; numerical population .npz files
    # and all JSON/reset contexts/results/logs are sufficient for initial review.
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if (not path.is_file() or path.is_symlink() or path == archive
                    or path.name.endswith((".pt", ".pyc", ".tmp", ".mp4"))
                    or path.name.startswith("slurm-")):
                continue
            tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
    print("Archive:", archive)
    if not complete:
        sys.exit(2)


if __name__ == "__main__":
    main()
