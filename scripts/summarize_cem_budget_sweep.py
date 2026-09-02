#!/usr/bin/env python3
"""Summarize a 2-D CEM (N,I) budget sweep into planning-efficiency metrics."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

PATTERN = re.compile(
    r"^(?P<label>.+)_n(?P<n>\d+)_i(?P<i>\d+)_k(?P<k>\d+)_ep(?P<ep>\d+)\.txt$"
)
SUCCESS_RE = re.compile(r"'success_rate':\s*([0-9.]+)")
TIME_RE = re.compile(r"evaluation_time:\s*([0-9.eE+-]+)\s*seconds")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reference-label", default="lewm")
    p.add_argument("--reference-n", type=int, default=300)
    p.add_argument("--reference-i", type=int, default=10)
    p.add_argument("--thresholds", nargs="+", type=float, default=[80, 90, 94, 95, 100])
    return p.parse_args()


def main():
    args = parse_args()
    indir, outdir = Path(args.input_dir), Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in sorted(indir.glob("*.txt")):
        m = PATTERN.match(path.name)
        if not m:
            continue
        txt = path.read_text(errors="replace")
        sm, tm = SUCCESS_RE.search(txt), TIME_RE.search(txt)
        if sm is None:
            raise RuntimeError(f"Could not parse success_rate from {path}")
        n, it, k, ep = map(int, (m.group("n"), m.group("i"), m.group("k"), m.group("ep")))
        rows.append({
            "label": m.group("label"), "num_samples": n, "iterations": it,
            "topk": k, "elite_fraction": k / n, "episodes": ep,
            "budget_NxI": n * it, "success_rate": float(sm.group(1)),
            "evaluation_time_s": float(tm.group(1)) if tm else math.nan,
            "source_file": path.name,
        })

    ref = [
        r for r in rows if r["label"] == args.reference_label
        and r["num_samples"] == args.reference_n
        and r["iterations"] == args.reference_i
    ]
    if len(ref) != 1:
        raise RuntimeError(f"Expected one reference row, got {len(ref)}")
    ref = ref[0]
    ref_budget, ref_success = int(ref["budget_NxI"]), float(ref["success_rate"])
    ten_x_budget = ref_budget / 10.0

    for r in rows:
        r["reduction_vs_reference_x"] = ref_budget / r["budget_NxI"]
        r["within_10x_budget"] = bool(r["budget_NxI"] <= ten_x_budget)
        r["meets_reference_success"] = bool(r["success_rate"] >= ref_success)
        r["ten_x_and_meets_reference"] = bool(
            r["within_10x_budget"] and r["meets_reference_success"]
        )
    rows.sort(key=lambda r: (r["label"], r["budget_NxI"], r["num_samples"], r["iterations"]))

    csv_path = outdir / "budget_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    labels = sorted({r["label"] for r in rows})
    thresholds, pareto = {}, {}
    for label in labels:
        lr = [r for r in rows if r["label"] == label]
        thresholds[label] = {}
        for t in args.thresholds:
            hit = [r for r in lr if r["success_rate"] >= t]
            thresholds[label][str(t)] = (
                min(hit, key=lambda r: (r["budget_NxI"], -r["success_rate"]))
                if hit else None
            )
        frontier, best_success = [], -1.0
        for b in sorted({r["budget_NxI"] for r in lr}):
            cand = [r for r in lr if r["budget_NxI"] <= b]
            best = max(cand, key=lambda r: (r["success_rate"], -r["budget_NxI"]))
            if best["success_rate"] > best_success:
                frontier.append({
                    "budget_NxI": b, "success_rate": best["success_rate"],
                    "num_samples": best["num_samples"],
                    "iterations": best["iterations"], "topk": best["topk"],
                })
                best_success = best["success_rate"]
        pareto[label] = frontier

    payload = {
        "reference": {
            "label": args.reference_label, "num_samples": args.reference_n,
            "iterations": args.reference_i, "budget_NxI": ref_budget,
            "success_rate": ref_success, "ten_x_budget_ceiling": ten_x_budget,
        },
        "interpretation": {
            "primary_goal": (
                "Reach/exceed LeWM high-budget reference success with >=10x fewer "
                "candidate evaluations (N*I)."
            ),
            "secondary_goal": "Preserve/improve high-budget success, ideally 100%.",
            "elite_rule": "topk fixed to 10% of N.",
        },
        "threshold_best": thresholds, "pareto_frontier": pareto, "rows": rows,
    }
    json_path = outdir / "budget_summary.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print("===== CEM PLANNING EFFICIENCY =====")
    print(
        f"Reference: {args.reference_label} N={args.reference_n} I={args.reference_i} "
        f"B={ref_budget} success={ref_success:.1f}%"
    )
    print(f"10x budget ceiling: B <= {ten_x_budget:.0f}\n")
    for label in labels:
        print(f"[{label}]")
        for r in [x for x in rows if x["label"] == label]:
            flag = " <== 10x+ & >=ref" if r["ten_x_and_meets_reference"] else ""
            print(
                f" N={r['num_samples']:3d} I={r['iterations']:2d} K={r['topk']:2d} "
                f"B={r['budget_NxI']:4d} S={r['success_rate']:5.1f}% "
                f"reduction={r['reduction_vs_reference_x']:5.1f}x{flag}"
            )
        print()
    print(f"Saved: {csv_path}\nSaved: {json_path}")


if __name__ == "__main__":
    main()
