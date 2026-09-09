#!/usr/bin/env python3
"""Summarize paired MH vs teacher-replacement Exp1 CEM results."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

PATTERN = re.compile(
    r"^(?P<label>mh|exp1)_seed(?P<seed>\d+)_b(?P<budget>500|1000)\.txt$"
)
SUCCESS_RE = re.compile(r"'success_rate':\s*([0-9.]+)")
TIME_RE = re.compile(r"evaluation_time:\s*([0-9.eE+-]+)\s*seconds")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(indir.glob("*.txt")):
        m = PATTERN.match(path.name)
        if not m:
            continue
        txt = path.read_text(errors="replace")
        sm = SUCCESS_RE.search(txt)
        tm = TIME_RE.search(txt)
        if sm is None:
            raise RuntimeError(f"Could not parse success_rate from {path}")
        rows.append({
            "label": m.group("label"),
            "seed": int(m.group("seed")),
            "budget": int(m.group("budget")),
            "success_rate": float(sm.group(1)),
            "evaluation_time_s": float(tm.group(1)) if tm else math.nan,
            "source_file": path.name,
        })

    expected = {(label, seed, budget)
                for label in ("mh", "exp1")
                for seed in range(42, 47)
                for budget in (500, 1000)}
    got = {(r["label"], r["seed"], r["budget"]) for r in rows}
    missing = sorted(expected - got)
    if missing:
        raise RuntimeError(f"Missing paired results: {missing}")

    rows.sort(key=lambda r: (r["budget"], r["seed"], r["label"]))
    csv_path = outdir / "paired_seed_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    aggregate = {}
    paired = {}
    for budget in (500, 1000):
        aggregate[str(budget)] = {}
        for label in ("mh", "exp1"):
            vals = [
                r["success_rate"] for r in rows
                if r["budget"] == budget and r["label"] == label
            ]
            aggregate[str(budget)][label] = {
                "mean": mean(vals),
                "std": stdev(vals),
                "min": min(vals),
                "max": max(vals),
                "values": vals,
            }

        deltas = []
        per_seed = []
        for seed in range(42, 47):
            mh = next(r["success_rate"] for r in rows
                      if r["budget"] == budget and r["seed"] == seed
                      and r["label"] == "mh")
            ex = next(r["success_rate"] for r in rows
                      if r["budget"] == budget and r["seed"] == seed
                      and r["label"] == "exp1")
            d = ex - mh
            deltas.append(d)
            per_seed.append({
                "seed": seed,
                "mh": mh,
                "exp1": ex,
                "delta_exp1_minus_mh": d,
            })
        paired[str(budget)] = {
            "mean_delta": mean(deltas),
            "std_delta": stdev(deltas),
            "wins": sum(d > 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas),
            "losses": sum(d < 0 for d in deltas),
            "per_seed": per_seed,
        }

    payload = {
        "aggregate": aggregate,
        "paired": paired,
        "definition": {
            "mh": "original formal MH-ALD",
            "exp1": "LeWM-init MH-ALD trained with frozen original MH-ALD teacher",
            "budgets": {
                "500": {"num_samples": 100, "iterations": 5, "topk": 10},
                "1000": {"num_samples": 100, "iterations": 10, "topk": 10},
            },
            "seeds": [42, 43, 44, 45, 46],
        },
    }
    json_path = outdir / "paired_summary.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print("===== EXP1 TEACHER REPLACEMENT: PAIRED CEM =====")
    for budget in (500, 1000):
        a = aggregate[str(budget)]
        p = paired[str(budget)]
        print(
            f"B={budget}: MH={a['mh']['mean']:.2f}±{a['mh']['std']:.2f}, "
            f"Exp1={a['exp1']['mean']:.2f}±{a['exp1']['std']:.2f}, "
            f"delta={p['mean_delta']:+.2f} pp, "
            f"W/T/L={p['wins']}/{p['ties']}/{p['losses']}"
        )
        for x in p["per_seed"]:
            print(
                f"  seed{x['seed']}: {x['mh']:.1f} -> {x['exp1']:.1f} "
                f"({x['delta_exp1_minus_mh']:+.1f})"
            )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
