#!/usr/bin/env python3
"""Summarize paired multi-seed MH-ALD key-point planning repeats."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(
    r"^(?P<label>.+)_n(?P<n>\d+)_i(?P<i>\d+)_k(?P<k>\d+)"
    r"_seed(?P<seed>\d+)_ep(?P<ep>\d+)\.txt$"
)
SUCCESS = re.compile(r"'success_rate':\s*([0-9.]+)")
TIME = re.compile(r"evaluation_time:\s*([0-9.eE+-]+)\s*seconds")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def stats(x):
    a = np.asarray(x, dtype=np.float64)
    return {
        "count": int(len(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def main():
    a = parse_args()
    inp = Path(a.input_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(inp.glob("*.txt")):
        m = PAT.match(p.name)
        if not m:
            continue
        txt = p.read_text(errors="replace")
        sm = SUCCESS.search(txt)
        if sm is None:
            raise RuntimeError(f"Could not parse success_rate from {p}")
        tm = TIME.search(txt)
        n = int(m.group("n"))
        it = int(m.group("i"))
        rows.append({
            "label": m.group("label"),
            "num_samples": n,
            "iterations": it,
            "topk": int(m.group("k")),
            "seed": int(m.group("seed")),
            "episodes": int(m.group("ep")),
            "budget_NxI": n * it,
            "success_rate": float(sm.group(1)),
            "evaluation_time_s": float(tm.group(1)) if tm else math.nan,
            "source_file": p.name,
        })

    if not rows:
        raise RuntimeError(f"No repeat result files found under {inp}")

    rows.sort(key=lambda r: (
        r["num_samples"], r["iterations"], r["seed"], r["label"]
    ))
    write_csv(out / "per_run.csv", rows)

    grouped = defaultdict(list)
    for r in rows:
        key = (r["label"], r["num_samples"], r["iterations"], r["topk"])
        grouped[key].append(r)

    agg = []
    for (label, n, it, k), rr in sorted(grouped.items()):
        s = stats([x["success_rate"] for x in rr])
        agg.append({
            "label": label,
            "num_samples": n,
            "iterations": it,
            "topk": k,
            "budget_NxI": n * it,
            "num_seeds": s["count"],
            "success_mean": s["mean"],
            "success_std": s["std"],
            "success_min": s["min"],
            "success_max": s["max"],
            "seeds": ",".join(str(x["seed"]) for x in rr),
        })
    write_csv(out / "aggregate.csv", agg)

    by_key = {
        (r["num_samples"], r["iterations"], r["seed"], r["label"]): r
        for r in rows
    }
    points = sorted({(r["num_samples"], r["iterations"]) for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    paired = []
    for n, it in points:
        for seed in seeds:
            triple = {
                label: by_key.get((n, it, seed, label))
                for label in ("lewm", "ald", "mh_ald")
            }
            if not all(triple.values()):
                continue
            l = triple["lewm"]["success_rate"]
            d = triple["ald"]["success_rate"]
            m = triple["mh_ald"]["success_rate"]
            paired.append({
                "num_samples": n,
                "iterations": it,
                "budget_NxI": n * it,
                "seed": seed,
                "lewm_success": l,
                "ald_success": d,
                "mh_ald_success": m,
                "ald_minus_lewm": d - l,
                "mh_minus_lewm": m - l,
                "mh_minus_ald": m - d,
            })
    write_csv(out / "paired_deltas.csv", paired)

    paired_summary = []
    for n, it in points:
        rr = [x for x in paired if x["num_samples"] == n and x["iterations"] == it]
        if not rr:
            continue
        for key in ("ald_minus_lewm", "mh_minus_lewm", "mh_minus_ald"):
            s = stats([x[key] for x in rr])
            paired_summary.append({
                "num_samples": n,
                "iterations": it,
                "budget_NxI": n * it,
                "comparison": key,
                "num_seeds": s["count"],
                "delta_mean": s["mean"],
                "delta_std": s["std"],
                "delta_min": s["min"],
                "delta_max": s["max"],
            })
    write_csv(out / "paired_delta_summary.csv", paired_summary)

    payload = {
        "per_run": rows,
        "aggregate": agg,
        "paired_deltas": paired,
        "paired_delta_summary": paired_summary,
        "interpretation": {
            "primary": "Use paired seed-level deltas, not only pooled mean success.",
            "positive_mh_minus_ald": "MH-ALD improved success at the same CEM setting and seed.",
        },
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2))

    print("===== MH-ALD KEY-POINT REPEATS =====")
    for r in agg:
        print(
            f"{r['label']:<7} N={r['num_samples']:3d} I={r['iterations']:2d} "
            f"B={r['budget_NxI']:4d} "
            f"S={r['success_mean']:5.1f} +/- {r['success_std']:4.1f}"
        )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
