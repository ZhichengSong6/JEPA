#!/usr/bin/env python3
"""Compare ALD vs MH-ALD on identical planner-visited CEM populations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ald-cem", required=True)
    p.add_argument("--mh-cem", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def summarize_source(source, rows):
    out = []
    iterations = sorted({int(r["cem_iteration"]) for r in rows})
    higher = [
        "rho_pred_phys",
        "rho_pred_enc",
        "pred_elite_overlap_phys",
        "pred_elite_update_cos_phys",
    ]
    lower = [
        "pred_selected_phys_percentile",
        "pred_selection_regret",
        "pred_oracle_best_rank_percentile",
    ]

    for it in iterations:
        rr = [r for r in rows if int(r["cem_iteration"]) == it]
        by_key = {}
        for r in rr:
            key = (r["trace_file"], int(r["solve_index"]))
            by_key.setdefault(key, {})[r["model"]] = r

        pairs = [v for v in by_key.values() if "ald" in v and "mh_ald" in v]
        row = {
            "trace_source": source,
            "cem_iteration": it,
            "paired_populations": len(pairs),
        }

        for metric in higher:
            av = np.asarray([f(p["ald"], metric) for p in pairs], dtype=float)
            mv = np.asarray([f(p["mh_ald"], metric) for p in pairs], dtype=float)
            mask = np.isfinite(av) & np.isfinite(mv)
            d = mv[mask] - av[mask]
            row["ald_" + metric] = float(np.mean(av[mask])) if mask.any() else float("nan")
            row["mh_" + metric] = float(np.mean(mv[mask])) if mask.any() else float("nan")
            row["mh_improvement_" + metric] = float(np.mean(d)) if len(d) else float("nan")
            row["mh_win_fraction_" + metric] = float(np.mean(d > 0)) if len(d) else float("nan")

        for metric in lower:
            av = np.asarray([f(p["ald"], metric) for p in pairs], dtype=float)
            mv = np.asarray([f(p["mh_ald"], metric) for p in pairs], dtype=float)
            mask = np.isfinite(av) & np.isfinite(mv)
            d = av[mask] - mv[mask]
            row["ald_" + metric] = float(np.mean(av[mask])) if mask.any() else float("nan")
            row["mh_" + metric] = float(np.mean(mv[mask])) if mask.any() else float("nan")
            row["mh_improvement_" + metric] = float(np.mean(d)) if len(d) else float("nan")
            row["mh_win_fraction_" + metric] = float(np.mean(d > 0)) if len(d) else float("nan")

        out.append(row)
    return out


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ald_rows = read_csv(Path(a.ald_cem) / "population_metrics.csv")
    mh_rows = read_csv(Path(a.mh_cem) / "population_metrics.csv")

    rows = summarize_source("ald_cem", ald_rows)
    rows += summarize_source("mh_ald_cem", mh_rows)
    rows.sort(key=lambda r: (r["trace_source"], r["cem_iteration"]))
    write_csv(out / "distribution_comparison.csv", rows)

    payload = {
        "rows": rows,
        "sign_convention": (
            "All mh_improvement_* fields are positive when MH-ALD is better. "
            "For rho/elite metrics this is MH-ALD minus ALD; for percentile/regret "
            "metrics this is ALD minus MH-ALD."
        ),
        "scientific_question": (
            "Does MH-ALD increasingly outperform ALD on the exact candidate "
            "distributions visited during iterative CEM refinement?"
        ),
    }
    (out / "distribution_comparison.json").write_text(json.dumps(payload, indent=2))

    print("===== ALD vs MH-ALD ON ACTUAL CEM POPULATIONS =====")
    for r in rows:
        print(
            f"{r['trace_source']:<10} it={r['cem_iteration']:2d} "
            f"n={r['paired_populations']:2d} "
            f"dRho={r.get('mh_improvement_rho_pred_phys', float('nan')):+.3f} "
            f"dElite={r.get('mh_improvement_pred_elite_overlap_phys', float('nan')):+.3f} "
            f"dPct={r.get('mh_improvement_pred_selected_phys_percentile', float('nan')):+.3f} "
            f"dRegret={r.get('mh_improvement_pred_selection_regret', float('nan')):+.3f}"
        )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
