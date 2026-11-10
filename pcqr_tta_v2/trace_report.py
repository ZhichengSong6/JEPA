"""Inspect v2 numeric snapshots produced before adaptation, even on FAILED runs."""
from __future__ import annotations

import argparse
from pathlib import Path

from .trace_audit import compare_pair


KEY_ORDER = (
    "issued_action_prefix",
    "first_cem_population",
    "solver_state",
    "env_state",
    "solver_proprio",
    "env_proprio",
    "solver_pixels",
    "env_pixels",
    "solver_action",
    "env_action",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--reference", default="frozen")
    ap.add_argument("--modes", nargs="+", default=["obs", "l2", "absolute", "response"])
    args = ap.parse_args()

    trace = args.run / "numeric_traces"
    if not trace.is_dir():
        raise SystemExit(f"No numeric_traces directory at {trace}. Re-run with TRACE_FIRST_TWO=1.")
    case = f"case{args.case:04d}"
    for replan in (0, 1):
        ref = trace / f"{args.reference}_{case}_replan{replan}.npz"
        if not ref.is_file():
            print(f"Reference missing: {ref.name}")
            continue
        print(f"\n===== {case} replan {replan} (reference={args.reference}) =====")
        for mode in args.modes:
            other = trace / f"{mode}_{case}_replan{replan}.npz"
            if not other.is_file():
                print(f"{mode}: not captured")
                continue
            print(f"\n{mode} vs {args.reference}")
            stats = compare_pair(ref, other)
            for key in [*KEY_ORDER, *sorted(set(stats) - set(KEY_ORDER))]:
                if key not in stats:
                    continue
                d = stats[key]
                if d["status"] == "exact_equal":
                    print(f"  {key}: EXACT SAME")
                elif d["status"] == "different":
                    print(f"  {key}: DIFFERENT | {d['num_different']}/{d['total_values']} elements "
                          f"| MAE={d['mae']:.9g} | MAX={d['max_abs']:.9g} "
                          f"| L2={d['l2']:.9g}")
                else:
                    print(f"  {key}: {d}")


if __name__ == "__main__":
    main()
