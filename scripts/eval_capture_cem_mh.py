#!/usr/bin/env python3
"""Run the repository's eval.py unchanged, capturing metrics and dataset starts.

The only evaluation argument replaced by this recording hook is video_path,
so parallel processes cannot write videos into the same checkpoint directory.
No model, planner, cost, RNG seed, reset, action or success rule is modified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported metric type: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = args.overrides
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    repo = Path(args.repo).resolve()
    record = Path(args.record).resolve()
    video_dir = Path(args.video_dir).resolve()
    if record.exists():
        raise FileExistsError(f"Refusing to overwrite: {record}")
    if not (repo / "eval.py").is_file():
        raise FileNotFoundError(repo / "eval.py")
    record.parent.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(repo))

    import stable_worldmodel as swm

    original = swm.World.evaluate_from_dataset
    calls = 0
    payload = None

    def capture(world, *positional, **kwargs):
        nonlocal calls, payload
        calls += 1
        if calls != 1:
            raise RuntimeError("Expected exactly one evaluate_from_dataset call")
        selection = {
            k: jsonable(kwargs.get(k)) for k in
            ("episodes_idx", "start_steps", "goal_offset_steps", "eval_budget")
        }
        kwargs["video_path"] = video_dir
        start = time.perf_counter()
        metrics = original(world, *positional, **kwargs)
        elapsed = time.perf_counter() - start
        payload = {
            "status": "complete",
            "hydra_overrides": overrides,
            "selection": selection,
            "metrics": jsonable(metrics),
            "world_evaluation_seconds": elapsed,
            "video_dir": str(video_dir),
        }
        return metrics

    swm.World.evaluate_from_dataset = capture
    sys.argv = [str(repo / "eval.py"), *overrides]
    try:
        runpy.run_path(str(repo / "eval.py"), run_name="__main__")
    finally:
        swm.World.evaluate_from_dataset = original
    if calls != 1 or payload is None:
        raise RuntimeError("Evaluation did not produce metrics")
    # Publish only after eval.py also finishes writing its official results.
    # Atomic, no-clobber publication prevents accidental concurrent overwrite.
    tmp = record.with_name(record.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("x") as f:
            f.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.link(tmp, record)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
