"""Lossless pre-adaptation numerical snapshots, strictly diagnostic.

Writes exact CEM population, transformed pixels and optional raw
observation fields before first/second CEM planning events. Never edits
the environment, action, cost, model, or accepted sample cohort.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def as_numeric_array(value: Any):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype.kind not in "biufc":
        return None
    return np.ascontiguousarray(arr).copy()


def save_before_update(
    output_dir: str | Path,
    label: str,
    replan: int,
    *,
    solver_info: dict,
    candidates,
    issued_actions: list,
    raw_info: dict | None = None,
    physics_snapshot: dict | None = None,
) -> Path:
    """Exclusively create a per-mode/case/replan .npz (allow_pickle=False)."""
    if not label or not all(c.isascii() and (c.isalnum() or c in "-_") for c in label):
        raise ValueError("Invalid diagnostic label")
    if replan not in (0, 1):
        raise ValueError("This diagnostic captures only pre-update plans 0 and 1")
    path = Path(output_dir) / f"{label}_replan{replan}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

    pixels = solver_info["pixels"]
    if not torch.is_tensor(pixels) or pixels.ndim != 6 or pixels.shape[0] != 1:
        raise ValueError("Expected solver pixels with [B,Q,T,C,H,W]")
    if not torch.is_tensor(candidates) or candidates.ndim != 4 or candidates.shape[0] != 1:
        raise ValueError("Expected CEM candidates [B,Q,H,A]")

    data = {
        "solver_pixels": as_numeric_array(pixels[0, 0]),
        "first_cem_population": as_numeric_array(candidates[0]),
        "raw_steps": np.asarray(len(issued_actions), dtype=np.int64),
    }
    for name in ("state", "proprio", "action"):
        arr = as_numeric_array(solver_info.get(name))
        if arr is not None:
            # Avoid saving the repeated Q copies from CEM's expanded info.
            if arr.ndim >= 3 and arr.shape[1] == candidates.shape[1]:
                arr = arr[:, 0].copy()
            data[f"solver_{name}"] = arr

    for name in ("state", "pixels", "proprio", "action"):
        value = (raw_info or {}).get(name)
        arr = as_numeric_array(value)
        if arr is not None:
            data[f"env_{name}"] = arr

    for name, value in (physics_snapshot or {}).items():
        arr = as_numeric_array(value)
        if arr is not None:
            data[f"physics_{name}"] = arr

    action_rows = [as_numeric_array(a).reshape(-1) for a in issued_actions]
    if action_rows:
        if any(a.shape != action_rows[0].shape for a in action_rows):
            raise ValueError("Action trace has inconsistent shapes")
        data["issued_action_prefix"] = np.stack(action_rows)
    else:
        data["issued_action_prefix"] = np.empty((0, 0), dtype=np.float32)

    if any(x is None for x in data.values()):
        raise RuntimeError("Unexpected missing numeric snapshot component")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **data)
    return path


def compare_pair(reference: str | Path, observed: str | Path) -> dict:
    """Summarize differences; NO thresholds or implicit acceptance."""
    result = {}
    with np.load(reference, allow_pickle=False) as x, np.load(observed, allow_pickle=False) as y:
        for key in sorted(set(x.files) | set(y.files)):
            if key not in x or key not in y:
                result[key] = {"status": "missing", "reference": key in x, "observed": key in y}
                continue
            a, b = x[key], y[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                result[key] = {"status": "shape_or_dtype_mismatch",
                               "reference": [str(a.dtype), list(a.shape)],
                               "observed": [str(b.dtype), list(b.shape)]}
                continue
            same = np.array_equal(a, b)
            diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
            absdiff = np.abs(diff)
            result[key] = {
                "status": "exact_equal" if same else "different",
                "dtype": str(a.dtype),
                "shape": list(a.shape),
                "num_different": int(np.count_nonzero(a != b)),
                "total_values": int(a.size),
                "mae": float(absdiff.mean()) if a.size else 0.,
                "max_abs": float(absdiff.max()) if a.size else 0.,
                "l2": float(np.linalg.norm(diff.ravel())),
            }
    return result
