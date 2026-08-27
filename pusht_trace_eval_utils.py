#!/usr/bin/env python3
"""Shared helpers for PushT CEM-trace diagnostics."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np


def latest_vec(x):
    x = np.asarray(x)
    while x.ndim > 1:
        x = x[-1]
    return np.asarray(x, dtype=np.float64)


def maybe_inverse_state(x, scaler, mode="auto"):
    x = latest_vec(x)
    if mode == "raw":
        return x
    if mode == "standardized":
        return scaler.inverse_transform(x[None])[0]
    if np.nanmax(np.abs(x[:4])) < 20.0:
        return scaler.inverse_transform(x[None])[0]
    return x


def decode_normalized_plan(mean_norm, action_scaler, action_block, clip=True):
    m = np.asarray(mean_norm, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"Expected [H,D] plan, got {m.shape}")
    h, d = m.shape
    if d != int(action_block) * 2:
        raise ValueError(f"Expected D={int(action_block)*2}, got {d}")
    flat = m.reshape(h, int(action_block), 2).reshape(-1, 2)
    raw = action_scaler.inverse_transform(flat).astype(np.float32)
    return np.clip(raw, -1.0, 1.0) if clip else raw


def decode_normalized_candidates(cands_norm, action_scaler, action_block):
    x = np.asarray(cands_norm, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected [N,H,D] candidates, got {x.shape}")
    n, h, d = x.shape
    if d != int(action_block) * 2:
        raise ValueError(f"Expected D={int(action_block)*2}, got {d}")
    raw = action_scaler.inverse_transform(
        x.reshape(n, h, int(action_block), 2).reshape(-1, 2)
    ).reshape(n, h * int(action_block), 2).astype(np.float32)
    clipped = np.clip(raw, -1.0, 1.0)
    return raw, clipped


def load_traces(trace_dir, max_solves=0, stride=1):
    fs = sorted(Path(trace_dir).glob("solve_*.npz"))
    fs = fs[::max(1, int(stride))]
    if int(max_solves) > 0:
        fs = fs[: int(max_solves)]
    if not fs:
        raise FileNotFoundError(f"No solve_*.npz under {trace_dir}")
    return fs


def rankdata(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[m]), rankdata(b[m])
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(np.dot(ra, rb) / den) if den > 1e-12 else float("nan")


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def selection_percentile(reference, score):
    r = np.asarray(reference, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    idx = int(np.nanargmin(s))
    ranks = rankdata(r) - 1.0
    return float(ranks[idx] / max(len(r) - 1, 1)), idx


def elite_metrics(reference, score, deltas, topk):
    reference = np.asarray(reference, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    k = int(min(max(1, topk), len(score)))
    iscore = np.argsort(score)[:k]
    iref = np.argsort(reference)[:k]
    overlap = len(set(iscore.tolist()) & set(iref.tolist())) / k
    us = deltas[iscore].mean(axis=0)
    ur = deltas[iref].mean(axis=0)
    return {
        "elite_overlap": float(overlap),
        "elite_update_cosine": cosine(us, ur),
        "score_elite_reference_mean": float(np.mean(reference[iscore])),
        "oracle_elite_reference_mean": float(np.mean(reference[iref])),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_csv_gz(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def numeric_summary(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }
