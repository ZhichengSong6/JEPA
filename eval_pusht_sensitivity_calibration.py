#!/usr/bin/env python3
"""Post-hoc calibration analysis for eval_pusht_horizon_directional.py.

Consumes candidate_metrics.npz already produced by the validated horizon
directional evaluator.  No simulator, checkpoint loading, or GPU is required.

For every symmetric pair U+=U0+r v, U-=U0-r v, define
    d_phys = C_phys(U+) - C_phys(U-)
    d_enc  = C_enc (U+) - C_enc (U-)
    d_pred = C_pred(U+) - C_pred(U-)

Primary reference is d_enc because C_enc and C_pred use the same model-specific
latent goal cost.  d_phys is kept as an oracle task-space check.

Interpretation:
  strong |d_enc|, weak |d_pred| -> sensitivity loss / collapse-like behavior
  weak   |d_enc|, strong |d_pred| -> over-response / sensitivity leakage
  sign(d_enc) != sign(d_pred) -> directional distortion

This is a planner-facing scalar diagnostic.  It does not by itself prove full
latent-Jacobian spurious expansion; if leakage is found, follow with a vector
JVP/singular-spectrum diagnostic.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="candidate_metrics.npz from eval_pusht_horizon_directional.py")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--margin-frac", type=float, default=0.02)
    p.add_argument("--weak-quantile", type=float, default=0.25)
    p.add_argument("--strong-quantile", type=float, default=0.75)
    p.add_argument("--num-bins", type=int, default=5)
    return p.parse_args()


def rankdata(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        r[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return r


def corr(a, b, rank=False):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    if rank:
        a, b = rankdata(a), rankdata(b)
    a, b = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else np.nan


def iqr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    q25, q75 = np.percentile(x, [25, 75])
    return float(q75 - q25)


def direction_acc(ref, est, margin):
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    m = np.isfinite(ref) & np.isfinite(est) & (np.abs(ref) > margin)
    if m.sum() == 0:
        return np.nan, 0
    r, e = ref[m], est[m]
    tie = np.abs(e) <= 1e-12
    good = np.where(tie, 0.5, (np.sign(r) == np.sign(e)).astype(float))
    return float(good.mean()), int(len(good))


def med(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def slope_origin(ref, pred):
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(ref) & np.isfinite(pred)
    if m.sum() < 2:
        return np.nan
    r, p = ref[m], pred[m]
    den = np.dot(r, r)
    return float(np.dot(r, p) / den) if den > 1e-12 else np.nan


def nrmse(ref, pred):
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(ref) & np.isfinite(pred)
    if m.sum() < 2:
        return np.nan
    r, p = ref[m], pred[m]
    den = np.sqrt(np.mean(r ** 2))
    return float(np.sqrt(np.mean((p - r) ** 2)) / den) if den > 1e-12 else np.nan


def contact_mode(cp, cm):
    if not cp and not cm:
        return "no_contact"
    if cp and cm:
        return "both_contact"
    return "contact_switch"


def find_pairs(base_radius, direction, sign):
    out = {}
    valid = np.isfinite(base_radius) & (direction >= 0) & (sign != 0)
    for r in np.unique(base_radius[valid]):
        for d in np.unique(direction[valid]):
            ip = np.where(valid & np.isclose(base_radius, r) &
                          (direction == d) & (sign > 0))[0]
            im = np.where(valid & np.isclose(base_radius, r) &
                          (direction == d) & (sign < 0))[0]
            if len(ip) == 1 and len(im) == 1:
                out[(float(r), int(d))] = (int(ip[0]), int(im[0]))
    return out


def summarize(denc, dpred, dphys, args):
    denc = np.asarray(denc, dtype=np.float64)
    dpred = np.asarray(dpred, dtype=np.float64)
    dphys = np.asarray(dphys, dtype=np.float64)
    m = np.isfinite(denc) & np.isfinite(dpred) & np.isfinite(dphys)
    denc, dpred, dphys = denc[m], dpred[m], dphys[m]
    if len(denc) == 0:
        return {"n_pairs": 0}

    em = args.margin_frac * max(iqr(denc), 1e-12)
    pm = args.margin_frac * max(iqr(dphys), 1e-12)
    dir_pe, n_pe = direction_acc(denc, dpred, em)
    dir_pp, n_pp = direction_acc(dphys, dpred, pm)
    dir_ep, n_ep = direction_acc(dphys, denc, pm)

    ae, ap = np.abs(denc), np.abs(dpred)
    qweak = float(np.quantile(ae, args.weak_quantile))
    qstrong = float(np.quantile(ae, args.strong_quantile))
    weak, strong = ae <= qweak, ae >= qstrong

    informative = ae > em
    eps = max(1e-8 * max(qstrong, 1.0), 1e-12)
    gain = ap[informative] / np.maximum(ae[informative], eps)
    log2gain = np.log2(np.maximum(gain, 1e-12)) if len(gain) else np.array([])
    strong_dir, n_strong = direction_acc(denc[strong], dpred[strong], 0.0)

    return {
        "n_pairs": int(len(denc)),
        "pred_enc_dir_acc": dir_pe,
        "pred_enc_dir_n": n_pe,
        "pred_phys_dir_acc": dir_pp,
        "pred_phys_dir_n": n_pp,
        "enc_phys_dir_acc": dir_ep,
        "enc_phys_dir_n": n_ep,
        "pearson_pred_enc": corr(denc, dpred, False),
        "spearman_pred_enc": corr(denc, dpred, True),
        "slope_pred_on_enc": slope_origin(denc, dpred),
        "nrmse_pred_vs_enc": nrmse(denc, dpred),
        "median_abs_enc_effect": med(ae),
        "median_abs_pred_effect": med(ap),
        "weak_ref_threshold": qweak,
        "strong_ref_threshold": qstrong,
        "weak_ref_pairs": int(weak.sum()),
        "strong_ref_pairs": int(strong.sum()),
        "weak_ref_overresponse_rate": float(np.mean(ap[weak] >= qstrong)) if weak.any() else np.nan,
        "strong_ref_underresponse_rate": float(np.mean(ap[strong] <= qweak)) if strong.any() else np.nan,
        "strong_ref_direction_acc": strong_dir,
        "strong_ref_direction_n": n_strong,
        "median_gain_pred_over_enc": med(gain),
        "median_log2_gain_pred_over_enc": med(log2gain),
    }


def calibration_bins(denc, dpred, dphys, args):
    denc = np.asarray(denc, dtype=np.float64)
    dpred = np.asarray(dpred, dtype=np.float64)
    dphys = np.asarray(dphys, dtype=np.float64)
    m = np.isfinite(denc) & np.isfinite(dpred) & np.isfinite(dphys)
    denc, dpred, dphys = denc[m], dpred[m], dphys[m]
    if len(denc) == 0:
        return []
    mag = np.abs(denc)
    edges = np.unique(np.quantile(mag, np.linspace(0, 1, args.num_bins + 1)))
    if len(edges) < 2:
        return []
    em = args.margin_frac * max(iqr(denc), 1e-12)
    pm = args.margin_frac * max(iqr(dphys), 1e-12)
    rows = []
    for bi in range(len(edges) - 1):
        lo, hi = float(edges[bi]), float(edges[bi + 1])
        bm = ((mag >= lo) & (mag <= hi)) if bi == len(edges) - 2 else ((mag >= lo) & (mag < hi))
        if not bm.any():
            continue
        dpe, npe = direction_acc(denc[bm], dpred[bm], em)
        dpp, npp = direction_acc(dphys[bm], dpred[bm], pm)
        dep, nep = direction_acc(dphys[bm], denc[bm], pm)
        eps = max(1e-8 * max(hi, 1.0), 1e-12)
        gain = np.abs(dpred[bm]) / np.maximum(np.abs(denc[bm]), eps)
        rows.append({
            "bin": bi,
            "q_lo_effect": lo,
            "q_hi_effect": hi,
            "n_pairs": int(bm.sum()),
            "median_abs_enc_effect": med(np.abs(denc[bm])),
            "median_abs_pred_effect": med(np.abs(dpred[bm])),
            "median_abs_phys_effect": med(np.abs(dphys[bm])),
            "median_gain_pred_over_enc": med(gain),
            "median_log2_gain_pred_over_enc": med(np.log2(np.maximum(gain, 1e-12))),
            "pred_enc_dir_acc": dpe,
            "pred_enc_dir_n": npe,
            "pred_phys_dir_acc": dpp,
            "pred_phys_dir_n": npp,
            "enc_phys_dir_acc": dep,
            "enc_phys_dir_n": nep,
        })
    return rows


def main():
    args = parse_args()
    if not (0 < args.weak_quantile < args.strong_quantile < 1):
        raise ValueError("Require 0 < weak_quantile < strong_quantile < 1")

    inp = Path(args.input)
    outdir = Path(args.output_dir) if args.output_dir else inp.parent / "sensitivity_calibration"
    outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(inp, allow_pickle=True)

    need = ["horizons", "labels", "candidate_base_radius", "candidate_direction",
            "candidate_sign", "physical_cost", "enc_costs", "pred_costs"]
    for k in need:
        if k not in z:
            raise KeyError(f"Missing key {k!r} in {inp}")

    horizons = np.asarray(z["horizons"]).astype(int)
    labels = [str(x) for x in np.asarray(z["labels"]).tolist()]
    base_r = np.asarray(z["candidate_base_radius"])
    dirs = np.asarray(z["candidate_direction"])
    signs = np.asarray(z["candidate_sign"])
    phys = np.asarray(z["physical_cost"], dtype=np.float64)
    enc = np.asarray(z["enc_costs"], dtype=np.float64)
    pred = np.asarray(z["pred_costs"], dtype=np.float64)
    contact = np.asarray(z["candidate_had_contact"]).astype(bool) if "candidate_had_contact" in z else np.zeros_like(phys, dtype=bool)

    if enc.shape != pred.shape or enc.ndim != 4 or enc.shape[1:] != phys.shape:
        raise ValueError(f"Unexpected shapes: enc={enc.shape}, pred={pred.shape}, phys={phys.shape}")

    pair_rows = []
    for mi, label in enumerate(labels):
        for hi, H in enumerate(horizons):
            for ai in range(phys.shape[1]):
                for (r, d), (ip, im) in find_pairs(base_r[hi, ai], dirs[hi, ai], signs[hi, ai]).items():
                    cp, cm = bool(contact[hi, ai, ip]), bool(contact[hi, ai, im])
                    dphys = float(phys[hi, ai, ip] - phys[hi, ai, im])
                    denc = float(enc[mi, hi, ai, ip] - enc[mi, hi, ai, im])
                    dpred = float(pred[mi, hi, ai, ip] - pred[mi, hi, ai, im])
                    pair_rows.append({
                        "model": label, "horizon": int(H), "anchor": ai,
                        "base_radius": r, "direction": d,
                        "contact_mode": contact_mode(cp, cm),
                        "d_phys": dphys, "d_enc": denc, "d_pred": dpred,
                        "abs_d_phys": abs(dphys), "abs_d_enc": abs(denc), "abs_d_pred": abs(dpred),
                    })

    if not pair_rows:
        raise RuntimeError("No +/- pairs found")

    with (outdir / "pair_effects.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        w.writeheader(); w.writerows(pair_rows)

    model = np.array([r["model"] for r in pair_rows], dtype=object)
    hcol = np.array([r["horizon"] for r in pair_rows], dtype=int)
    rcol = np.array([r["base_radius"] for r in pair_rows], dtype=float)
    ccol = np.array([r["contact_mode"] for r in pair_rows], dtype=object)
    dphys = np.array([r["d_phys"] for r in pair_rows], dtype=float)
    denc = np.array([r["d_enc"] for r in pair_rows], dtype=float)
    dpred = np.array([r["d_pred"] for r in pair_rows], dtype=float)

    groups = ["all", "no_contact", "both_contact", "contact_switch"]
    summary_rows, bin_rows = [], []
    for label in labels:
        for H in horizons:
            for r in sorted(np.unique(rcol[(model == label) & (hcol == H)])):
                base = (model == label) & (hcol == H) & np.isclose(rcol, r)
                for group in groups:
                    idx = np.where(base & ((ccol == group) if group != "all" else True))[0]
                    if len(idx) == 0:
                        continue
                    s = summarize(denc[idx], dpred[idx], dphys[idx], args)
                    summary_rows.append({"model": label, "horizon": int(H), "base_radius": float(r), "group": group, **s})
                    for b in calibration_bins(denc[idx], dpred[idx], dphys[idx], args):
                        bin_rows.append({"model": label, "horizon": int(H), "base_radius": float(r), "group": group, **b})

    def write_csv(path, rows):
        if not rows:
            return
        keys = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    write_csv(outdir / "calibration_summary.csv", summary_rows)
    write_csv(outdir / "calibration_bins.csv", bin_rows)
    with (outdir / "summary.json").open("w") as f:
        json.dump({"input": str(inp), "primary_reference": "d_enc", "rows": summary_rows}, f, indent=2)

    print("\nPrimary reference: d_enc = C_enc(U+) - C_enc(U-)")
    print("Ideal: dirPE=1, rhoPE=1, slope=1, nrmse=0, log2g=0, weakOver=0, strongUnder=0\n")
    hdr = (f"{'model':<14} {'H':>2} {'r':>5} {'n':>6} {'dirPE':>7} {'dirPPhys':>9} "
           f"{'rhoPE':>7} {'slope':>7} {'nrmse':>7} {'log2g':>7} {'weakOver':>9} "
           f"{'strongUnder':>11} {'strongDir':>9}")
    print(hdr); print("-" * len(hdr))
    for row in summary_rows:
        if row["group"] != "all":
            continue
        fmt = lambda x: "nan" if x is None or not np.isfinite(float(x)) else f"{float(x):.3f}"
        print(f"{row['model']:<14} {row['horizon']:>2d} {row['base_radius']:>5.2f} {row['n_pairs']:>6d} "
              f"{fmt(row.get('pred_enc_dir_acc')):>7} {fmt(row.get('pred_phys_dir_acc')):>9} "
              f"{fmt(row.get('spearman_pred_enc')):>7} {fmt(row.get('slope_pred_on_enc')):>7} "
              f"{fmt(row.get('nrmse_pred_vs_enc')):>7} {fmt(row.get('median_log2_gain_pred_over_enc')):>7} "
              f"{fmt(row.get('weak_ref_overresponse_rate')):>9} {fmt(row.get('strong_ref_underresponse_rate')):>11} "
              f"{fmt(row.get('strong_ref_direction_acc')):>9}")

    print(f"\nSaved to {outdir}")
    print("  pair_effects.csv")
    print("  calibration_summary.csv")
    print("  calibration_bins.csv")
    print("  summary.json")


if __name__ == "__main__":
    main()
