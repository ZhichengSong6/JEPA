#!/usr/bin/env python3
"""Re-analyze fixed-H5 goal-proximity results by *actual latent distance*.

This deliberately removes timestamp from the independent variable.  It consumes
anchor_goal_metrics.csv produced by eval_pusht_fixed_horizon_goal_proximity.py
and bins samples by d_l2 = ||z_endpoint-z_goal||.

No model/simulator rerun is required.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np


def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3: return float("nan")
    def ranks(x):
        o=np.argsort(x,kind="mergesort"); r=np.empty(len(x),float); i=0
        while i<len(x):
            j=i+1
            while j<len(x) and x[o[j]]==x[o[i]]: j+=1
            r[o[i:j]]=0.5*((i+1)+j); i=j
        return r
    ra=ranks(a[m]); rb=ranks(b[m]); ra-=ra.mean(); rb-=rb.mean()
    den=np.linalg.norm(ra)*np.linalg.norm(rb)
    return float(np.dot(ra,rb)/den) if den>1e-12 else float("nan")


def mean(rows,key):
    x=np.asarray([float(r[key]) for r in rows],float); x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="anchor_goal_metrics.csv")
    p.add_argument("--num-bins",type=int,default=5)
    p.add_argument("--output-dir",default=None)
    a=p.parse_args()
    rows=list(csv.DictReader(open(a.input)))
    out=Path(a.output_dir) if a.output_dir else Path(a.input).parent/"distance_binned_order_transition"
    out.mkdir(parents=True,exist_ok=True)
    labels=sorted(set(r["model"] for r in rows))
    summary=[]; corr=[]
    metrics=[
      "exact_odd_to_even_rms","exact_odd_to_evenvar_rms","rho_qtrue_enc",
      "rho_qtrue_phys","rho_pred_phys","rho_debiased_phys",
      "debias_rho_phys_gain","endpoint_mse","tilt_to_quadvar_rms"
    ]
    for label in labels:
        rr=[r for r in rows if r["model"]==label and np.isfinite(float(r["d_l2"]))]
        d=np.asarray([float(r["d_l2"]) for r in rr])
        edges=np.quantile(d,np.linspace(0,1,a.num_bins+1))
        edges=np.maximum.accumulate(edges)
        # Include the largest value in the last bin.
        ids=np.searchsorted(edges[1:-1],d,side="right")
        for bi in range(a.num_bins):
            br=[r for r,i in zip(rr,ids) if i==bi]
            if not br: continue
            rec={"model":label,"bin":bi,"n":len(br),
                 "d_min":float(np.min([float(r["d_l2"]) for r in br])),
                 "d_max":float(np.max([float(r["d_l2"]) for r in br])),
                 "d_mean":mean(br,"d_l2")}
            for k in metrics: rec[k+"_mean"]=mean(br,k)
            summary.append(rec)
        corr.append({
          "model":label,"n":len(rr),
          "rho_d_vs_odd_even":spearman(d,[float(r["exact_odd_to_even_rms"]) for r in rr]),
          "rho_d_vs_odd_evenvar":spearman(d,[float(r["exact_odd_to_evenvar_rms"]) for r in rr]),
          "rho_d_vs_qtrue_enc":spearman(d,[float(r["rho_qtrue_enc"]) for r in rr]),
          "rho_d_vs_qtrue_phys":spearman(d,[float(r["rho_qtrue_phys"]) for r in rr]),
          "rho_d_vs_debias_gain":spearman(d,[float(r["debias_rho_phys_gain"]) for r in rr]),
        })
    fields=list(summary[0].keys()) if summary else []
    with open(out/"distance_bins.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
    with open(out/"distance_correlations.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(corr[0].keys())); w.writeheader(); w.writerows(corr)
    json.dump({"bins":summary,"correlations":corr},open(out/"summary.json","w"),indent=2)
    print("\nDistance-conditioned order transition")
    print(f"{'model':<12} {'bin':>3} {'n':>4} {'d':>8} {'odd/even':>10} {'rhoQ/E':>8} {'rhoRaw':>8} {'rhoDeb':>8}")
    for r in summary:
        print(f"{r['model']:<12} {r['bin']:>3} {r['n']:>4} {r['d_mean']:>8.3f} "
              f"{r['exact_odd_to_even_rms_mean']:>10.3f} {r['rho_qtrue_enc_mean']:>8.3f} "
              f"{r['rho_pred_phys_mean']:>8.3f} {r['rho_debiased_phys_mean']:>8.3f}")
    print(f"Saved to {out}")


if __name__=="__main__": main()
