#!/usr/bin/env python3
"""Evaluate learned-vs-physical local landscape at exact CEM-visited centers.

Pipeline
--------
1) Run official PushT evaluation with solver=traced_cem to record mu_i, Sigma_i.
2) This script loads those traces.
3) For selected CEM iterations it converts the normalized coarse-action mean
   back to raw PushT actions, creates exact symmetric first-block perturbations,
   physically executes them in the simulator, and compares LeWM-family models
   on the SAME centers.

The simulator rollouts are ORACLE DIAGNOSTICS only; they never enter planning.

Primary questions
-----------------
* Does raw ranking fidelity degrade as CEM moves away from mu_0?
* Does oracle endpoint debiasing still recover ranking at later centers?
* If debiased ranking also degrades, is local counterfactual response J itself
  becoming wrong off the expert/planner-training neighborhood?
* Does the physical/encoder landscape become more even/quadratic during CEM?

Outputs
-------
planner_center_metrics.csv
summary.csv
summary.json
"""
import argparse, csv, json, os, time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing

from eval import img_transform
from eval_pusht_fixed_action_response_horizon import (
    _make_fixed_first_block_candidates,
    _rollout_checkpoints,
)
from eval_pusht_horizon_directional import (
    _current_goal_images, _encode, _label, _normalize_actions,
    _physical_cost, _predict, _spearman,
)
from eval_pusht_linear_tilt_quadratic_signal import _score_metrics


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--trace-dir",required=True)
    p.add_argument("--policies",nargs="+",required=True)
    p.add_argument("--labels",nargs="+",default=None)
    p.add_argument("--config",default="config/eval/pusht.yaml")
    p.add_argument("--dataset",default=None)
    p.add_argument("--iterations",nargs="+",type=int,default=[0,1,3,5,10,30])
    p.add_argument("--max-solves",type=int,default=50)
    p.add_argument("--solve-stride",type=int,default=1)
    p.add_argument("--radius",type=float,default=0.1565)
    p.add_argument("--num-directions",type=int,default=64)
    p.add_argument("--elite-frac",type=float,default=0.10)
    p.add_argument("--state-space",choices=["auto","raw","standardized"],default="auto")
    p.add_argument("--env-seed",type=int,default=0)
    p.add_argument("--model-batch-size",type=int,default=64)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--output-dir",default=None)
    return p.parse_args()


def rms(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x*x))) if len(x) else float("nan")


def cos(a,b):
    a=np.asarray(a,float).reshape(-1); b=np.asarray(b,float).reshape(-1)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-12 else float("nan")


def gram_cos(A,B):
    A=np.asarray(A,float); B=np.asarray(B,float)
    GA=A@A.T; GB=B@B.T
    return cos(GA,GB)


def write_csv(path,rows):
    if not rows: return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def latest_vec(x):
    x=np.asarray(x)
    while x.ndim>1:
        x=x[-1]
    return np.asarray(x,dtype=np.float64)


def maybe_inverse_state(x,scaler,mode):
    x=latest_vec(x)
    if mode=="raw":
        return x
    if mode=="standardized":
        return scaler.inverse_transform(x[None])[0]
    # PushT x/y coordinates are O(10^2); standardized values are O(1).
    if np.nanmax(np.abs(x[:4])) < 20.0:
        return scaler.inverse_transform(x[None])[0]
    return x


def decode_plan(mean_norm,action_scaler,action_block):
    m=np.asarray(mean_norm,dtype=np.float32)
    h,d=m.shape
    if d != action_block*2:
        raise RuntimeError(f"Expected coarse action dim {action_block*2}, got {d}")
    flat=m.reshape(h,action_block,2).reshape(-1,2)
    raw=action_scaler.inverse_transform(flat).astype(np.float32)
    return np.clip(raw,-1.0,1.0)


def load_traces(trace_dir,max_solves,stride):
    fs=sorted(Path(trace_dir).glob("solve_*.npz"))
    fs=fs[::max(1,int(stride))]
    if max_solves>0: fs=fs[:max_solves]
    if not fs: raise FileNotFoundError(f"No solve_*.npz in {trace_dir}")
    return fs


def aggregate(rows):
    keys=[
      "center_shift_l2","sigma_l2","d_l2","e_l2","endpoint_mse",
      "response_cosine","response_gain","response_gram_cosine",
      "rho_pred_phys","rho_debiased_phys","debias_rho_phys_gain",
      "elite_overlap_pred_phys","elite_overlap_debiased_phys",
      "pred_selected_phys_percentile","debiased_selected_phys_percentile",
      "exact_odd_to_even_rms","rho_qtrue_enc","rho_qtrue_phys",
      "tilt_to_quadvar_rms"
    ]
    out={"n":len(rows)}
    for k in keys:
        v=np.asarray([r.get(k,np.nan) for r in rows],float); v=v[np.isfinite(v)]
        out[k+"_mean"]=float(v.mean()) if len(v) else float("nan")
        out[k+"_median"]=float(np.median(v)) if len(v) else float("nan")
    return out


def main():
    a=parse_args()
    if a.labels is not None and len(a.labels)!=len(a.policies):
        raise ValueError("--labels length must equal --policies length")
    cfg=OmegaConf.load(a.config)
    dataset_name=a.dataset or str(cfg.eval.dataset_name)
    cache=Path(os.environ.get("STABLEWM_HOME",swm.data.utils.get_cache_dir()))
    out=Path(a.output_dir) if a.output_dir else Path(a.trace_dir)/"planner_center_landscape"
    out.mkdir(parents=True,exist_ok=True)

    ds=swm.data.HDF5Dataset(dataset_name,keys_to_cache=["action","state"],cache_dir=cache)
    action=np.asarray(ds.get_col_data("action"),dtype=np.float32)
    state=np.asarray(ds.get_col_data("state"),dtype=np.float64)
    action_scaler=preprocessing.StandardScaler().fit(action[np.isfinite(action).all(axis=1)])
    state_scaler=preprocessing.StandardScaler().fit(state[np.isfinite(state).all(axis=1)])

    device=torch.device(a.device)
    transform=img_transform(cfg)
    labels=[_label(p,a.labels,i) for i,p in enumerate(a.policies)]
    models=[]
    for label,policy in zip(labels,a.policies):
        print(f"Loading [{label}] {policy}")
        m=swm.policy.AutoCostModel(policy).to(device).eval()
        m.requires_grad_(False); m.interpolate_pos_encoding=True
        models.append(m)

    traces=load_traces(a.trace_dir,a.max_solves,a.solve_stride)
    env=gym.make(str(cfg.world.env_name),render_mode="rgb_array")
    rows=[]; tstart=time.time()
    try:
        for si,path in enumerate(traces):
            tr=np.load(path,allow_pickle=True)
            if "info_state" not in tr or "info_goal_state" not in tr:
                raise RuntimeError(
                    f"{path} lacks info_state/info_goal_state. "
                    "Trace must be produced by traced_cem.py through the official policy path."
                )
            init_state=maybe_inverse_state(tr["info_state"],state_scaler,a.state_space)
            goal_state=maybe_inverse_state(tr["info_goal_state"],state_scaler,a.state_space)
            means=np.asarray(tr["mean"],dtype=np.float32)
            vars_=np.asarray(tr["var"],dtype=np.float32)
            action_block=int(np.asarray(tr["action_block"]).item())
            horizon=int(np.asarray(tr["horizon"]).item())
            raw_horizon=horizon*action_block
            valid_iters=[i for i in sorted(set(a.iterations)) if 0<=i<len(means)]
            if not valid_iters: continue
            mu0=means[0]
            seed=a.env_seed+si
            current_image,goal_image=_current_goal_images(env,init_state,goal_state,seed)

            for it in valid_iters:
                center_raw=decode_plan(means[it],action_scaler,action_block)
                rng=np.random.default_rng(31_337_001*(si+1)+it)
                cands,_,dmeta,smeta,first_deltas,eqerr=_make_fixed_first_block_candidates(
                    center_raw,a.radius,a.num_directions,rng,action_block
                )
                normalized=_normalize_actions(cands,action_scaler,horizon,action_block)
                nC=len(cands)
                real_states=np.empty((nC,state.shape[-1]),dtype=np.float64)
                real_images=[None]*nC
                for ci in range(nC):
                    rr=_rollout_checkpoints(
                        env,init_state,goal_state,cands[ci],[raw_horizon],seed
                    )
                    real_states[ci]=rr[raw_horizon]["state"]
                    real_images[ci]=rr[raw_horizon]["image"]
                phys_cost,*_=_physical_cost(real_states,goal_state)

                for label,model in zip(labels,models):
                    zg=_encode(model,transform,[goal_image],device,a.model_batch_size)[0]
                    zr=_encode(model,transform,real_images,device,a.model_batch_size)
                    zp=_predict(model,transform,current_image,normalized,device,a.model_batch_size)
                    enc_cost=torch.sum((zr-zg[None])**2,dim=-1).cpu().numpy().astype(float)
                    pred_cost=torch.sum((zp-zg[None])**2,dim=-1).cpu().numpy().astype(float)
                    z0,zhat0=zr[0],zp[0]
                    d=z0-zg; e=zhat0-z0

                    spur=np.zeros(nC,float); q=np.zeros(nC,float)
                    odd=np.zeros(nC,float); even=np.zeros(nC,float)
                    jtrue=[]; jpred=[]; gains=[]; coss=[]
                    for di in range(a.num_directions):
                        ip=np.where((dmeta==di)&(smeta>0))[0]
                        im=np.where((dmeta==di)&(smeta<0))[0]
                        if len(ip)!=1 or len(im)!=1: raise RuntimeError("Malformed +/- pair")
                        ip,im=int(ip[0]),int(im[0])
                        jt=(zr[ip]-zr[im])/(2*a.radius)
                        jp=(zp[ip]-zp[im])/(2*a.radius)
                        jtrue.append(jt.cpu().numpy()); jpred.append(jp.cpu().numpy())
                        coss.append(cos(jp.cpu().numpy(),jt.cpu().numpy()))
                        gains.append(float(torch.linalg.vector_norm(jp).item()/max(torch.linalg.vector_norm(jt).item(),1e-12)))
                        ls=2*a.radius*float(torch.dot(e,jp).item())
                        qq=(a.radius**2)*float(torch.dot(jt,jt).item())
                        spur[ip]=ls; spur[im]=-ls; q[ip]=qq; q[im]=qq
                        oo=0.5*(enc_cost[ip]-enc_cost[im])
                        ee=0.5*(enc_cost[ip]+enc_cost[im])-enc_cost[0]
                        odd[ip]=oo; odd[im]=-oo; even[ip]=ee; even[im]=ee

                    deb=pred_cost-spur
                    sl=slice(1,None)
                    qvar=q[sl]-np.mean(q[sl])
                    row={
                      "model":label,"trace_file":path.name,"solve_index":int(np.asarray(tr["solve_index"]).item()),
                      "cem_iteration":it,"horizon":horizon,"action_block":action_block,
                      "radius":a.radius,"num_directions":a.num_directions,
                      "max_equal_norm_error":eqerr,
                      "center_shift_l2":float(np.linalg.norm(means[it]-mu0)),
                      "sigma_l2":float(np.linalg.norm(vars_[it])),
                      "d_l2":float(torch.linalg.vector_norm(d).item()),
                      "e_l2":float(torch.linalg.vector_norm(e).item()),
                      "endpoint_mse":float(e.pow(2).mean().item()),
                      "response_cosine":float(np.nanmean(coss)),
                      "response_gain":float(np.nanmean(gains)),
                      "response_gram_cosine":gram_cos(np.asarray(jpred),np.asarray(jtrue)),
                      "exact_odd_to_even_rms":rms(odd[sl])/max(rms(even[sl]),1e-12),
                      "rho_qtrue_enc":_spearman(q[sl],enc_cost[sl]),
                      "rho_qtrue_phys":_spearman(q[sl],phys_cost[sl]),
                      "tilt_to_quadvar_rms":rms(spur[sl])/max(rms(qvar),1e-12),
                    }
                    row.update(_score_metrics(phys_cost,enc_cost,pred_cost,first_deltas,a.elite_frac,"pred"))
                    row.update(_score_metrics(phys_cost,enc_cost,deb,first_deltas,a.elite_frac,"debiased"))
                    row["debias_rho_phys_gain"]=row["rho_debiased_phys"]-row["rho_pred_phys"]
                    rows.append(row)
            print(f"trace {si+1}/{len(traces)} {path.name} done")
    finally:
        env.close()

    write_csv(out/"planner_center_metrics.csv",rows)
    summary=[]
    for label in labels:
        its=sorted(set(r["cem_iteration"] for r in rows if r["model"]==label))
        for it in its:
            rr=[r for r in rows if r["model"]==label and r["cem_iteration"]==it]
            summary.append({"model":label,"cem_iteration":it,**aggregate(rr)})
    write_csv(out/"summary.csv",summary)

    drift=[]
    for label in labels:
        rr=[r for r in rows if r["model"]==label]
        shift=[r["center_shift_l2"] for r in rr]
        drift.append({
          "model":label,
          "rho_shift_vs_raw_ranking":_spearman(shift,[r["rho_pred_phys"] for r in rr]),
          "rho_shift_vs_debiased_ranking":_spearman(shift,[r["rho_debiased_phys"] for r in rr]),
          "rho_shift_vs_response_cosine":_spearman(shift,[r["response_cosine"] for r in rr]),
          "rho_shift_vs_odd_even":_spearman(shift,[r["exact_odd_to_even_rms"] for r in rr]),
        })
    json.dump({"summary":summary,"drift_correlations":drift,"elapsed_seconds":time.time()-tstart},
              open(out/"summary.json","w"),indent=2)

    print("\nPlanner-center landscape summary")
    print(f"{'model':<12} {'it':>3} {'shift':>8} {'rhoRaw':>8} {'rhoDeb':>8} {'respCos':>8} {'gain':>8} {'odd/even':>9}")
    for s in summary:
        print(f"{s['model']:<12} {s['cem_iteration']:>3} {s['center_shift_l2_mean']:>8.3f} "
              f"{s['rho_pred_phys_mean']:>8.3f} {s['rho_debiased_phys_mean']:>8.3f} "
              f"{s['response_cosine_mean']:>8.3f} {s['response_gain_mean']:>8.3f} "
              f"{s['exact_odd_to_even_rms_mean']:>9.3f}")
    print(f"Saved to {out}")


if __name__=="__main__": main()
