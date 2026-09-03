#!/usr/bin/env python3
"""Diagnostic A: explain case-77 ALD regression at fixed official CEM populations.

For e=z_pred-z_real and d=z_real-z_goal:
  C_pred-C_enc = 2 d^T e + ||e||^2.
This separates total endpoint MSE from goal-radial error that distorts CEM cost.
No planner/checkpoint is modified.
"""
from __future__ import annotations
import json, os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import gymnasium as gym, hydra, numpy as np, torch
from omegaconf import DictConfig
from eval import get_dataset, img_transform
from eval_lowbudget_failure_autopsy import _build_process,_load_start_goal_states,_physical_cost,_spearman
from eval_b3000_paired_failure_analysis import _extract_variations,_normalized_to_raw,_slice_info
from eval_b3000_critical_jepa_mechanism import _find_trace_for_env,_goal_embedding_from_solver,_predict_processed_context,_record_map
from eval_b3000_hardstate_encoder_geometry import _component_costs,_selection_metrics
from eval_pusht_action_ranking import _encode_images
from eval_pusht_official_diagnostic import load_recording,model_fingerprint,replay,write_csv,write_json

def inversion_rate(score,target,margin_frac=.02):
    s,t=np.asarray(score),np.asarray(target); i,j=np.triu_indices(len(s),1)
    dt,ds=t[i]-t[j],s[i]-s[j]; q25,q75=np.percentile(t,[25,75])
    keep=np.abs(dt)>margin_frac*max(float(q75-q25),1e-12)
    if not np.any(keep): return float("nan"),0
    dt,ds=dt[keep],ds[keep]
    inv=np.where(np.abs(ds)<=1e-12,.5,(np.sign(ds)!=np.sign(dt)).astype(float))
    return float(inv.mean()),int(len(inv))

@hydra.main(version_base=None,config_path="./config/eval",config_name="pusht")
def run(cfg:DictConfig):
    dc=cfg.get("diagnostic",{}); run_dir=Path(str(dc.get("run_dir",""))).expanduser().resolve()
    if not (run_dir/"run_identity.json").is_file(): raise ValueError("Set +diagnostic.run_dir to the complete official diagnostic run")
    out=Path(str(dc.get("output_dir","outputs/pusht_diag_A_case77"))).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    case=int(dc.get("case",77)); iterations=[int(x) for x in dc.get("iterations",[0,3,9,19,29])]
    solves=[int(x) for x in dc.get("solves",[0,1])]; sources=[str(x) for x in dc.get("sources",["lewm","ald_tf"])]
    batch=int(dc.get("model_batch_size",64)); margin=float(dc.get("pair_margin_frac",.02))
    identity=json.loads((run_dir/"run_identity.json").read_text()); p=identity["protocol"]; device=torch.device(str(cfg.solver.device))
    runs={k:load_recording(run_dir/"recordings"/f"{k}.pt",device) for k in ("lewm","ald_tf")}
    models={k:v["model"] for k,v in runs.items()}; sha={k:model_fingerprint(v,visual_only=True) for k,v in models.items()}
    if len(set(sha.values()))!=1: raise RuntimeError(f"LeWM/ALD encoders differ: {sha}")
    dataset=get_dataset(cfg,str(p["dataset_name"])); process,transform=_build_process(cfg,dataset),img_transform(cfg)
    _,goals=_load_start_goal_states(dataset,np.array([identity["episodes"][case]]),np.array([identity["start_steps"][case]]),int(p["goal_offset_steps"]))
    goal,action_block,topk=goals[0],int(p["action_block"]),int(p["topk"]); rows=[]; arrays={}
    env=gym.make(str(p["env_name"]),render_mode="rgb_array")
    try:
      for src in sources:
       source=runs[src]
       for solve in solves:
        tr,li=_find_trace_for_env(source["solver"].trace,case,solve)
        if li is None: raise RuntimeError(f"Missing case={case} source={src} solve={solve}")
        info=_slice_info(tr["solver_info"],li); rec=_record_map(source["recorder"],case)
        step=int(source["recorder"].solve_step_by_env[(solve,case)]); start=np.asarray(rec[step]["state"],dtype=np.float64)
        variations=_extract_variations(info); px=torch.as_tensor(info["pixels"])[0,-1:]
        zgs={k:_goal_embedding_from_solver(m,info,device).float() for k,m in models.items()}
        for it in iterations:
         cand=tr["candidates"][li,it]; raw=_normalized_to_raw(cand,process["action"],action_block)
         final,images,contacts,hits=replay(env,start,goal,raw,variations,42)
         phys=_component_costs(final,goal)["official_cost"]; endpoint_success=_physical_cost(final,goal)[3]
         for label,model in models.items():
          zr=_encode_images(model,transform,images,device,batch).float()
          zp=_predict_processed_context(model,px,np.empty((0,2),np.float32),cand,process["action"],action_block,device,batch).float()
          d,e=zr-zgs[label][None],zp-zr; enc=d.square().sum(-1); pred=(d+e).square().sum(-1)
          cross=2*(d*e).sum(-1); enorm=e.square().sum(-1); mse=e.square().mean(-1)
          radial=(d*e).sum(-1)/torch.clamp(torch.linalg.vector_norm(d,dim=-1),min=1e-12)
          audit=torch.max(torch.abs((pred-enc)-cross-enorm)).item()
          if audit>2e-4: raise RuntimeError(f"cost decomposition audit failed: {audit}")
          enc,pred,cross,enorm,mse,radial=[x.detach().cpu().numpy().astype(np.float64) for x in (enc,pred,cross,enorm,mse,radial)]
          native=float("nan")
          if label==src:
           native=float(np.max(np.abs(pred-np.asarray(tr["predicted_costs"][li,it]))))
           if native>2e-4: raise RuntimeError(f"native CEM cost audit failed: {native}")
          inv_e,pairs_e=inversion_rate(pred,enc,margin); inv_p,pairs_p=inversion_rate(pred,phys,margin)
          se=_selection_metrics(pred,enc,min(topk,len(pred))); sp=_selection_metrics(pred,phys,min(topk,len(pred)))
          rows.append(dict(case=case,source=src,solve=solve,cem_iteration=it,model=label,endpoint_mse=float(mse.mean()),
            pred_encoder_rho=_spearman(pred,enc),encoder_physical_rho=_spearman(enc,phys),predictor_physical_rho=_spearman(pred,phys),
            encoder_pair_inversion=inv_e,encoder_pairs=pairs_e,physical_pair_inversion=inv_p,physical_pairs=pairs_p,
            cross_term_mean=float(cross.mean()),cross_term_std=float(cross.std()),error_norm_mean=float(enorm.mean()),
            cost_distortion_std=float((pred-enc).std()),radial_abs_mean=float(np.abs(radial).mean()),
            selected_encoder_percentile=float(se["selected_target_percentile"]),selected_physical_percentile=float(sp["selected_target_percentile"]),
            physical_topk_recall=float(sp["top10_recall"]),ever_success_fraction=float(np.mean(hits)),endpoint_success_fraction=float(np.mean(endpoint_success)),
            contact_fraction=float(np.mean(contacts>0)),native_recomputed_max_abs=native,decomposition_max_abs=audit))
          key=f"{src}_s{solve}_i{it}_{label}"
          arrays[f"{key}_encoder_cost"],arrays[f"{key}_predictor_cost"]=enc,pred
          arrays[f"{key}_physical_cost"],arrays[f"{key}_endpoint_mse"]=phys,mse
          arrays[f"{key}_cross_term"],arrays[f"{key}_error_norm"],arrays[f"{key}_radial_projection"]=cross,enorm,radial
          print(f"A {key}: mse={mse.mean():.5f} rho(pred,enc)={_spearman(pred,enc):.3f} inv={inv_e:.3f}",flush=True)
    finally: env.close()
    write_csv(out/"population_summary.csv",rows); np.savez_compressed(out/"candidate_terms.npz",**arrays)
    write_json(out/"summary.json",dict(status="complete",case=case,sources=sources,solves=solves,iterations=iterations,
      shared_encoder_sha256=next(iter(sha.values())),identity="Cpred-Cenc=2*(z-zg)^T*(zhat-z)+||zhat-z||^2",
      interpretation="Lower MSE with worse predictor/encoder ranking or more pair inversions implicates goal-radial cost distortion rather than total prediction magnitude."))
    print(f"=== DIAGNOSTIC A DONE ===\nResults: {out}",flush=True)
if __name__=="__main__": run()
