#!/usr/bin/env python3
"""Diagnostic B: isolate case-93 CEM candidate-coverage failure.

Fixes the exact recorded ALD solve-1 state/goal. Replays the saved official
population, then diagnosis-only reruns vary search resources:
300x5, 900x5, 300x7, 300x10 (num_samples x coarse horizon), with independent
restarts. Model cost/checkpoints are unchanged.
"""
from __future__ import annotations
import json,os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import gymnasium as gym,hydra,numpy as np,stable_worldmodel as swm,torch
from gymnasium.spaces import Box
from omegaconf import DictConfig
from eval import get_dataset
from eval_lowbudget_failure_autopsy import _build_process,_load_start_goal_states,_physical_cost,_spearman
from eval_b3000_paired_failure_analysis import CrossTraceCEMSolver,_extract_variations,_normalized_to_raw,_slice_info
from eval_b3000_critical_jepa_mechanism import _find_trace_for_env,_record_map
from eval_b3000_hardstate_encoder_geometry import _component_costs,_selection_metrics
from eval_pusht_official_diagnostic import load_recording,replay,write_csv,write_json

def parse_designs(x):
    items=[s.strip() for s in (x.split(",") if isinstance(x,str) else x) if str(s).strip()]; out=[]
    for item in items:
        n,h=str(item).lower().split("x",1); out.append((int(n),int(h)))
    return out

def summarize(rows):
    out=[]
    for design in sorted(set(r["design"] for r in rows if r["design"]!="recorded_official")):
        rr=[r for r in rows if r["design"]==design]
        out.append(dict(design=design,num_samples=rr[0]["num_samples"],horizon=rr[0]["horizon"],
          checked_populations=len(rr),restarts=len(set(r["restart"] for r in rr)),any_success=any(r["any_ever_success"] for r in rr),
          max_success_fraction=max(r["ever_success_fraction"] for r in rr),min_physical_cost=min(r["oracle_best_physical_cost"] for r in rr),
          best_selected_physical_percentile=min(r["selected_physical_percentile"] for r in rr),
          mean_predictor_physical_rho=float(np.mean([r["predictor_physical_rho"] for r in rr]))))
    return out

@hydra.main(version_base=None,config_path="./config/eval",config_name="pusht")
def run(cfg:DictConfig):
    dc=cfg.get("diagnostic",{}); run_dir=Path(str(dc.get("run_dir",""))).expanduser().resolve()
    if not (run_dir/"run_identity.json").is_file(): raise ValueError("Set +diagnostic.run_dir to the complete official run")
    out=Path(str(dc.get("output_dir","outputs/pusht_diag_B_case93"))).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    mode=str(dc.get("mode","formal")); case=int(dc.get("case",93)); solve=int(dc.get("solve",1)); source_label=str(dc.get("source","ald_tf"))
    defaults=("300x5,900x5,300x7,300x10",4,30,[0,3,9,19,29],30) if mode=="formal" else ("30x5",1,2,[0,1],3)
    designs=parse_designs(dc.get("designs",defaults[0])); restarts=int(dc.get("restarts",defaults[1])); n_steps=int(dc.get("n_steps",defaults[2]))
    replay_iters=[int(x) for x in dc.get("replay_iterations",defaults[3])]; topk_fixed=int(dc.get("topk",defaults[4])); base_seed=int(dc.get("base_seed",42))
    identity=json.loads((run_dir/"run_identity.json").read_text()); p=identity["protocol"]; device=torch.device(str(cfg.solver.device))
    source=load_recording(run_dir/"recordings"/f"{source_label}.pt",device); model=source["model"]
    dataset=get_dataset(cfg,str(p["dataset_name"])); process=_build_process(cfg,dataset)
    _,goals=_load_start_goal_states(dataset,np.array([identity["episodes"][case]]),np.array([identity["start_steps"][case]]),int(p["goal_offset_steps"])); goal=goals[0]
    tr,li=_find_trace_for_env(source["solver"].trace,case,solve)
    if li is None: raise RuntimeError(f"Missing case={case} solve={solve}")
    info=_slice_info(tr["solver_info"],li); rec=_record_map(source["recorder"],case)
    step=int(source["recorder"].solve_step_by_env[(solve,case)]); start=np.asarray(rec[step]["state"],dtype=np.float64)
    variations=_extract_variations(info); action_block=int(p["action_block"])
    warm=torch.as_tensor(tr["prev_mean"][li,0][None],dtype=torch.float32); rows=[]; mean_rows=[]
    env=gym.make(str(p["env_name"]),render_mode="rgb_array")
    def score(design,restart,horizon,it,cand,pred):
        raw=_normalized_to_raw(cand,process["action"],action_block); final,_,contacts,hits=replay(env,start,goal,raw,variations,42)
        phys=_component_costs(final,goal)["official_cost"]; end=_physical_cost(final,goal)[3]; pred=np.asarray(pred,dtype=np.float64)
        sel=_selection_metrics(pred,phys,min(topk_fixed,len(pred)))
        row=dict(case=case,source=source_label,solve=solve,design=design,restart=restart,num_samples=len(cand),horizon=horizon,cem_iteration=it,
          num_success_ever=int(np.sum(hits)),num_success_endpoint=int(np.sum(end)),any_ever_success=bool(np.any(hits)),
          ever_success_fraction=float(np.mean(hits)),endpoint_success_fraction=float(np.mean(end)),oracle_best_physical_cost=float(np.min(phys)),
          predictor_physical_rho=_spearman(pred,phys),selected_physical_percentile=float(sel["selected_target_percentile"]),
          physical_topk_recall=float(sel["top10_recall"]),contact_fraction=float(np.mean(contacts>0)),
          block_motion_mean_px=float(np.linalg.norm(final[:,2:4]-start[2:4],axis=1).mean()))
        rows.append(row); print(f"B {design} r={restart} i={it}: success={row['num_success_ever']}/{len(cand)} best={row['oracle_best_physical_cost']:.4f}",flush=True)
    try:
      for it in replay_iters:
        if it<tr["candidates"].shape[1]: score("recorded_official",-1,int(tr["candidates"].shape[3]),it,tr["candidates"][li,it],tr["predicted_costs"][li,it])
      for di,(n,h) in enumerate(designs):
       for r in range(restarts):
        topk=min(topk_fixed,n); solver=CrossTraceCEMSolver(model=model,batch_size=1,num_samples=n,var_scale=float(p.get("var_scale",1.0)),
          n_steps=n_steps,topk=topk,device=str(cfg.solver.device),seed=base_seed+1000*di+r,state_scaler=process.get("state"))
        plan=swm.PlanConfig(horizon=h,receding_horizon=h,action_block=action_block)
        solver.configure(action_space=Box(-1.,1.,shape=(h,2),dtype=np.float32),n_envs=1,config=plan)
        result=solver.solve(info,init_action=warm); nt=solver.trace[0]
        for it in replay_iters:
            if it<n_steps: score(f"{n}x{h}",r,h,it,nt["candidates"][0,it],nt["predicted_costs"][0,it])
        mean_norm=np.asarray(result["actions"],dtype=np.float32); raw_mean=_normalized_to_raw(mean_norm,process["action"],action_block)
        mf,_,mc,mh=replay(env,start,goal,raw_mean,variations,42); mend=_physical_cost(mf,goal)[3]
        mean_rows.append(dict(design=f"{n}x{h}",restart=r,mean_ever_success=bool(mh[0]),mean_endpoint_success=bool(mend[0]),
          mean_physical_cost=float(_component_costs(mf,goal)["official_cost"][0]),mean_contact_steps=int(mc[0]),
          mean_block_motion_px=float(np.linalg.norm(mf[0,2:4]-start[2:4]))))
    finally: env.close()
    summary=summarize(rows); recorded=[r for r in rows if r["design"]=="recorded_official"]
    recmax=max([r["ever_success_fraction"] for r in recorded] or [0.]); base=next((r for r in summary if r["design"]=="300x5"),None)
    bigger=next((r for r in summary if r["design"]=="900x5"),None); longer=[r for r in summary if r["horizon"]>5 and r["num_samples"]==300]
    bmax=base["max_success_fraction"] if base else 0.; nmax=bigger["max_success_fraction"] if bigger else 0.
    hmax=max([r["max_success_fraction"] for r in longer] or [0.]); eps=.005
    flags=dict(restart_or_basin=(bmax>recmax+eps),sample_budget=(nmax>bmax+eps),longer_horizon=(hmax>bmax+eps))
    write_csv(out/"coverage_rows.csv",rows); write_csv(out/"design_summary.csv",summary); write_csv(out/"mean_plan_rows.csv",mean_rows)
    write_json(out/"diagnosis.json",dict(status="complete",case=case,source=source_label,solve=solve,designs=[f"{n}x{h}" for n,h in designs],
      restarts=restarts,n_steps=n_steps,replay_iterations=replay_iters,recorded_max_success_fraction=recmax,rerun_300x5_max=bmax,
      rerun_900x5_max=nmax,rerun_long_horizon_max=hmax,material_change_threshold=eps,flags=flags,
      interpretation="300x5 restart helps => local basin/random coverage; 900x5 helps => sample budget; longer horizon helps => horizon; none helps => test action parameterization/reachability next."))
    print(f"=== DIAGNOSTIC B DONE ===\nResults: {out}",flush=True)
if __name__=="__main__": run()
