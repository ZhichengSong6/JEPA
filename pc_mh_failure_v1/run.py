#!/usr/bin/env python3
"""Read-only PC-MH vs Stepwise failure analysis using existing exact replay tools.

No training, planner replacement, reward head, or counterfactual supervision.
Survey always uses all 100 official starts per seed. The native-reset pinning
makes this a NEW matched diagnostic cohort, not recovered historical results.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analysis_core import (LABELS, sha, dump, plain, write_csv, group, choose_cases,
                           cost_stats, decomposition, flags, make_joint_probes,
                           interaction_rows)

BASE = '9efd2f963fbdc6548faf00c7e23d5d79494f88a3'
# Fail closed if the reused runtime differs from the version inspected for this release.
BLOBS = {
    'eval_cem_mh_diagnostics.py': 'f4817661cd1312eefdbc004d8b067dc91e601b77',
    'cem_mh_diag_core.py': 'fc9aaaabc8d2316e8a69a8de86f6c010c103a501',
    'cem_mh_pixel_checks.py': '0120bebfa25e8667ddc86adad6ec88cd82d91920',
    'eval_lowbudget_failure_autopsy.py': 'c9cf43dc0d652ca903e5bf49511fc14ecbb0533d',
    'pusht_exact_replay.py': '58ae8c8b10b1d4f6bedd980a6204aac684b912a1',
    'planner_context_mh.py': '017e3c6ec5fdcd7149367097b9ef0c204b14b9b6',
    'pc_mh_stepwise_v1/planner_context_stepwise_mh.py': 'c8329c1e0d554d1a532abe617b294b2945e61db7',
    'jepa.py': '246b085b7ebedf5505671b0cc7cccb07d182b1ac',
}


def verify_source():
    values={}
    for rel, expected in BLOBS.items():
        data=(REPO/rel).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Upstream changed: {rel}. Do not reset local changes; reconcile first.')
        values[rel]=sha(REPO/rel)
    for rel in ('eval.py','config/eval/pusht.yaml','config/eval/solver/cem.yaml'):
        values[rel]=sha(REPO/rel)
    for rel in ('run.py','analysis_core.py'):
        values['pc_mh_failure_v1/'+rel]=sha(HERE/rel)
    return values


def safe_path(root, rel):
    rel=Path(str(rel))
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'Expected a relative project path: {rel}')
    p=(root/rel).resolve()
    if root not in p.parents:
        raise ValueError('Path escapes project data root')
    return p


def read_json(p):
    return json.loads(Path(p).read_text())


def prepare(args):
    from omegaconf import OmegaConf
    root=Path(os.environ['STABLEWM_HOME']).expanduser().resolve()
    out=Path(args.output).expanduser().resolve()
    if root not in out.parents:
        raise ValueError('Output must be strictly inside STABLEWM_HOME')
    if out.exists():
        raise FileExistsError(f'{out}: preserve existing results; use a new tag')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s not in range(42,47) for s in args.seeds):
        raise ValueError('Use unique seeds from 42..46')
    cfgpath=Path(args.baseline_config).resolve()
    completed=Path(args.stepwise_complete).resolve()
    cfg=OmegaConf.load(cfgpath)
    info=read_json(completed)
    execution_path=completed.parent/'execution.json'
    execution=read_json(execution_path)
    if execution.get('mode') != 'formal' or execution.get('baseline_config_sha256') != sha(cfgpath):
        raise RuntimeError('Stepwise was not prepared from this exact baseline RUN config')
    if info.get('mode') != 'formal':
        raise ValueError('Stepwise source must be a FORMAL completion record')
    if not cfg.mh_ald.enabled or not cfg.mh_ald.planner_context.enabled:
        raise ValueError('Baseline config is not PC-MH')
    if str(cfg.mh_ald.init_policy) != str(cfg.mh_ald.teacher_policy):
        raise ValueError('Baseline teacher and initialization differ')
    for key in ('cem_aligned','goal_sensitive','goal_coupled'):
        if bool(OmegaConf.select(cfg, f'mh_ald.{key}.enabled', default=False)):
            raise ValueError('Expected pure original PC-MH')
    baseline=str(Path(str(cfg.subdir))/f'{cfg.output_model_name}_epoch_{int(cfg.trainer.max_epochs)}')
    policies={'pcmh':baseline,'stepwise':info['policy'],'teacher':str(cfg.mh_ald.teacher_policy)}
    checkpoints={k:safe_path(root,v+'_object.ckpt') for k,v in policies.items()}
    for p in checkpoints.values():
        if not p.is_file(): raise FileNotFoundError(p)
    if checkpoints['stepwise'] != Path(info['final_checkpoint']).resolve():
        raise RuntimeError('Completion policy and checkpoint path disagree')
    hashes={k:sha(v) for k,v in checkpoints.items()}
    if hashes['stepwise'] != info.get('final_checkpoint_sha256'):
        raise RuntimeError('Stepwise checksum differs from training completion record')
    if hashes['pcmh'] == hashes['stepwise']:
        raise ValueError('The comparison resolves to the same checkpoint')
    if args.max_cases<1 or args.radius<=0: raise ValueError('Invalid case count or probe radius')
    sources=verify_source()
    h5=root/'pusht_expert_train.h5'
    if not h5.is_file(): raise FileNotFoundError(h5)
    version=importlib.metadata.version('stable-worldmodel')
    if version!='0.0.6': raise RuntimeError(f'Expected installed stable-worldmodel==0.0.6, got {version}')
    out.mkdir(parents=True, exist_ok=False)
    payload={'schema':1,'baseline_config':str(cfgpath),'baseline_config_sha256':sha(cfgpath),
             'stepwise_complete':str(completed),'stepwise_complete_sha256':sha(completed),
             'stepwise_execution':str(execution_path),'stepwise_execution_sha256':sha(execution_path),
             'root':str(root),'repo':str(REPO),'output':str(out),'policies':policies,
             'checkpoint_paths':{k:str(v) for k,v in checkpoints.items()},'checkpoint_sha256':hashes,
             'source_sha256':sources,'reference_commit':BASE,'stable_worldmodel':version,
             'dataset_path':str(h5),'dataset_size':h5.stat().st_size,'dataset_mtime_ns':h5.stat().st_mtime_ns,
             'seeds':args.seeds,'selection_seed':20261012,'max_cases':args.max_cases,
             'iterations':[0,9],'radius':args.radius,'pairs':[[0,1],[0,4],[2,4]],
             'training_used_simulator_labels':False,'optimizer_created':False,
             'cohort':'NEW explicit native-reset seeds; not historical success replication',
             'oracle':'diagnosis-only, fixed 25-step endpoint and ever-success reported separately'}
    dump(out/'protocol.json',payload)
    print(out, flush=True)


def verify_protocol(out):
    m=read_json(out/'protocol.json')
    if Path(m['repo']).resolve()!=REPO or Path(m['output']).resolve()!=out:
        raise RuntimeError('Prepared repository/output moved')
    if Path(os.environ['STABLEWM_HOME']).resolve()!=Path(m['root']):
        raise RuntimeError('STABLEWM_HOME differs from prepared run')
    if verify_source()!=m['source_sha256']: raise RuntimeError('Code changed since preparation')
    for key in ('baseline_config','stepwise_complete','stepwise_execution'):
        if sha(m[key])!=m[key+'_sha256']: raise RuntimeError(f'Source changed: {key}')
    for label,p in m['checkpoint_paths'].items():
        if sha(p)!=m['checkpoint_sha256'][label]: raise RuntimeError(f'Checkpoint changed: {label}')
    p=Path(m['dataset_path']); st=p.stat()
    if (st.st_size,st.st_mtime_ns)!=(m['dataset_size'],m['dataset_mtime_ns']):
        raise RuntimeError('Dataset file changed since preparation')
    if importlib.metadata.version('stable-worldmodel')!=m['stable_worldmodel']:
        raise RuntimeError('Installed worldmodel version changed')
    return m


def state_digest(model):
    import torch
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        t=v.detach().cpu().contiguous()
        if (t.is_floating_point() or t.is_complex()) and not torch.isfinite(t).all():
            raise RuntimeError('Nonfinite checkpoint tensor: '+k)
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_models(m, device):
    import torch
    import stable_worldmodel as swm
    models={}
    for label,policy in m['policies'].items():
        model=swm.policy.AutoCostModel(policy).to(device).eval().requires_grad_(False)
        model.interpolate_pos_encoding=True
        for attr in ('coordinate_adapter','factor_heads','inverse_dynamics_head'):
            if getattr(model,attr,None) is not None: raise RuntimeError(f'Unexpected inference module {attr}')
        state_digest(model)
        models[label]=model
    for label in ('stepwise','teacher'):
        for groupname in ('encoder','projector'):
            a=getattr(models['pcmh'],groupname).state_dict(); b=getattr(models[label],groupname).state_dict()
            if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):
                raise RuntimeError(f'Frozen latent frames differ: {label}.{groupname}')
    return models


def build_data(seed):
    from omegaconf import OmegaConf
    from eval import get_dataset
    from eval_lowbudget_failure_autopsy import _prepare_eval_rows, _build_process
    cfg=OmegaConf.load(REPO/'config/eval/pusht.yaml')
    if 'defaults' in cfg: del cfg['defaults']
    cfg.solver=OmegaConf.load(REPO/'config/eval/solver/cem.yaml')
    cfg.seed=seed; cfg.eval.num_eval=100; cfg.world.max_episode_steps=100
    cfg.solver.num_samples=100; cfg.solver.n_steps=10; cfg.solver.topk=10
    cfg.solver.batch_size=1; cfg.solver.var_scale=1.0; cfg.solver.device='cuda:0'
    cfg.solver._target_='stable_worldmodel.solver.CEMSolver'
    cfg.world.history_size=1; cfg.world.frame_skip=1
    cfg.plan_config.horizon=5; cfg.plan_config.receding_horizon=5; cfg.plan_config.action_block=5
    cfg.eval.dataset_name='pusht_expert_train'; cfg.eval.img_size=224
    cfg.eval.goal_offset_steps=25; cfg.eval.eval_budget=50
    dataset=get_dataset(cfg, cfg.eval.dataset_name)
    _, rows, ep, st=_prepare_eval_rows(cfg,dataset)
    cfg.diag_episodes=[int(x) for x in ep]; cfg.diag_start=[int(x) for x in st]
    return cfg,dataset,_build_process(cfg,dataset),rows


def close_dataset(dataset):
    f=getattr(dataset,'h5_file',None)
    if f is not None: f.close()


def survey(out,m,models):
    from omegaconf import OmegaConf
    import eval_cem_mh_diagnostics as legacy
    import stable_worldmodel as swm
    folder=out/'survey'; folder.mkdir(exist_ok=False)
    legacy.save_runtime_sources(folder,swm)
    rows=[]; index={}
    for seed in m['seeds']:
        cfg,ds,process,indices=build_data(seed)
        seed_dir=folder/f'seed{seed}'; seed_dir.mkdir()
        try:
            results={}
            for label in LABELS:
                print(f'SURVEY seed={seed} model={label}; all 100 environments',flush=True)
                r,tr=legacy.run_closed(cfg,ds,process,models[label],seed,set(),set(m['iterations']),False)
                if tr: raise RuntimeError('Unexpected traces in survey')
                dump(seed_dir/f'{label}.json',r); results[label]=r
            legacy.check_initial(results['pcmh'],results['stepwise'])
            dump(seed_dir/'config.json',OmegaConf.to_container(cfg,resolve=True))
            for i,(a,b) in enumerate(zip(results['pcmh']['successes'],results['stepwise']['successes'])):
                rows.append({'seed':seed,'eval_index':i,'dataset_row':int(indices[i]),
                             'episode_idx':int(cfg.diag_episodes[i]),'start_step':int(cfg.diag_start[i]),
                             'pcmh_success':bool(a),'stepwise_success':bool(b),'group':group(a,b)})
            for p in seed_dir.glob('*.json'): index[str(p.relative_to(out))]=sha(p)
        finally: close_dataset(ds)
    selected=choose_cases(rows,m['max_cases'],m['selection_seed'])
    dump(out/'cases.json',rows); write_csv(out/'cases.csv',rows)
    dump(out/'selected_cases.json',selected); write_csv(out/'selected_cases.csv',selected)
    for p in ('cases.json','selected_cases.json'): index[p]=sha(out/p)
    dump(out/'survey_complete.json',{'artifacts':index,'cases':len(rows),'selected':len(selected)})
    print(f'SURVEY PASS: {len(rows)} cases, {len(selected)} selected for detailed replay',flush=True)


def verify_stage(out,name):
    marker=read_json(out/(name+'_complete.json'))
    for rel,expected in marker['artifacts'].items():
        if sha(out/rel)!=expected: raise RuntimeError(f'Artifact changed: {rel}')
    return marker


def trace(out,m,models):
    import torch
    import eval_cem_mh_diagnostics as legacy
    from cem_mh_diag_core import repeat_comparison
    from cem_mh_pixel_checks import dataset_start_images
    from omegaconf import OmegaConf
    verify_stage(out,'survey')
    selected=read_json(out/'selected_cases.json')
    folder=out/'capture'; folder.mkdir(exist_ok=False); index={}
    for seed in m['seeds']:
        cases=[r for r in selected if r['seed']==seed]
        if not cases: continue
        cfg,ds,process,_=build_data(seed)
        d=folder/f'seed{seed}'; d.mkdir()
        try:
            if OmegaConf.to_container(cfg,resolve=True)!=read_json(out/f'survey/seed{seed}/config.json'):
                raise RuntimeError('Survey and trace configuration differ')
            images=dataset_start_images(ds,cfg,cases)
            reference=read_json(out/f'survey/seed{seed}/pcmh.json')
            for label in LABELS:
                print(f'TRACE seed={seed} model={label}; still all 100 environments',flush=True)
                r,traces=legacy.run_closed(cfg,ds,process,models[label],seed,
                                           {r['eval_index'] for r in cases},set(m['iterations']),True)
                dump(d/f'{label}.json',r)
                legacy.check_initial(reference,r)
                old=read_json(out/f'survey/seed{seed}/{label}.json')
                gate=repeat_comparison(old,r)
                gate['reset_counts_match']=old['native_reset_counts']==r['native_reset_counts']
                gate['passed']=gate['passed'] and gate['reset_counts_match']
                dump(d/f'{label}_repeatability.json',gate)
                p=d/f'{label}_traces.pt'
                torch.save({'env_spec':r['env_spec'],'traces':traces},p)
                if not gate['passed']: raise RuntimeError(f'Repeatability failed: seed{seed}/{label}; analysis BLOCKED')
                legacy.validate_traces(r['env_spec'],traces,cfg,process,images,d,label)
            for p in d.rglob('*'):
                if p.is_file(): index[str(p.relative_to(out))]=sha(p)
        finally: close_dataset(ds)
    dump(out/'trace_complete.json',{'artifacts':index,'passed':True})
    print('TRACE/REPEATABILITY/PIXEL/PREFIX PASS',flush=True)


def replay_sequence(raw,snapshot,actions,collect=False):
    """Fixed endpoint after 25 native steps; no auto-reset/TimeLimit, even after success."""
    import numpy as np
    from cem_mh_diag_core import restore_prefix
    restore_prefix(raw,snapshot)
    first=None; frames=[]; terminal=False
    if np.asarray(actions).shape!=(25,2): raise ValueError('Expected exactly 25 raw 2-D actions')
    for k,a in enumerate(actions,1):
        _,_,terminal,trunc,_=raw.step(a)
        if trunc: raise RuntimeError('Native truncation during fixed-horizon diagnostic')
        if terminal and first is None: first=k
        if collect and k%5==0: frames.append(np.asarray(raw.render()).copy())
    return {'ever_success':first is not None,'first_success_step':first,
            'endpoint_success':bool(terminal),'executed_steps':25,
            'official_endpoint_distance':float(raw.eval_state(raw.goal_state,raw._get_obs())[1]),
            'state':np.asarray(raw._get_obs()).copy(),'image':np.asarray(raw.render()).copy(),
            'frames':frames}


def model_rollouts(model,z0,plans,device):
    import torch
    from planner_context_mh import planner_context_rollout
    with torch.inference_mode():
        a=torch.as_tensor(plans,dtype=torch.float32,device=device)
        z=z0.to(device).expand(len(a),-1)
        return planner_context_rollout(model,z,a,3).float().cpu().numpy()


def reference_targets(teacher,z0,plans,center,observed,teacher_pred,teacher_center,device):
    import numpy as np
    import torch
    from pc_mh_stepwise_v1.planner_context_stepwise_mh import stepwise_observation_targets
    y_pc=observed[None,1:]+teacher_pred-teacher_center
    n=len(plans)
    with torch.inference_mode():
        obs=torch.as_tensor(observed,dtype=torch.float32,device=device)[None].expand(n,-1,-1)
        u=torch.as_tensor(center,dtype=torch.float32,device=device)[None].expand(n,-1,-1)
        up=torch.as_tensor(plans,dtype=torch.float32,device=device)
        y_sw=stepwise_observation_targets(teacher,obs,u,up,max_history=3).float().cpu().numpy()
    if not np.isfinite(y_pc).all() or not np.isfinite(y_sw).all():
        raise RuntimeError('Nonfinite diagnostic reference targets')
    return {'pcmh':y_pc,'stepwise':y_sw}


def save_rgb(path, image):
    import numpy as np
    from PIL import Image
    x=np.asarray(plain(image))
    if x.ndim!=3 or x.shape[-1]!=3: raise ValueError('Only real HWC RGB frames may be exported')
    if x.min()<0 or x.max()>255: raise ValueError('RGB range mismatch')
    Image.fromarray(x.astype(np.uint8)).save(path)


def analyze_population(cfg,process,models,device,tr,pop,center_data,out,key):
    import numpy as np
    import torch
    import eval_cem_mh_diagnostics as legacy
    from eval import img_transform
    transform=img_transform(cfg)
    raw,z0,goal,center,observed,tc=center_data
    plans=np.asarray(pop['candidates'],dtype=np.float32)
    predictions={l:model_rollouts(model,z0,plans,device) for l,model in models.items()}
    targets=reference_targets(models['teacher'],z0,plans,center,observed,
                              predictions['teacher'],tc,device)
    score,elite={},{}
    for l in LABELS:
        score[l],elite[l]=legacy.model_scores(models[l],tr['info'],plans,device)
        direct=((predictions[l][:,-1]-goal)**2).sum(-1)
        if not np.allclose(score[l],direct,rtol=1e-5,atol=1e-5):
            raise RuntimeError('Latent rollout and actual cost disagree: '+l)
    source=key['source']
    if not np.allclose(score[source],np.asarray(pop['source_cost'].float()),rtol=1e-5,atol=1e-5):
        raise RuntimeError('Source rescore mismatch')
    if not np.array_equal(elite[source],np.asarray(pop['source_elite'])):
        raise RuntimeError('Source GPU topk mismatch')
    replays=[replay_sequence(raw,tr['snapshot'],a) for a in legacy.decode(plans,process['action'])]
    real=legacy.encode_images(models['pcmh'],transform,[r['image'] for r in replays],device).numpy()
    oracle=np.square(real-goal).sum(-1)
    elite['encoder']=torch.topk(torch.as_tensor(oracle,device=device),10,largest=False).indices.cpu().numpy()
    ever=np.array([r['ever_success'] for r in replays]); endpoint=np.array([r['endpoint_success'] for r in replays])
    ct=torch.as_tensor(plans,device=device)
    means=torch.stack([ct[torch.as_tensor(elite[l],device=device)].mean(0) for l in (*LABELS,'encoder')]).cpu().numpy()
    if pop['iteration']==9 and not np.allclose(means[LABELS.index(source)],center,rtol=0,atol=1e-6):
        raise RuntimeError('Final elite mean is not the official returned plan')
    mean_runs={l:replay_sequence(raw,tr['snapshot'],a) for l,a in zip((*LABELS,'encoder'),legacy.decode(means,process['action']))}
    scope=np.unique(np.concatenate([elite[l] for l in (*LABELS,'encoder')]))
    rows=[]
    for l in LABELS:
        target_cost=np.square(targets[l][:,-1]-goal).sum(-1)
        d=decomposition(predictions[l][:,-1],targets[l][:,-1],real)
        near=decomposition(predictions[l][scope,-1],targets[l][scope,-1],real[scope])
        r={**key,'iteration':int(pop['iteration']),'scoring_model':l,
           **flags(score[l],oracle,ever,elite[l],mean_runs[l]['ever_success']),
           'source_top1_endpoint_success':bool(endpoint[int(np.argmin(score[l]))]),
           'encoder_top1_endpoint_success':bool(endpoint[int(np.argmin(oracle))]),
           'mean_endpoint_success':mean_runs[l]['endpoint_success'],
           'encoder_mean_ever_success':mean_runs['encoder']['ever_success'],
           'encoder_mean_endpoint_success':mean_runs['encoder']['endpoint_success'],
           'elite_union_size':len(scope), **d,
           **{'elite_union_'+k:v for k,v in near.items()},
           **{'student_'+k:v for k,v in cost_stats(score[l],oracle).items()},
           **{'reference_'+k:v for k,v in cost_stats(target_cost,oracle).items()},
           'reference_scope':'ON_POLICY_RETURNED_PLAN_CENTER; not a measured training loss'}
        rows.append(r)
    paths=out/'populations'; paths.mkdir(exist_ok=True)
    stem=f"s{key['seed']}_{source}_c{key['eval_index']}_r{key['solve_no']}_i{pop['iteration']}"
    dump(paths/(stem+'.json'),rows)
    np.savez_compressed(paths/(stem+'.npz'),candidates=plans,encoder_cost=oracle,
        ever_success=ever,endpoint_success=endpoint,first_success_step=[r['first_success_step'] or -1 for r in replays],
        official_endpoint_distance=[r['official_endpoint_distance'] for r in replays],
        real_terminal=real,pcmh_prediction=predictions['pcmh'][:,-1],stepwise_prediction=predictions['stepwise'][:,-1],
        pcmh_reference=targets['pcmh'][:,-1],stepwise_reference=targets['stepwise'][:,-1],
        pcmh_cost=score['pcmh'],stepwise_cost=score['stepwise'],
        pcmh_elite=elite['pcmh'],stepwise_elite=elite['stepwise'],encoder_elite=elite['encoder'],elite_means=means)
    if pop['iteration']==9:
        images=out/'images'/stem; images.mkdir(parents=True,exist_ok=False)
        save_rgb(images/'start_native.png',tr['snapshot']['image'])
        save_rgb(images/'goal_dataset.png',tr['raw_policy_goal'])
        for l,sc in [(source,score[source]),('encoder',oracle)]:
            save_rgb(images/f'{l}_best_endpoint.png',replays[int(np.argmin(sc))]['image'])
            save_rgb(images/f'{l}_elite_mean_endpoint.png',mean_runs[l]['image'])
        dump(images/'labels.json',{'these_are':'real replay endpoints, not model-rendered predictions',
            'initial_input_origin':'dataset frame at solve0, wrapped live frame later',
            'mean_outcomes':{l:{k:v for k,v in r.items() if k not in ('image','frames')} for l,r in mean_runs.items()}})
    return rows


def analyze_joint(cfg,process,models,device,tr,center_data,out,key,m):
    import numpy as np
    import eval_cem_mh_diagnostics as legacy
    from eval import img_transform
    raw,z0,goal,center,observed,tc=center_data
    raw_center=legacy.decode(center[None],process['action'])[0].reshape(5,10)
    probes,specs=make_joint_probes(raw_center,m['selection_seed']+10000*key['seed']+100*key['eval_index']+key['solve_no'],
                                 m['radius'],tuple(map(tuple,m['pairs'])))
    normalized=process['action'].transform(probes.reshape(-1,2)).reshape(len(probes),5,10).astype(np.float32)
    if not np.allclose(normalized[0],center,rtol=0,atol=2e-6): raise RuntimeError('Action normalization round trip failed')
    # Preserve the exact source center; other probes are diagnostic synthetic actions.
    normalized[0]=center
    replays=[replay_sequence(raw,tr['snapshot'],a.reshape(25,2),True) for a in probes]
    frames=[im for r in replays for im in r['frames']]
    actual=legacy.encode_images(models['pcmh'],img_transform(cfg),frames,device).numpy().reshape(len(probes),5,-1)
    outputs={l:model_rollouts(model,z0,normalized,device) for l,model in models.items()}
    targets=reference_targets(models['teacher'],z0,normalized,center,observed,outputs['teacher'],tc,device)
    outputs.update({'pcmh_reference':targets['pcmh'],'stepwise_reference':targets['stepwise'],'replay':actual})
    rows=[{**key,**r} for r in interaction_rows(outputs,specs)]
    folder=out/'joint'; folder.mkdir(exist_ok=True)
    stem=f"s{key['seed']}_{key['source']}_c{key['eval_index']}_r{key['solve_no']}"
    dump(folder/(stem+'.json'),{'rows':rows,'specs':specs,
        'note':'finite bounded interventions on a diagnostic replay; no training labels; small/full single controls included'})
    np.savez_compressed(folder/(stem+'.npz'),raw_actions=probes,normalized_actions=normalized,**outputs)
    return rows


def analyze(out,m,models,device):
    import gymnasium as gym
    import numpy as np
    import torch
    import eval_cem_mh_diagnostics as legacy
    from cem_mh_pixel_checks import dataset_start_images
    from eval import img_transform
    verify_stage(out,'survey'); verify_stage(out,'trace')
    selected=read_json(out/'selected_cases.json')
    work=out/'analysis'; work.mkdir(exist_ok=False)
    population_rows=[]; joint_rows=[]
    for seed in m['seeds']:
        cases=[r for r in selected if r['seed']==seed]
        if not cases: continue
        cfg,ds,process,_=build_data(seed)
        try:
            images=dataset_start_images(ds,cfg,cases)
            for source in LABELS:
                # Only trusted trace files from this run, verified by SHA256 above.
                payload=torch.load(out/f'capture/seed{seed}/{source}_traces.pt',map_location='cpu',weights_only=False)
                env=gym.make(payload['env_spec']['id'],**payload['env_spec']['kwargs'])
                try:
                    for tr in payload['traces']:
                        case=next(r for r in cases if r['eval_index']==tr['eval_index'])
                        key={**case,'source':source,'solve_no':int(tr['solve_no'])}
                        if tr['solve_no']>1: raise RuntimeError('Unexpected >2 replans for this fixed protocol')
                        print(f"REPLAY seed={seed} {source} case={tr['eval_index']} solve={tr['solve_no']}",flush=True)
                        legacy.validate_trace_execution(env.unwrapped,tr,img_transform(cfg),process['action'],images[tr['eval_index']])
                        with torch.inference_mode():
                            z0=models['pcmh'].encode({'pixels':tr['info']['pixels'].to(device)})['emb'][:,-1].cpu()
                            goal=models['pcmh'].encode({'pixels':tr['info']['goal'].to(device)})['emb'][:,-1].cpu().numpy()
                        center=np.asarray(tr['returned_actions'],dtype=np.float32)
                        cr=replay_sequence(env.unwrapped,tr['snapshot'],legacy.decode(center[None],process['action'])[0],True)
                        future=legacy.encode_images(models['pcmh'],img_transform(cfg),cr['frames'],device).numpy()
                        observed=np.concatenate([z0.numpy(),future],axis=0)
                        tc=model_rollouts(models['teacher'],z0,center[None],device)
                        center_data=(env.unwrapped,z0,goal,center,observed,tc)
                        for pop in tr['populations']:
                            population_rows+=analyze_population(cfg,process,models,device,tr,pop,center_data,work,key)
                        joint_rows+=analyze_joint(cfg,process,models,device,tr,center_data,work,key,m)
                        # Incremental tables aid inspection if a later case blocks. Completion remains separate.
                        write_csv(work/'populations.csv',population_rows)
                        write_csv(work/'joint_response.csv',joint_rows)
                finally: env.close()
        finally: close_dataset(ds)
    index={str(p.relative_to(out)):sha(p) for p in work.rglob('*') if p.is_file()}
    dump(out/'analyze_complete.json',{'artifacts':index,'population_rows':len(population_rows),
                                     'joint_rows':len(joint_rows),'selected_cases':len(selected)})
    print('REPLAY/POPULATION/TARGET/JOINT ANALYSIS PASS',flush=True)


def summarize(out):
    import csv
    import numpy as np
    verify_stage(out,'survey')
    cases=read_json(out/'cases.json'); selected=read_json(out/'selected_cases.json')
    text=['# PC-MH / Stepwise failure-analysis report','',
          'This is a NEW paired native-reset diagnostic cohort. It does not replace the historical 93.6% / 92.8% benchmark.',
          'All training checkpoints were read only. Simulator outcomes are diagnosis-only.','',
          '## Cohort (all cases, not just selected failures)']
    cohort=[]
    for seed in sorted({r['seed'] for r in cases}):
        rows=[r for r in cases if r['seed']==seed]
        d={'seed':seed,'cases':len(rows),**{l+'_success_percent':100*np.mean([r[l+'_success'] for r in rows]) for l in LABELS},
           **{g:sum(r['group']==g for r in rows) for g in ('both_fail','regressed','rescued','both_success')}}
        cohort.append(d)
        text.append(f"- Seed {seed}: PC-MH {d['pcmh_success_percent']:.1f}%, Stepwise {d['stepwise_success_percent']:.1f}%; "
                    f"both-fail={d['both_fail']}, regressions={d['regressed']}, rescues={d['rescued']}.")
    write_csv(out/'cohort_summary.csv',cohort)
    text+=['','## Selected mechanism cases',f'{len(selected)} fixed-seed, group-stratified cases. Their mechanism frequencies are NOT unbiased benchmark proportions.']
    if (out/'analyze_complete.json').is_file():
        verify_stage(out,'analyze')
        rows=list(csv.DictReader((out/'analysis/populations.csv').open()))
        only=[r for r in rows if r['scoring_model']==r['source']]
        for flag in ('predictor_top1_miss','encoder_top1_task_mismatch','no_success_in_recorded_population','elite_mean_failure'):
            count=sum(r[flag]=='True' for r in only)
            text.append(f'- {flag}: {count}/{len(only)} source-population rows. Flags overlap; not episode-level causes.')
        text+=['','## How to decide the next method experiment',
          'Read analysis/populations.csv and images for the SAME source history/population. Never pool candidate sets from different states.',
          'reference_fit_mse is an ON-POLICY diagnostic reference gap, not historical training loss. prediction_mse = reference_fit_mse + reference_target_mse + cross_term.',
          'Read joint_response.csv only on causally_joint_affected=True rows. Check real_interaction_mse before relative errors; use matched_single_prediction_mse to control total perturbation norm.',
          'A joint-calibration training experiment is not automatically recommended. First establish a recurring predictor-selection miss and useful teacher/reference information on the same cases.',
          'No successful sampled candidate does not prove global infeasibility. An encoder metric miss does not prove all predictor training useless. No significance test is inferred from these selected rows.',
          'Fixed 25-step endpoint metrics and ever-success differ. Replay continues natively after success only for this diagnostic. It does not change closed-loop evaluation.']
    else: text+=['','Detailed replay is NOT COMPLETE. Do not infer mechanism conclusions from this report.']
    text+=['','## Files','cases.csv; selected_cases.csv; analysis/populations.csv; analysis/joint_response.csv; analysis/images/.',
           'Do not use diagnostic simulator outcomes as training labels. Raw local .pt traces use pickle and must not be loaded from untrusted sources.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--output',required=True)
    p.add_argument('--baseline-config',required=True); p.add_argument('--stepwise-complete',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,47)))
    p.add_argument('--max-cases',type=int,default=12); p.add_argument('--radius',type=float,default=0.1565)
    p=sub.add_parser('run'); p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['all','survey','trace','analyze'],default='all')
    p=sub.add_parser('summarize'); p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare': return prepare(args)
    out=Path(args.output).resolve()
    if args.command=='summarize': return summarize(out)
    m=verify_protocol(out)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Run inside the allocated GPU job')
    if int(os.environ.get('SLURM_NTASKS','1'))!=1: raise RuntimeError('Use ONE process/GPU, not training DDP')
    torch.set_num_threads(2)
    models=load_models(m,'cuda:0'); before={l:state_digest(v) for l,v in models.items()}
    stages=('survey','trace','analyze') if args.stage=='all' else (args.stage,)
    try:
        for stage in stages:
            if (out/(stage+'_complete.json')).is_file():
                verify_stage(out,stage); print(f'REUSING completed {stage}',flush=True); continue
            {'survey':survey,'trace':trace,'analyze':lambda o,m,x:analyze(o,m,x,'cuda:0')}[stage](out,m,models)
        if before!={l:state_digest(v) for l,v in models.items()}: raise RuntimeError('Read-only model state changed')
        summarize(out)
        if (out/'analyze_complete.json').is_file():
            dump(out/'COMPLETE.json',{'finished_unix':time.time(),'checkpoints_unchanged':True,
                                     'note':'Diagnostic completion only, not a new planning success claim'})
        print(f'FAILURE DIAGNOSTIC {args.stage.upper()} PASS: {out}',flush=True)
    except Exception as exc:
        dump(out/'BLOCKED.json',{'stage':stage if 'stage' in locals() else args.stage,
            'exception':type(exc).__name__,'message':str(exc),'details':getattr(exc,'details',None)})
        raise


if __name__=='__main__':
    main()
