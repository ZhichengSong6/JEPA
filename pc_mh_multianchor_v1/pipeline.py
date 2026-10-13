#!/usr/bin/env python3
"""Prepare frozen residual memory, episode-held-out gate, official CEM query pool.
Usage: python pc_mh_multianchor_v1/pipeline.py --base-config ... --output ... [--smoke]
The formal student is blocked unless both held-out audit and query coverage pass.
"""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from core import (Memory, episode_partition, save_npz, load_npz, dump, finite,
                  pair_records, select_gate, accepted, masked_metrics, correct_queries)
import runtime as rt


def extract(ds,indices,teacher,device):
    from planner_context_mh import planner_context_rollout
    parts={k:[] for k in ('z','u','future','teacher','residual','episode','start','clip')}
    invalid=0
    for s in range(0,len(indices),16):
        ix=indices[s:s+16]
        items=[ds[int(i)] for i in ix]
        pixels=torch.stack([torch.as_tensor(x['pixels']) for x in items])
        actions=torch.stack([torch.as_tensor(x['action']) for x in items]).float()
        if pixels.shape[1]!=8 or actions.shape[1:]!=(8,10): raise ValueError('Unexpected dataset window/layout')
        good=torch.isfinite(actions).all((1,2));invalid+=int((~good).sum())
        if not good.any(): continue
        ix=ix[good.numpy()];pixels=pixels[good].to(device);u=actions[good,2:7].to(device)
        with torch.inference_mode():
            emb=teacher.encode({'pixels':pixels})['emb'].float()
            if emb.shape[1:]!=(8,192): raise ValueError('Expected LeWM latent [B,8,192]')
            pred=planner_context_rollout(teacher,emb[:,2],u,3).float()
            arrays={'z':emb[:,2].cpu().numpy(),'u':u.cpu().numpy(),
                    'future':emb[:,3:8].cpu().numpy(),'teacher':pred.cpu().numpy(),
                    'residual':(emb[:,3:8]-pred).cpu().numpy()}
        for k,v in arrays.items(): finite(v,k);parts[k].append(v)
        clips=np.asarray([ds.clip_indices[int(i)] for i in ix],dtype=np.int64)
        parts['episode'].append(clips[:,0]);parts['start'].append(clips[:,1]);parts['clip'].append(ix)
        if s%1024==0: print(f'CACHE {s}/{len(indices)} invalid={invalid}',flush=True)
    if not parts['z']: raise ValueError('No finite windows')
    result={k:np.concatenate(v) for k,v in parts.items()}
    if len(np.unique(np.column_stack([result['episode'],result['start']]),axis=0))!=len(result['z']):
        raise ValueError('Duplicate transition starts')
    return result,invalid


def cache(out,cfg,m,t,device):
    ds,ti,vi,mean,std=rt.data(cfg)
    try:
        clips=np.asarray(ds.clip_indices,dtype=np.int64)
        if clips.shape!=(len(ds),2): raise ValueError('Unexpected clip index layout')
        split=episode_partition(clips[ti,0],m['seed'])
        save_npz(out/'original_split.npz',train=ti,validation=vi)
        save_npz(out/'normalization.npz',mean=mean,std=std)
        counts=[int(.6*m['samples']),int(.2*m['samples'])]
        counts.append(m['samples']-sum(counts))
        rng=np.random.default_rng(m['seed']+1);report={}
        files=['original_split.npz','normalization.npz','parent_state.json']
        # No diagnostic outputs are loaded anywhere in this module.
        for (name,eps),count in zip(split.items(),counts):
            candidates=ti[np.isin(clips[ti,0],eps)]
            if len(candidates)<count: raise ValueError('Insufficient original train windows: '+name)
            ix=np.sort(rng.choice(candidates,size=count,replace=False))
            result,rejected=extract(ds,ix,t,device)
            if len(result['z'])<count*.95: raise ValueError('More than 5% invalid actions; investigate data')
            save_npz(out/(name+'.npz'),**result);files.append(name+'.npz')
            report[name]={'windows':len(result['z']),'invalid':rejected,'episodes':len(np.unique(result['episode']))}
        dump(out/'split_report.json',{'counts':report,'episode_ids':{k:v.tolist() for k,v in split.items()},
             'note':'Resid estimator held-out episodes ONLY; not unseen by original LeWM; original validation CLIPS are never inputs to the new memory'})
        files.append('split_report.json')
        rt.seal(out,'cache_complete',files)
    finally: rt.close_data(ds)


def audit(out,m,device):
    bank,cal,aud=[load_npz(out/(n+'.npz')) for n in ('bank','calibration','audit')]
    for a,b in ((bank,cal),(bank,aud),(cal,aud)):
        if np.intersect1d(a['episode'],b['episode']).size: raise ValueError('Internal episode leakage')
    mem=Memory(bank,device,k=m['k'])
    save_npz(out/'scales.npz',**mem.scales)
    gates=[];cal_report=[];audit_report=[];audit_arrays={};nmin=8 if m['smoke'] else 64
    total_b=total_a=total_n=0
    for h in range(1,6):
        print(f'CALIBRATION horizon={h}',flush=True)
        c=pair_records(mem,cal,h);g=select_gate(c,min_rows=nmin);gates.append(g)
        cal_report.append(g)
        # Thresholds are now frozen. Do not search again on audit results.
        a=pair_records(mem,aud,h);mask=accepted(a['stats'],g)
        metric=masked_metrics(a,mask);audit_report.append(metric)
        audit_arrays.update({f'h{h}_{k}':v for k,v in a.items()});audit_arrays[f'h{h}_accepted']=mask
        if metric['n']:
            total_b+=float(a['before'][mask].sum());total_a+=float(a['after'][mask].sum());total_n+=metric['n']
        print(f'AUDIT h={h}: {metric}',flush=True)
    terminal=audit_report[-1];overall_gain=1-total_a/total_b if total_b>1e-12 else None
    passed=(overall_gain is not None and overall_gain>=.1 and terminal['n']>=nmin
            and terminal['coverage']>=.2 and terminal['gain'] is not None and terminal['gain']>=.1)
    report={'passed':bool(passed),'calibration':cal_report,'audit':audit_report,
            'aggregate_accepted_gain':overall_gain,'accepted_cells':total_n,
            'requirements':{'aggregate_gain':.1,'terminal_gain':.1,'terminal_coverage':.2,'min_terminal_pairs':nmin},
            'no_claim':'Descriptive development gate, not significance or counterfactual correctness'}
    dump(out/'gates.json',gates);dump(out/'audit_report.json',report)
    save_npz(out/'audit_pairs.npz',**audit_arrays)
    rt.seal(out,'audit_complete',['scales.npz','gates.json','audit_report.json','audit_pairs.npz'])
    return passed


def queries(out,cfg,m,t,device):
    import stable_worldmodel as swm
    from planner_context_mh import planner_context_rollout
    from queries import generate,validate_pixel_cost
    bank=load_npz(out/'bank.npz');norm=load_npz(out/'normalization.npz')
    mem=Memory(bank,device,k=m['k'],scales=load_npz(out/'scales.npz'))
    gates=json.loads((out/'gates.json').read_text())
    # Only memory-partition episodes produce student query supervision. Audit
    # and calibration trajectories never become student query anchors here.
    if len(bank['z'])<m['queries']: raise ValueError('Not enough query anchors')
    ids=np.random.default_rng(m['seed']+2).choice(len(bank['z']),m['queries'],replace=False)
    ds,_,_,_,_=rt.data(cfg)
    outq=out/'query_chunks';outq.mkdir()
    (out/'installed_cem.py').write_text(inspect.getsource(swm.solver.CEMSolver))
    files=['installed_cem.py'];all_masks=[];semantics=None
    try:
        for offset in range(0,len(ids),32):
            fields={k:[] for k in ('z','u0','u','old_target','target','mask','gate_stats','episode','clip','kind')}
            for counter,b in enumerate(ids[offset:offset+32],offset):
                q,kinds,p,c=generate(t,bank['z'][b],bank['future'][b,-1],m['seed']+counter,device)
                if semantics is None:
                    semantics=validate_pixel_cost(t,ds[int(bank['clip'][b])],p[:8],bank['z'][b],bank['future'][b,-1],device)
                    print(f'OFFICIAL CEM / PIXEL COST CHECK PASS: max_abs={semantics}',flush=True)
                z=np.repeat(bank['z'][b:b+1],4,0);u0=np.repeat(bank['u'][b:b+1],4,0)
                ep=np.repeat(bank['episode'][b:b+1],4,0)
                with torch.inference_mode():
                    pred=planner_context_rollout(t,torch.as_tensor(z,device=device),torch.as_tensor(q,device=device),3).float().cpu().numpy()
                old=bank['future'][b:b+1]+(pred-bank['teacher'][b:b+1])
                mean=np.tile(norm['mean'].reshape(-1),5);std=np.tile(norm['std'].reshape(-1),5)
                new,mask,stats=correct_queries(mem,z,u0,q,ep,gates,old,q*std+mean,u0*std+mean)
                values={'z':z,'u0':u0,'u':q,'old_target':old,'target':new,'mask':mask,'gate_stats':stats,
                        'episode':ep,'clip':np.repeat(bank['clip'][b:b+1],4,0),'kind':np.asarray(kinds)}
                for k,v in values.items(): fields[k].append(v)
                all_masks.append(mask)
            path=f'query_chunks/chunk_{offset:05d}.npz'
            save_npz(out/path,**{k:np.concatenate(v) for k,v in fields.items()});files.append(path)
            print(f'QUERY CACHE {min(offset+32,len(ids))}/{len(ids)}',flush=True)
    finally: rt.close_data(ds)
    mask=np.concatenate(all_masks)
    coverage=float(mask[:,-1].mean())
    report={'anchors':len(ids),'queries':len(mask),'per_horizon_coverage':mask.mean(0).tolist(),
            'terminal_coverage':coverage,'required_terminal_coverage':.2,'passed':coverage>=.2,
            'pixel_cost_max_abs':semantics,'candidate_source':'Frozen LeWM10, official installed CEM N100 I10 K10',
            'out_of_domain':'No action clipping; unsupported causal prefixes get mask=False'}
    dump(out/'query_report.json',report);files.append('query_report.json')
    rt.seal(out,'queries_complete',files)
    return report['passed']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=Path(args.output).resolve()
    cfg,m=rt.initialize(out,args.base_config,args.smoke)
    if not torch.cuda.is_available(): raise RuntimeError('Use a single allocated GPU for artifact preparation')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    try:
        t=rt.teacher(cfg,'cuda:0');before=rt.digest(t)
        dump(out/'parent_state.json',{'state_sha256':before})
        from queries import generate, validate_pixel_cost
        ds,ti,_,_,_=rt.data(cfg)
        try:
            item=ds[int(ti[0])]
            with torch.inference_mode():
                emb=t.encode({'pixels':torch.as_tensor(item['pixels'])[None].to('cuda:0')})['emb'].float().cpu().numpy()[0]
            _,_,plans,_=generate(t,emb[2],emb[7],m['seed'],'cuda:0')
            err=validate_pixel_cost(t,item,plans[:8],emb[2],emb[7],'cuda:0')
            dump(out/'native_cem_precheck.json',{'passed':True,'pixel_cost_max_abs':err})
            print('NATIVE CEM / PIXEL COST PREFLIGHT PASS',flush=True)
        finally: rt.close_data(ds)
        cache(out,cfg,m,t,'cuda:0')
        apass=audit(out,m,'cuda:0')
        if not apass and not args.smoke:
            dump(out/'REJECTED.json',{'stage':'residual_audit','reason':'Frozen independent gate failed; no formal student training authorized'})
            print('RESIDUAL AUDIT REJECTED: inspect audit_report.json; no thresholds changed',flush=True)
            return
        qpass=queries(out,cfg,m,t,'cuda:0')
        if rt.digest(t)!=before: raise RuntimeError('Frozen teacher changed')
        ready=bool(apass and qpass and not args.smoke)
        dump(out/('READY.json' if ready else 'SMOKE_COMPLETE.json' if args.smoke else 'REJECTED.json'),
             {'formal_ready':ready,'audit_pass':bool(apass),'query_coverage_pass':bool(qpass),'smoke':args.smoke,
              'parent_unchanged':True,'note':'No planning improvement is claimed'})
        print(f'PREPARATION {"READY" if ready else "SMOKE COMPLETE" if args.smoke else "REJECTED"}: {out}',flush=True)
    except Exception as exc:
        dump(out/'BLOCKED.json',{'error':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__': main()
