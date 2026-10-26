#!/usr/bin/env python3
"""Evaluate ONLY the original LeWM10 on an already completed paired cohort.

The existing PC-MH/anchor evaluator and its result directory are read-only.
No re-sampling, training, candidate tracing, or counterfactual replay.
"""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import hashlib
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PREVIOUS = REPO / 'pc_mh_anchor_eval_v1'
sys.path.insert(0, str(PREVIOUS))
import comparison as C
import run_eval as E
if Path(C.__file__).resolve().parent != PREVIOUS or Path(E.__file__).resolve().parent != PREVIOUS:
    raise RuntimeError('Wrong comparison/run_eval import; use the sibling evaluation directory')

OLD_BLOBS = {'run_eval.py':'68c67c4cbd2b378e58f18c66f918a2867c441386',
             'comparison.py':'6220e7d39ebbd727992b34da7396f13a7d2b943b',
             'submit.sh':'a3cfe89d2571a9846e5263382fbc0fee6cd50044'}
LABELS = ('lewm10', 'anchor', 'pcmh')
PAIRS = (('anchor','lewm10'), ('pcmh','lewm10'), ('pcmh','anchor'))


def sources():
    for name, wanted in OLD_BLOBS.items():
        raw = (PREVIOUS/name).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if actual != wanted:
            raise RuntimeError('Previous evaluation code changed: '+name)
    values = E.sources()
    values.update({str(HERE.name+'/'+n):C.sha(HERE/n) for n in ('run_parent.py','submit.sh')})
    return values


def artifact_path(folder, relative):
    folder = Path(folder).resolve(); rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Artifact path escapes its run directory')
    path = (folder/rel).resolve()
    if folder not in path.parents:
        raise ValueError('Artifact path escapes its run directory')
    return path


def validate_cohort(cases, excluded):
    expected = [(s,i) for s in C.SEEDS for i in range(C.N)]
    if [(c['seed'],c['eval_index']) for c in cases] != expected:
        raise RuntimeError('Existing cohort order/count differs; never re-sample')
    keys = ('seed','eval_index','dataset_row','episode_idx','start_step','goal_step')
    if any(type(c[k]) is not int or c[k] < 0 for c in cases for k in keys):
        raise RuntimeError('Invalid existing cohort identifiers')
    if len({c['dataset_row'] for c in cases}) != len(cases):
        raise RuntimeError('Duplicate dataset row in existing cohort')
    if any(c['goal_step'] != c['start_step']+25 or c['episode_idx'] in excluded for c in cases):
        raise RuntimeError('Existing cohort violates goal/exclusion protocol')
    for seed in C.SEEDS:
        rows = [c['dataset_row'] for c in cases if c['seed']==seed]
        if rows != sorted(rows):
            raise RuntimeError('Existing cohort is not sorted within seed')


def validate_config(d, cases, seed, budget):
    n, iterations, k = C.BUDGETS[budget]
    checks = ((d['seed'],seed), (d['eval']['num_eval'],C.N),
        (d['eval']['eval_budget'],50), (d['eval']['goal_offset_steps'],25),
        (d['eval']['dataset_name'],'pusht_expert_train'), (d['eval']['img_size'],224),
        (d['world']['num_envs'],C.N), (d['world']['max_episode_steps'],100),
        (d['world']['history_size'],1), (d['world']['frame_skip'],1),
        (d['plan_config']['horizon'],5), (d['plan_config']['receding_horizon'],5),
        (d['plan_config']['action_block'],5), (d['solver']['num_samples'],n),
        (d['solver']['n_steps'],iterations), (d['solver']['topk'],k),
        (d['solver']['batch_size'],1), (d['solver']['seed'],seed),
        (d['solver']['device'],'cuda:0'), (d['solver']['var_scale'],1.0),
        (d['solver']['_target_'],'stable_worldmodel.solver.CEMSolver'),
        (d['diag_episodes'],[c['episode_idx'] for c in cases]),
        (d['diag_start'],[c['start_step'] for c in cases]))
    if any(a != b for a,b in checks):
        raise RuntimeError(f'Existing configuration/cohort mismatch: B{budget}, seed={seed}')


def source_bundle(source):
    """Validate old completed results without calling any writer or evaluator."""
    source = Path(source).resolve()
    meta = E.verify(source)
    complete = C.read(source/'COMPLETE.json')
    if (complete.get('runs') != 20 or complete.get('paired_cases_per_budget') != 500
            or complete.get('checkpoints_unchanged') is not True
            or complete.get('initial_conditions_verified') is not True):
        raise RuntimeError('Source is not a complete verified 20-run comparison')
    required = {'protocol.json','cohort.json','cohort.csv','summary.csv','model_audit.json'}
    required |= {f'runs/B{b}_seed{s}_{label}.json'
                 for b in C.BUDGETS for s in C.SEEDS for label in C.LABELS}
    artifacts = complete['artifacts']
    if not required <= artifacts.keys():
        raise RuntimeError('Source completion is missing required artifacts')
    for rel,wanted in artifacts.items():
        if C.sha(artifact_path(source,rel)) != wanted:
            raise RuntimeError('Completed source artifact changed: '+rel)
    cohort = C.sealed_read(source/'cohort.json')
    if cohort['protocol_sha256'] != C.sha(source/'protocol.json'):
        raise RuntimeError('Source cohort/protocol binding mismatch')
    cases = cohort['cases']; validate_cohort(cases, meta['excluded_episodes'])
    cells = {}
    for seed in C.SEEDS:
        part = [c for c in cases if c['seed']==seed]
        for budget in C.BUDGETS:
            for label in C.LABELS:
                cell = E.load_cell(source,budget,seed,label)
                if cell['policy'] != meta['policies'][label]:
                    raise RuntimeError('Source cell/checkpoint policy mismatch')
                validate_config(cell['config'],part,seed,budget)
                cells[budget,seed,label] = cell
            if cells[budget,seed,'pcmh']['config'] != cells[budget,seed,'anchor']['config']:
                raise RuntimeError('Saved two-model configurations differ')
    return meta, complete, cases, cells


def parent_info(meta):
    """Resolve the exact shared training parent, not a guessed checkpoint."""
    root = Path(meta['root'])
    _, training, paths, policies, _ = E.training_inputs(root, meta['anchor_run'])
    if {k:str(v) for k,v in paths.items()} != meta['checkpoints'] or policies != meta['policies']:
        raise RuntimeError('Completed source models differ from training provenance')
    parent = E.within(root,training['parent_checkpoint'])
    policy = E.checkpoint_policy(root,parent)
    wanted = training['parent_checkpoint_sha256']
    if policy != 'lewm_epoch_10' or meta['input_sha256'].get(str(parent)) != wanted:
        raise RuntimeError('Expected the exact shared LeWM10 parent')
    if C.sha(parent) != wanted:
        raise RuntimeError('Parent checkpoint changed')
    return {'path':str(parent),'policy':policy,'sha256':wanted}


def prepare(args):
    root = Path(os.environ['STABLEWM_HOME']).resolve()
    source, out = E.within(root,args.source), E.within(root,args.output)
    if (out.exists() or out.parent != root or source == out or source in out.parents
            or out in source.parents or args.node not in E.NODES):
        raise ValueError('Use a fresh sibling output and an allowed node; never overwrite source')
    code = sources()
    old, _, _, _ = source_bundle(source)
    parent = parent_info(old)
    out.mkdir(parents=False,exist_ok=False)
    (out/'runs').mkdir(); (out/'errors').mkdir()
    # Byte-for-byte snapshots. The cohort still belongs to the OLD protocol.
    for original, name in (('COMPLETE.json','source_COMPLETE.json'),
                           ('protocol.json','source_protocol.json'),
                           ('cohort.json','cohort.json'), ('cohort.csv','cohort.csv')):
        (out/name).write_bytes((source/original).read_bytes())
    meta = {'schema':1,'root':str(root),'repo':str(REPO),'output':str(out),'source':str(source),
        'node':args.node,'created_unix':time.time(),'parent':parent,'code_sha256':code,
        'versions':E.versions(),'source_complete_sha256':C.sha(source/'COMPLETE.json'),
        'source_protocol_sha256':C.sha(source/'protocol.json'),
        'cohort_sha256':C.sha(source/'cohort.json'),'cohort_csv_sha256':C.sha(source/'cohort.csv'),
        'seeds':list(C.SEEDS),'budgets':{str(k):list(v) for k,v in C.BUDGETS.items()},
        'new_runs':10,'reuse_runs':20,'new_model':'lewm10','training':False,
        'resampled':False,'trace_enabled':False,'counterfactual_replay':False}
    C.dump(out/'protocol.json',meta)
    print('PREPARED LEWM10-ONLY EVALUATION: '+str(out),flush=True)
    print('Reuse 20 completed PC-MH/anchor runs; execute only 10 LeWM10 runs.',flush=True)


def verify(out):
    out = Path(out).resolve(); m = C.read(out/'protocol.json')
    if (str(out)!=m['output'] or str(REPO)!=m['repo']
            or str(Path(os.environ['STABLEWM_HOME']).resolve())!=m['root']):
        raise RuntimeError('Prepared locations changed')
    if sources()!=m['code_sha256'] or E.versions()!=m['versions']:
        raise RuntimeError('Code/environment changed; do not mix evaluations')
    source = Path(m['source'])
    for path,wanted in ((source/'COMPLETE.json',m['source_complete_sha256']),
        (source/'protocol.json',m['source_protocol_sha256']),
        (source/'cohort.json',m['cohort_sha256']),
        (out/'source_COMPLETE.json',m['source_complete_sha256']),
        (out/'source_protocol.json',m['source_protocol_sha256']),
        (out/'cohort.json',m['cohort_sha256']), (out/'cohort.csv',m['cohort_csv_sha256'])):
        if C.sha(path)!=wanted: raise RuntimeError('Source/snapshot changed: '+str(path))
    old,_,cases,cells = source_bundle(source)
    if parent_info(old)!=m['parent']:
        raise RuntimeError('Parent identity changed')
    if (m['seeds']!=list(C.SEEDS) or m['budgets']!={str(k):list(v) for k,v in C.BUDGETS.items()}
            or m['new_runs']!=10 or m['reuse_runs']!=20):
        raise RuntimeError('Evaluation matrix changed')
    return m, old, cases, cells


def result_path(out, budget, seed):
    return Path(out)/'runs'/f'B{budget}_seed{seed}_lewm10.json'


def identity(out, m, budget, seed):
    return {'protocol_sha256':C.sha(Path(out)/'protocol.json'),
            'source_complete_sha256':m['source_complete_sha256'],
            'cohort_sha256':m['cohort_sha256'],'parent_sha256':m['parent']['sha256'],
            'budget':budget,'seed':seed,'label':'lewm10'}


def load_parent_cell(out,m,budget,seed,reference):
    cell = C.sealed_read(result_path(out,budget,seed))
    if (cell['identity']!=identity(out,m,budget,seed) or cell['policy']!=m['parent']['policy']
            or cell['config']!=reference['config'] or cell.get('initial_conditions_verified') is not True):
        raise RuntimeError('Saved parent cell provenance/config mismatch')
    C.successes(cell['result'])
    return cell


def validate_parent_pair(result, cells, budget, seed):
    for label in C.LABELS:
        E.check_pair(cells[budget,seed,label]['result'],result)
    # The budgets share initial conditions, not trajectories or success outcomes.
    E.check_pair(cells[1000,seed,'pcmh']['result'],result)


def evaluate_missing(out,m,cases,cells,dataset,process,model,make_cfg):
    """The only evaluation loop: one parent call per missing budget/seed."""
    before = E.model_digest(model)
    for seed in C.SEEDS:
        for budget in C.BUDGETS:
            ref = cells[budget,seed,'pcmh']; path = result_path(out,budget,seed)
            if path.exists():
                cell = load_parent_cell(out,m,budget,seed,ref)
                validate_parent_pair(cell['result'],cells,budget,seed)
                print(f'REUSE LeWM10 B{budget} seed={seed}',flush=True)
                continue
            cfg = make_cfg(copy.deepcopy(ref['config']))
            print(f'EVAL LeWM10 B{budget} seed={seed}; unchanged 100-case cohort',flush=True)
            start = time.monotonic()
            result = E.run_one(cfg,dataset,process,model,seed)
            validate_parent_pair(result,cells,budget,seed)
            if E.model_digest(model)!=before:
                raise RuntimeError('LeWM10 state changed during inference')
            C.sealed_write(path,{'identity':identity(out,m,budget,seed),'policy':m['parent']['policy'],
                'config':copy.deepcopy(ref['config']),'result':result,
                'elapsed_seconds':time.monotonic()-start,'initial_conditions_verified':True})
            print(f'FINISHED LeWM10 B{budget} seed={seed}: {100*C.successes(result).mean():.2f}%; PAIRED RESET PASS',flush=True)
    if E.model_digest(model)!=before:
        raise RuntimeError('LeWM10 state changed')
    return before


def execute(out,m,old,cases,cells):
    # Preserve the successful evaluator's import order; no custom resolver registration.
    import stable_pretraining  # noqa: F401
    import torch
    import stable_worldmodel as swm
    from omegaconf import OmegaConf as O
    from eval import get_dataset
    from eval_lowbudget_failure_autopsy import _build_process
    cfg0 = O.create(copy.deepcopy(cells[1000,C.SEEDS[0],'pcmh']['config']))
    dataset = get_dataset(cfg0,cfg0.eval.dataset_name)
    try:
        col = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
        order = np.argsort([c['dataset_row'] for c in cases]); ordered=[cases[i] for i in order]
        data = dataset.get_row_data(np.asarray([c['dataset_row'] for c in ordered]))
        if (not np.array_equal(np.asarray(data[col]).reshape(-1),[c['episode_idx'] for c in ordered])
                or not np.array_equal(np.asarray(data['step_idx']).reshape(-1),[c['start_step'] for c in ordered])):
            raise RuntimeError('Frozen cohort no longer matches dataset rows')
        process = _build_process(cfg0,dataset)
        model = swm.policy.AutoCostModel(m['parent']['policy']).to('cuda:0').eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        for attr in ('coordinate_adapter','factor_heads','inverse_dynamics_head'):
            if getattr(model,attr,None) is not None: raise RuntimeError('Unexpected parent component: '+attr)
        # Load one reference on CPU only for frame/identity checks; never evaluate it.
        reference = swm.policy.AutoCostModel(old['policies']['anchor']).cpu().eval().requires_grad_(False)
        expected = C.read(Path(m['source'])/'model_audit.json')['initial_state_sha256']['anchor']
        if E.model_digest(reference)!=expected: raise RuntimeError('Reference model state differs from old evaluation')
        for name in ('encoder','projector'):
            a,b = getattr(model,name).state_dict(),getattr(reference,name).state_dict()
            if a.keys()!=b.keys() or any(not torch.equal(a[k].cpu(),b[k].cpu()) for k in a):
                raise RuntimeError('Parent/reference frozen visual frame differs: '+name)
        del reference
        before = evaluate_missing(out,m,cases,cells,dataset,process,model,O.create)
        C.dump(out/'model_audit.json',{'parent_state_sha256':before,'parent_unchanged':True,
                                      'same_frozen_visual_frame':True,'evaluated_models':['lewm10']})
    finally:
        if getattr(dataset,'h5_file',None) is not None: dataset.h5_file.close()


def tables(rows):
    """Use exactly the old paired episode-cluster bootstrap for all three pairs."""
    per_seed, stats, effects = [], [], []
    for a,b in PAIRS:
        mapped = [{**r,'pcmh_success':r[a+'_success'],'anchor_success':r[b+'_success']} for r in rows]
        by_seed, totals = C.summary(mapped)
        if (a,b)==PAIRS[0]:
            for q in by_seed:
                part=[r for r in rows if r['budget']==q['budget'] and r['seed']==q['seed']]
                per_seed.append({'budget':q['budget'],'seed':q['seed'],'n':len(part),
                    **{k+'_success_pct':float(100*np.mean([r[k+'_success'] for r in part])) for k in LABELS}})
        for r in totals:
            effects.append({'budget':r['budget'],'model_a':a,'model_b':b,'n':r['n'],
                'unique_dataset_episodes':r['unique_dataset_episodes'],
                'a_minus_b_pp':r['pcmh_minus_anchor_pp'],'a_only':r['pcmh_only'],'b_only':r['anchor_only'],
                'both_success':r['both_success'],'both_fail':r['both_fail'],
                'episode_cluster_ci95_low_pp':r['episode_cluster_ci95_low_pp'],
                'episode_cluster_ci95_high_pp':r['episode_cluster_ci95_high_pp']})
    for budget in C.BUDGETS:
        part=[r for r in rows if r['budget']==budget]; q=[r for r in per_seed if r['budget']==budget]
        row={'budget':budget,'n':len(part),'unique_dataset_episodes':len({r['episode_idx'] for r in part})}
        for k in LABELS:
            row[k+'_mean_pct']=float(100*np.mean([r[k+'_success'] for r in part]))
            row[k+'_seed_sd_pp']=float(np.std([r[k+'_success_pct'] for r in q],ddof=1))
        for a,b in PAIRS:
            row[a+'_minus_'+b+'_pp']=next(r['a_minus_b_pp'] for r in effects if (r['budget'],r['model_a'],r['model_b'])==(budget,a,b))
        stats.append(row)
    return per_seed,stats,effects


def summarize(out,m,cases,cells,check_initial=True):
    rows=[]; artifacts={}
    for seed in C.SEEDS:
        part=[c for c in cases if c['seed']==seed]
        for budget in C.BUDGETS:
            parent=load_parent_cell(out,m,budget,seed,cells[budget,seed,'pcmh'])['result']
            if check_initial: validate_parent_pair(parent,cells,budget,seed)
            values={'lewm10':C.successes(parent),
                    **{k:C.successes(cells[budget,seed,k]['result']) for k in C.LABELS}}
            rows.extend({**c,'budget':budget,**{k+'_success':int(values[k][i]) for k in LABELS}}
                        for i,c in enumerate(part))
            path=result_path(out,budget,seed);artifacts[str(path.relative_to(out))]=C.sha(path)
    per_seed,stats,effects=tables(rows)
    # Recomputed PC-MH/anchor results must reproduce the saved result table exactly (up to roundoff).
    with (Path(m['source'])/'summary.csv').open(newline='') as f: original=list(csv.DictReader(f))
    for old in original:
        b=int(old['budget']); r=next(x for x in effects if (x['budget'],x['model_a'],x['model_b'])==(b,'pcmh','anchor'))
        for new_key,old_key in (('a_minus_b_pp','pcmh_minus_anchor_pp'),('a_only','pcmh_only'),('b_only','anchor_only'),
            ('both_success','both_success'),('both_fail','both_fail'),
            ('episode_cluster_ci95_low_pp','episode_cluster_ci95_low_pp'),('episode_cluster_ci95_high_pp','episode_cluster_ci95_high_pp')):
            if not np.isclose(r[new_key],float(old[old_key]),atol=1e-10,rtol=0):
                raise RuntimeError('Old comparison changed during merge: '+old_key)
    for name,data in (('paired_cases.csv',rows),('per_seed.csv',per_seed),('summary.csv',stats),('pairwise.csv',effects)):
        C.write_csv(out/name,data)
    C.dump(out/'summary.json',{'models':stats,'paired_effects':effects,
        'uncertainty':'Descriptive dataset-episode cluster percentile bootstrap, 5000 draws per pair; fixed checkpoints; no training-seed uncertainty.',
        'inference_note':'Unadjusted descriptive intervals for multiple contrasts. No automatic significance/equivalence claim.'})
    text=['# LeWM10 / PC-anchor-only / PC-MH', '',
        'Only LeWM10 was newly evaluated (10 runs). The completed 20 other runs were reused unchanged.',
        'Identical frozen cohort, goals, native reset protocol and budgets; no re-sampling.', '',
        '| Budget | LeWM10 | Anchor-only | PC-MH | Anchor - LeWM10 | PC-MH - LeWM10 | PC-MH - Anchor |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        text.append(f"| B{r['budget']} | {r['lewm10_mean_pct']:.2f}% | {r['anchor_mean_pct']:.2f}% | {r['pcmh_mean_pct']:.2f}% | {r['anchor_minus_lewm10_pp']:+.2f} pp | {r['pcmh_minus_lewm10_pp']:+.2f} pp | {r['pcmh_minus_anchor_pp']:+.2f} pp |")
    text += ['', 'B1000 primary; B3000 secondary. The same 500 starts are reused across budgets.',
        'pairwise.csv includes all three paired contrasts with win/loss counts and descriptive episode-cluster intervals.',
        'One checkpoint per model; intervals do NOT quantify training-initialization uncertainty and are not multiplicity-adjusted.',
        'Parent comparisons include ten additional Stage-2 epochs; they do not isolate extra training from a specific calibration mechanism.',
        'This is the existing 142-146 cohort, not the historical benchmark and not a held-out-pretraining claim.',
        'No training, candidate-population capture or counterfactual replay. Timing includes reset/recording checks.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    for name in ('protocol.json','source_COMPLETE.json','source_protocol.json','cohort.json','cohort.csv',
                 'paired_cases.csv','per_seed.csv','summary.csv','pairwise.csv','summary.json','REPORT.md','model_audit.json'):
        artifacts[name]=C.sha(out/name)
    print('\n'.join(text),flush=True)
    return artifacts


def run(out):
    out=Path(out).resolve()
    if (out/'COMPLETE.json').exists(): raise RuntimeError('Already complete; read REPORT.md')
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            m,old,cases,cells=verify(out)
            node=os.environ.get('SLURMD_NODENAME',socket.gethostname().split('.')[0])
            job=os.environ.get('SLURM_JOB_ID')
            if not job or node!=m['node'] or node not in E.NODES or int(os.environ.get('SLURM_NTASKS','1'))!=1:
                raise RuntimeError('Use the one-GPU Slurm launcher on the prepared node')
            import torch
            torch.cuda.init()
            if torch.cuda.device_count()!=1: raise RuntimeError('Need one Slurm-visible GPU; do not overwrite CUDA_VISIBLE_DEVICES')
            torch.cuda.set_device(0);torch.ones(1,device='cuda:0').sum().item();torch.set_num_threads(2)
            C.dump(out/'STATUS.json',{'state':'RUNNING','job_id':job})
            execute(out,m,old,cases,cells)
            verify(out)  # Original results, model files and source remain unchanged.
            artifacts=summarize(out,m,cases,cells)
            C.dump(out/'COMPLETE.json',{'job_id':job,'finished_unix':time.time(),
                'new_runs':10,'reused_runs':20,'paired_cases_per_budget':500,
                'source_unchanged':True,'checkpoints_unchanged':True,'initial_conditions_verified':True,
                'source_complete_sha256':m['source_complete_sha256'],'artifacts':artifacts})
            C.dump(out/'STATUS.json',{'state':'COMPLETE','job_id':job})
            print('LEWM10 SUPPLEMENT / THREE-MODEL COMPARISON PASS: '+str(out),flush=True)
        except Exception as exc:
            error={'state':'FAILED','job_id':os.environ.get('SLURM_JOB_ID'),'error':type(exc).__name__,
                   'message':str(exc),'traceback':traceback.format_exc()}
            C.dump(out/'errors'/f'job_{os.environ.get("SLURM_JOB_ID")}_{time.time_ns()}.json',error)
            C.dump(out/'STATUS.json',error)
            raise


def read_report(out):
    out=Path(out).resolve(); complete=C.read(out/'COMPLETE.json')
    if complete.get('new_runs')!=10 or complete.get('reused_runs')!=20:
        raise RuntimeError('Incomplete supplemental evaluation')
    for rel,wanted in complete['artifacts'].items():
        if C.sha(artifact_path(out,rel))!=wanted: raise RuntimeError('Result artifact changed: '+rel)
    print((out/'REPORT.md').read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--source',required=True);q.add_argument('--output',required=True)
    q.add_argument('--node',choices=E.NODES,required=True)
    for cmd in ('run','verify','summarize'):sub.add_parser(cmd).add_argument('--output',required=True)
    sub.add_parser('check-source')
    a=p.parse_args()
    if a.command=='prepare':prepare(a)
    elif a.command=='run':run(a.output)
    elif a.command=='verify':verify(a.output);print('LEWM10 SOURCE/COHORT/PARENT CHECK PASS')
    elif a.command=='check-source':sources();print('PINNED LEWM10 EVALUATION SOURCE CHECK PASS')
    else:read_report(a.output)


if __name__=='__main__':main()
