#!/usr/bin/env python3
"""Read-only PC-MH / PC-anchor-only paired CEM evaluation. No Trainer or replay."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import numpy as np
from comparison import (SEEDS, BUDGETS, LABELS, N, sha, read, dump, object_hash,
                        sealed_write, sealed_read, write_csv, excluded_episodes,
                        select_cohort, successes, paired_rows, summary)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault('MUJOCO_GL', 'egl')
NODES = ('4090node3','4090node2','3090node1','3090node2','3090node3')
PINNED = {
    'eval.py':'81b95b03b58c707b69968f789dbf76281a8ac5a4',
    'config/eval/pusht.yaml':'6584ef029381214eca1ff71a6dbc2964d69870e4',
    'config/eval/solver/cem.yaml':'8d24fdab34e0dcf45777d418663f8077a433fb98',
    'eval_cem_mh_diagnostics.py':'f4817661cd1312eefdbc004d8b067dc91e601b77',
    'eval_lowbudget_failure_autopsy.py':'c9cf43dc0d652ca903e5bf49511fc14ecbb0533d',
    'cem_mh_diag_core.py':'fc9aaaabc8d2316e8a69a8de86f6c010c103a501',
    'cem_mh_pixel_checks.py':'0120bebfa25e8667ddc86adad6ec88cd82d91920',
    'pusht_exact_replay.py':'58ae8c8b10b1d4f6bedd980a6204aac684b912a1',
    'jepa.py':'246b085b7ebedf5505671b0cc7cccb07d182b1ac',
}


def sources():
    values = {}
    for name, wanted in PINNED.items():
        raw = (REPO/name).read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob != wanted:
            raise RuntimeError(f'Source differs: {name}; reconcile, do not reset local work')
        values[name] = sha(REPO/name)
    for name in ('run_eval.py','comparison.py','submit.sh'):
        values[str(HERE.name+'/'+name)] = sha(HERE/name)
    return values


def versions():
    names = ('torch','numpy','stable-worldmodel','stable-pretraining','gymnasium','omegaconf','hydra-core')
    return {p:importlib.metadata.version(p) for p in names}


def within(root, path):
    p = Path(path).expanduser().resolve()
    if root not in p.parents:
        raise ValueError(f'Path must be below STABLEWM_HOME: {p}')
    return p


def checkpoint_policy(root, path):
    p = within(root, path)
    name = str(p.relative_to(root))
    if not name.endswith('_object.ckpt'):
        raise ValueError('Expected plain JEPA *_object.ckpt')
    return name[:-len('_object.ckpt')]


def training_inputs(root, anchor):
    anchor = within(root, anchor)
    cpath, ppath = anchor/'training_complete.json', anchor/'protocol.json'
    c, p = read(cpath), read(ppath)
    if c.get('mode') != 'formal' or c.get('epochs') != 10 or c.get('all_ranks_finished') is not True:
        raise RuntimeError('Need completed formal 10-epoch anchor-only, not smoke')
    if p.get('mode') != 'formal' or c['protocol_sha256'] != sha(ppath):
        raise RuntimeError('Training protocol mismatch')
    if (anchor/'BLOCKED.json').exists() or list(anchor.glob('FAILED_rank*.json')):
        raise RuntimeError('Formal training has failure markers')
    if c['baseline_config_sha256'] != p['baseline_config_sha256'] or c['source_sha256'] != p['source_sha256']:
        raise RuntimeError('Training completion/protocol disagreement')
    paths = {'anchor':within(root,c['final_checkpoint']), 'pcmh':within(root,p['comparison_checkpoint'])}
    policies = {k:checkpoint_policy(root,v) for k,v in paths.items()}
    if c['policy'] != policies['anchor'] or paths['anchor'].parent != anchor:
        raise RuntimeError('Anchor policy/checkpoint mismatch')
    checks = {str(cpath):sha(cpath), str(ppath):sha(ppath),
        str(paths['anchor']):c['final_checkpoint_sha256'],
        str(paths['pcmh']):p['comparison_checkpoint_sha256'],
        str(within(root,p['parent_checkpoint'])):p['parent_checkpoint_sha256'],
        str(within(root,p['baseline_config'])):p['baseline_config_sha256']}
    for path, expected in checks.items():
        if sha(path) != expected:
            raise RuntimeError('Changed training artifact: '+path)
    if checks[str(paths['anchor'])] == checks[str(paths['pcmh'])]:
        raise RuntimeError('Comparison resolves to identical checkpoints')
    if sha(REPO/'jepa.py') != c['source_sha256']['jepa.py']:
        raise RuntimeError('JEPA code differs from completed training')
    return c, p, paths, policies, checks


def prepare(args):
    root = Path(os.environ['STABLEWM_HOME']).resolve()
    out = within(root,args.output)
    if args.node not in NODES or out.exists() or out.parent != root:
        raise ValueError('Use an allowed node and fresh evaluation output')
    c,p,paths,policies,checks = training_inputs(root,args.anchor_run)
    excluded_path = within(root,args.exclude_cases)
    excluded = excluded_episodes(excluded_path)
    checks[str(excluded_path)] = sha(excluded_path)
    dataset = root/'pusht_expert_train.h5'; st = dataset.stat()
    if str(dataset) != p['dataset'] or (st.st_size,st.st_mtime_ns) != (p['dataset_size'],p['dataset_mtime_ns']):
        raise RuntimeError('Dataset differs from completed training')
    ver = versions()
    if ver['stable-worldmodel'] != '0.0.6':
        raise RuntimeError('Use recorded stable-worldmodel 0.0.6; do not upgrade')
    src = sources()
    if any(Path(path) == out or out in Path(path).parents for path in checks):
        raise ValueError('Output would contain an input artifact')
    out.mkdir(parents=True,exist_ok=False)
    for d in ('runs','errors'): (out/d).mkdir()
    protocol = {'schema':1,'root':str(root),'repo':str(REPO),'output':str(out),'node':args.node,
        'created_unix':time.time(),'anchor_run':str(Path(args.anchor_run).resolve()),
        'checkpoints':{k:str(v) for k,v in paths.items()},'policies':policies,'input_sha256':checks,
        'source_sha256':src,'versions':ver,'dataset':str(dataset),
        'dataset_size':st.st_size,'dataset_mtime_ns':st.st_mtime_ns,
        'seeds':list(SEEDS),'budgets':{str(k):list(v) for k,v in BUDGETS.items()},'num_eval':N,
        'excluded_episodes':excluded,'excluded_cases_source':str(excluded_path),
        'cohort_note':'New paired native-reset cohort; full selected diagnostic episodes excluded. Not unseen pretraining data.',
        'training_job':c.get('job_id'),'training_steps':c['global_step'],
        'effect_direction':'PC-MH minus PC-anchor-only, percentage points',
        'trace_enabled':False,'counterfactual_replay':False,'training':False}
    dump(out/'protocol.json',protocol)
    print('PREPARED PAIRED EVALUATION: '+str(out),flush=True)
    print('N=100/300, I=10, K=10/30; seeds=142..146; 100 cases/seed; 20 runs total',flush=True)


def verify(out):
    out = Path(out).resolve(); m = read(out/'protocol.json')
    if str(out) != m['output'] or str(REPO) != m['repo'] or str(Path(os.environ['STABLEWM_HOME']).resolve()) != m['root']:
        raise RuntimeError('Prepared location changed')
    if sources() != m['source_sha256'] or versions() != m['versions']:
        raise RuntimeError('Code/environment changed; do not mix evaluations')
    for path, wanted in m['input_sha256'].items():
        if sha(path) != wanted: raise RuntimeError('Input changed: '+path)
    st = Path(m['dataset']).stat()
    if (st.st_size,st.st_mtime_ns) != (m['dataset_size'],m['dataset_mtime_ns']):
        raise RuntimeError('Dataset changed')
    if (m['seeds'] != list(SEEDS) or m['num_eval'] != N or
            m['budgets'] != {str(k):list(v) for k,v in BUDGETS.items()}):
        raise RuntimeError('Evaluation matrix changed')
    return m


def make_config(seed,budget):
    # Import dependency before reading configuration; no duplicate eval resolver.
    import stable_pretraining  # noqa: F401
    from omegaconf import OmegaConf as O
    cfg = O.load(REPO/'config/eval/pusht.yaml')
    if 'defaults' in cfg: del cfg['defaults']
    cfg.solver = O.load(REPO/'config/eval/solver/cem.yaml')
    cfg.seed = int(seed); cfg.eval.num_eval = N; cfg.world.max_episode_steps = 100
    cfg.solver.num_samples,cfg.solver.n_steps,cfg.solver.topk = BUDGETS[budget]
    cfg.solver.seed = int(seed); cfg.solver.device = 'cuda:0'
    if (cfg.plan_config.horizon,cfg.plan_config.action_block,cfg.plan_config.receding_horizon,
            cfg.world.history_size,cfg.world.frame_skip,cfg.eval.goal_offset_steps,cfg.eval.eval_budget,
            cfg.solver.batch_size) != (5,5,5,1,1,25,50,1):
        raise RuntimeError('Unexpected official evaluation semantics')
    return cfg


def model_digest(model):
    import torch
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        x = value.detach().cpu().contiguous()
        if (x.is_floating_point() or x.is_complex()) and not bool(torch.isfinite(x).all()):
            raise RuntimeError('Nonfinite model: '+key)
        h.update(key.encode());h.update(str((x.dtype,tuple(x.shape))).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_models(m):
    import torch
    import stable_worldmodel as swm
    models = {}
    for label in LABELS:
        model = swm.policy.AutoCostModel(m['policies'][label]).to('cuda:0').eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        for attr in ('coordinate_adapter','factor_heads','inverse_dynamics_head'):
            if getattr(model,attr,None) is not None: raise RuntimeError('Unexpected component: '+attr)
        models[label] = model
    for component in ('encoder','projector'):
        a,b = [getattr(models[k],component).state_dict() for k in LABELS]
        if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):
            raise RuntimeError('Models do not share frozen visual frame: '+component)
    return models


def run_one(cfg,dataset,process,model,seed):
    """Unchanged existing closed-loop evaluator with all candidate tracing OFF.

    The N=100 hard-coded assertions in legacy tapped_cost are inactive. CPU adapter fixtures cover N=100 and N=300; the real
    installed GPU environment is not tested here. No altered solver, extra simulator rollout or Trainer.
    """
    import eval_cem_mh_diagnostics as legacy
    with __import__('torch').inference_mode():
        result,traces = legacy.run_closed(cfg,dataset,process,model,seed,set(),set(),False)
    if traces: raise RuntimeError('Unexpected candidate capture')
    result = legacy.plain(result)
    successes(result)
    for key in ('contexts','initial_physics','initial_recipes','initial_images','native_steps'):
        if len(result[key]) != N: raise RuntimeError('Incomplete initial/action records: '+key)
    return result


def check_pair(a,b):
    import eval_cem_mh_diagnostics as legacy
    successes(a); successes(b)
    legacy.check_initial(a,b)
    # Includes _set_goal_state, reset kwargs and state setters, not just position.
    if a['initial_recipes'] != b['initial_recipes'] or a['env_spec'] != b['env_spec']:
        raise RuntimeError('Initial reset/goal recipe differs across models/budgets')


def result_path(out,budget,seed,label):
    return out/'runs'/f'B{budget}_seed{seed}_{label}.json'


def cell_identity(out,budget,seed,label):
    return {'protocol_sha256':sha(out/'protocol.json'),'cohort_sha256':sha(out/'cohort.json'),
            'budget':budget,'seed':seed,'label':label}


def load_cell(out,budget,seed,label):
    cell = sealed_read(result_path(out,budget,seed,label))
    if cell['identity'] != cell_identity(out,budget,seed,label):
        raise RuntimeError('Mismatched cell provenance')
    successes(cell['result'])
    return cell


def execute(out,m):
    import torch
    from omegaconf import OmegaConf as O
    from eval import get_dataset
    from eval_lowbudget_failure_autopsy import _build_process
    cfg0 = make_config(SEEDS[0],1000)
    dataset = get_dataset(cfg0,cfg0.eval.dataset_name)
    try:
        col = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
        cohort_file = out/'cohort.json'
        if cohort_file.exists():
            cohort = sealed_read(cohort_file)
            if cohort['protocol_sha256'] != sha(out/'protocol.json'):
                raise RuntimeError('Cohort belongs to another protocol')
        else:
            cases = select_cohort(dataset.get_col_data(col),dataset.get_col_data('step_idx'),m['excluded_episodes'])
            cohort = {'protocol_sha256':sha(out/'protocol.json'),'cases':cases}
            sealed_write(cohort_file,cohort)
            write_csv(out/'cohort.csv',cases)
        cases = cohort['cases']
        if not (out/'cohort.csv').exists(): write_csv(out/'cohort.csv',cases)
        if len(cases)!=len(SEEDS)*N: raise RuntimeError('Incomplete cohort')
        # Verify row identities against the read-only dataset, not sampled outcomes.
        rows = np.array([x['dataset_row'] for x in cases]); order = np.argsort(rows)
        data = dataset.get_row_data(rows[order])
        if not np.array_equal(np.asarray(data[col]).reshape(-1),[cases[i]['episode_idx'] for i in order]):
            raise RuntimeError('Cohort episode identity mismatch')
        if not np.array_equal(np.asarray(data['step_idx']).reshape(-1),[cases[i]['start_step'] for i in order]):
            raise RuntimeError('Cohort step identity mismatch')
        process = _build_process(cfg0,dataset)
        models = load_models(m); before = {k:model_digest(v) for k,v in models.items()}
        dump(out/'model_audit.json',{'initial_state_sha256':before,'same_visual_frame':True})
        for seed in SEEDS:
            part = [c for c in cases if c['seed']==seed]
            for budget in BUDGETS:
                cfg = make_config(seed,budget)
                cfg.diag_episodes = [c['episode_idx'] for c in part]
                cfg.diag_start = [c['start_step'] for c in part]
                # Alternate order deterministically; both calls reset RNGs internally.
                labels = LABELS if seed%2==0 else LABELS[::-1]
                for label in labels:
                    path = result_path(out,budget,seed,label)
                    if path.exists():
                        load_cell(out,budget,seed,label)
                        print(f'REUSE B{budget} seed={seed} model={label}',flush=True)
                        continue
                    print(f'EVAL B{budget} seed={seed} model={label}; 100 environments',flush=True)
                    start = time.monotonic()
                    result = run_one(cfg,dataset,process,models[label],seed)
                    if model_digest(models[label]) != before[label]:
                        raise RuntimeError('Inference changed model state')
                    payload = {'identity':cell_identity(out,budget,seed,label),
                        'policy':m['policies'][label],'config':O.to_container(cfg,resolve=True),
                        'elapsed_seconds':time.monotonic()-start,'result':result}
                    sealed_write(path,payload)
                    print(f'FINISHED B{budget} seed={seed} {label}: {100*successes(result).mean():.2f}%',flush=True)
                a,b = [load_cell(out,budget,seed,k)['result'] for k in LABELS]
                check_pair(a,b)
                if budget==3000:
                    check_pair(load_cell(out,1000,seed,'pcmh')['result'],a)
                print(f'PAIRED INITIAL STATE/GOAL/RESET PASS: B{budget} seed={seed}',flush=True)
        if {k:model_digest(v) for k,v in models.items()} != before:
            raise RuntimeError('Model state changed')
    finally:
        if getattr(dataset,'h5_file',None) is not None: dataset.h5_file.close()


def summarize(out,validate_initial=False):
    m = read(out/'protocol.json'); cohort = sealed_read(out/'cohort.json'); rows=[]; artifacts={}
    for seed in SEEDS:
        part = [c for c in cohort['cases'] if c['seed']==seed]
        if len(part)!=N: raise RuntimeError('Wrong case count')
        for budget in BUDGETS:
            a,b = [load_cell(out,budget,seed,k)['result'] for k in LABELS]
            if validate_initial:
                check_pair(a,b)
                if budget==3000: check_pair(load_cell(out,1000,seed,'pcmh')['result'],a)
            rows.extend(paired_rows(part,a,b,budget))
            for k in LABELS:
                p=result_path(out,budget,seed,k); artifacts[str(p.relative_to(out))]=sha(p)
    per_seed,totals = summary(rows)
    write_csv(out/'paired_cases.csv',rows);write_csv(out/'per_seed.csv',per_seed);write_csv(out/'summary.csv',totals)
    dump(out/'summary.json',{'by_budget':totals,'direction':m['effect_direction'],
        'uncertainty':'Descriptive paired dataset-episode cluster percentile bootstrap (5000 draws); fixed checkpoints, not training-seed uncertainty.'})
    text=['# PC-MH vs PC-anchor-only', '', 'NEW paired native-reset cohort; not the historical 93.6% benchmark.',
          'Seeds 142-146; 100 starts each; selected diagnostic episodes excluded.',
          'One trained checkpoint per method. Positive difference favors PC-MH.', '',
          '| Budget | PC-MH | Anchor-only | PC-MH minus anchor | Paired wins / losses |',
          '|---|---:|---:|---:|---:|']
    for r in totals:
        text.append(f"| B{r['budget']} | {r['pcmh_mean_pct']:.2f}% | {r['anchor_mean_pct']:.2f}% | {r['pcmh_minus_anchor_pp']:+.2f} pp | {r['pcmh_only']} / {r['anchor_only']} |")
    text += ['', 'B1000 is primary; B3000 secondary. No automatic significance or equivalence claim.',
             'CI in summary.csv resamples dataset episodes jointly across both methods; it does not measure training-initialization uncertainty.',
             'No candidate-population capture, counterfactual replay, teacher query, or training occurred.',
             'Timing includes existing native recording and reset audits; not a clean deployment-latency benchmark.',
             'See paired_cases.csv for every paired outcome and runs/*.json for reset contexts and executed actions.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    for name in ('protocol.json','cohort.json','cohort.csv','paired_cases.csv','per_seed.csv','summary.csv','summary.json','REPORT.md','model_audit.json'):
        artifacts[name]=sha(out/name)
    print('\n'.join(text),flush=True)
    return artifacts


def run(out):
    out=Path(out).resolve();m=verify(out)
    if (out/'COMPLETE.json').exists(): raise RuntimeError('Already complete; read REPORT.md, do not resubmit')
    import torch
    node=os.environ.get('SLURMD_NODENAME',socket.gethostname().split('.')[0])
    if not os.environ.get('SLURM_JOB_ID') or node!=m['node'] or node not in NODES:
        raise RuntimeError('Use the one-GPU Slurm launcher on the prepared allowed node')
    if int(os.environ.get('SLURM_NTASKS','1'))!=1: raise RuntimeError('Evaluation is single-process, not DDP')
    torch.cuda.init()
    if torch.cuda.device_count()!=1: raise RuntimeError('Require exactly one Slurm-visible GPU; do not rewrite CUDA_VISIBLE_DEVICES')
    torch.cuda.set_device(0); torch.ones(1,device='cuda:0').sum().item()
    torch.set_num_threads(2)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            dump(out/'STATUS.json',{'state':'RUNNING','job_id':os.environ['SLURM_JOB_ID']})
            execute(out,m)
            verify(out)
            artifacts=summarize(out,validate_initial=True)
            dump(out/'COMPLETE.json',{'job_id':os.environ['SLURM_JOB_ID'],'finished_unix':time.time(),
                'runs':20,'paired_cases_per_budget':500,'checkpoints_unchanged':True,'initial_conditions_verified':True,
                'artifacts':artifacts})
            dump(out/'STATUS.json',{'state':'COMPLETE','job_id':os.environ['SLURM_JOB_ID']})
            print('PC-MH / ANCHOR PAIRED EVALUATION PASS: '+str(out),flush=True)
        except Exception as exc:
            err={'state':'FAILED','job_id':os.environ.get('SLURM_JOB_ID'),'error':type(exc).__name__,
                 'message':str(exc),'traceback':traceback.format_exc()}
            dump(out/'errors'/f'job_{os.environ.get("SLURM_JOB_ID")}_{time.time_ns()}.json',err)
            dump(out/'STATUS.json',err)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--anchor-run',required=True);q.add_argument('--exclude-cases',required=True)
    q.add_argument('--output',required=True);q.add_argument('--node',required=True,choices=NODES)
    for cmd in ('run','verify','summarize'):
        sub.add_parser(cmd).add_argument('--output',required=True)
    sub.add_parser('check-source')
    a=p.parse_args()
    if a.cmd=='prepare':prepare(a)
    elif a.cmd=='run':run(a.output)
    elif a.cmd=='verify':verify(Path(a.output));print('EVALUATION INPUT/SOURCE CHECK PASS')
    elif a.cmd=='check-source':sources();print('PINNED EVALUATION SOURCE CHECK PASS')
    else:
        out=Path(a.output); c=read(out/'COMPLETE.json')
        for name,wanted in c['artifacts'].items():
            if sha(out/name)!=wanted:raise RuntimeError('Saved result changed: '+name)
        print((out/'REPORT.md').read_text())


if __name__=='__main__':main()
