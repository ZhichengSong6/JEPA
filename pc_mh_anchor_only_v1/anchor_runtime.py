"""Configuration/provenance helpers. Never writes upstream code or old runs."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import operator
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
BASE = '22ff594681c1e491b47598de0347d75931969e73'
BLOBS = {
    'jepa.py': '246b085b7ebedf5505671b0cc7cccb07d182b1ac',
    'planner_context_mh.py': '017e3c6ec5fdcd7149367097b9ef0c204b14b9b6',
    'train_pc_mh_ald.py': '89f0505266cfd3ec610851c3e93ea11d066ced0e',
    'train_mh_ald.py': '94d31e21bf3134b2c6a6a9376245b508f04f5a03',
    'anchored_local_dynamics.py': 'af5f2aa992ccc686d1abdbc86720dcf36a77d787',
    'stage1_bias_calibration.py': 'e1e30b3ad01bb21d2353c0073a9681b9d60801a6',
    'utils.py': 'eaaf24a4074c5f29344988d6e6d6ff9f0dd082a2',
}
NODES = ('4090node2', '4090node3', '3090node1', '3090node2', '3090node3')
CODE = ('anchor_objective.py', 'anchor_runtime.py', 'train_anchor.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def arithmetic(text):
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv}
    def visit(x):
        if isinstance(x, ast.Expression): return visit(x.body)
        if isinstance(x, ast.Constant) and type(x.value) in (int, float): return x.value
        if isinstance(x, ast.BinOp) and type(x.op) in ops: return ops[type(x.op)](visit(x.left), visit(x.right))
        if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub): return -visit(x.operand)
        raise ValueError('Only numeric config arithmetic is allowed')
    return visit(ast.parse(str(text), mode='eval'))


def omega():
    # This dependency unconditionally registers eval on its first import.
    # Import it first; never clear resolvers/reload modules/edit site-packages.
    import stable_pretraining  # noqa: F401
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', arithmetic, replace=True)
    return OmegaConf


def sources():
    out = {}
    for name, expected in BLOBS.items():
        b = (REPO/name).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Upstream changed: {name}; reconcile without overwriting local work')
        out[name] = sha(REPO/name)
    out.update({str(HERE.name+'/'+n): sha(HERE/n) for n in CODE})
    return out


def versions():
    return {p: importlib.metadata.version(p) for p in
            ('torch', 'numpy', 'lightning', 'omegaconf', 'stable-worldmodel', 'stable-pretraining')}


def inside(root, relative):
    root, r = Path(root).resolve(), Path(str(relative))
    if r.is_absolute() or '..' in r.parts or not r.parts:
        raise ValueError('Expected nonempty relative path')
    p = (root/r).resolve()
    if root not in p.parents:
        raise ValueError('Path escapes STABLEWM_HOME')
    return p


def validate(d):
    if not d['mh_ald']['enabled'] or not d['mh_ald']['planner_context']['enabled']:
        raise ValueError('Need original PC-MH config')
    mh, pc, t = d['mh_ald'], d['mh_ald']['planner_context'], d['trainer']
    ds = d['data']['dataset']
    if (d['wm']['history_size'], d['wm']['num_preds'], mh['rollout_horizon'],
            pc['observation_history'], pc['predictor_max_history'], ds['frameskip'], ds['num_steps']) != (3,1,5,1,3,5,8):
        raise ValueError('Expected unchanged PushT H5/history3/frame-skip5 protocol')
    if ds['name'] != 'pusht_expert_train' or not set(ds['keys_to_load']) <= {'pixels','action','proprio'}:
        raise ValueError('Wrong dataset or extra/privileged supervision')
    if mh['init_policy'] != 'lewm_epoch_10' or mh['teacher_policy'] != mh['init_policy']:
        raise ValueError('Use original LeWM10, not IDM or a Stage-2 checkpoint')
    for k in ('cem_aligned', 'goal_sensitive', 'goal_coupled'):
        if mh.get(k, {}).get('enabled', False): raise ValueError('Not pure PC-MH')
    for k in ('tf_weight', 'rollout_weight', 'weight'):
        if not math.isfinite(float(mh[k])) or float(mh[k]) != 1.0:
            raise ValueError('This control targets the recorded 1/1/1 PC-MH run')
    if type(mh['directions_per_position']) is not int or mh['directions_per_position'] < 1:
        raise ValueError('Invalid equal probe count')
    if type(t['devices']) is not int or not 1 <= t['devices'] <= 4:
        raise ValueError('Saved actual GPU count must be explicit, not auto')
    if t.get('num_nodes',1) != 1 or t.get('sync_batchnorm',False) or t['max_epochs'] != 10:
        raise ValueError('Expected one node, frozen BN and 10 Stage-2 epochs')
    if t.get('accumulate_grad_batches',1) != 1 or t.get('max_steps',-1) != -1:
        raise ValueError('Expected baseline no accumulation and no step truncation')
    if t.get('limit_train_batches',1.0) != 1.0 or t.get('fast_dev_run',False) or t.get('overfit_batches',0) != 0:
        raise ValueError('Baseline is a shortened/debug run')
    if type(d['loader']['batch_size']) is not int or d['loader']['batch_size'] < 1:
        raise ValueError('Invalid per-rank batch')


def config(path):
    O = omega()
    d = O.to_container(O.load(path), resolve=True)
    validate(d)
    return d


def derive(base, root, output, mode):
    validate(base)
    if mode not in ('smoke','formal'): raise ValueError('mode must be smoke/formal')
    root, output = Path(root).resolve(), Path(output).resolve()
    if root not in output.parents: raise ValueError('Output must be under STABLEWM_HOME')
    d = copy.deepcopy(base)
    d['subdir'] = str(output.relative_to(root))
    d['trainer']['default_root_dir'] = str(output)  # isolate Lightning checkpoints too
    d['output_model_name'] += '_anchor_only' + ('_smoke' if mode == 'smoke' else '')
    if d.get('wandb', {}).get('enabled'):
        w = d['wandb']['config']
        w.update(id=d['subdir'], name=d['output_model_name'], resume='never')
    if mode == 'smoke':
        d['trainer'].update(max_epochs=1, max_steps=2, limit_train_batches=2, limit_val_batches=1)
    return d


def differences(a, b, prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        rows = []
        for k in sorted(a.keys() | b.keys()):
            key = prefix+'.'+k if prefix else k
            if k not in a or k not in b: rows.append({'key':key,'before':a.get(k),'after':b.get(k)})
            else: rows.extend(differences(a[k],b[k],key))
        return rows
    return [] if a == b else [{'key':prefix,'before':a,'after':b}]


def smoke_gate(path, baseline_sha, code, world):
    p = Path(path)
    c = read(p/'training_complete.json')
    m = read(p/'protocol.json')
    if (p/'BLOCKED.json').exists() or list(p.glob('FAILED_rank*.json')):
        raise RuntimeError('Smoke has a failure marker')
    if (c.get('mode') != 'smoke' or not c.get('all_ranks_finished') or
            c.get('global_step') != 2 or c.get('baseline_config_sha256') != baseline_sha or
            c.get('source_sha256') != code or c.get('world_size') != world or
            c.get('protocol_sha256') != sha(p/'protocol.json')):
        raise RuntimeError('Smoke completion does not match this source/config/world')
    if sha(c['final_checkpoint']) != c['final_checkpoint_sha256']:
        raise RuntimeError('Smoke checkpoint changed')
    if m['source_sha256'] != code: raise RuntimeError('Smoke protocol changed')
    if versions() != m['versions'] or sha(m['parent_checkpoint']) != m['parent_checkpoint_sha256']:
        raise RuntimeError('Smoke environment/parent no longer matches')
    return {'directory':str(p.resolve()),'completion_sha256':sha(p/'training_complete.json')}


def prepare(args):
    root = Path(os.environ['STABLEWM_HOME']).resolve()
    out, base_path = Path(args.output).resolve(), Path(args.base_config).resolve()
    if args.node not in NODES: raise ValueError('Node is not permitted for long jobs')
    if root not in out.parents or out.exists() or out == base_path.parent or base_path.parent in out.parents:
        raise FileExistsError('Use a fresh sibling output; no overwrite/automatic resume')
    code, d = sources(), config(base_path)
    ver = versions()
    if ver['stable-worldmodel'] != '0.0.6': raise RuntimeError('Keep the recorded stable-worldmodel 0.0.6')
    parent = inside(root, d['mh_ald']['init_policy']+'_object.ckpt')
    old_model = inside(root, str(Path(d['subdir'])/(d['output_model_name']+'_epoch_10_object.ckpt')))
    dataset = root/'pusht_expert_train.h5'
    for p in (parent, old_model, dataset):
        if not p.is_file(): raise FileNotFoundError(p)
    bsha = sha(base_path)
    gate = None
    if args.mode == 'formal':
        if not args.smoke_run: raise ValueError('Formal requires the matching completed smoke run')
        gate = smoke_gate(args.smoke_run, bsha, code, d['trainer']['devices'])
    cfg = derive(d, root, out, args.mode)
    out.mkdir(parents=True, exist_ok=False)
    (out/'baseline_config.yaml').write_bytes(base_path.read_bytes())
    omega().save(omega().create(cfg), out/'launch_config.yaml')
    st = dataset.stat()
    m = {'schema':1,'mode':args.mode,'node':args.node,'repo':str(REPO),'output':str(out),
         'baseline_config':str(base_path),'baseline_config_sha256':bsha,
         'launch_config_sha256':sha(out/'launch_config.yaml'),'source_sha256':code,
         'versions':ver,'parent_checkpoint':str(parent),'parent_checkpoint_sha256':sha(parent),
         'comparison_checkpoint':str(old_model),'comparison_checkpoint_sha256':sha(old_model),
         'dataset':str(dataset),'dataset_size':st.st_size,'dataset_mtime_ns':st.st_mtime_ns,
         'world_size':d['trainer']['devices'],'local_batch':d['loader']['batch_size'],
         'global_batch':d['trainer']['devices']*d['loader']['batch_size'],
         'cpus_per_task':max(2,int(d['loader']['num_workers'])+2),
         'max_epochs':cfg['trainer']['max_epochs'],'seed':d['seed'],
         'config_changes':differences(d,cfg),'smoke_gate':gate,'reference_commit':BASE,
         'objective':'TF + PC factual rollout + original-causal-mask center MSE',
         'teacher_forward':False,'perturbed_actions':False,'equal_compute_claim':False}
    dump(out/'protocol.json',m)
    print(f"PREPARED {args.mode}: {out}; GPUs={m['world_size']} local/global batch={m['local_batch']}/{m['global_batch']}",flush=True)


def verify(out):
    out = Path(out).resolve(); m = read(out/'protocol.json')
    if str(REPO) != m['repo'] or str(out) != m['output']:
        raise RuntimeError('Prepared repository or output moved')
    if Path(os.environ['STABLEWM_HOME']).resolve() not in out.parents:
        raise ValueError('STABLEWM_HOME changed')
    if sources() != m['source_sha256'] or versions() != m['versions']:
        raise RuntimeError('Source/environment changed after preparation')
    for p, h in ((m['baseline_config'],m['baseline_config_sha256']),
                 (out/'baseline_config.yaml',m['baseline_config_sha256']),
                 (out/'launch_config.yaml',m['launch_config_sha256']),
                 (m['parent_checkpoint'],m['parent_checkpoint_sha256'])):
        if sha(p) != h: raise RuntimeError('Source artifact changed: '+str(p))
    st = Path(m['dataset']).stat()
    if (st.st_size,st.st_mtime_ns) != (m['dataset_size'],m['dataset_mtime_ns']):
        raise RuntimeError('Dataset changed')
    O = omega()
    return O.load(out/'launch_config.yaml'), m
