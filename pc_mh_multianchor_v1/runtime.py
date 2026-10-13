"""Repository adapter; source/version checks protect matched comparisons."""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
import operator
import os
from pathlib import Path
import sys
import numpy as np
import torch
from core import sha, dump, finite

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
BLOBS={
 'jepa.py':'246b085b7ebedf5505671b0cc7cccb07d182b1ac',
 'planner_context_mh.py':'017e3c6ec5fdcd7149367097b9ef0c204b14b9b6',
 'train_pc_mh_ald.py':'89f0505266cfd3ec610851c3e93ea11d066ced0e',
 'train_mh_ald.py':'94d31e21bf3134b2c6a6a9376245b508f04f5a03',
 'anchored_local_dynamics.py':'af5f2aa992ccc686d1abdbc86720dcf36a77d787',
 'stage1_bias_calibration.py':'e1e30b3ad01bb21d2353c0073a9681b9d60801a6',
 'utils.py':'eaaf24a4074c5f29344988d6e6d6ff9f0dd082a2'}


def sources():
    out={}
    for name,expected in BLOBS.items():
        b=(REPO/name).read_bytes()
        if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=expected:
            raise RuntimeError(f'Upstream changed: {name}; reconcile, never auto-reset')
        out[name]=sha(REPO/name)
    for name in ('core.py','runtime.py','pipeline.py','queries.py'):
        out['pc_mh_multianchor_v1/'+name]=sha(HERE/name)
    return out


def arithmetic(text):
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
         ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv}
    def walk(n):
        if isinstance(n,ast.Expression): return walk(n.body)
        if isinstance(n,ast.Constant) and type(n.value) in (int,float): return n.value
        if isinstance(n,ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](walk(n.left),walk(n.right))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub): return -walk(n.operand)
        raise ValueError('Only numeric configuration interpolation is allowed')
    return walk(ast.parse(str(text),mode='eval'))


def omega():
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver('eval'): OmegaConf.register_new_resolver('eval',arithmetic)
    return OmegaConf


def config(path):
    O=omega(); cfg=O.load(path)
    if str(cfg.data.dataset.name)!='pusht_expert_train': raise ValueError('This release is PushT only')
    if not cfg.mh_ald.enabled or not cfg.mh_ald.planner_context.enabled: raise ValueError('Need PC-MH baseline')
    if (int(cfg.wm.history_size),int(cfg.wm.num_preds),int(cfg.mh_ald.rollout_horizon),
        int(cfg.data.dataset.frameskip),int(cfg.mh_ald.planner_context.observation_history),
        int(cfg.mh_ald.planner_context.predictor_max_history)) != (3,1,5,5,1,3):
        raise ValueError('Expected unchanged H5 / history3 / frame-skip5 / planner history1 protocol')
    if int(cfg.data.dataset.num_steps)!=8: raise ValueError('Expected 8 observation frames')
    if str(cfg.mh_ald.init_policy)!=str(cfg.mh_ald.teacher_policy): raise ValueError('Different parents')
    if str(cfg.mh_ald.init_policy)!='lewm_epoch_10': raise ValueError('Use the plain LeWM10 PC-MH source, not IDM/Stepwise')
    for k in ('cem_aligned','goal_sensitive','goal_coupled'):
        if O.select(cfg,f'mh_ald.{k}.enabled',default=False): raise ValueError('Need pure PC-MH')
    if not set(cfg.data.dataset.keys_to_load)<= {'pixels','action','proprio'}: raise ValueError('Privileged columns forbidden')
    if type(cfg.trainer.devices) is not int or cfg.trainer.devices<1: raise ValueError('Actual baseline GPU count must be explicit')
    if int(cfg.trainer.get('num_nodes',1))!=1 or cfg.trainer.get('sync_batchnorm',False): raise ValueError('Expected one node and frozen BN')
    return cfg


def data(cfg):
    import stable_worldmodel as swm
    import stable_pretraining as spt
    from utils import get_column_normalizer,get_img_preprocessor
    from train_mh_ald import _raw_action_stats
    ds=swm.data.HDF5Dataset(**cfg.data.dataset,transform=None)
    mean,std=_raw_action_stats(ds)
    finite(mean,'raw mean');finite(std,'raw std')
    if bool((std<=0).any()): raise ValueError('Degenerate raw action normalization')
    transforms=[get_img_preprocessor(source='pixels',target='pixels',img_size=cfg.img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if not col.startswith('pixels'): transforms.append(get_column_normalizer(ds,col,col))
    ds.transform=spt.data.transforms.Compose(*transforms)
    g=torch.Generator().manual_seed(int(cfg.seed))
    tr,va=spt.data.random_split(ds,lengths=[cfg.train_split,1-cfg.train_split],generator=g)
    if not hasattr(tr,'indices') or not hasattr(va,'indices'):
        raise RuntimeError('Installed split does not expose indices; refuse to guess split')
    ti=np.asarray(tr.indices,dtype=np.int64); vi=np.asarray(va.indices,dtype=np.int64)
    if np.intersect1d(ti,vi).size or len(ti)+len(vi)!=len(ds): raise RuntimeError('Malformed baseline split')
    return ds,ti,vi,mean.numpy(),std.numpy()


def close_data(ds):
    f=getattr(ds,'h5_file',None)
    if f is not None: f.close()


def digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        t=v.detach().cpu().contiguous()
        if (t.is_floating_point() or t.is_complex()) and not torch.isfinite(t).all():
            raise RuntimeError('Nonfinite model: '+k)
        h.update(k.encode());h.update(str((t.dtype,tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def teacher(cfg,device):
    from train_mh_ald import _load_policy_model
    t=_load_policy_model(str(cfg.mh_ald.teacher_policy)).to(device).eval().requires_grad_(False)
    for a in ('factor_heads','coordinate_adapter','inverse_dynamics_head'):
        if getattr(t,a,None) is not None: raise ValueError('Unexpected parent component: '+a)
    digest(t)
    return t


def initialize(out,base,smoke):
    root=Path(os.environ['STABLEWM_HOME']).resolve();out=Path(out).resolve();base=Path(base).resolve()
    if root not in out.parents or out==base.parent or base.parent in out.parents:
        raise ValueError('Use a new sibling output under STABLEWM_HOME')
    cfg=config(base)
    if importlib.metadata.version('stable-worldmodel')!='0.0.6': raise RuntimeError('Keep installed stable-worldmodel 0.0.6')
    src=sources()
    checkpoint=root/(str(cfg.mh_ald.init_policy)+'_object.ckpt')
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    dataset=root/'pusht_expert_train.h5'; st=dataset.stat()
    out.mkdir(parents=True,exist_ok=False)
    (out/'baseline_config.yaml').write_bytes(base.read_bytes())
    m={'base_config':str(base),'base_sha256':sha(base),'source_sha256':src,'output':str(out),
       'dataset':str(dataset),'dataset_size':st.st_size,'dataset_mtime_ns':st.st_mtime_ns,
       'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'smoke':bool(smoke),
       'seed':20261013,'samples':1024 if smoke else 100000,'queries':8 if smoke else 4096,
       'k':8,'lambda_query':.5,'query_batch_per_rank':16,'min_coverage':.2,'min_gain':.1,
       'versions':{p:importlib.metadata.version(p) for p in ('torch','numpy','stable-worldmodel','stable-pretraining')},
       'supervision':'Only original training clips; never load diagnostic replays or task-state labels',
       'inference':'No new modules; original JEPA and CEM'}
    dump(out/'protocol.json',m)
    return cfg,m


def verify(out):
    out=Path(out).resolve();m=json.loads((out/'protocol.json').read_text())
    if str(out)!=m['output'] or Path(os.environ['STABLEWM_HOME']).resolve() not in out.parents: raise ValueError('Moved artifact root')
    if sources()!=m['source_sha256']: raise RuntimeError('Preparation sources changed')
    if sha(m['base_config'])!=m['base_sha256'] or sha(out/'baseline_config.yaml')!=m['base_sha256']: raise RuntimeError('Baseline changed')
    if sha(m['checkpoint'])!=m['checkpoint_sha256']: raise RuntimeError('Parent checkpoint changed')
    st=Path(m['dataset']).stat()
    if (st.st_size,st.st_mtime_ns)!=(m['dataset_size'],m['dataset_mtime_ns']): raise RuntimeError('Dataset changed')
    for p,v in m['versions'].items():
        if importlib.metadata.version(p)!=v: raise RuntimeError('Environment changed: '+p)
    return config(out/'baseline_config.yaml'),m


def seal(out,name,files,extra=None):
    dump(Path(out)/(name+'.json'),{'artifacts':{f:sha(Path(out)/f) for f in files},**(extra or {})})


def check_seal(out,name):
    m=json.loads((Path(out)/(name+'.json')).read_text())
    for f,h in m['artifacts'].items():
        if sha(Path(out)/f)!=h: raise RuntimeError('Changed artifact: '+f)
    return m
