#!/usr/bin/env python3
"""CPU math/fixtures; --with-repo additionally tests pinned real functions.

No server data, trained weights, simulator or GPU is used by these tests.
The fresh config/import check reads only the saved YAML when --with-repo is used.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf
import anchor_runtime as rt
from anchor_objective import center_mask_loss, anchor_only_forward
import train_anchor as runner

torch.set_num_threads(1)


def base_config():
    return {'mh_ald':{'enabled':True,'init_policy':'lewm_epoch_10','teacher_policy':'lewm_epoch_10',
        'rollout_horizon':5,'directions_per_position':1,'perturb_radius':.1565,'probes_per_chunk':5,
        'tf_weight':1.,'rollout_weight':1.,'weight':1.,
        'planner_context':{'enabled':True,'observation_history':1,'predictor_max_history':3}},
        'wm':{'history_size':3,'num_preds':1},'data':{'dataset':{'name':'pusht_expert_train',
        'frameskip':5,'num_steps':8,'keys_to_load':['pixels','action','proprio']}},
        'trainer':{'devices':4,'max_epochs':10,'precision':'bf16','num_nodes':1,'sync_batchnorm':False},
        'loader':{'batch_size':64,'num_workers':6},'subdir':'old','output_model_name':'lewm_old',
        'seed':3072,'train_split':.9,'optimizer':{'type':'AdamW','lr':5e-5,'weight_decay':1e-3},
        'wandb':{'enabled':False}}


# Small interface fixtures below; NOT the installed simulator/LeWM.
def mask(positions,horizon,dtype,device):
    return (torch.arange(horizon,device=device)[None] >= positions[:,None]).to(dtype)[None,:,:,None]


def masked_mse(error,valid):
    m=valid.expand(error.shape[0],-1,-1,-1)
    return (error.square().mean(-1,keepdim=True)*m).sum()/m.sum().clamp_min(1.)


def frozen_encode(model,batch):
    model.encoder.eval();model.projector.eval()
    with torch.no_grad():return model.encode({'pixels':batch['pixels']})['emb'].detach()


def rollout(model,z,u,history):
    pred=[];zs=z[:,None]
    for h in range(u.shape[1]):
        nxt=model.predict(zs[:,-history:],model.action_encoder(u[:,:h+1])[:,-history:])[:,-1:]
        pred.append(nxt);zs=torch.cat([zs,nxt],1)
    return torch.cat(pred,1)


@contextlib.contextmanager
def fixtures():
    a=NS(_mh_causal_mask=mask,_mh_masked_mse=masked_mse,_frozen_visual_encode=frozen_encode)
    p=NS(planner_context_rollout=rollout)
    with patch.dict(sys.modules,{'anchored_local_dynamics':a,'planner_context_mh':p}):yield


class Action(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Linear(2,4);self.patch_embed=NS(in_channels=2)
    def forward(self,u):return self.net(u)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=torch.nn.Linear(4,4);self.projector=torch.nn.BatchNorm1d(4)
        self.action_encoder=Action();self.predictor=torch.nn.Linear(8,4)
        self.pred_proj=torch.nn.BatchNorm1d(4)
    def encode(self,b):
        x=b['pixels'];n,t,d=x.shape
        return {'emb':self.projector(self.encoder(x).reshape(-1,d)).reshape(n,t,d)}
    def predict(self,z,a):
        y=self.predictor(torch.cat([z,a],-1));n,t,d=y.shape
        return self.pred_proj(y.reshape(-1,d)).reshape(n,t,d)


def module_and_batch():
    torch.manual_seed(8);m=Toy().eval()
    m.encoder.requires_grad_(False);m.projector.requires_grad_(False)
    teacher=copy.deepcopy(m).eval().requires_grad_(False)
    obj=NS(model=m,teacher_model=teacher,log_dict=lambda *a,**k:None)
    batch={'pixels':torch.randn(2,8,4),'action':torch.randn(2,8,2)}
    return obj,batch


class MathTests(unittest.TestCase):
    def compare(self,h,directions=1,dtype=torch.float64):
        e=torch.randn(3,h,7,dtype=dtype,requires_grad=True)
        with fixtures():v=center_mask_loss(e,directions)
        weights=torch.arange(1,h+1,dtype=dtype)
        expected=(e.square().mean((0,2))*weights).sum()/weights.sum()
        torch.testing.assert_close(v,expected)
        torch.testing.assert_close(torch.autograd.grad(v,e,retain_graph=True)[0],torch.autograd.grad(expected,e)[0])
    def test_h1(self):self.compare(1)
    def test_h3(self):self.compare(3)
    def test_h5(self):self.compare(5)
    def test_multiple_directions(self):self.compare(5,4)
    def test_float32(self):self.compare(5,dtype=torch.float32)
    def test_zero_error(self):
        with fixtures():self.assertEqual(center_mask_loss(torch.zeros(2,5,4)).item(),0.)
    def test_no_sampler_rng(self):
        torch.manual_seed(5);e=torch.ones(2,5,4);state=torch.get_rng_state().clone()
        with fixtures():center_mask_loss(e)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
    def test_invalid_shape(self):
        with fixtures(),self.assertRaises(ValueError):center_mask_loss(torch.ones(2,5))
    def test_empty_shape(self):
        with fixtures(),self.assertRaises(ValueError):center_mask_loss(torch.ones(0,5,4))
    def test_invalid_directions(self):
        for d in (0,-1,1.5,True):
            with fixtures(),self.assertRaises(ValueError):center_mask_loss(torch.ones(2,5,4),d)
    def test_objective_exact(self):
        obj,b=module_and_batch();cfg=OmegaConf.create(base_config())
        with fixtures():out=anchor_only_forward(obj,copy.deepcopy(b),'train',cfg,None,None)
        emb=frozen_encode(obj.model,b);pred=obj.model.predict(emb[:,:3],obj.model.action_encoder(b['action'][:,:3]))
        e=rollout(obj.model,emb[:,2],b['action'][:,2:7],3)-emb[:,3:8]
        v=(pred-emb[:,1:4]).square().mean()+e.square().mean()
        v+=(e.square().mean((0,2))*torch.arange(1,6)).sum()/15
        torch.testing.assert_close(out['loss'],v)
    def test_frozen_grads_and_buffers(self):
        obj,b=module_and_batch();obj.model.train();before=(runner.digest(obj.model.encoder),runner.digest(obj.model.projector),runner.digest(obj.model.pred_proj))
        with fixtures():anchor_only_forward(obj,b,'train',OmegaConf.create(base_config()),None,None)['loss'].backward()
        runner.check_parameters(obj,gradients=True)
        self.assertEqual(before,(runner.digest(obj.model.encoder),runner.digest(obj.model.projector),runner.digest(obj.model.pred_proj)))
    def test_teacher_not_called(self):
        obj,b=module_and_batch()
        obj.teacher_model.predict=lambda *a:(_ for _ in ()).throw(AssertionError('teacher called'))
        obj.teacher_model.encode=obj.teacher_model.predict
        with fixtures():anchor_only_forward(obj,b,'train',OmegaConf.create(base_config()),None,None)
    def test_finite_direct_metrics(self):
        obj,b=module_and_batch()
        with fixtures():out=anchor_only_forward(obj,b,'train',OmegaConf.create(base_config()),None,None)
        self.assertEqual(set(out),set(obj._anchor_last['train']))
        self.assertTrue(all(not x.requires_grad and torch.isfinite(x) for x in obj._anchor_last['train'].values()))
    def test_nan_baseline_action_handling(self):
        obj,b=module_and_batch();b['action'][0,0,0]=float('nan')
        with fixtures():out=anchor_only_forward(obj,b,'train',OmegaConf.create(base_config()),None,None)
        self.assertTrue(torch.isfinite(out['loss']))
    def test_nonfinite_loss_stops(self):
        obj,b=module_and_batch();b['pixels'][0,0,0]=float('nan')
        with fixtures(),self.assertRaises(RuntimeError):anchor_only_forward(obj,b,'train',OmegaConf.create(base_config()),None,None)
    def test_pickleable_forward(self):self.assertIs(pickle.loads(pickle.dumps(anchor_only_forward)),anchor_only_forward)


class ConfigTests(unittest.TestCase):
    def test_formal_changes_only_paths(self):
        b=base_config();d=rt.derive(b,'/tmp/root','/tmp/root/new','formal')
        self.assertEqual({r['key'] for r in rt.differences(b,d)},
                         {'subdir','output_model_name','trainer.default_root_dir'})
    def test_smoke_preserves_gpu_batch(self):
        b=base_config();d=rt.derive(b,'/tmp/root','/tmp/root/new','smoke')
        self.assertEqual(d['trainer']['devices'],4);self.assertEqual(d['loader'],b['loader'])
        self.assertEqual(d['trainer']['max_steps'],2)
    def test_not_mutate_base(self):
        b=base_config();old=copy.deepcopy(b);rt.derive(b,'/tmp/root','/tmp/root/new','formal');self.assertEqual(b,old)
    def test_wandb_separate(self):
        b=base_config();b['wandb']={'enabled':True,'config':{'id':'old','name':'old','resume':'allow'}}
        d=rt.derive(b,'/tmp/root','/tmp/root/new','formal')
        self.assertEqual(d['wandb']['config']['id'],'new');self.assertEqual(d['wandb']['config']['resume'],'never')
    def test_auto_gpu_rejected(self):
        b=base_config();b['trainer']['devices']='auto'
        with self.assertRaises(ValueError):rt.validate(b)
    def test_wrong_parent_rejected(self):
        b=base_config();b['mh_ald']['init_policy']='lewm_idm'
        with self.assertRaises(ValueError):rt.validate(b)
    def test_bad_weight_rejected(self):
        b=base_config();b['mh_ald']['weight']=0
        with self.assertRaises(ValueError):rt.validate(b)
    def test_privileged_state_rejected(self):
        b=base_config();b['data']['dataset']['keys_to_load'].append('state')
        with self.assertRaises(ValueError):rt.validate(b)
    def test_paths(self):
        for s in ('../old','/absolute',''):
            with self.assertRaises(ValueError):rt.inside('/tmp/root',s)
    def test_arithmetic(self):
        self.assertEqual(rt.arithmetic('3+5'),8)
        with self.assertRaises(ValueError):rt.arithmetic('__import__("os")')
    def test_startup_order_fresh_process(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'stable_pretraining.py').write_text("from omegaconf import OmegaConf\nOmegaConf.register_new_resolver('eval', eval)\n")
            code="import anchor_runtime as r; O=r.omega(); import stable_pretraining; assert O.create({'n':'${eval:3+5}'}).n==8; r.omega()"
            env={**os.environ,'PYTHONPATH':str(rt.HERE)+os.pathsep+d}
            subprocess.run([sys.executable,'-c',code],env=env,check=True,capture_output=True,text=True,timeout=30)
    def test_smoke_gate_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            rt.dump(Path(d)/'training_complete.json',{'mode':'smoke'})
            rt.dump(Path(d)/'protocol.json',{})
            with self.assertRaises(RuntimeError):rt.smoke_gate(d,'hash',{},4)
    def test_node_whitelist(self):
        self.assertEqual(set(rt.NODES),{'4090node2','4090node3','3090node1','3090node2','3090node3'})
    def test_submit_rejects_disallowed_before_dependency(self):
        for node in ('4090node1','4090node2,4090node3','bad'):
            r=subprocess.run(['bash',str(rt.HERE/'submit.sh'),'smoke','v1'],env={**os.environ,'NODE':node},capture_output=True,text=True)
            self.assertEqual(r.returncode,2);self.assertIn('Allowed:',r.stderr)
    def test_shell_syntax_no_mem(self):
        p=rt.HERE/'submit.sh';subprocess.run(['bash','-n',str(p)],check=True)
        s=p.read_text();self.assertNotIn('#SBATCH --mem',s);self.assertIn('NODE:-4090node2',s)
        self.assertIn('/tmp/pcao.',s)


class ExecutionTests(unittest.TestCase):
    def test_state_digest_buffers(self):
        m=Toy();a=runner.digest(m);m.projector.running_mean.add_(1);self.assertNotEqual(a,runner.digest(m))
    def test_baseline_delegation_export_metrics(self):
        obj,batch=module_and_batch();parent=copy.deepcopy(obj.model)
        baseline=NS(pc_mh_ald_forward=lambda *a:None,_load_policy_model=lambda n:copy.deepcopy(parent))
        class Callback:
            def __init__(self,dirpath,filename,epoch_interval):pass
        baseline.ModelObjectCallBack=Callback
        def run(cfg):
            s=baseline._load_policy_model('lewm_epoch_10');t=baseline._load_policy_model('lewm_epoch_10').requires_grad_(False)
            mod=NS(model=s,teacher_model=t,log_dict=lambda *a,**k:None)
            cb=baseline.ModelObjectCallBack(dirpath=out,filename=cfg.output_model_name,epoch_interval=1)
            tr=NS(world_size=1,is_global_zero=True,current_epoch=0,num_training_batches=2,global_step=0,interrupted=False)
            cb.on_fit_start(tr,mod);cb.on_train_epoch_start(tr,mod)
            opt=torch.optim.AdamW([p for p in s.parameters() if p.requires_grad],lr=.01)
            for i in range(2):
                opt.zero_grad();y=baseline.pc_mh_ald_forward(mod,copy.deepcopy(batch),'train',cfg,None,None)
                y['loss'].backward();cb.on_after_backward(tr,mod);opt.step();tr.global_step+=1
                cb.on_train_batch_end(tr,mod,y,batch,i)
            cb.on_train_epoch_end(tr,mod);cb.on_fit_end(tr,mod)
        baseline.run=run;original=baseline.pc_mh_ald_forward
        with tempfile.TemporaryDirectory() as d,fixtures(),patch.dict(sys.modules,{'train_pc_mh_ald':baseline}),contextlib.redirect_stdout(io.StringIO()):
            out=Path(d);c=rt.derive(base_config(),out.parent,out,'smoke');cfg=OmegaConf.create(c)
            m={'world_size':1,'local_batch':2,'global_batch':2,'mode':'smoke'}
            result=runner.execute(cfg,m,out)
            self.assertEqual(result['global_step'],2)
            self.assertIs(baseline.pc_mh_ald_forward,original)
            self.assertEqual(len(list(out.glob('*object.ckpt'))),1)
            self.assertIn('pc_anchor_loss', (out/'training_progress.csv').read_text())
            self.assertNotIn(',,,',(out/'training_progress.csv').read_text())
            self.assertTrue((out/'epoch_metrics.csv').is_file())
    def test_baseline_patch_restored_on_exception(self):
        def fail(cfg):raise RuntimeError('sentinel')
        b=NS(pc_mh_ald_forward=lambda *a:None,ModelObjectCallBack=type('C',(),{}),_load_policy_model=lambda n:None,run=fail)
        before=(b.pc_mh_ald_forward,b.ModelObjectCallBack,b._load_policy_model)
        with patch.dict(sys.modules,{'train_pc_mh_ald':b}),self.assertRaisesRegex(RuntimeError,'sentinel'):
            runner.execute(None,{},Path('.'))
        self.assertEqual((b.pc_mh_ald_forward,b.ModelObjectCallBack,b._load_policy_model),before)
    def test_incomplete_epoch_stops(self):
        class Base:pass
        cfg=OmegaConf.create(rt.derive(base_config(),'/tmp/root','/tmp/root/new','smoke'))
        callback=runner.callback_type(Base,cfg,{},Path('.'),[])();callback.batches=1
        with self.assertRaisesRegex(RuntimeError,'Incomplete epoch'):
            callback.on_train_epoch_end(NS(num_training_batches=2),None)
    def test_bad_optional_component(self):
        m=Toy();m.inverse_dynamics_head=torch.nn.Linear(4,2)
        with self.assertRaises(RuntimeError):runner.plain_model(m)


def real_repo_equivalence():
    """Use actual pinned old forward, replacing ONLY its probe outputs by U0.

    Small Toy parameters are used, not trained LeWM weights. Value AND parameter
    gradients must match our new forward. No old sampler is called with radius=0.
    """
    rt.sources()
    import planner_context_mh as pc
    original_pair=pc._paired_planner_context_predictions
    def centers(**kw):
        s,t=kw['student'],kw['teacher'];h=kw['horizon'];c=kw['history_size']-1
        u=kw['normalized_actions'][:,c:c+h]
        sp=pc.planner_context_rollout(s,kw['current_emb'],u,kw['predictor_max_history'])
        with torch.no_grad():tp=pc.planner_context_rollout(t,kw['current_emb'],u,kw['predictor_max_history'])
        pos=torch.arange(h,device=sp.device).repeat_interleave(kw['directions_per_position'])
        a=sp[:,None].expand(-1,len(pos),-1,-1);b=tp[:,None].expand_as(a)
        return a,a,b,b,torch.ones(a.shape[:2],device=sp.device),pos
    cfg=OmegaConf.create(base_config())
    a,batch=module_and_batch();b=copy.deepcopy(a)
    with patch.object(pc,'_paired_planner_context_predictions',centers):
        old=pc.pc_mh_ald_forward(a,copy.deepcopy(batch),'train',cfg,None,None)
    new=anchor_only_forward(b,copy.deepcopy(batch),'train',cfg,None,None)
    torch.testing.assert_close(old['tf_loss'],new['tf_loss'])
    torch.testing.assert_close(old['pc_mh_rollout_loss'],new['pc_rollout_loss'])
    torch.testing.assert_close(old['pc_mh_loss'],new['pc_anchor_loss'])
    torch.testing.assert_close(old['loss'],new['loss'],atol=1e-6,rtol=1e-6)
    old['loss'].backward();new['loss'].backward()
    for (ka,pa),(kb,pb) in zip(a.model.named_parameters(),b.model.named_parameters()):
        assert ka==kb
        if pa.grad is None:assert pb.grad is None
        else:torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-5)
    assert pc._paired_planner_context_predictions is original_pair
    print('PINNED REAL FORWARD ZERO-PROBE VALUE / GRADIENT EQUIVALENCE PASS')


def main():
    p=argparse.ArgumentParser();p.add_argument('--with-repo',action='store_true');p.add_argument('--base-config')
    a=p.parse_args()
    if a.with_repo:
        base=a.base_config or os.environ.get('BASE_CONFIG')
        if not base:p.error('--with-repo requires --base-config or BASE_CONFIG')
        subprocess.run([sys.executable,str(rt.HERE/'train_anchor.py'),'check','--base-config',base],check=True)
        real_repo_equivalence()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                             for c in (MathTests,ConfigTests,ExecutionTests))
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    if not r.wasSuccessful():raise SystemExit(1)
    print(f'ANCHOR CPU TESTS PASS: {r.testsRun}; real LeWM weights/data/GPU/DDP NOT RUN')


if __name__=='__main__':main()
