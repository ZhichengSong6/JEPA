#!/usr/bin/env python3
"""CPU numerical / isolation tests. Not a real LeWM or Slurm benchmark."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import core
from core import Memory,features,fit_scales,select_gate,accepted,query_loss
from queries import select_queries
from train import augment_forward,QueryPool
import runtime

torch.set_num_threads(1)


def toy(n=40,h=3,d=4):
    rng=np.random.default_rng(44)
    z=rng.normal(size=(n,d)).astype('float32')
    u=rng.normal(size=(n,h,2)).astype('float32')*.1
    residual=np.repeat(u.cumsum(1)[...,:1],d,2)
    return {'z':z,'u':u,'residual':residual,'episode':np.arange(n,dtype=np.int64)}


class MathTests(unittest.TestCase):
    def test_episode_isolation(self):
        s=core.episode_partition(np.repeat(np.arange(100),3),12)
        self.assertEqual([len(v) for v in s.values()],[60,20,20])
        self.assertEqual(len(set(np.concatenate(list(s.values())))),100)
        for a,b in (('bank','audit'),('bank','calibration'),('audit','calibration')):
            self.assertFalse(np.intersect1d(s[a],s[b]).size)
    def test_episode_determinism(self):
        a=core.episode_partition(np.arange(30),7);b=core.episode_partition(np.arange(30)[::-1],7)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
    def test_episode_small_rejected(self):
        with self.assertRaises(ValueError):core.episode_partition([1,2],1)
    def test_scales_bank_only(self):
        b=toy();s=fit_scales(b);np.testing.assert_allclose(s['z_mean'],b['z'].mean(0))
    def test_constant_features_finite(self):
        b=toy();b['z'][:]=1;b['u'][:]=0;s=fit_scales(b)
        self.assertTrue(np.isfinite(features(b['z'],b['u'],2,s)[0]).all())
    def test_causal_features(self):
        b=toy();s=fit_scales(b);u=b['u'].copy();u[:,1:]=999
        np.testing.assert_array_equal(features(b['z'],u,1,s)[1],features(b['z'],b['u'],1,s)[1])
    def test_nonfinite_rejected(self):
        b=toy();b['z'][0,0]=np.nan
        with self.assertRaises(ValueError):fit_scales(b)
    def test_distinct_neighbors_and_exclusion(self):
        b=toy();b['episode']=np.arange(40)//2;m=Memory(b,k=8)
        _,_,ix=m.estimate(b['z'][:5],b['u'][:5],2,b['episode'][:5])
        for q,j in enumerate(ix):
            e=b['episode'][j];self.assertEqual(len(set(e)),8);self.assertNotIn(b['episode'][q],e)
    def test_insufficient_episodes_rejected(self):
        b=toy();b['episode'][:]=0
        with self.assertRaises(ValueError):Memory(b,k=8)
    def test_memory_constant_residual(self):
        b=toy();b['residual'][:]=.3;m=Memory(b,k=8)
        r,s,_=m.estimate(b['z'][:3],b['u'][:3],1,b['episode'][:3])
        np.testing.assert_allclose(r,.3,atol=1e-6);np.testing.assert_allclose(s[:,2],0,atol=1e-10)
    def test_memory_prefix_not_future(self):
        b=toy();m=Memory(b,k=8);q=b['u'][:3].copy()
        a=m.estimate(b['z'][:3],q,1,b['episode'][:3]);q[:,1:]=500
        c=m.estimate(b['z'][:3],q,1,b['episode'][:3])
        for x,y in zip(a,c):np.testing.assert_array_equal(x,y)
    def test_memory_does_not_mutate(self):
        b=toy();old=copy.deepcopy(b);m=Memory(b,k=8)
        m.estimate(b['z'][:2],b['u'][:2],2,b['episode'][:2])
        for k in b:np.testing.assert_array_equal(b[k],old[k])
    def test_gate_rejects_no_improvement(self):
        r={'before':np.ones(100),'after':np.ones(100)*2,'stats':np.zeros((100,5))}
        self.assertFalse(select_gate(r)['enabled'])
    def test_gate_accepts_information(self):
        r={'before':np.ones(100),'after':np.ones(100)*.5,'stats':np.zeros((100,5))}
        g=select_gate(r);self.assertTrue(g['enabled']);self.assertTrue(accepted(r['stats'],g).all())
    def test_disabled_gate_has_no_labels(self):
        self.assertFalse(accepted(np.zeros((20,5)),{'enabled':False}).any())
    def test_audit_is_not_recalibrated(self):
        c={'before':np.ones(100),'after':np.ones(100)*.5,'stats':np.ones((100,5))}
        g=select_gate(c);before=json.dumps(g)
        self.assertFalse(accepted(np.ones((20,5))*2,g).any());self.assertEqual(before,json.dumps(g))
    def test_reference_cancels_exactly(self):
        b=toy();m=Memory(b,k=8);n=3;old=np.ones((n,3,4),dtype='float32')
        g=[{'enabled':True,'selected':{'thresholds':[1e5]*5}}]*3
        y,mask,_=core.correct_queries(m,b['z'][:n],b['u'][:n],b['u'][:n],b['episode'][:n],g,old,b['u'][:n],b['u'][:n])
        np.testing.assert_array_equal(y,old);self.assertTrue(mask.all())
    def test_outside_bounds_not_clipped(self):
        b=toy();m=Memory(b,k=8);q=b['u'][:2].copy();q[:,1,0]=1.3;before=q.copy()
        g=[{'enabled':True,'selected':{'thresholds':[1e5]*5}}]*3
        _,mask,_=core.correct_queries(m,b['z'][:2],b['u'][:2],q,b['episode'][:2],g,np.ones((2,3,4)),q,b['u'][:2])
        self.assertTrue(mask[:,0].all());self.assertFalse(mask[:,1:].any());np.testing.assert_array_equal(q,before)
    def test_pair_reference_is_different_episode(self):
        b=toy(40);v=toy(15);v['episode']+=100;m=Memory(b,k=8)
        r=core.pair_records(m,v,2)
        self.assertTrue(np.all(v['episode']!=v['episode'][r['ref']]))
    def test_pair_labels_not_neighbor_inputs(self):
        b=toy();v=toy(15);v['episode']+=100;m=Memory(b,k=8)
        a=core.pair_records(m,v,1);v['residual'][:]=999;c=core.pair_records(m,v,1)
        np.testing.assert_array_equal(a['ref'],c['ref']);np.testing.assert_array_equal(a['stats'],c['stats'])
    def test_masked_loss_gradient(self):
        p=torch.ones(2,3,4,requires_grad=True);y=torch.zeros_like(p,requires_grad=True)
        mask=torch.tensor([[1,0,0],[0,0,0]],dtype=torch.bool)
        loss=query_loss(p,y,mask);loss.backward();self.assertAlmostEqual(loss.item(),1.)
        self.assertIsNone(y.grad);self.assertEqual(p.grad[1].abs().sum(),0)
    def test_zero_mask_zero_gradient(self):
        p=torch.randn(2,3,4,requires_grad=True)
        l=query_loss(p,torch.zeros_like(p),torch.zeros(2,3,dtype=torch.bool));l.backward()
        self.assertEqual(l.item(),0.);self.assertEqual(p.grad.abs().sum().item(),0.)
    def test_ddp_global_denominator(self):
        p=torch.ones(1,1,2,requires_grad=True)
        def reduce(count): count.add_(3)
        with patch('torch.distributed.is_available',return_value=True), patch('torch.distributed.is_initialized',return_value=True), patch('torch.distributed.get_world_size',return_value=2), patch('torch.distributed.all_reduce',side_effect=reduce):
            loss=query_loss(p,torch.zeros_like(p),torch.ones(1,1,dtype=torch.bool))
        self.assertAlmostEqual(loss.item(),.5)
        loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_bad_loss_shapes(self):
        with self.assertRaises(ValueError):query_loss(torch.ones(2,3,4),torch.ones(2,3,4),torch.ones(2,4))
    def test_loss_nonfinite_rejected(self):
        p=torch.zeros(1,2,2);p[0,0,0]=float('inf')
        with self.assertRaises(ValueError):query_loss(p,p,torch.ones(1,2))
    def test_npz_closes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';core.save_npz(p,x=np.arange(10))
            r=core.load_npz(p);np.testing.assert_array_equal(r['x'],np.arange(10))
            with self.assertRaises(FileExistsError):core.save_npz(p,x=np.ones(2))
            p.unlink();self.assertFalse(list(Path(d).iterdir()))
    def test_query_selection_unique(self):
        p=np.random.default_rng(1).normal(size=(100,5,10)).astype('float32')
        q,k=select_queries(p,np.arange(100,dtype='float32'),p[0],3)
        self.assertEqual(q.shape,(4,5,10));self.assertEqual(len(np.unique(q.reshape(4,-1),axis=0)),4)
        np.testing.assert_array_equal(q[0],p[0]);self.assertEqual(k[0],'returned_mean')
    def test_query_selection_collapsed_rejected(self):
        with self.assertRaises(RuntimeError):select_queries(np.zeros((100,5,10)),np.arange(100),np.zeros((5,10)),1)
    def test_no_rng_mutation_from_numpy_probes(self):
        np.random.seed(8);state=np.random.get_state()
        select_queries(np.arange(5000,dtype='float32').reshape(100,5,10),np.arange(100),np.ones((5,10)),42)
        now=np.random.get_state();np.testing.assert_array_equal(state[1],now[1])
    def test_pool_private_rng_and_with_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'query_chunks').mkdir();p=Path(d)/'query_chunks/chunk_00000.npz'
            core.save_npz(p,z=np.ones((4,4),dtype='float32'),u=np.ones((4,5,2),dtype='float32'),
                target=np.ones((4,5,4),dtype='float32'),old_target=np.zeros((4,5,4),dtype='float32'),mask=np.ones((4,5),dtype=bool))
            a=QueryPool(d,7);b=QueryPool(d,7);state=torch.get_rng_state().clone()
            x=a.sample(16,'cpu');y=b.sample(16,'cpu')
            self.assertTrue(torch.equal(x['u'],y['u']));self.assertTrue(torch.equal(state,torch.get_rng_state()))
    def test_safe_arithmetic(self):
        self.assertEqual(runtime.arithmetic('3+5'),8)
        with self.assertRaises(ValueError):runtime.arithmetic('__import__("os")')


class AdapterTests(unittest.TestCase):
    def test_official_interface_fixture(self):
        from queries import generate
        class Solver:
            def __init__(self,model,**kw):
                self.model=model;self.kw=kw
                self.rng=torch.Generator().manual_seed(kw['seed'])
            def configure(self,**kw):
                self.config=kw
                assert kw['n_envs']==1 and kw['action_space'].shape==(1,2)
            def __call__(self,info):
                mean=torch.zeros(1,5,10);var=torch.ones_like(mean)
                ex={k:v[:,None].expand(1,100,*v.shape[1:]) for k,v in info.items()}
                for _ in range(10):
                    a=torch.randn(1,100,5,10,generator=self.rng)*var[:,None]+mean[:,None]
                    a[:,0]=mean
                    c=self.model.get_cost(ex,a)
                    elite=a[0,torch.topk(c[0],10,largest=False).indices]
                    mean=elite.mean(0)[None];var=elite.std(0)[None]
                return {'actions':mean}
        fake=SimpleNamespace(solver=SimpleNamespace(CEMSolver=Solver),PlanConfig=lambda **kw:SimpleNamespace(**kw))
        gym=SimpleNamespace(spaces=SimpleNamespace(Box=lambda *a,**kw:SimpleNamespace(shape=kw['shape'])))
        rollout=SimpleNamespace(planner_context_rollout=lambda m,z,u,h:z[:,None]+u[:,:,:4].cumsum(1))
        with patch.dict(sys.modules,{'stable_worldmodel':fake,'gymnasium':gym,'planner_context_mh':rollout}):
            q,k,p,c=generate(torch.nn.Identity(),np.zeros(4),np.ones(4),17,'cpu')
        self.assertEqual(q.shape,(4,5,10));self.assertEqual(p.shape,(100,5,10));self.assertTrue(np.isfinite(c).all())


class ForwardTests(unittest.TestCase):
    def test_production_forward_is_pickleable(self):
        import pickle
        from functools import partial
        from train import ma_forward
        self.assertIs(pickle.loads(pickle.dumps(ma_forward)),ma_forward)
        self.assertTrue(pickle.dumps(partial(ma_forward,cfg=None,action_mean=None,action_std=None)))

    def test_original_objective_and_query_gradients(self):
        model=torch.nn.Linear(2,4,bias=False)
        pool=SimpleNamespace(sample=lambda n,d:{'z':torch.zeros(n,4),'u':torch.ones(n,5,2),
             'mask':torch.ones(n,5,dtype=torch.bool),'target':torch.ones(n,5,4),'old_target':torch.zeros(n,5,4)})
        calls=[];last={}
        def original(self,batch,stage,*a):
            calls.append(stage);return {'loss':self.model.weight.square().mean()}
        module=SimpleNamespace(model=model)
        fake=SimpleNamespace(planner_context_rollout=lambda mod,z,u,h:mod(u))
        with patch.dict(sys.modules,{'planner_context_mh':fake}):
            f=augment_forward(original,pool,3,'corrected',last)
            result=f(module,{},'train',None,None,None)
            expected=model.weight.square().mean()+.5*(model(torch.ones(3,5,2))-1).square().mean()
            torch.testing.assert_close(result['loss'],expected)
            result['loss'].backward();self.assertTrue(torch.isfinite(model.weight.grad).all())
            self.assertIn('loss',last);self.assertEqual(calls,['train'])
    def test_validation_does_not_sample_query_labels(self):
        pool=SimpleNamespace(sample=lambda *a:(_ for _ in ()).throw(AssertionError('validation must not query')))
        original=lambda *a:{'loss':torch.tensor(2.)}
        fake=SimpleNamespace(planner_context_rollout=None)
        with patch.dict(sys.modules,{'planner_context_mh':fake}):
            out=augment_forward(original,pool,4,'corrected',{})(None,{},'val',None,None,None)
        self.assertEqual(out['loss'],2.)
    def test_control_uses_same_mask_but_old_target(self):
        model=torch.nn.Linear(2,4,bias=False)
        q={'z':torch.zeros(2,4),'u':torch.ones(2,5,2),'mask':torch.ones(2,5,dtype=torch.bool),
           'target':torch.ones(2,5,4)*20,'old_target':torch.zeros(2,5,4)}
        pool=SimpleNamespace(sample=lambda *a:q)
        fake=SimpleNamespace(planner_context_rollout=lambda m,z,u,h:m(u))
        with patch.dict(sys.modules,{'planner_context_mh':fake}):
            f=augment_forward(lambda *a:{'loss':model.weight.sum()*0},pool,2,'same_queries',{})
            out=f(SimpleNamespace(model=model),{},'train',None,None,None)
        torch.testing.assert_close(out['loss'],.5*model(q['u']).square().mean())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-repo',action='store_true')
    parser.add_argument('--base-config',help='Saved baseline config for a read-only startup check')
    args=parser.parse_args()
    # Separate processes are essential: importing train_pc_mh_ald in this test
    # process first used to conceal the production config-before-training bug.
    startup=[sys.executable,str(Path(__file__).with_name('test_resolver_startup.py'))]
    if args.with_repo:
        runtime.sources()
        startup.append('--with-repo')
    if args.base_config:
        if not args.with_repo: parser.error('--base-config requires --with-repo')
        startup.extend(['--base-config',args.base_config])
    subprocess.run(startup,check=True)
    if args.with_repo:
        import planner_context_mh, train_pc_mh_ald
        print('PINNED REPOSITORY HASH / IMPORT CHECK PASS',flush=True)
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    print(f'MULTIANCHOR CPU TESTS PASS: {result.testsRun}; real LeWM/GPU/Slurm pipeline NOT RUN')
