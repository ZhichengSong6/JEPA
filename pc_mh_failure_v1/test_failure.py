#!/usr/bin/env python3
"""Local synthetic tests only; no claim of real PushT/CUDA validation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analysis_core import *
spec=importlib.util.spec_from_file_location('failure_runner',HERE/'run.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)

class CoreTests(unittest.TestCase):
    def test_groups(self):
        self.assertEqual([group(a,b) for a,b in [(1,1),(0,0),(0,1),(1,0)]],
                         ['both_success','both_fail','rescued','regressed'])

    def rows(self):
        return [{'seed':42+i//100,'eval_index':i%100,'group':g}
                for i,g in enumerate(['both_fail']*20+['regressed']*20+['rescued']*20+['both_success']*140)]

    def test_selection_repeatable(self):
        a=choose_cases(self.rows()); b=choose_cases(list(reversed(self.rows())))
        self.assertEqual(a,b); self.assertEqual(len(a),12)
        self.assertEqual({g:sum(r['group']==g for r in a) for g in ('both_fail','regressed','rescued','both_success')},
                         {'both_fail':4,'regressed':3,'rescued':3,'both_success':2})

    def test_selection_shortage(self):
        rows=[{'seed':42,'eval_index':i,'group':'both_success'} for i in range(3)]
        self.assertEqual(len(choose_cases(rows)),3)
        self.assertEqual(len(choose_cases(self.rows(),1)),1)

    def test_duplicate_id_rejected(self):
        r=self.rows(); r.append(r[0])
        with self.assertRaises(ValueError): choose_cases(r)

    def test_decomposition_identity(self):
        rng=np.random.default_rng(4)
        a,b,c=[rng.normal(size=(100,192)) for _ in range(3)]
        d=decomposition(a,b,c)
        self.assertLess(d['identity_error'],1e-12)
        self.assertGreater(d['reference_fit_mse'],0)

    def test_decomposition_nonfinite(self):
        with self.assertRaises(ValueError): decomposition([[float('nan')]],[[0]],[[0]])

    def test_ranking(self):
        self.assertEqual(cost_stats([3,2,1],[1,2,3])['inversion_rate'],1)
        self.assertEqual(cost_stats([1,2,3],[1,2,3])['inversion_rate'],0)
        self.assertIsNone(cost_stats([1,1,1],[1,1,1])['inversion_rate'])

    def test_flags_predictor(self):
        d=flags([1,2,3],[2,1,3],[False,True,False],[0,1],False)
        self.assertTrue(d['predictor_top1_miss']); self.assertTrue(d['elite_mean_failure'])
        self.assertFalse(d['encoder_top1_task_mismatch'])

    def test_flags_metric(self):
        d=flags([1,2,3],[1,2,3],[False,True,False],[0,1],False)
        self.assertTrue(d['encoder_top1_task_mismatch']); self.assertFalse(d['predictor_top1_miss'])

    def test_no_success_is_local_flag(self):
        d=flags([1,2],[1,2],[False,False],[0],False)
        self.assertTrue(d['no_success_in_recorded_population'])
        self.assertNotIn('unreachable',d)

    def test_probe_shape_norm_bounds(self):
        center=np.zeros((5,10)); probes,specs=make_joint_probes(center,seed=3)
        self.assertEqual(probes.shape,(37,5,10))
        self.assertTrue(np.abs(probes).max()<=1)
        for s in specs:
            ix=s['indices']
            for name in ('++','+-','-+','--','P+','P-','Q+','Q-'):
                self.assertAlmostEqual(np.linalg.norm(probes[ix[name]]-probes[0]),s['joint_radius'],places=6)
            self.assertTrue(np.allclose(probes[ix['p+']]+probes[ix['p-']],2*probes[0]))
            self.assertTrue(np.allclose(probes[ix['++']],probes[ix['p+']]+probes[ix['q+']]-probes[0]))

    def test_probe_boundary_and_skip(self):
        p,s=make_joint_probes(np.ones((5,10)),seed=9)
        self.assertEqual(len(p),1); self.assertTrue(all(x['skipped'] for x in s))
        c=np.ones((5,10))*.999
        p,s=make_joint_probes(c,seed=9)
        self.assertTrue(np.abs(p).max()<=1+1e-6)
        self.assertTrue(all(x['joint_radius']<.1565 for x in s))

    def test_probe_does_not_modify_global_rng(self):
        np.random.seed(5); state=np.random.get_state()
        make_joint_probes(np.zeros((5,10)),seed=9)
        now=np.random.get_state()
        self.assertTrue(np.array_equal(state[1],now[1])); self.assertEqual(state[2],now[2])

    def test_probe_invalid_center(self):
        with self.assertRaises(ValueError): make_joint_probes(np.ones((5,10))*2,seed=9)

    def test_interaction_linear_zero(self):
        p,s=make_joint_probes(np.zeros((5,10)),seed=9)
        f=np.cumsum(p.sum(-1),axis=1)[...,None]
        rows=interaction_rows({'replay':f,'teacher':f.copy()},s)
        self.assertTrue(all(r['real_interaction_mse']<1e-10 for r in rows))
        self.assertTrue(all(r['interaction_relative_rmse'] is None for r in rows))

    def test_interaction_nonzero_teacher_better(self):
        p,s=make_joint_probes(np.zeros((5,10)),seed=9,pairs=((0,1),))
        x=p[:,0,0]; y=p[:,1,0]
        linear=np.repeat((x+y)[:,None,None],5,axis=1)
        true=linear.copy(); true[:,1:,0]+=5*(x*y)[:,None]
        rows=interaction_rows({'replay':true,'teacher':true,'student':linear},s)
        h=[r for r in rows if r['horizon']==5]
        t=next(r for r in h if r['model']=='teacher'); v=next(r for r in h if r['model']=='student')
        self.assertGreater(t['real_interaction_mse'],1e-10)
        self.assertEqual(t['interaction_error_mse'],0)
        self.assertGreater(v['interaction_error_mse'],0)

    def test_json_csv(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d)/'a.json'; dump(p,{'array':np.array([1,2])})
            self.assertEqual(json.loads(p.read_text()),{'array':[1,2]})
            write_csv(Path(d)/'a.csv',[{'a':1},{'b':2}])
            self.assertIn('a,b',(Path(d)/'a.csv').read_text())
            with self.assertRaises(ValueError): dump(p,{'bad':float('nan')})

    def test_safe_paths(self):
        root=HERE.resolve()
        self.assertEqual(runner.safe_path(root,'abc'),root/'abc')
        for p in ('../escape','/tmp/escape','.'):
            with self.assertRaises(ValueError): runner.safe_path(root,p)

    def test_stage_hash_check(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d); (p/'a.txt').write_text('a')
            dump(p/'survey_complete.json',{'artifacts':{'a.txt':sha(p/'a.txt')}})
            runner.verify_stage(p,'survey')
            (p/'a.txt').write_text('b')
            with self.assertRaises(RuntimeError): runner.verify_stage(p,'survey')

class IntegrationTests(unittest.TestCase):
    def test_survey_keeps_all_environments(self):
        from omegaconf import OmegaConf
        calls=[]
        models={l:object() for l in LABELS}
        cfg=OmegaConf.create({'world':{'num_envs':100},'diag_episodes':list(range(100)),
                             'diag_start':[0]*100})
        def closed(cfg,ds,process,model,seed,selected,iterations,enabled):
            calls.append((model,len(cfg.diag_episodes),selected,enabled))
            ok=[True]*100
            if model is models['pcmh']: ok[0]=False
            else: ok[1]=False
            return {'successes':ok},[]
        legacy=types.SimpleNamespace(run_closed=closed,check_initial=lambda a,b:None,
                                     save_runtime_sources=lambda p,s:None)
        with tempfile.TemporaryDirectory(dir=HERE) as d, patch.dict(sys.modules,{'eval_cem_mh_diagnostics':legacy,'stable_worldmodel':types.SimpleNamespace()}), \
             patch.object(runner,'build_data',return_value=(cfg,object(),{},np.arange(100))),patch.object(runner,'close_dataset'):
            p=Path(d)
            runner.survey(p,{'seeds':[42],'iterations':[0,9],'max_cases':4,'selection_seed':2},models)
            self.assertEqual(len(calls),2)
            self.assertTrue(all(n==100 and not sel and not en for _,n,sel,en in calls))
            rows=json.loads((p/'cases.json').read_text())
            self.assertEqual(len(rows),100)
            self.assertEqual(rows[0]['group'],'rescued')
            self.assertEqual(rows[1]['group'],'regressed')

    def test_native_replay_continues_after_success_but_rejects_truncation(self):
        class Raw:
            def __init__(self): self.k=0; self.goal_state=np.array([1]); self.truncate=False
            def step(self,a):
                self.k+=1
                return {},0,self.k==3,self.truncate,{}
            def render(self): return np.zeros((4,4,3),dtype=np.uint8)
            def _get_obs(self): return np.array([self.k])
            def eval_state(self,g,s): return False,abs(s[0]-g[0])
        raw=Raw()
        fake=types.SimpleNamespace(restore_prefix=lambda r,s:setattr(r,'k',0))
        with patch.dict(sys.modules,{'cem_mh_diag_core':fake}):
            r=runner.replay_sequence(raw,{},np.zeros((25,2)),True)
            self.assertEqual(raw.k,25); self.assertEqual(len(r['frames']),5)
            self.assertTrue(r['ever_success']); self.assertFalse(r['endpoint_success'])
            self.assertEqual(r['first_success_step'],3)
            raw.truncate=True
            with self.assertRaises(RuntimeError): runner.replay_sequence(raw,{},np.zeros((25,2)))

    def test_reference_decomposition_and_no_grad(self):
        import torch
        fake=types.SimpleNamespace(stepwise_observation_targets=lambda t,o,u,p,max_history:o[:,1:]+p[:,:,:2]-u[:,:,:2])
        observed=np.arange(12,dtype=np.float32).reshape(6,2)
        plans=np.ones((3,5,10),np.float32); center=np.zeros((5,10),np.float32)
        pred=np.ones((3,5,2),np.float32)*2; tc=np.ones((1,5,2),np.float32)
        with patch.dict(sys.modules,{'pc_mh_stepwise_v1.planner_context_stepwise_mh':fake}):
            targets=runner.reference_targets(None,torch.zeros(1,2),plans,center,observed,pred,tc,'cpu')
        self.assertTrue(np.array_equal(targets['pcmh'],targets['stepwise']))
        self.assertEqual(targets['pcmh'].shape,(3,5,2))

    def test_model_digest_includes_buffers(self):
        import torch
        m=torch.nn.BatchNorm1d(2).eval()
        before=runner.state_digest(m)
        m.running_mean[0]+=1
        self.assertNotEqual(before,runner.state_digest(m))

    def test_source_score_mismatch_blocks_before_oracle(self):
        import torch
        model=object()
        fake=types.SimpleNamespace(model_scores=lambda *a:(np.zeros(100),np.arange(10)))
        fake_eval=types.SimpleNamespace(img_transform=lambda cfg:None)
        tr={'info':{}}
        pop={'candidates':np.zeros((100,5,10),np.float32),'source_cost':torch.ones(100),
             'source_elite':torch.arange(10),'iteration':9}
        cd=(None,torch.zeros(1,2),np.zeros((1,2)),np.zeros((5,10)),np.zeros((6,2)),np.zeros((1,5,2)))
        with patch.dict(sys.modules,{'eval_cem_mh_diagnostics':fake,'eval':fake_eval}),\
             patch.object(runner,'model_rollouts',return_value=np.zeros((100,5,2))),\
             patch.object(runner,'reference_targets',return_value={l:np.zeros((100,5,2)) for l in LABELS}),\
             patch.object(runner,'replay_sequence') as replay:
            with self.assertRaisesRegex(RuntimeError,'Source rescore mismatch'):
                runner.analyze_population(None,{},dict.fromkeys((*LABELS,'teacher'),model),'cpu',tr,pop,cd,HERE,{'source':'pcmh'})
            replay.assert_not_called()

    def test_full_population_outputs_and_final_mean(self):
        import torch
        plans=np.zeros((100,5,10),dtype=np.float32)
        plans[:,0,0]=np.arange(100)/100
        predicted=np.cumsum(plans[:,:,:2],axis=1)
        costs=np.square(predicted[:,-1]).sum(-1)
        elite=torch.topk(torch.from_numpy(costs),10,largest=False).indices.numpy()
        center=plans[elite].mean(0)
        goal=np.zeros((1,2),dtype=np.float32)
        fake=types.SimpleNamespace(
            model_scores=lambda *a:(costs.copy(),elite.copy()),
            decode=lambda a,sc:np.asarray(a).reshape(-1,25,2),
            encode_images=lambda model,tf,imgs,dev:torch.tensor([[float(im[0,0,0])/100,0] for im in imgs]))
        fake_eval=types.SimpleNamespace(img_transform=lambda cfg:None)
        def replay(raw,snap,acts,*args):
            x=float(np.asarray(acts).sum())
            return {'ever_success':x<.2,'endpoint_success':x<.2,'first_success_step':1 if x<.2 else None,
                    'official_endpoint_distance':x,'state':np.array([x]),'frames':[],
                    'image':np.full((4,4,3),int(x*100),dtype=np.uint8)}
        tr={'info':{},'snapshot':{'image':np.zeros((4,4,3),dtype=np.uint8)},
            'raw_policy_goal':np.zeros((4,4,3),dtype=np.uint8)}
        pop={'candidates':plans,'source_cost':torch.from_numpy(costs),'source_elite':torch.from_numpy(elite),'iteration':9}
        cd=(None,torch.zeros(1,2),goal,center,np.zeros((6,2)),np.zeros((1,5,2)))
        key={'source':'pcmh','seed':42,'eval_index':1,'solve_no':0,'group':'both_fail'}
        with tempfile.TemporaryDirectory(dir=HERE) as d, patch.dict(sys.modules,{'eval_cem_mh_diagnostics':fake,'eval':fake_eval}),\
             patch.object(runner,'model_rollouts',return_value=predicted),\
             patch.object(runner,'reference_targets',return_value={l:predicted.copy() for l in LABELS}),\
             patch.object(runner,'replay_sequence',side_effect=replay):
            r=runner.analyze_population(None,{'action':None},dict.fromkeys((*LABELS,'teacher'),object()),
                                      'cpu',tr,pop,cd,Path(d),key)
            self.assertEqual(len(r),2)
            self.assertTrue(list((Path(d)/'populations').glob('*.npz')))
            self.assertTrue(list((Path(d)/'images').rglob('*mean_endpoint.png')))
            self.assertLess(r[0]['identity_error'],1e-12)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if not result.wasSuccessful(): sys.exit(1)
    print(f'LOCAL SYNTHETIC TESTS PASS: {result.testsRun}; real GPU/PushT integration NOT RUN')
