#!/usr/bin/env python3
"""CPU bookkeeping/interface fixtures only; optional pinned repository check."""
import argparse
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch
import comparison as c
import run_eval as r


@contextlib.contextmanager
def module_override(name, obj):
    missing=object(); old=sys.modules.get(name,missing);sys.modules[name]=obj
    try:yield
    finally:
        if old is missing:sys.modules.pop(name,None)
        else:sys.modules[name]=old


def result(success=True):
    return {'successes':[success]*100,'contexts':[{}]*100,'initial_physics':[[0.,1.]]*100,
            'initial_recipes':[{'goal':[1,2]}]*100,'initial_images':['hash']*100,
            'native_steps':[[] for _ in range(100)],'env_spec':{'id':'fixture'}}


def small_rows():
    rows=[]
    for budget in c.BUDGETS:
        for seed in (1,2):
            for i in range(3):
                rows.append({'budget':budget,'seed':seed,'eval_index':i,'episode_idx':i,
                             'pcmh_success':1 if i!=2 else 0,'anchor_success':1 if i==0 else 0})
    return rows


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.e=np.repeat(np.arange(100),60);self.s=np.tile(np.arange(60),100)
    def test_deterministic_and_disjoint_rows(self):
        a=c.select_cohort(self.e,self.s,[3,7]);b=c.select_cohort(self.e,self.s,[3,7])
        self.assertEqual(a,b);self.assertEqual(len(a),500)
        self.assertEqual(len({x['dataset_row'] for x in a}),500)
    def test_excluded_whole_episode_and_valid_future(self):
        a=c.select_cohort(self.e,self.s,[3,7])
        self.assertTrue(all(x['episode_idx'] not in (3,7) and x['goal_step']<=59 for x in a))
    def test_sorted_per_seed(self):
        a=c.select_cohort(self.e,self.s,[])
        for seed in c.SEEDS:
            p=[x['dataset_row'] for x in a if x['seed']==seed]
            self.assertEqual(p,sorted(p));self.assertEqual(len(p),100)
    def test_global_rng_unchanged(self):
        np.random.seed(4);before=np.random.get_state();c.select_cohort(self.e,self.s,[])
        after=np.random.get_state();np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])
    def test_bad_inputs(self):
        for e,s in (([1],[1,2]),([float('nan')],[0]),([1],[.5]),([1],[-1])):
            with self.assertRaises(ValueError):c.select_cohort(e,s,[],seeds=(1,),n=1)
    def test_too_few(self):
        with self.assertRaises(ValueError):c.select_cohort([1]*20,list(range(20)),[])
    def test_duplicate_seeds(self):
        with self.assertRaises(ValueError):c.select_cohort(self.e,self.s,[],seeds=(1,1))
    def test_exclusion_csv_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';p.write_text('episode_idx,group\n2,failed\n2,success\n5,failed\n')
            self.assertEqual(c.excluded_episodes(p),[2,5]);p.unlink()
    def test_no_outcome_in_selection(self):
        # Only two input columns and an episode exclusion set define selection.
        a=c.select_cohort(self.e,self.s,[1]);self.assertNotIn('success',str(a[0]))


class OutputTests(unittest.TestCase):
    def test_seal_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';c.sealed_write(p,{'x':[1,2]});self.assertEqual(c.sealed_read(p),{'x':[1,2]})
            with self.assertRaises(FileExistsError):c.sealed_write(p,{})
    def test_seal_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';c.sealed_write(p,{'x':1});j=c.read(p);j['payload']['x']=2;c.dump(p,j)
            with self.assertRaises(RuntimeError):c.sealed_read(p)
    def test_nan_write_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json'
            with self.assertRaises(ValueError):c.sealed_write(p,{'x':float('nan')})
            self.assertFalse(p.exists())
    def test_success_not_inferred_from_mean(self):
        for x in ({'success_rate':.95},{'successes':[1]*100},{'successes':[True]*99}):
            with self.assertRaises(ValueError):c.successes(x)
    def test_effect_sign_and_discordance(self):
        tables,totals=c.summary(small_rows(),seeds=(1,2),n=3)
        self.assertEqual(len(tables),4)
        for t in totals:
            self.assertAlmostEqual(t['pcmh_minus_anchor_pp'],100/3)
            self.assertEqual(t['pcmh_only'],2);self.assertEqual(t['anchor_only'],0)
    def test_missing_and_duplicate_cells_fail(self):
        rows=small_rows()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(RuntimeError):c.summary(bad,seeds=(1,2),n=3)
    def test_bootstrap_reproducible_and_zero_difference(self):
        rows=small_rows()
        for row in rows:row['anchor_success']=row['pcmh_success']
        a=c.summary(rows,seeds=(1,2),n=3);self.assertEqual(a,c.summary(rows,seeds=(1,2),n=3))
        for t in a[1]:self.assertEqual((t['episode_cluster_ci95_low_pp'],t['episode_cluster_ci95_high_pp']),(0,0))
    def test_cell_identity_checked(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'runs').mkdir();c.dump(out/'protocol.json',{});c.dump(out/'cohort.json',{})
            identity=r.cell_identity(out,1000,142,'pcmh')
            c.sealed_write(r.result_path(out,1000,142,'pcmh'),{'identity':identity,'result':result()})
            self.assertEqual(r.load_cell(out,1000,142,'pcmh')['identity'],identity)
            c.dump(out/'protocol.json',{'changed':1})
            with self.assertRaises(RuntimeError):r.load_cell(out,1000,142,'pcmh')


class AdapterTests(unittest.TestCase):
    def test_budget_forwarding_and_no_trace(self):
        calls=[]
        def fake(cfg,ds,process,model,seed,selected,iterations,trace):
            calls.append((cfg.solver.num_samples,selected,iterations,trace,torch.is_grad_enabled()))
            return result(),[]
        with module_override('eval_cem_mh_diagnostics',NS(run_closed=fake,plain=lambda x:x)):
            for n in (100,300):r.run_one(NS(solver=NS(num_samples=n)),None,{},torch.nn.Identity(),142)
        self.assertEqual(calls,[(100,set(),set(),False,False),(300,set(),set(),False,False)])
    def test_unexpected_trace_rejected(self):
        f=NS(run_closed=lambda *a:(result(),[1]),plain=lambda x:x)
        with module_override('eval_cem_mh_diagnostics',f),self.assertRaises(RuntimeError):r.run_one(None,None,{},None,142)
    def test_check_pair_keeps_legacy_gate(self):
        calls=[];a=result();b=result(False)
        with module_override('eval_cem_mh_diagnostics',NS(check_initial=lambda x,y:calls.append(1))):r.check_pair(a,b)
        self.assertEqual(calls,[1])
    def test_goal_recipe_mismatch_rejected(self):
        a=result();b=copy.deepcopy(a);b['initial_recipes'][0]={'goal':[9,9]}
        with module_override('eval_cem_mh_diagnostics',NS(check_initial=lambda *a:None)),self.assertRaises(RuntimeError):r.check_pair(a,b)
    def test_environment_mismatch_rejected(self):
        a=result();b=copy.deepcopy(a);b['env_spec']['id']='changed'
        with module_override('eval_cem_mh_diagnostics',NS(check_initial=lambda *a:None)),self.assertRaises(RuntimeError):r.check_pair(a,b)
    def test_nonfinite_model_rejected(self):
        m=torch.nn.Linear(1,1);m.weight.data.fill_(float('nan'))
        with self.assertRaises(RuntimeError):r.model_digest(m)
    def test_digest_includes_buffers(self):
        m=torch.nn.BatchNorm1d(3);a=r.model_digest(m);m.running_mean.add_(1);self.assertNotEqual(a,r.model_digest(m))
    def test_runtime_paths(self):
        with self.assertRaises(ValueError):r.within(Path('/tmp/root'),'/tmp/elsewhere')
        self.assertEqual(r.checkpoint_policy(Path('/tmp/root'),'/tmp/root/run/model_object.ckpt'),'run/model')
    def test_training_rejects_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'smoke';p.mkdir();c.dump(p/'training_complete.json',{'mode':'smoke'});c.dump(p/'protocol.json',{})
            with self.assertRaises(RuntimeError):r.training_inputs(root,p)
    def test_launcher_syntax_and_single_gpu(self):
        p=r.HERE/'submit.sh';subprocess.run(['bash','-n',str(p)],check=True);s=p.read_text()
        self.assertIn('NODE:-4090node3',s);self.assertIn('#SBATCH --gres=gpu:1',s)
        self.assertIn('#SBATCH --ntasks=1',s);self.assertNotIn('#SBATCH --mem',s)
        self.assertNotIn('export CUDA_VISIBLE_DEVICES',s);self.assertIn('/tmp/pcaev.',s)
    def test_bad_node_fails_before_conda(self):
        for node in ('4090node1','4090node2,4090node3','bad'):
            p=subprocess.run(['bash',str(r.HERE/'submit.sh'),'run','test'],env={**os.environ,'NODE':node},capture_output=True,text=True)
            self.assertEqual(p.returncode,2)
    def test_full_summary_from_sealed_fixtures(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'runs').mkdir();cases=[]
            c.dump(out/'protocol.json',{'effect_direction':'PC-MH minus anchor'})
            for seed in c.SEEDS:
                for i in range(100):cases.append({'seed':seed,'eval_index':i,'dataset_row':100*seed+i,
                    'episode_idx':i,'start_step':0,'goal_step':25})
            c.sealed_write(out/'cohort.json',{'cases':cases});c.write_csv(out/'cohort.csv',cases)
            c.dump(out/'model_audit.json',{})
            for b in c.BUDGETS:
                for seed in c.SEEDS:
                    for label in c.LABELS:
                        c.sealed_write(r.result_path(out,b,seed,label),{'identity':r.cell_identity(out,b,seed,label),
                            'result':result(label=='pcmh')})
            with contextlib.redirect_stdout(io.StringIO()):artifacts=r.summarize(out)
            totals=c.read(out/'summary.json')['by_budget']
            self.assertTrue(all(x['pcmh_minus_anchor_pp']==100 for x in totals));self.assertEqual(len(artifacts),29)
            with (out/'paired_cases.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),1000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--with-repo',action='store_true');args=p.parse_args()
    if args.with_repo:r.sources();print('PINNED EVALUATION SOURCE CHECK PASS',flush=True)
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result_=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result_.wasSuccessful():raise SystemExit(1)
    print(f'PAIRED EVAL CPU TESTS PASS: {result_.testsRun}; real LeWM/PushT/GPU evaluation NOT RUN')
