#!/usr/bin/env python3
"""CPU bookkeeping/integration fixtures; no real LeWM, PushT, CUDA or Slurm."""
from __future__ import annotations
import argparse
import contextlib
import copy
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import run_parent as R
C,E=R.C,R.E


def config(part,seed,budget):
    n,i,k=C.BUDGETS[budget]
    return {'seed':seed,'policy':'random',
        'eval':{'num_eval':100,'eval_budget':50,'goal_offset_steps':25,
                'dataset_name':'pusht_expert_train','img_size':224,'callables':[]},
        'world':{'env_name':'swm/PushT-v1','num_envs':100,'max_episode_steps':100,'history_size':1,'frame_skip':1},
        'plan_config':{'horizon':5,'receding_horizon':5,'action_block':5},
        'solver':{'_target_':'stable_worldmodel.solver.CEMSolver','num_samples':n,'n_steps':i,'topk':k,
                  'batch_size':1,'seed':seed,'device':'cuda:0','var_scale':1.},
        'diag_episodes':[x['episode_idx'] for x in part],'diag_start':[x['start_step'] for x in part]}


def result(label,seed):
    offset={'lewm10':1,'anchor':2,'pcmh':3}[label]
    return {'successes':[bool((i+seed+offset)%7) for i in range(100)],
        'contexts':[{'variation_count':1,'seed':seed} for _ in range(100)],
        'initial_physics':[[float(seed),i] for i in range(100)],
        'initial_recipes':[{'goal':i,'seed':seed} for i in range(100)],
        'initial_images':[str(i) for i in range(100)],'native_steps':[[] for _ in range(100)],
        'env_spec':{'id':'fixture'},'metrics':{}}


def pair_check(a,b):
    C.successes(a);C.successes(b)
    for k in ('contexts','initial_physics','initial_recipes','initial_images','env_spec'):
        if a[k]!=b[k]:raise RuntimeError('Initial mismatch: '+k)


class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.source=self.root/'existing';self.source.mkdir();(self.source/'runs').mkdir()
        self.out=self.root/'new';self.cases=[];self.cells={}
        for j,s in enumerate(C.SEEDS):
            for i in range(100):self.cases.append({'seed':s,'eval_index':i,'dataset_row':j*100+i,
                'episode_idx':(j*100+i)//2,'start_step':i,'goal_step':i+25})
        self.parent=self.root/'lewm_epoch_10_object.ckpt';self.parent.write_text('parent')
        self.meta={'root':str(self.root),'repo':str(R.REPO),'output':str(self.source),'excluded_episodes':[],
            'policies':{'pcmh':'old_model','anchor':'new_model'},'anchor_run':str(self.root/'trained'),
            'checkpoints':{'pcmh':str(self.root/'old_model_object.ckpt'),'anchor':str(self.root/'new_model_object.ckpt')},
            'input_sha256':{str(self.parent):C.sha(self.parent)}}
        C.dump(self.source/'protocol.json',self.meta)
        C.sealed_write(self.source/'cohort.json',{'protocol_sha256':C.sha(self.source/'protocol.json'),'cases':self.cases})
        C.write_csv(self.source/'cohort.csv',self.cases)
        original_rows=[]
        for seed in C.SEEDS:
            part=[c for c in self.cases if c['seed']==seed]
            for budget in C.BUDGETS:
                for label in C.LABELS:
                    cell={'identity':E.cell_identity(self.source,budget,seed,label),
                          'policy':self.meta['policies'][label], 'config':config(part,seed,budget),
                          'result':result(label,seed),'elapsed_seconds':1.0}
                    C.sealed_write(E.result_path(self.source,budget,seed,label),cell)
                    self.cells[budget,seed,label]=cell
                original_rows.extend(C.paired_rows(part,result('pcmh',seed),result('anchor',seed),budget))
        _,totals=C.summary(original_rows);C.write_csv(self.source/'summary.csv',totals)
        C.dump(self.source/'model_audit.json',{'initial_state_sha256':{'anchor':'model'}})
        self.seal()
        self.train={'parent_checkpoint':str(self.parent),'parent_checkpoint_sha256':C.sha(self.parent)}
        self.info={'path':str(self.parent),'policy':'lewm_epoch_10','sha256':C.sha(self.parent)}

    def seal(self):
        artifacts={str(p.relative_to(self.source)):C.sha(p) for p in self.source.rglob('*')
                   if p.is_file() and p.name!='COMPLETE.json'}
        C.dump(self.source/'COMPLETE.json',{'runs':20,'paired_cases_per_budget':500,
            'checkpoints_unchanged':True,'initial_conditions_verified':True,'artifacts':artifacts})

    def prepare(self):
        with patch.dict(os.environ,{'STABLEWM_HOME':str(self.root)}),\
             patch.object(R,'sources',return_value={'code':'1'}),\
             patch.object(E,'versions',return_value={'test':'1'}),\
             patch.object(E,'verify',return_value=self.meta),\
             patch.object(R,'parent_info',return_value=self.info),contextlib.redirect_stdout(io.StringIO()):
            R.prepare(types.SimpleNamespace(source=self.source,output=self.out,node='4090node3'))
        self.m=C.read(self.out/'protocol.json')

    def run(self, failure=False):
        calls=[]
        def one(cfg,ds,proc,model,seed):
            calls.append((cfg['solver']['num_samples'],seed,model))
            if failure:raise RuntimeError('fixture failure')
            return result('lewm10',seed)
        with patch.object(E,'run_one',side_effect=one),patch.object(E,'model_digest',return_value='stable'),\
             patch.object(E,'check_pair',side_effect=pair_check),contextlib.redirect_stdout(io.StringIO()):
            R.evaluate_missing(self.out,self.m,self.cases,self.cells,None,None,'PARENT',lambda x:x)
        return calls


class ParentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name)

    def test_source_readonly_and_full_validation(self):
        before={str(p):C.sha(p) for p in self.f.source.rglob('*') if p.is_file()}
        with patch.object(E,'verify',return_value=self.f.meta):
            m,complete,cases,cells=R.source_bundle(self.f.source)
        self.assertEqual((len(cases),len(cells),complete['runs']),(500,20,20))
        self.assertEqual(before,{str(p):C.sha(p) for p in self.f.source.rglob('*') if p.is_file()})

    def test_source_not_complete_rejected(self):
        c=C.read(self.f.source/'COMPLETE.json');c['runs']=19;C.dump(self.f.source/'COMPLETE.json',c)
        with patch.object(E,'verify',return_value=self.f.meta),self.assertRaisesRegex(RuntimeError,'complete'):
            R.source_bundle(self.f.source)

    def test_tampered_old_cell_rejected(self):
        E.result_path(self.f.source,1000,142,'anchor').write_text('{}')
        with patch.object(E,'verify',return_value=self.f.meta),self.assertRaisesRegex(RuntimeError,'changed'):
            R.source_bundle(self.f.source)

    def test_missing_old_cell_rejected(self):
        E.result_path(self.f.source,3000,146,'anchor').unlink()
        with patch.object(E,'verify',return_value=self.f.meta),self.assertRaises(FileNotFoundError):
            R.source_bundle(self.f.source)

    def test_missing_artifact_coverage_rejected(self):
        c=C.read(self.f.source/'COMPLETE.json');del c['artifacts']['runs/B1000_seed142_anchor.json']
        C.dump(self.f.source/'COMPLETE.json',c)
        with patch.object(E,'verify',return_value=self.f.meta),self.assertRaisesRegex(RuntimeError,'missing'):
            R.source_bundle(self.f.source)

    def test_wrong_config_rejected_even_if_resealed(self):
        p=E.result_path(self.f.source,1000,142,'anchor');x=C.sealed_read(p);p.unlink();x['config']['solver']['seed']=99
        C.sealed_write(p,x);self.f.seal()
        with patch.object(E,'verify',return_value=self.f.meta),self.assertRaisesRegex(RuntimeError,'configuration'):
            R.source_bundle(self.f.source)

    def test_wrong_cell_policy_rejected(self):
        p=E.result_path(self.f.source,1000,142,'anchor');x=C.sealed_read(p);p.unlink();x['policy']='wrong'
        C.sealed_write(p,x);self.f.seal()
        with patch.object(E,'verify',return_value=self.f.meta),self.assertRaisesRegex(RuntimeError,'policy'):
            R.source_bundle(self.f.source)

    def test_parent_from_training_hash(self):
        paths={k:Path(p) for k,p in self.f.meta['checkpoints'].items()}
        with patch.object(E,'training_inputs',return_value=({},self.f.train,paths,self.f.meta['policies'],{})):
            self.assertEqual(R.parent_info(self.f.meta),self.f.info)

    def test_parent_replaced_rejected(self):
        self.f.parent.write_text('changed')
        paths={k:Path(p) for k,p in self.f.meta['checkpoints'].items()}
        with patch.object(E,'training_inputs',return_value=({},self.f.train,paths,self.f.meta['policies'],{})),\
             self.assertRaisesRegex(RuntimeError,'Parent checkpoint changed'):
            R.parent_info(self.f.meta)

    def test_wrong_parent_name_rejected(self):
        p=self.f.root/'lewm_epoch_20_object.ckpt';p.write_text('other');self.f.train['parent_checkpoint']=str(p)
        paths={k:Path(p) for k,p in self.f.meta['checkpoints'].items()}
        with patch.object(E,'training_inputs',return_value=({},self.f.train,paths,self.f.meta['policies'],{})),self.assertRaises(RuntimeError):
            R.parent_info(self.f.meta)

    def test_prepare_preserves_cohort_bytes_and_old_files(self):
        before=C.sha(self.f.source/'COMPLETE.json');self.f.prepare()
        self.assertEqual(C.sha(self.f.source/'cohort.json'),C.sha(self.f.out/'cohort.json'))
        self.assertEqual(before,C.sha(self.f.source/'COMPLETE.json'))
        self.assertEqual(self.f.m['new_runs'],10)

    def test_fresh_output_required(self):
        self.f.prepare()
        with self.assertRaises(ValueError):self.f.prepare()

    def test_only_ten_parent_calls_and_no_resampling(self):
        self.f.prepare()
        with patch.object(C,'select_cohort',side_effect=AssertionError('resampled')):
            calls=self.f.run()
        self.assertEqual(len(calls),10);self.assertEqual({x[2] for x in calls},{'PARENT'})
        self.assertEqual({x[0] for x in calls},{100,300})
        self.assertEqual(len(list((self.f.out/'runs').glob('*.json'))),10)
        self.assertEqual(len(list((self.f.source/'runs').glob('*.json'))),20)

    def test_full_resume_has_zero_evaluations(self):
        self.f.prepare();self.f.run();self.assertEqual(self.f.run(),[])

    def test_partial_resume_only_missing_cell(self):
        self.f.prepare();self.f.run();R.result_path(self.f.out,3000,145).unlink()
        self.assertEqual(self.f.run(),[(300,145,'PARENT')])

    def test_corrupt_parent_seal_blocks_reuse(self):
        self.f.prepare();self.f.run();p=R.result_path(self.f.out,1000,142)
        x=C.read(p);x['payload']['policy']='wrong';C.dump(p,x)
        with self.assertRaisesRegex(RuntimeError,'checksum'):self.f.run()

    def test_resealed_wrong_parent_config_blocks_reuse(self):
        self.f.prepare();self.f.run();p=R.result_path(self.f.out,1000,142)
        x=C.sealed_read(p);p.unlink();x['config']['solver']['topk']=11;C.sealed_write(p,x)
        with self.assertRaisesRegex(RuntimeError,'provenance/config'):self.f.run()

    def test_initial_mismatch_not_written_as_valid(self):
        self.f.prepare()
        with patch.object(E,'check_pair',side_effect=RuntimeError('reset mismatch')),\
             patch.object(E,'run_one',return_value=result('lewm10',142)),\
             patch.object(E,'model_digest',return_value='stable'),self.assertRaisesRegex(RuntimeError,'reset mismatch'):
            R.evaluate_missing(self.f.out,self.f.m,self.f.cases,self.f.cells,None,None,'PARENT',lambda x:x)
        self.assertFalse(list((self.f.out/'runs').glob('*.json')))

    def test_model_mutation_blocks_result(self):
        self.f.prepare()
        with patch.object(E,'check_pair',side_effect=pair_check),patch.object(E,'run_one',return_value=result('lewm10',142)),\
             patch.object(E,'model_digest',side_effect=['a','b']),self.assertRaisesRegex(RuntimeError,'state changed'):
            R.evaluate_missing(self.f.out,self.f.m,self.f.cases,self.f.cells,None,None,'PARENT',lambda x:x)
        self.assertFalse(list((self.f.out/'runs').glob('*.json')))

    def test_summarize_three_models_reproduces_old_effect(self):
        self.f.prepare();self.f.run();C.dump(self.f.out/'model_audit.json',{'fixture':True})
        with patch.object(E,'check_pair',side_effect=pair_check),contextlib.redirect_stdout(io.StringIO()):
            artifacts=R.summarize(self.f.out,self.f.m,self.f.cases,self.f.cells)
        for p,h in artifacts.items():self.assertEqual(C.sha(self.f.out/p),h)
        with (self.f.out/'summary.csv').open() as f:rows=list(csv.DictReader(f))
        with (self.f.out/'pairwise.csv').open() as f:effects=list(csv.DictReader(f))
        self.assertEqual(len(rows),2);self.assertEqual(len(effects),6)
        with (self.f.out/'paired_cases.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),1000)
        for row in rows:
            self.assertAlmostEqual(float(row['pcmh_minus_lewm10_pp']),
                float(row['anchor_minus_lewm10_pp'])+float(row['pcmh_minus_anchor_pp']))

    def test_missing_parent_no_partial_overall_mean(self):
        self.f.prepare();self.f.run();R.result_path(self.f.out,3000,146).unlink()
        with patch.object(E,'check_pair',side_effect=pair_check),self.assertRaises(FileNotFoundError):
            R.summarize(self.f.out,self.f.m,self.f.cases,self.f.cells)
        self.assertFalse((self.f.out/'summary.csv').exists())

    def test_old_summary_change_rejected(self):
        self.f.prepare();self.f.run()
        with (self.f.source/'summary.csv').open() as f:rows=list(csv.DictReader(f))
        rows[0]['pcmh_minus_anchor_pp']='90';C.write_csv(self.f.source/'summary.csv',rows)
        with patch.object(E,'check_pair',side_effect=pair_check),self.assertRaisesRegex(RuntimeError,'Old comparison changed'):
            R.summarize(self.f.out,self.f.m,self.f.cases,self.f.cells)

    def test_full_execute_adapter_only_parent_evaluated(self):
        from omegaconf import OmegaConf as O
        import torch
        self.f.prepare()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Linear(1,1);self.projector=torch.nn.Linear(1,1)
            def to(self,*a,**kw):return self
        model=Model();ref=copy.deepcopy(model)
        C.dump(self.f.source/'model_audit.json',{'initial_state_sha256':{'anchor':E.model_digest(ref)}})
        class Handle:
            closed=False
            def close(self):self.closed=True
        class Data:
            h5_file=Handle();column_names=['episode_idx','step_idx']
            def get_row_data(s,rows):
                return {'episode_idx':np.array([self.f.cases[i]['episode_idx'] for i in rows]),
                        'step_idx':np.array([self.f.cases[i]['start_step'] for i in rows])}
        ds=Data();loads=[];calls=[]
        def load(policy):
            loads.append(policy);return model if policy=='lewm_epoch_10' else ref
        def one(cfg,dataset,process,m,seed):
            self.assertIs(m,model);self.assertIs(dataset,ds)
            calls.append((int(cfg.solver.num_samples),seed));return result('lewm10',seed)
        mods={'stable_pretraining':types.ModuleType('stable_pretraining'),
              'stable_worldmodel':types.SimpleNamespace(policy=types.SimpleNamespace(AutoCostModel=load)),
              'eval':types.SimpleNamespace(get_dataset=lambda *a:ds),
              'eval_lowbudget_failure_autopsy':types.SimpleNamespace(_build_process=lambda *a:{})}
        # Restore only these module names (do not discard lazy PyTorch imports).
        original={k:sys.modules.get(k) for k in mods}
        sys.modules.update(mods)
        try:
            with patch.object(E,'run_one',side_effect=one),patch.object(E,'check_pair',side_effect=pair_check),\
                 contextlib.redirect_stdout(io.StringIO()):
                R.execute(self.f.out,self.f.m,self.f.meta,self.f.cases,self.f.cells)
        finally:
            for k,v in original.items():
                if v is None:sys.modules.pop(k,None)
                else:sys.modules[k]=v
        self.assertEqual(loads,['lewm_epoch_10','new_model']);self.assertEqual(len(calls),10)
        self.assertTrue(ds.h5_file.closed)


class SmallTests(unittest.TestCase):
    def cases(self):
        return [{'seed':s,'eval_index':i,'dataset_row':j*100+i,'episode_idx':j*100+i,
                 'start_step':0,'goal_step':25} for j,s in enumerate(C.SEEDS) for i in range(100)]
    def test_cohort_valid(self):R.validate_cohort(self.cases(),[])
    def test_cohort_reorder(self):
        cases=self.cases();cases[0],cases[1]=cases[1],cases[0]
        with self.assertRaises(RuntimeError):R.validate_cohort(cases,[])
    def test_cohort_duplicate_row(self):
        cases=self.cases();cases[1]['dataset_row']=cases[0]['dataset_row']
        with self.assertRaises(RuntimeError):R.validate_cohort(cases,[])
    def test_excluded_episode(self):
        with self.assertRaises(RuntimeError):R.validate_cohort(self.cases(),[0])
    def test_goal_offset(self):
        cases=self.cases();cases[1]['goal_step']=26
        with self.assertRaises(RuntimeError):R.validate_cohort(cases,[])
    def test_integer_ids(self):
        cases=self.cases();cases[1]['episode_idx']=1.5
        with self.assertRaises(RuntimeError):R.validate_cohort(cases,[])
    def test_paths(self):
        for rel in ('../escape','/tmp/escape','.'):
            with self.assertRaises(ValueError):R.artifact_path('/tmp/root',rel)
    def test_disallowed_nodes_rejected_before_conda(self):
        for node in ('4090node1','bad','4090node2,4090node3'):
            p=subprocess.run(['bash',str(R.HERE/'submit.sh'),'run','v1'],env={**os.environ,'NODE':node},capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertIn('Allowed',p.stderr)
    def test_shell_single_gpu_no_mem_and_no_visibility_override(self):
        script=R.HERE/'submit.sh';subprocess.run(['bash','-n',str(script)],check=True)
        text=script.read_text();self.assertIn('NODE:-4090node3',text)
        self.assertIn('--gres=gpu:1',text);self.assertIn('--ntasks=1',text)
        self.assertNotIn('--mem',text);self.assertNotIn('CUDA_VISIBLE_DEVICES=',text)
    def test_pairwise_sign_and_zero(self):
        rows=[{**c,'budget':b,'lewm10_success':0,'anchor_success':1,'pcmh_success':1}
              for b in C.BUDGETS for c in self.cases()]
        _,stats,effects=R.tables(rows)
        self.assertEqual(stats[0]['anchor_minus_lewm10_pp'],100)
        for r in effects:
            if r['model_b']=='anchor':self.assertEqual(r['a_minus_b_pp'],0)
            else:self.assertEqual(r['a_only'],500)
    def test_partial_stats_rejected(self):
        rows=[{**c,'budget':b,'lewm10_success':0,'anchor_success':1,'pcmh_success':1}
              for b in C.BUDGETS for c in self.cases()][:-1]
        with self.assertRaises(RuntimeError):R.tables(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--with-repo',action='store_true');a=p.parse_args()
    if a.with_repo:R.sources();print('PINNED LEWM10 EVALUATION SOURCE CHECK PASS')
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    print(f'LEWM10 SUPPLEMENT CPU TESTS PASS: {result.testsRun}; real LeWM/PushT/CUDA/Slurm NOT RUN')


if __name__=='__main__':main()
