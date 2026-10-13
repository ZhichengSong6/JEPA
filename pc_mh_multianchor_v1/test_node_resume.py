#!/usr/bin/env python3
"""CPU fixture checks only: node policy and interrupted-cache recovery plumbing."""
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import core
import resume_prepare as recovery

HERE = Path(__file__).resolve().parent


def fixture(n=20):
    rng = np.random.default_rng(17)
    arrays = {k: rng.normal(size=(n, *shape)).astype('float32')
              for k, shape in recovery.KEY_SHAPES.items() if k not in ('episode', 'start', 'clip')}
    arrays['residual'] = arrays['future'] - arrays['teacher']
    arrays.update(episode=np.arange(n, dtype=np.int64), start=np.zeros(n, dtype=np.int64), clip=np.arange(n, dtype=np.int64))
    clips = np.column_stack((arrays['episode'], arrays['start']))
    return arrays, clips


class CacheTests(unittest.TestCase):
    def test_valid_cache_unchanged(self):
        a,c=fixture(); before=copy.deepcopy(a)
        recovery.validate_cached(a,a['clip'],c)
        for k in a: np.testing.assert_array_equal(a[k],before[k])

    def test_partial_group_is_not_reused(self):
        a,c=fixture(); a={k:v[:-1] for k,v in a.items()}
        with self.assertRaises(ValueError): recovery.validate_cached(a,np.arange(20),c)

    def test_wrong_clip_selection_blocked(self):
        a,c=fixture(); a['clip']=a['clip'][::-1]
        with self.assertRaises(RuntimeError): recovery.validate_cached(a,np.arange(20),c)

    def test_wrong_episode_blocked(self):
        a,c=fixture(); a['episode'][0]=99
        with self.assertRaises(RuntimeError): recovery.validate_cached(a,np.arange(20),c)

    def test_nan_and_residual_corruption_blocked(self):
        for change in ('nan','residual'):
            a,c=fixture(); a['residual'][0,0,0]=np.nan if change=='nan' else 999
            with self.assertRaises((ValueError,RuntimeError)): recovery.validate_cached(a,np.arange(20),c)

    def test_reencoding_good_and_bad(self):
        a,c=fixture(); inds=a['clip']
        def extract(ds,ix,teacher,device): return {k:v[ix].copy() for k,v in a.items()},0
        self.assertEqual(recovery.verify_reencoding(a,inds,None,None,'cpu',extract),16)
        def bad(*args):
            data,n=extract(*args); data['z']+=1; return data,n
        with self.assertRaises(RuntimeError): recovery.verify_reencoding(a,inds,None,None,'cpu',bad)

    def test_snapshot_tampering_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('first');h={'x':core.sha(p)}
            recovery.verify_snapshot(d,h);p.write_text('changed')
            with self.assertRaises(RuntimeError): recovery.verify_snapshot(d,h)

    def test_two_complete_groups_reused_third_recomputed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'old';out=root/'new';source.mkdir();out.mkdir()
            a,c=fixture();ids=a['clip'];dataset=SimpleNamespace(clip_indices=c)
            for name in ('original_split.npz','normalization.npz'):
                for folder in (source,out): core.save_npz(folder/name,x=ids)
            for name in ('bank','calibration'): core.save_npz(source/(name+'.npz'),**a)
            (source/'audit.partial').write_text('incomplete; must not be loaded')
            hashes={name:core.sha(source/name) for name in ('bank.npz','calibration.npz')}
            calls=[];records=[]
            def extract(ds,ix,t,dev):
                calls.append(len(ix)); return {k:v[ix].copy() for k,v in a.items()},0
            with patch.object(recovery.pipeline,'extract',extract),contextlib.redirect_stdout(io.StringIO()):
                with recovery.reuse_extract(source,out,hashes,records):
                    for _ in range(3):
                        data,invalid=recovery.pipeline.extract(dataset,ids,None,'cpu')
                        self.assertEqual(invalid,0);np.testing.assert_array_equal(data['z'],a['z'])
                self.assertIs(recovery.pipeline.extract,extract)
            self.assertEqual(calls,[16,16,20])
            self.assertEqual([r['action'] for r in records],['reused_complete','reused_complete','recomputed'])
            self.assertEqual((source/'audit.partial').read_text(),'incomplete; must not be loaded')

    def test_original_extract_restored_on_failure(self):
        before=recovery.pipeline.extract
        with self.assertRaises(RuntimeError):
            with recovery.reuse_extract(Path('.'),Path('.'),{},[]): raise RuntimeError('check failure')
        self.assertIs(recovery.pipeline.extract,before)

    def test_paths_and_completed_stage_blocked(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'STABLEWM_HOME':d}):
            old=Path(d)/'old';old.mkdir();new=Path(d)/'new'
            for output in (old,old/'nested',Path(d),Path(d).parent/'outside'):
                with self.assertRaises(ValueError): recovery.validate_source(old,output)
            new.mkdir()
            with self.assertRaises(FileExistsError): recovery.validate_source(old,new)
            new.rmdir();(old/'cache_complete.json').write_text('{}')
            with patch.object(recovery.rt,'verify',return_value=(None,{'smoke':False})):
                with self.assertRaises(RuntimeError): recovery.validate_source(old,new)

    def test_failed_audit_does_not_generate_queries(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            with patch.object(recovery,'reuse_extract',return_value=contextlib.nullcontext()), \
                 patch.object(recovery,'verify_snapshot'),patch.object(recovery.rt,'check_seal'), \
                 patch.object(recovery.pipeline,'cache'),patch.object(recovery.pipeline,'audit',return_value=False), \
                 patch.object(recovery.pipeline,'queries',side_effect=AssertionError('must not run')), \
                 contextlib.redirect_stdout(io.StringIO()):
                result=recovery.continue_preparation(out,None,{},None,out,{},[])
            self.assertEqual(result,(False,False));self.assertTrue((out/'REJECTED.json').exists())
            self.assertFalse((out/'READY.json').exists())


class NodeTests(unittest.TestCase):
    def run_launcher(self, mode, node=None):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup)
        root=Path(td.name);repo=root/'repo';here=repo/'pc_mh_multianchor_v1';here.mkdir(parents=True)
        data=root/'data';data.mkdir();bindir=root/'bin';bindir.mkdir()
        conda=root/'conda.sh';conda.write_text('conda() { :; }\n')
        script=(HERE/'submit.sh').read_text().replace(
            'CONDA_SH=/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh',
            'CONDA_SH='+str(conda))
        (here/'submit.sh').write_text(script)
        for name,body in {'python':'#!/bin/bash\nprintf "4 4\\n"\n',
                          'sbatch':'#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGS_FILE"\necho 12345\n'}.items():
            path=bindir/name;path.write_text(body);path.chmod(0o700)
        prepared=data/'prepared';prepared.mkdir()
        for name in ('queries_complete.json','READY.json','protocol.json'): (prepared/name).write_text('{}')
        env=os.environ.copy();env.pop('NODE',None)
        env.update(STABLEWM_HOME=str(data),SOURCE_OUT=str(prepared),PATH=str(bindir)+':'+env['PATH'],
                   ARGS_FILE=str(root/'args'),SBATCH_NODELIST='4090node1')
        if mode.startswith('train') or mode=='control': env['DATA_OUT']=str(prepared)
        if node is not None:env['NODE']=node
        result=subprocess.run(['bash',str(here/'submit.sh'),mode,'test'],env=env,text=True,capture_output=True)
        jobs=list(data.glob('pcmh_ma_logs/*.sbatch'))
        return result,(jobs[0].read_text() if jobs else ''),(root/'args').read_text() if (root/'args').exists() else ''

    def test_all_modes_default_to_permitted_node(self):
        for mode in ('prepare','prepare-resume','prepare-smoke','train-smoke','train','control'):
            with self.subTest(mode=mode):
                r,job,args=self.run_launcher(mode)
                self.assertEqual(r.returncode,0,r.stderr)
                self.assertIn('#SBATCH --nodelist=4090node2',job)
                self.assertIn('#SBATCH --exclude=4090node1',job)
                self.assertIn('--nodelist=4090node2',args)
                self.assertIn('--exclude=4090node1',args)
                self.assertNotIn('#SBATCH --mem',job)
                if mode=='prepare-resume': self.assertIn('resume_prepare.py',job)

    def test_all_five_allowed_nodes(self):
        for node in sorted(recovery.ALLOWED_NODES):
            r,job,args=self.run_launcher('prepare',node)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--nodelist='+node,args)

    def test_disallowed_or_multiple_nodes_fail_before_submission(self):
        for node in ('4090node1','4090node4','4090node[2-3]','4090node2,4090node3','bad node'):
            r,job,args=self.run_launcher('prepare',node)
            self.assertEqual(r.returncode,5);self.assertEqual(job,'');self.assertEqual(args,'')


if __name__=='__main__':
    unittest.main(verbosity=2)
