#!/usr/bin/env python3
"""CPU-only regression checks for the diagnostic itself, not GPU health tests."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
import check_gpu_startup as probe


class Tests(unittest.TestCase):
    def test_four_visible(self):
        self.assertEqual(probe.check_counts(4,4,4,4,3), [])
    def test_count_changes_not_hidden(self):
        self.assertTrue(probe.check_counts(4,3,3,4,3))
    def test_three_gpus_not_accepted_for_four(self):
        self.assertTrue(probe.check_counts(3,3,3,4,0))
    def test_per_task_single_gpu_not_accepted(self):
        self.assertTrue(probe.check_counts(1,1,1,4,0))
    def test_runtime_disagrees(self):
        self.assertTrue(probe.check_counts(4,3,4,4,0))
    def test_invalid_local_rank(self):
        self.assertTrue(probe.check_counts(4,4,4,4,4))
    def test_missing_runtime_private_api(self):
        self.assertEqual(probe.check_counts(4,None,4,4,0), [])
    def test_only_postinit_ordinals_queried(self):
        ids=[];initialized=[]
        def count(): return 3 if initialized else 4
        def props(i):
            ids.append(i)
            if i>=3:raise AssertionError('Invalid device id')
            return NS(name='fixture',total_memory=100,uuid=str(i))
        torch=NS(cuda=NS(device_count=count,init=lambda:initialized.append(True),
                        get_device_properties=props), _C=NS(_cuda_getDeviceCount=lambda:3))
        old=dict(os.environ);record={}
        probe.inspect_cuda(torch,4,3,record)
        self.assertEqual(ids,[0,1,2]);self.assertFalse(record['passed'])
        self.assertEqual(old,dict(os.environ))
    def test_passing_summary_and_missing_rank_rejected(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);a=NS(output=d,expected=2)
            self.assertEqual(probe.summarize(a),1)
            for rank in range(2):
                probe.write_json(Path(d)/f'rank{rank}_complete.json',
                    {phase:{'returncode':0} for phase in ('trainer','cuda')})
                probe.write_json(root/f'rank{rank}_trainer.json',{'passed':True})
                probe.write_json(root/f'rank{rank}_cuda.json',{'passed':True,
                    'devices':[{'uuid':'GPU-a'},{'uuid':'GPU-b'}]})
            self.assertEqual(probe.summarize(a),0)
            result=json.loads((root/'GPU_CHECK.json').read_text())
            self.assertTrue(result['passed'])
    def test_rank_mapping_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            for rank in range(2):
                probe.write_json(Path(d)/f'rank{rank}_complete.json',
                    {phase:{'returncode':0} for phase in ('trainer','cuda')})
                probe.write_json(Path(d)/f'rank{rank}_trainer.json',{'passed':True})
                probe.write_json(Path(d)/f'rank{rank}_cuda.json',{'passed':True,
                    'devices':[{'uuid':f'GPU-{rank}'},{'uuid':f'GPU-{1-rank}'}]})
            self.assertEqual(probe.summarize(NS(output=d,expected=2)),1)
    def test_failed_trainer_never_accepted(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            probe.write_json(Path(d)/'rank0_complete.json',
                {phase:{'returncode':0} for phase in ('trainer','cuda')})
            probe.write_json(Path(d)/'rank0_trainer.json',{'passed':False,'message':'Invalid device id'})
            probe.write_json(Path(d)/'rank0_cuda.json',{'passed':True})
            self.assertEqual(probe.summarize(NS(output=d,expected=1)),1)
    def test_child_nonzero_exit_never_accepted(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            probe.write_json(Path(d)/'rank0_complete.json',
                {'cuda':{'returncode':0},'trainer':{'returncode':1}})
            for phase in ('cuda','trainer'):
                probe.write_json(Path(d)/f'rank0_{phase}.json',{'passed':True})
            self.assertEqual(probe.summarize(NS(output=d,expected=1)),1)
    def test_node_limit_and_shell_syntax(self):
        path=Path(__file__).with_name('submit_gpu_check.sh')
        subprocess.run(['bash','-n',str(path)],check=True)
        for node in ('4090node1','bad','4090node2,4090node3'):
            p=subprocess.run(['bash',str(path),'v1'],env={**os.environ,'NODE':node},capture_output=True)
            self.assertEqual(p.returncode,2)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    print(f'GPU CHECK CPU TESTS PASS: {result.testsRun}; real allocated GPUs/Trainer NOT RUN')
