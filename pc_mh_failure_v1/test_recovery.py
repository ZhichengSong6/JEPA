#!/usr/bin/env python3
"""Synthetic recovery tests. No installed simulator, real captures or GPU is used."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from recover_analysis import (center_audit, require_valid, CenterInputError,
                              validate_paths, create_readonly_view, LINKS, OLD_ERROR,
                              joint_with_domain_audit, patched_joint, verify_source_run,
                              preflight_centers)


def identity_decode(value, scaler):
    return np.asarray(value).reshape(len(value), 25, 2).copy()


class RecoveryTests(unittest.TestCase):
    def test_eligible_and_input_unchanged(self):
        center = np.linspace(-1, 1, 50).reshape(5, 10)
        before = center.copy()
        report = center_audit(center, .1565)
        self.assertEqual(report['status'], 'eligible')
        require_valid(report)
        np.testing.assert_array_equal(center, before)

    def test_original_boundary_tolerance_unchanged(self):
        center = np.ones((5, 10)) * (1 + .5e-6)
        self.assertEqual(center_audit(center, .1565)['status'], 'eligible')
        center[0, 0] = 1 + 2e-6
        self.assertEqual(center_audit(center, .1565)['status'], 'outside_bounded_probe_domain')

    def test_outside_reports_magnitude(self):
        center = np.zeros((5, 10)); center[0, 0] = -1.2; center[4, 9] = 1.03
        r = center_audit(center, .1565)
        self.assertEqual(r['outside_with_tolerance'], 2)
        self.assertAlmostEqual(r['max_bound_excess'], .2)
        self.assertEqual(r['raw_min'], -1.2)
        self.assertEqual(r['raw_max'], 1.03)
        require_valid(r)

    def test_nan_and_inf_still_fail(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            center = np.zeros((5, 10)); center[0, 0] = value
            report = center_audit(center, .1565)
            self.assertEqual(report['status'], 'nonfinite_center')
            with self.assertRaises(CenterInputError): require_valid(report)
            json.dumps(report, allow_nan=False)

    def test_invalid_shape_still_fails(self):
        report = center_audit(np.zeros((1, 5, 10)), .1565)
        with self.assertRaises(CenterInputError): require_valid(report)

    def test_invalid_radius_still_fails(self):
        for radius in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(CenterInputError):
                require_valid(center_audit(np.zeros((5, 10)), radius))

    def test_eligible_delegates_exact_inputs(self):
        calls = []
        original = lambda *a: calls.append(a) or ['unchanged']
        center = np.zeros((5, 10), dtype=np.float32)
        args = (None, {'action': None}, {}, 'cpu', {}, (None,None,None,center,None,None),
                Path('.'), {'seed':42,'source':'pcmh','eval_index':1,'solve_no':0},
                {'radius':.1565,'pairs':[[0,1]]})
        result = joint_with_domain_audit(original, identity_decode)(*args)
        self.assertEqual(result, ['unchanged'])
        self.assertIs(calls[0][5][3], center)

    def test_outside_skips_only_joint_and_preserves_values(self):
        with tempfile.TemporaryDirectory() as d:
            center = np.zeros((5, 10), dtype=np.float32); center[0, 0] = 1.25
            old = center.copy(); calls = []
            wrapper = joint_with_domain_audit(lambda *a: calls.append(a), identity_decode)
            key = {'seed':45,'source':'pcmh','eval_index':36,'solve_no':0}
            with contextlib.redirect_stdout(io.StringIO()):
                rows = wrapper(None, {'action':None}, {}, 'cpu', {},
                               (None,None,None,center,None,None),Path(d),key,
                               {'radius':.1565,'pairs':[[0,1],[0,4],[2,4]]})
            self.assertEqual(calls, [])
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r['skipped'] for r in rows))
            self.assertTrue(all('interaction_error_mse' not in r for r in rows))
            np.testing.assert_array_equal(center, old)
            archive_path = next((Path(d)/'joint').glob('*.npz'))
            # Close the NPZ reader and its file before TemporaryDirectory cleanup.
            # An open deleted file can leave .nfs* entries on the server's NFS mount.
            with archive_path.open('rb') as archive_file:
                with np.load(archive_file, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['raw_center'], old)
                    np.testing.assert_array_equal(saved['normalized_center'], old)
            self.assertTrue(archive_file.closed)
            self.assertEqual(json.loads(next((Path(d)/'joint').glob('*.json')).read_text())
                             ['center_audit']['raw_max'], 1.25)

    def test_nonfinite_not_silently_skipped(self):
        center = np.zeros((5,10)); center[0,0] = np.nan
        calls = []
        with self.assertRaises(CenterInputError):
            joint_with_domain_audit(lambda *a: calls.append(a), identity_decode)(
                None, {'action':None}, {}, 'cpu', {}, (None,None,None,center,None,None),
                Path('.'), {}, {'radius':.1565})
        self.assertEqual(calls, [])

    def test_patch_restored_on_error(self):
        old = lambda *a: None
        driver = SimpleNamespace(analyze_joint=old)
        with self.assertRaises(RuntimeError):
            with patched_joint(driver, identity_decode):
                self.assertIsNot(driver.analyze_joint, old)
                raise RuntimeError('other replay failure must still propagate')
        self.assertIs(driver.analyze_joint, old)

    def test_paths_reject_overwrite_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'old'; source.mkdir()
            validate_paths(root,source,root/'new')
            for output in (source, source/'nested', root, root.parent/'escaped'):
                with self.assertRaises(ValueError): validate_paths(root,source,output)
            target = root/'new'; target.mkdir()
            with self.assertRaises(FileExistsError): validate_paths(root,source,target)

    def test_readonly_view_preserves_original(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/'source'; source.mkdir(); output=Path(d)/'new'
            for name in LINKS:
                if name in ('survey','capture'): (source/name).mkdir()
                else: (source/name).write_text('{}')
            (source/'protocol.json').write_text('{"original":true}')
            (source/'BLOCKED.json').write_text('{"original_error":true}')
            create_readonly_view(source,output)
            self.assertTrue((output/'capture').is_symlink())
            self.assertFalse((output/'analysis').exists())
            self.assertEqual((output/'source_protocol.json').read_bytes(),
                             (source/'protocol.json').read_bytes())
            self.assertFalse((output/'protocol.json').exists())
            self.assertEqual((source/'BLOCKED.json').read_text(), '{"original_error":true}')

    def test_reuse_requires_trace_gate_and_exact_error(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)
            driver=SimpleNamespace(verify_protocol=lambda out:{'source':'test'},
                verify_stage=lambda out,name:{'passed':True},
                read_json=lambda p:{'stage':'analyze','exception':'ValueError','message':OLD_ERROR})
            self.assertEqual(verify_source_run(driver,source),{'source':'test'})
            driver.verify_stage=lambda out,name:{'passed':False}
            with self.assertRaises(RuntimeError): verify_source_run(driver,source)
            driver.verify_stage=lambda out,name:{'passed':True}
            driver.read_json=lambda p:{'stage':'analyze','exception':'RuntimeError','message':'other'}
            with self.assertRaises(RuntimeError): verify_source_run(driver,source)

    def test_completed_source_is_not_reprocessed(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d); (source/'COMPLETE.json').write_text('{}')
            driver=SimpleNamespace(verify_protocol=lambda out:{},
                                   verify_stage=lambda out,name:{'passed':True})
            with self.assertRaises(RuntimeError): verify_source_run(driver,source)

    def test_preflight_reads_captures_without_running_cem(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source'; output=Path(d)/'new'
            source.mkdir(); output.mkdir(); (source/'capture/seed45').mkdir(parents=True)
            for label in ('pcmh','stepwise'):
                center=np.zeros((5,10),dtype=np.float32)
                if label=='pcmh': center[1,2]=1.125
                torch.save({'traces':[{'returned_actions':center,'eval_index':36,'solve_no':0}]},
                           source/f'capture/seed45/{label}_traces.pt')
            closed=[]; tables=[]
            driver=SimpleNamespace(read_json=lambda p:[{'seed':45}],
                build_data=lambda s:(None,'dataset',{'action':None},None),
                close_dataset=lambda ds:closed.append(ds),
                write_csv=lambda p,rows:tables.append([r.copy() for r in rows]))
            with contextlib.redirect_stdout(io.StringIO()):
                result=preflight_centers(driver,source,output,{'seeds':[45],'radius':.1565},identity_decode)
            self.assertEqual(result['reference_centers'],2)
            self.assertEqual(result['outside_domain_centers'],1)
            self.assertEqual(result['eligible_centers'],1)
            self.assertEqual(closed,['dataset'])
            self.assertEqual(tables[-1][0]['raw_max'],1.125)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)
    print(f'RECOVERY SYNTHETIC TESTS PASS: {result.testsRun}; real server captures/GPU NOT RUN')
