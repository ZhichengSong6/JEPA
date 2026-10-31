import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pcqr_tta_v1.run_pusht import evaluate_case, jsonable, resolve_policy, sha256


class RunnerUtilityTests(unittest.TestCase):
    def test_checkpoint_directory_unique_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            run = home / 'run'
            run.mkdir()
            obj = run / 'model_epoch_10_object.ckpt'
            obj.write_bytes(b'test artifact, not a loadable checkpoint')
            prefix, path = resolve_policy(None, str(run), home)
            self.assertEqual(prefix, 'run/model_epoch_10')
            self.assertEqual(path, obj)
            self.assertEqual(len(sha256(obj)), 64)

    def test_ambiguous_directory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            for n in ('a_epoch_10_object.ckpt', 'b_epoch_10_object.ckpt'):
                (home / n).write_bytes(b'')
            with self.assertRaises(RuntimeError):
                resolve_policy(None, str(home), home)

    def test_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_policy('not_present', None, Path(tmp))

    def test_explicit_checkpoint_and_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / 'source_object.ckpt').write_bytes(b'')
            self.assertEqual(resolve_policy('source_object.ckpt', None, home)[0], 'source')

    def test_json_preserves_metrics_and_replaces_nonfinite(self):
        value = {'tensor': torch.tensor([1, 0]), 'array': np.array([1., np.nan]),
                 'scalar': np.float32(2.5)}
        encoded = jsonable(value)
        self.assertEqual(encoded, {'tensor': [1, 0], 'array': [1., None], 'scalar': 2.5})
        json.dumps(encoded, allow_nan=False)


class EvaluationCallTests(unittest.TestCase):
    def evaluate(self, world, out_dir, mode='response', index=0):
        return evaluate_case(
            world, self.dataset, case={'episode': 12, 'start': 7},
            goal_offset_steps=25, eval_budget=50, callables=self.callables,
            out_dir=out_dir, mode=mode, case_index=index)

    def setUp(self):
        self.dataset = object()
        self.callables = [{'method': '_set_state', 'args': {'state': {'value': 'state'}}}]

    def test_legacy_path_conversion_with_explicit_video_disabled(self):
        # Reproduce the server failure contract without CUDA, SWM or ffmpeg.
        # Construct Path BEFORE inspecting save_video to cover stricter versions.
        class LegacyWorld:
            def evaluate_from_dataset(self, dataset, *, episodes_idx, start_steps,
                                      goal_offset_steps, eval_budget, callables,
                                      save_video=True, video_path='./'):
                self.path = Path(video_path)
                if save_video:
                    raise AssertionError('Smoke must not invoke the video writer')
                self.arguments = (dataset, episodes_idx, start_steps, goal_offset_steps,
                                  eval_budget, callables)
                return {'success_rate': 100.0, 'episode_successes': [True]}

        with tempfile.TemporaryDirectory() as tmp:
            world = LegacyWorld()
            result = self.evaluate(world, Path(tmp))
            self.assertEqual(world.path, Path(tmp).resolve() / 'videos/response_case0000')
            self.assertEqual(world.arguments,
                             (self.dataset, [12], [7], 25, 50, self.callables))
            self.assertEqual(result, {'success_rate': 100.0, 'episode_successes': [True]})
            self.assertFalse((Path(tmp) / 'videos').exists())

    def test_case_and_mode_video_paths_are_distinct(self):
        class World:
            def evaluate_from_dataset(self, dataset, **kw):
                self.kw = kw
                return kw['video_path']
        with tempfile.TemporaryDirectory() as tmp:
            world = World()
            paths = {self.evaluate(world, Path(tmp), mode, index)
                     for mode in ('frozen', 'obs', 'response') for index in (0, 1)}
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(Path(p).is_absolute() for p in paths))
            self.assertIs(world.kw['save_video'], False)

    def test_evaluation_failure_propagates_without_retry(self):
        class World:
            calls = 0
            def evaluate_from_dataset(self, dataset, **kw):
                self.calls += 1
                raise TypeError('internal environment failure')
        with tempfile.TemporaryDirectory() as tmp:
            world = World()
            with self.assertRaisesRegex(TypeError, 'internal environment failure'):
                self.evaluate(world, Path(tmp))
            self.assertEqual(world.calls, 1)

    def test_runner_uses_the_tested_evaluation_helper(self):
        import ast
        import inspect
        import pcqr_tta_v1.run_pusht as runner
        tree = ast.parse(inspect.getsource(runner.main))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertEqual(sum(isinstance(n.func, ast.Name) and n.func.id == 'evaluate_case'
                             for n in calls), 1)
        self.assertFalse(any(isinstance(n.func, ast.Attribute)
                             and n.func.attr == 'evaluate_from_dataset' for n in calls))


if __name__ == '__main__':
    unittest.main()
