import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pcqr_tta_v1.run_pusht import jsonable, resolve_policy, sha256


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


if __name__ == '__main__':
    unittest.main()
