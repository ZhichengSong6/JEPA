#!/usr/bin/env python3
"""Fresh-process resolver regression tests; optional REAL installed import check.

Synthetic cases emulate stable_pretraining's unguarded import-time resolver
registration using real OmegaConf. --with-repo additionally reads the real saved
config BEFORE importing the actual training modules, without reading datasets,
loading checkpoints, creating a Trainer, or requiring CUDA.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = Path(__file__).resolve().parent
FIXTURE_CONFIG = '''\
wm:
  history_size: 3
  num_preds: 1
mh_ald:
  enabled: true
  rollout_horizon: 5
  init_policy: lewm_epoch_10
  teacher_policy: lewm_epoch_10
  planner_context:
    enabled: true
    observation_history: 1
    predictor_max_history: 3
data:
  dataset:
    name: pusht_expert_train
    frameskip: 5
    num_steps: ${eval:'${wm.history_size}+${mh_ald.rollout_horizon}'}
    keys_to_load: [pixels, action, proprio]
trainer:
  devices: 4
  num_nodes: 1
  sync_batchnorm: false
loader:
  batch_size: 64
  num_workers: 6
subdir: baseline_run
output_model_name: baseline_model
'''


def child(script, *args):
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script), *map(str, args)],
        text=True, capture_output=True, timeout=120,
    )
    if result.returncode:
        raise AssertionError(
            f'Fresh process exited {result.returncode}\n'
            f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
        )
    return result


class ResolverStartupTests(unittest.TestCase):
    def fixture(self, body, dependency_source=None):
        with tempfile.TemporaryDirectory(prefix='pcma_resolver_') as directory:
            root = Path(directory)
            pkg = root / 'stable_pretraining'
            pkg.mkdir()
            if dependency_source is None:
                dependency_source = (
                    'from omegaconf import OmegaConf\n'
                    'OmegaConf.register_new_resolver("eval", eval)\n'
                    'IMPORT_COUNT = 1\n'
                )
            (pkg / '__init__.py').write_text(dependency_source)
            (root / 'train_mh_ald.py').write_text(
                'import stable_pretraining as spt\n'
                'IMPORTED = True\n'
            )
            path = root / 'baseline.yaml'
            path.write_text(FIXTURE_CONFIG)
            before = path.read_bytes()
            script = (
                'import sys\nfrom pathlib import Path\n'
                'sys.path.insert(0, sys.argv[1])\n'
                'import runtime\n'
                # runtime adds its repository to sys.path. Give only these
                # isolated synthetic cases explicit fixture precedence.
                'sys.path.insert(0, sys.argv[2])\n'
                'from omegaconf import OmegaConf\n'
                'config_path = Path(sys.argv[2]) / "baseline.yaml"\n'
                + textwrap.dedent(body)
            )
            result = child(script, HERE, root)
            self.assertEqual(path.read_bytes(), before, 'Input config was modified')
            return result

    def test_old_order_reproduces_exact_duplicate(self):
        self.fixture('''
            OmegaConf.register_new_resolver('eval', runtime.arithmetic)
            try:
                import train_mh_ald
            except ValueError as exc:
                assert "resolver 'eval' is already registered" in str(exc), str(exc)
            else:
                raise AssertionError('Old order did not reproduce the collision')
        ''')

    def test_config_first_then_training_import(self):
        self.fixture('''
            assert 'stable_pretraining' not in sys.modules
            cfg = runtime.config(config_path)
            import train_mh_ald
            assert train_mh_ald.IMPORTED
            assert cfg.data.dataset.num_steps == 8
            assert cfg.trainer.devices == 4 and cfg.loader.batch_size == 64
            assert cfg.subdir == 'baseline_run'
        ''')

    def test_dependency_first_and_repeated_calls(self):
        self.fixture('''
            import stable_pretraining as spt
            original_module = spt
            for _ in range(3):
                assert runtime.config(config_path).data.dataset.num_steps == 8
                assert runtime.omega() is OmegaConf
            import stable_pretraining as again
            assert again is original_module and again.IMPORT_COUNT == 1
        ''')

    def test_numeric_only_resolver_survives_late_import(self):
        self.fixture('''
            runtime.config(config_path)
            import train_mh_ald
            cfg = OmegaConf.create({'ok': "${eval:'3+5'}", 'bad': "${eval:'abs(-1)'}"})
            assert cfg.ok == 8
            try:
                cfg.bad
            except Exception as exc:
                assert 'Only numeric configuration interpolation' in str(exc), str(exc)
            else:
                raise AssertionError('A function call bypassed the arithmetic guard')
        ''')

    def test_unrelated_and_builtin_resolvers_preserved(self):
        self.fixture('''
            OmegaConf.register_new_resolver('test_keep', lambda value: value + 7)
            runtime.config(config_path)
            import train_mh_ald
            assert OmegaConf.create({'x': '${test_keep:3}'}).x == 10
            assert OmegaConf.has_resolver('oc.env')
        ''')

    def test_dependency_import_error_is_not_swallowed(self):
        self.fixture('''
            try:
                runtime.omega()
            except RuntimeError as exc:
                assert str(exc) == 'fixture dependency startup failed'
            else:
                raise AssertionError('Unexpected dependency failure was hidden')
            assert not OmegaConf.has_resolver('eval')
        ''', "raise RuntimeError('fixture dependency startup failed')\n")


def check_real_repository(base_config=None):
    if base_config is None:
        base_config = os.environ.get('BASE_CONFIG')
    if base_config is None:
        root = os.environ.get('STABLEWM_HOME')
        if not root:
            raise ValueError('Set STABLEWM_HOME/BASE_CONFIG or pass --base-config for --with-repo')
        base_config = str(Path(root) / 'pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4' / 'config.yaml')
    path = Path(base_config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Real saved baseline config required: {path}')
    # Do NOT preimport a training module in this process and assume it covers
    # production. Every check below starts from a fresh Python interpreter.
    result = child('''
        import sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        import runtime
        before = Path(sys.argv[2]).read_bytes()
        cfg = runtime.config(sys.argv[2])
        import stable_pretraining
        import train_mh_ald
        import train_pc_mh_ald
        import planner_context_mh
        import pipeline
        import train
        O = runtime.omega()
        resolved = O.to_container(cfg, resolve=True)
        cfg_again = runtime.config(sys.argv[2])
        assert resolved == O.to_container(cfg_again, resolve=True)
        assert resolved['data']['dataset']['num_steps'] == 8
        assert Path(sys.argv[2]).read_bytes() == before
        print('REAL CONFIG -> TRAINING IMPORT PASS: '
              f"frames={cfg.data.dataset.num_steps}, "
              f"GPUs={cfg.trainer.devices}, local_batch={cfg.loader.batch_size}", flush=True)
    ''', HERE, path)
    if result.stdout:
        print(result.stdout, end='', flush=True)
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr, flush=True)
    print('IMPORT CHECK ONLY: no dataset/checkpoint loading, no Trainer, no GPU work', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-repo', action='store_true')
    parser.add_argument('--base-config')
    args = parser.parse_args()
    if args.base_config and not args.with_repo:
        parser.error('--base-config requires --with-repo')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ResolverStartupTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f'RESOLVER STARTUP SYNTHETIC TESTS PASS: {result.testsRun}', flush=True)
    if args.with_repo:
        check_real_repository(args.base_config)
