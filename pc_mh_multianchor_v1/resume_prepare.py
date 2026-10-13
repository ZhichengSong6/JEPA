#!/usr/bin/env python3
"""Recover an interrupted cache extraction into a NEW sibling preparation run.

Reuse only complete, zero-invalid-action NPZ groups after protocol, selection,
shape, finite-value, residual-identity and deterministic re-encoding checks.
Old partial files are never loaded; an unfinished group is recomputed in full.
No source file is changed and no science/gate/optimizer parameter is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import core
import pipeline
import runtime as rt

GROUPS = ('bank', 'calibration', 'audit')
KEY_SHAPES = {'z': (192,), 'u': (5, 10), 'future': (5, 192),
              'teacher': (5, 192), 'residual': (5, 192),
              'episode': (), 'start': (), 'clip': ()}
ALLOWED_NODES = {'4090node2', '4090node3', '3090node1', '3090node2', '3090node3'}


def validate_source(source, output):
    root = Path(os.environ['STABLEWM_HOME']).resolve()
    source, output = Path(source).resolve(), Path(output).resolve()
    if root not in source.parents or root not in output.parents:
        raise ValueError('Source/output must be strictly inside STABLEWM_HOME')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Use a NEW sibling output; never overwrite or nest the source')
    if output.exists():
        raise FileExistsError(f'Preserve {output}; use a new tag')
    cfg, metadata = rt.verify(source)
    if metadata['smoke']:
        raise ValueError('This recovery is for the interrupted formal preparation')
    for name in ('cache_complete.json', 'audit_complete.json', 'queries_complete.json',
                 'READY.json', 'REJECTED.json'):
        if (source / name).exists():
            raise RuntimeError('Source progressed beyond cache extraction: ' + name)
    required = ('protocol.json', 'baseline_config.yaml', 'parent_state.json',
                'original_split.npz', 'normalization.npz', 'native_cem_precheck.json')
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if json.loads((source / 'native_cem_precheck.json').read_text()).get('passed') is not True:
        raise RuntimeError('Source CEM/pixel interface check did not pass')
    present = [g + '.npz' for g in GROUPS if (source / (g + '.npz')).is_file()]
    if not present:
        raise RuntimeError('No complete cache group to reuse; preserve source and use normal prepare')
    # The interrupted run has no final cache seal. These hashes record the
    # recovery-time snapshot, NOT a falsely claimed original completed seal.
    hashes = {name: core.sha(source / name) for name in (*required, *present)}
    return cfg, metadata, hashes


def verify_snapshot(source, hashes):
    for name, expected in hashes.items():
        if core.sha(Path(source) / name) != expected:
            raise RuntimeError('Source changed during recovery: ' + name)


def validate_cached(arrays, indices, clips):
    if set(arrays) != set(KEY_SHAPES):
        raise ValueError('Unexpected cache fields')
    n = len(indices)
    for key, shape in KEY_SHAPES.items():
        value = arrays[key]
        if value.shape != (n, *shape):
            raise ValueError(f'Incomplete/malformed cache {key}: {value.shape}; expected {(n, *shape)}')
        if key in ('episode', 'start', 'clip'):
            if not np.issubdtype(value.dtype, np.integer):
                raise ValueError('Cache identity must be integer: ' + key)
        elif value.dtype != np.dtype('float32'):
            raise ValueError('Expected original float32 cache: ' + key)
        core.finite(value, key)
    if not np.array_equal(arrays['clip'], indices):
        raise RuntimeError('Cached clip selection differs; do not change seeds or reuse partial groups')
    expected = np.asarray(clips, dtype=np.int64)[indices]
    if not np.array_equal(arrays['episode'], expected[:, 0]) or not np.array_equal(arrays['start'], expected[:, 1]):
        raise RuntimeError('Cached episode/start identities differ')
    if len(np.unique(expected, axis=0)) != n:
        raise RuntimeError('Duplicate cache identities')
    for left in range(0, n, 2048):
        sl = slice(left, left + 2048)
        if not np.allclose(arrays['residual'][sl], arrays['future'][sl] - arrays['teacher'][sl], rtol=0, atol=1e-6):
            raise RuntimeError('Cached residual != factual future - teacher prediction')


def verify_reencoding(arrays, indices, ds, model, device, extract):
    # Small deterministic provenance check only, not an estimator acceptance
    # test. It cannot prove all cached numbers correct; full NPZ CRC/finite and
    # selection checks are also required. No global RNG is consumed.
    rows = np.unique(np.linspace(0, len(indices) - 1, min(16, len(indices)), dtype=np.int64))
    fresh, invalid = extract(ds, indices[rows], model, device)
    if invalid:
        raise RuntimeError('Previously valid cached check windows now have invalid actions')
    for key in KEY_SHAPES:
        old = arrays[key][rows]
        new = fresh[key]
        if old.shape != new.shape:
            raise RuntimeError('Re-encoding shape mismatch: ' + key)
        if key in ('episode', 'start', 'clip', 'u'):
            match = np.array_equal(old, new)
        else:
            match = np.allclose(old, new, rtol=1e-5, atol=2e-5)
        if not match:
            raise RuntimeError('Re-encoding check failed: ' + key)
    return len(rows)


def same_npz(a, b):
    x, y = core.load_npz(a), core.load_npz(b)
    if x.keys() != y.keys() or any(not np.array_equal(x[k], y[k]) for k in x):
        raise RuntimeError(f'Recomputed split/normalization differs: {Path(a).name}')


@contextmanager
def reuse_extract(source, output, hashes, records):
    original = pipeline.extract
    count = 0

    def extract(ds, indices, teacher, device):
        nonlocal count
        if count >= len(GROUPS):
            raise RuntimeError('Unexpected extra cache extraction call')
        name = GROUPS[count]
        count += 1
        if count == 1:
            for artifact in ('original_split.npz', 'normalization.npz'):
                same_npz(Path(source) / artifact, Path(output) / artifact)
        filename = name + '.npz'
        path = Path(source) / filename
        if filename not in hashes:
            if path.exists():
                raise RuntimeError('A new source cache appeared after snapshot: ' + filename)
            print(f'RECOMPUTE GROUP {name}: {len(indices)} selected windows; RAM/partial progress is not reusable', flush=True)
            arrays, invalid = original(ds, indices, teacher, device)
            records.append({'group': name, 'action': 'recomputed', 'windows': len(arrays['z']), 'invalid': int(invalid)})
        else:
            if core.sha(path) != hashes[filename]:
                raise RuntimeError('Cache changed since snapshot: ' + filename)
            arrays = core.load_npz(path)  # Loads all arrays and checks ZIP CRC; closes readers.
            validate_cached(arrays, indices, ds.clip_indices)
            checked = verify_reencoding(arrays, indices, ds, teacher, device, original)
            records.append({'group': name, 'action': 'reused_complete', 'windows': len(indices),
                            'invalid': 0, 'source_sha256': hashes[filename], 'reencoded_check_rows': checked})
            print(f'REUSE CACHE {name}: {len(indices)} windows; exact selection and {checked} re-encodings checked', flush=True)
            invalid = 0
        core.dump(Path(output) / 'cache_recovery_progress.json', records)
        return arrays, invalid

    pipeline.extract = extract
    try:
        yield
        if count != len(GROUPS):
            raise RuntimeError('Expected all three cache groups')
    finally:
        pipeline.extract = original


def continue_preparation(output, cfg, metadata, teacher, source, hashes, records):
    with reuse_extract(source, output, hashes, records):
        pipeline.cache(output, cfg, metadata, teacher, 'cuda:0')
    verify_snapshot(source, hashes)
    rt.check_seal(output, 'cache_complete')
    print('CACHE RECOVERY COMPLETE: original source preserved', flush=True)
    passed = pipeline.audit(output, metadata, 'cuda:0')
    if not passed:
        core.dump(output / 'REJECTED.json', {'stage': 'residual_audit',
                  'reason': 'Original frozen independent gate failed; no student training authorized'})
        print('RESIDUAL AUDIT REJECTED: no thresholds changed', flush=True)
        return False, False
    query_pass = pipeline.queries(output, cfg, metadata, teacher, 'cuda:0')
    return bool(passed), bool(query_pass)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    node = os.environ.get('SLURMD_NODENAME', os.uname().nodename.split('.')[0])
    if not os.environ.get('SLURM_JOB_ID') or node not in ALLOWED_NODES:
        raise RuntimeError('Use the permitted-node Slurm launcher; 4090node1 is forbidden')
    if int(os.environ.get('SLURM_NTASKS', '1')) != 1 or not torch.cuda.is_available():
        raise RuntimeError('Preparation requires one allocated GPU/process')
    cfg, old, hashes = validate_source(source, output)
    cfg, metadata = rt.initialize(output, old['base_config'], False)
    # Only the output path may change in the scientific protocol.
    changed = [key for key in set(old) | set(metadata) if key != 'output' and old.get(key) != metadata.get(key)]
    if changed:
        raise RuntimeError('Recovery protocol mismatch: ' + ', '.join(changed))
    records = []
    try:
        core.dump(output / 'cache_recovery.json', {
            'source': str(source), 'source_snapshot_sha256': hashes,
            'recovery_code_sha256': core.sha(__file__), 'started_unix': time.time(),
            'source_has_final_cache_seal': False,
            'validation': 'Trust local user-generated files; CRC/selection/shape/finite/residual checks plus up to 16 re-encodings per reused group. Not a full recomputation.',
            'partial_files': 'preserved in source but never reused'})
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        teacher = rt.teacher(cfg, 'cuda:0')
        before = rt.digest(teacher)
        if before != json.loads((source / 'parent_state.json').read_text())['state_sha256']:
            raise RuntimeError('Teacher state differs from interrupted extraction')
        core.dump(output / 'parent_state.json', {'state_sha256': before})
        (output / 'native_cem_precheck.json').write_bytes((source / 'native_cem_precheck.json').read_bytes())
        passed, query_pass = continue_preparation(output, cfg, metadata, teacher, source, hashes, records)
        if rt.digest(teacher) != before:
            raise RuntimeError('Frozen teacher changed')
        verify_snapshot(source, hashes)
        ready = bool(passed and query_pass)
        if passed:
            core.dump(output / ('READY.json' if ready else 'REJECTED.json'), {
                'formal_ready': ready, 'audit_pass': passed, 'query_coverage_pass': query_pass,
                'smoke': False, 'parent_unchanged': True, 'note': 'No planning improvement is claimed'})
        print(f'PREPARATION {"READY" if ready else "REJECTED"}: {output}', flush=True)
        core.dump(output / 'recovery_complete.json', {'source_unchanged': True, 'parent_unchanged': True,
                   'groups': records, 'finished_unix': time.time(),
                   'note': 'Check READY/REJECTED separately. No automatic training.'})
    except Exception as exc:
        # A BLOCKED marker always takes precedence over any earlier status.
        core.dump(output / 'BLOCKED.json', {'error': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
