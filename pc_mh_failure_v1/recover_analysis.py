#!/usr/bin/env python3
"""Recover the v1 joint-probe domain failure without repeating survey/CEM capture.

The original run.py, analysis_core.py, protocol and capture hashes stay unchanged.
A new output directory links verified survey/capture files read-only and reruns
analysis only. Out-of-domain *finite* reference actions skip only the optional
bounded joint probes. Invalid shapes/NaN/Inf/radii still block. No clipping of
recorded actions, altered bounds, relaxed replay gates or new training is used.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED = {
    'run.py': '98aae7d4db8c1083133cb74c50f71229a05015ce',
    'analysis_core.py': 'e0cef647ddc01f62f0953e2e8e452943789508cd',
}
OLD_ERROR = 'Need finite raw [5,10] center in [-1,1] and positive radius'
BOUND_TOLERANCE = 1e-6  # SAME domain tolerance as v1, not a relaxed check.
LINKS = ('survey', 'capture', 'cases.json', 'cases.csv', 'selected_cases.json',
         'selected_cases.csv', 'survey_complete.json', 'trace_complete.json')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_driver():
    for name, expected in EXPECTED.items():
        data = (HERE / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Original {name} changed; this recovery targets the exact v1 source. '
                               'Do not edit protocol hashes or reset local files to bypass this.')
    for path in (HERE.parent, HERE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location('_pcmh_original_failure_driver', HERE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def center_audit(center, radius):
    """Separate v1's four combined failure conditions; do not modify the input."""
    c = np.asarray(center, dtype=np.float64)
    r = float(radius)
    finite = np.isfinite(c)
    values = c[finite]
    report = {
        'raw_shape': list(c.shape), 'shape_valid': c.shape == (5, 10),
        'all_finite': bool(finite.all()), 'nonfinite_count': int((~finite).sum()),
        'radius': r if np.isfinite(r) else None,
        'radius_valid': bool(np.isfinite(r) and r > 0),
        'raw_min': float(values.min()) if values.size else None,
        'raw_max': float(values.max()) if values.size else None,
        'below_minus_one': int((values < -1).sum()),
        'above_plus_one': int((values > 1).sum()),
        'outside_with_tolerance': int((np.abs(values) > 1 + BOUND_TOLERANCE).sum()),
        'max_bound_excess': float(np.maximum(np.abs(values) - 1, 0).max()) if values.size else None,
        'bound_tolerance': BOUND_TOLERANCE,
    }
    if not report['shape_valid']:
        status = 'invalid_shape'
    elif not report['all_finite']:
        status = 'nonfinite_center'
    elif not report['radius_valid']:
        status = 'invalid_radius'
    elif report['outside_with_tolerance']:
        status = 'outside_bounded_probe_domain'
    else:
        status = 'eligible'
    report['status'] = status
    return report


class CenterInputError(ValueError):
    def __init__(self, audit):
        self.details = audit
        super().__init__('Invalid joint-probe input: ' + json.dumps(audit, allow_nan=False))


def require_valid(audit):
    if audit['status'] not in ('eligible', 'outside_bounded_probe_domain'):
        raise CenterInputError(audit)


def validate_paths(root, source, output):
    root, source, output = (Path(p).expanduser().resolve() for p in (root, source, output))
    if root not in source.parents or root not in output.parents:
        raise ValueError('Source and new output must both be strictly inside STABLEWM_HOME.')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Use a separate sibling recovery directory, never overwrite/nest source.')
    if output.exists():
        raise FileExistsError(f'Preserve {output}; choose a new recovery tag.')
    return root, source, output


def verify_source_run(driver, source):
    metadata = driver.verify_protocol(source)
    driver.verify_stage(source, 'survey')
    gate = driver.verify_stage(source, 'trace')
    if gate.get('passed') is not True:
        raise RuntimeError('Trace/repeatability gate did not pass; cannot reuse captures.')
    if (source / 'analyze_complete.json').exists() or (source / 'COMPLETE.json').exists():
        raise RuntimeError('Source analysis is already complete; recovery is not needed.')
    blocked = driver.read_json(source / 'BLOCKED.json')
    if (blocked.get('stage'), blocked.get('exception'), blocked.get('message')) != (
            'analyze', 'ValueError', OLD_ERROR):
        raise RuntimeError('Source failed for a different reason; this recovery is not applicable.')
    return metadata


def create_readonly_view(source, output):
    # Original capture artifacts are not copied, rewritten, removed or re-labelled.
    for name in LINKS:
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    output.mkdir(parents=True, exist_ok=False)
    for name in LINKS:
        (output / name).symlink_to(source / name, target_is_directory=(source / name).is_dir())
    (output / 'source_protocol.json').write_bytes((source / 'protocol.json').read_bytes())
    (output / 'source_BLOCKED.json').write_bytes((source / 'BLOCKED.json').read_bytes())


def joint_with_domain_audit(original, decode):
    def wrapped(cfg, process, models, device, tr, center_data, out, key, metadata):
        normalized_center = np.asarray(center_data[3])
        raw = decode(normalized_center[None], process['action'])[0].reshape(5, 10)
        audit = center_audit(raw, metadata['radius'])
        require_valid(audit)
        if audit['status'] == 'eligible':
            return original(cfg, process, models, device, tr, center_data, out, key, metadata)
        # Do not clip the returned action and pretend it is the original plan.
        # Population scoring/replay already ran and is retained for this case.
        rows = [{**key, 'p': int(p), 'q': int(q), 'skipped': True,
                 'skip_reason': 'raw_center_outside_bounded_probe_domain',
                 'raw_min': audit['raw_min'], 'raw_max': audit['raw_max'],
                 'max_bound_excess': audit['max_bound_excess']}
                for p, q in metadata['pairs']]
        folder = Path(out) / 'joint'
        folder.mkdir(exist_ok=True)
        stem = f"s{key['seed']}_{key['source']}_c{key['eval_index']}_r{key['solve_no']}"
        json_write(folder / (stem + '.json'), {
            'rows': rows, 'specs': [], 'center_audit': audit,
            'note': 'Only bounded joint probes are unavailable. Original population/mean '
                    'actions and physical replays are unchanged. This is not zero interaction.'})
        np.savez_compressed(folder / (stem + '.npz'),
                            raw_center=raw, normalized_center=normalized_center)
        print(f"JOINT SKIPPED {stem}: raw_min={audit['raw_min']:.9g} "
              f"raw_max={audit['raw_max']:.9g} excess={audit['max_bound_excess']:.9g}; "
              'population analysis retained, no clipping', flush=True)
        return rows
    return wrapped


@contextlib.contextmanager
def patched_joint(driver, decode):
    original = driver.analyze_joint
    driver.analyze_joint = joint_with_domain_audit(original, decode)
    try:
        yield
    finally:
        driver.analyze_joint = original


def preflight_centers(driver, source, output, metadata, decode):
    """Inspect every saved center first, before model loading or physical replay."""
    import torch
    selected = driver.read_json(source / 'selected_cases.json')
    reports = []
    for seed in metadata['seeds']:
        cases = [r for r in selected if r['seed'] == seed]
        if not cases:
            continue
        _, dataset, process, _ = driver.build_data(seed)
        try:
            for label in ('pcmh', 'stepwise'):
                # Local, user-generated pickle; stage hashes were verified first.
                saved = torch.load(source / f'capture/seed{seed}/{label}_traces.pt',
                                   map_location='cpu', weights_only=False)
                for tr in saved['traces']:
                    plan = np.asarray(tr['returned_actions'], dtype=np.float32)
                    if plan.shape != (5, 10) or not np.isfinite(plan).all():
                        raise CenterInputError({'status': 'invalid_normalized_plan',
                                                'seed': seed, 'source': label,
                                                'eval_index': int(tr['eval_index']),
                                                'shape': list(plan.shape)})
                    raw = decode(plan[None], process['action'])[0].reshape(5, 10)
                    report = {'seed': int(seed), 'source': label,
                              'eval_index': int(tr['eval_index']), 'solve_no': int(tr['solve_no']),
                              **center_audit(raw, metadata['radius'])}
                    reports.append(report)
                    driver.write_csv(output / 'joint_center_audit.csv', reports)
                    require_valid(report)
                    if report['status'] != 'eligible':
                        print('CENTER AUDIT: ' + json.dumps(report, allow_nan=False), flush=True)
        finally:
            driver.close_dataset(dataset)
    summary = {'reference_centers': len(reports),
               'eligible_centers': sum(r['status'] == 'eligible' for r in reports),
               'outside_domain_centers': sum(r['status'] == 'outside_bounded_probe_domain' for r in reports),
               'note': 'Bounded joint statistics apply only to eligible centers. '
                       'Skipped centers are not zero-error/zero-interaction examples.'}
    json_write(output / 'joint_probe_coverage.json', summary)
    print('CENTER PREFLIGHT: ' + json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Failed, immutable v1 output directory')
    parser.add_argument('--output', required=True, help='New recovery output directory')
    args = parser.parse_args()
    _, source, output = validate_paths(os.environ['STABLEWM_HOME'], args.source, args.output)
    driver = load_driver()
    metadata = verify_source_run(driver, source)
    import torch
    import eval_cem_mh_diagnostics as legacy
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Use submit_recovery.sh inside a Slurm GPU allocation.')
    if int(os.environ.get('SLURM_NTASKS', '1')) != 1:
        raise RuntimeError('Recovery uses one process/GPU, not training DDP.')
    create_readonly_view(source, output)
    json_write(output / 'recovery_protocol.json', {
        'source': str(source), 'output': str(output),
        'source_protocol_sha256': file_sha(source / 'protocol.json'),
        'source_survey_marker_sha256': file_sha(source / 'survey_complete.json'),
        'source_trace_marker_sha256': file_sha(source / 'trace_complete.json'),
        'recovery_code_sha256': file_sha(__file__), 'original_git_blobs': EXPECTED,
        'stage': 'analyze_only', 'survey_and_capture': 'reused, with original SHA256 gates',
        'partial_old_analysis': 'preserved but NOT reused; physical replay is recomputed',
        'domain_policy': 'skip bounded joint probes only at finite out-of-domain centers; no clipping',
        'start_unix': time.time()})
    try:
        coverage = preflight_centers(driver, source, output, metadata, legacy.decode)
        torch.set_num_threads(2)
        models = driver.load_models(metadata, 'cuda:0')
        before = {name: driver.state_digest(model) for name, model in models.items()}
        print('REUSING verified survey + trace. No new CEM survey/capture is run.', flush=True)
        with patched_joint(driver, legacy.decode):
            driver.analyze(output, metadata, models, 'cuda:0')
        if before != {name: driver.state_digest(model) for name, model in models.items()}:
            raise RuntimeError('Read-only model state changed.')
        verify_source_run(driver, source)  # Original inputs and artifacts must remain unchanged.
        driver.verify_stage(output, 'analyze')
        driver.summarize(output)
        with (output / 'REPORT.md').open('a', encoding='utf-8') as f:
            f.write('\n## Recovery and bounded-probe coverage\n'
                    f"Survey/capture reused from `{source}`; physical analysis recomputed.\n"
                    f"Eligible centers: {coverage['eligible_centers']}/{coverage['reference_centers']}; "
                    f"out-of-domain centers: {coverage['outside_domain_centers']}.\n"
                    'All recorded populations and elite means are analyzed unchanged. '
                    'Bounded joint probes at out-of-domain centers are marked skipped, never clipped. '
                    'Do not treat skipped entries as zero effect or generalize eligible-only joint '
                    'statistics to all failures. See joint_center_audit.csv and joint_probe_coverage.json.\n')
        json_write(output / 'COMPLETE.json', {
            'finished_unix': time.time(), 'checkpoints_unchanged': True,
            'source_capture_unchanged': True, 'recovery_stage': 'analyze_only',
            'bounded_joint_coverage': coverage,
            'note': 'Population analysis completed. Some bounded joint diagnostics may be '
                    'unavailable; read coverage. Not a new benchmark result.'})
        print(f'RECOVERY ANALYSIS PASS: {output}; '
              f"joint centers skipped={coverage['outside_domain_centers']}", flush=True)
    except Exception as exc:
        json_write(output / 'BLOCKED.json', {
            'stage': 'recovery_analyze', 'exception': type(exc).__name__,
            'message': str(exc), 'details': getattr(exc, 'details', None)})
        raise


if __name__ == '__main__':
    main()
