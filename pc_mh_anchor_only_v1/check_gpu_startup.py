#!/usr/bin/env python3
"""Read-only allocated-GPU/Trainer-startup diagnosis for smoke job 18149.

No LeWM weights/dataset, forward/backward, Trainer.fit, GPU reset, CUDA mask
rewrite, callback disabling, or automatic training submission. The two phases
use separate fresh processes to avoid changing each other's CUDA init order.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'NVIDIA_VISIBLE_DEVICES',
            'PYTORCH_NVML_BASED_CUDA_CHECK', 'SLURM_JOB_ID', 'SLURM_JOB_NODELIST',
            'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURM_GPUS_PER_TASK',
            'SLURM_TRES_PER_TASK', 'SLURM_TRES_BIND', 'SLURM_GPU_BIND',
            'SLURM_NTASKS', 'SLURM_LOCALID', 'SLURM_PROCID')
NODES = ('4090node2', '4090node3', '3090node1', '3090node2', '3090node3')


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    tmp.replace(path)


def check_counts(before, native, after, expected, local_rank):
    reasons = []
    if before != after:
        reasons.append(f'CUDA count changed across initialization: {before} -> {after}')
    if native is not None and native != after:
        reasons.append(f'Native runtime count {native} differs from public count {after}')
    if after != expected:
        reasons.append(f'Expected {expected} visible GPUs per DDP rank; CUDA reports {after}')
    if not 0 <= local_rank < after:
        reasons.append(f'Local rank {local_rank} has no corresponding CUDA ordinal')
    return reasons


def inspect_cuda(torch, expected, local_rank, record):
    # Public count can use NVML before CUDA initialization. Record, do not patch.
    record['count_before_init'] = int(torch.cuda.device_count())
    native = getattr(getattr(torch, '_C', None), '_cuda_getDeviceCount', None)
    record['native_runtime_count'] = int(native()) if callable(native) else None
    torch.cuda.init()
    record['count_after_init'] = int(torch.cuda.device_count())
    record['devices'] = []
    for i in range(record['count_after_init']):
        props = torch.cuda.get_device_properties(i)
        record['devices'].append({'logical_id': i, 'name': props.name,
            'total_memory': int(props.total_memory), 'uuid': str(getattr(props, 'uuid', 'unavailable'))})
    record['problems'] = check_counts(record['count_before_init'],
        record['native_runtime_count'], record['count_after_init'], expected, local_rank)
    # Only a tiny tensor on this task's logical GPU; never use an unallocated ID.
    if not record['problems']:
        x = torch.ones(4, device=f'cuda:{local_rank}')
        record['tiny_tensor_sum'] = float(x.sum().item())
        torch.cuda.synchronize(local_rank)
        del x
        if record['tiny_tensor_sum'] != 4.0:
            record['problems'].append('Tiny CUDA arithmetic check failed')
    record['passed'] = not record['problems']


def child(args):
    record = {'phase': args.phase, 'pid': os.getpid(), 'host': socket.gethostname(),
              'env': {k: os.environ.get(k) for k in ENV_KEYS}, 'passed': False}
    try:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('GPU probes must run inside the submitted Slurm allocation')
        if args.phase == 'cuda':
            import torch
            record['torch_version'] = torch.__version__
            record['torch_cuda_build'] = torch.version.cuda
            inspect_cuda(torch, args.expected, int(os.environ['SLURM_LOCALID']), record)
        else:
            # Same config-first import path as the real script; no dataset/model.
            sys.path.insert(0, str(HERE))
            import anchor_runtime as rt
            cfg = rt.config(args.base_config)
            if cfg['trainer']['devices'] != args.expected:
                raise RuntimeError('Saved baseline GPU count differs from allocation')
            import stable_pretraining as spt
            import lightning as pl
            import torch
            record['versions'] = {p: importlib.metadata.version(p) for p in
                ('torch', 'lightning', 'stable-pretraining', 'omegaconf')}
            record['count_before_trainer'] = int(torch.cuda.device_count())
            src = Path(spt.__file__).resolve().parent/'callbacks/cpu_offload.py'
            record['cpu_offload_file'] = str(src)
            if src.is_file():
                content = src.read_bytes()
                record['cpu_offload_sha256'] = hashlib.sha256(content).hexdigest()
                # Exact installed source, not assumed to equal current upstream.
                lines = content.decode('utf-8').splitlines()
                record['cpu_offload_source_first_130_lines'] = '\n'.join(
                    f'{i+1}: {line}' for i, line in enumerate(lines[:130]))
            kw = dict(cfg['trainer'])
            kw.update(default_root_dir=str(Path(args.output).parent/'trainer_artifacts'),
                      logger=False, enable_progress_bar=False)
            # This triggers the installed external callback factories exactly as
            # Trainer initialization does. No callbacks are removed or patched.
            trainer = pl.Trainer(**kw)
            record['callbacks'] = [f'{type(c).__module__}.{type(c).__qualname__}'
                                   for c in trainer.callbacks]
            record['count_after_trainer'] = int(torch.cuda.device_count())
            record['passed'] = record['count_after_trainer'] == args.expected
            if not record['passed']:
                record['message'] = (f'Expected {args.expected} GPUs; Trainer process '
                                     f"reports {record['count_after_trainer']}")
            del trainer
    except Exception as exc:
        record['error'] = type(exc).__name__
        record['message'] = str(exc)
        record['traceback'] = traceback.format_exc()
    finally:
        record['finished_unix'] = time.time()
        write_json(args.output, record)
    return 0 if record['passed'] else 1


def worker(args):
    out = Path(args.output).resolve()
    node = os.environ.get('SLURMD_NODENAME', socket.gethostname().split('.')[0])
    if node not in NODES or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use an allowed Slurm compute-node allocation')
    if int(os.environ.get('SLURM_NTASKS', '0')) != args.expected:
        raise RuntimeError('Expected one Slurm task per saved baseline GPU')
    rank = int(os.environ['SLURM_PROCID'])
    reports = {}
    # Each phase gets a new interpreter; the CUDA probe cannot pre-initialize
    # CUDA in the Trainer probe and accidentally conceal the original failure.
    for phase in ('trainer', 'cuda'):
        dest = out/f'rank{rank}_{phase}.json'
        if dest.exists():
            raise FileExistsError(dest)
        command = [sys.executable, str(Path(__file__).resolve()), 'child',
            '--phase', phase, '--output', str(dest), '--expected', str(args.expected),
            '--base-config', str(Path(args.base_config).resolve())]
        with (out/f'rank{rank}_{phase}.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=180, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        if not dest.exists():
            write_json(dest, {'passed': False, 'phase': phase,
                             'error': 'ChildProcessFailure', 'returncode': code})
        reports[phase] = {'returncode': code, 'report': dest.name}
        print(f'GPU CHECK rank={rank} phase={phase} exit={code}', flush=True)
    write_json(out/f'rank{rank}_complete.json', reports)
    # Finish recording both phases for all ranks. The summary determines the
    # JOB's exit status; a failed child is never converted into a successful gate.
    return 0


def summarize(args):
    out = Path(args.output).resolve()
    rows = []; passed = True
    for rank in range(args.expected):
        done = out/f'rank{rank}_complete.json'
        completion = json.loads(done.read_text()) if done.exists() else {}
        row = {'rank': rank, 'child_completion': completion}
        for phase in ('trainer', 'cuda'):
            path = out/f'rank{rank}_{phase}.json'
            r = json.loads(path.read_text()) if path.exists() else {'passed': False, 'error': 'Missing report'}
            row[phase] = r
            passed = (passed and r.get('passed') is True
                      and completion.get(phase, {}).get('returncode') == 0)
        rows.append(row)
        cuda = row['cuda']; trainer = row['trainer']
        print(f"rank={rank} mask={cuda.get('env',{}).get('CUDA_VISIBLE_DEVICES')} "
              f"pre/native/post={cuda.get('count_before_init')}/"
              f"{cuda.get('native_runtime_count')}/{cuda.get('count_after_init')} "
              f"cuda_ok={cuda.get('passed')} trainer_ok={trainer.get('passed')}")
        for phase in ('cuda', 'trainer'):
            if not row[phase].get('passed'):
                print(f"  {phase}: {row[phase].get('problems') or row[phase].get('message') or row[phase].get('error')}")
    # For this devices=N / N-task DDP layout, ranks must see the same GPU order.
    mappings = [tuple(x['uuid'] for x in r['cuda'].get('devices', [])) for r in rows]
    verified_mapping = all(len(m) == args.expected and 'unavailable' not in m for m in mappings)
    if verified_mapping and (any(m != mappings[0] for m in mappings) or
                             len(set(mappings[0])) != args.expected):
        passed = False
        print('GPU UUID mapping differs across ranks or contains duplicate devices')
    report = {'passed': bool(passed), 'expected_gpus': args.expected,
        'uuid_mapping_available': verified_mapping, 'ranks': rows,
        'note': 'Startup diagnostic only. No dataset/model/Trainer.fit/DDP collectives or training run.'}
    write_json(out/'GPU_CHECK.json', report)
    print('GPU STARTUP CHECK ' + ('PASS' if passed else 'FAILED') + ': ' + str(out))
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('worker', 'child', 'summarize'))
    p.add_argument('--output', required=True)
    p.add_argument('--expected', type=int, required=True)
    p.add_argument('--base-config', default='')
    p.add_argument('--phase', choices=('cuda', 'trainer'))
    a = p.parse_args()
    if not 1 <= a.expected <= 4:
        p.error('Expected saved baseline GPU count in 1..4')
    return {'worker': worker, 'child': child, 'summarize': summarize}[a.command](a)


if __name__ == '__main__':
    raise SystemExit(main())
