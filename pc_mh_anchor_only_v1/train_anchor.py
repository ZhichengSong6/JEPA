#!/usr/bin/env python3
"""Fresh PC-anchor-only run using the unchanged PC-MH training entrypoint.

Only the forward and audited export/log callback are replaced in this process.
No edits to upstream files, no rejected residual artifacts, no automatic resume.
"""
from __future__ import annotations
import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import socket
import time
import traceback
import torch
import anchor_runtime as rt
from anchor_objective import anchor_only_forward


def digest(module):
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        t = v.detach().cpu().contiguous()
        if (t.is_floating_point() or t.is_complex()) and not bool(torch.isfinite(t).all()):
            raise RuntimeError('Nonfinite model tensor: '+k)
        h.update(k.encode()); h.update(str((t.dtype,tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def plain_model(model):
    for name in ('coordinate_adapter','factor_heads','inverse_dynamics_head'):
        if getattr(model,name,None) is not None:
            raise RuntimeError('Unexpected inference component: '+name)


def check_parameters(module, gradients=False):
    model = module.model
    allowed = ('action_encoder.', 'predictor.', 'pred_proj.')
    for name, p in model.named_parameters():
        if p.requires_grad != name.startswith(allowed):
            raise RuntimeError('Trainable parameter scope differs: '+name)
    frozen = (model.encoder, model.projector, module.teacher_model)
    if any(p.requires_grad for mod in frozen for p in mod.parameters()):
        raise RuntimeError('Frozen parameter became trainable')
    if gradients:
        if any(p.grad is not None for mod in frozen for p in mod.parameters()):
            raise RuntimeError('Gradient leaked into frozen state')
        for name in ('action_encoder','predictor','pred_proj'):
            params = list(getattr(model,name).parameters())
            gs = [p.grad for p in params if p.grad is not None]
            if params and (not gs or not all(bool(torch.isfinite(g).all()) for g in gs)):
                raise RuntimeError('Missing/nonfinite predictor gradient: '+name)


def csv_append(path, row):
    exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists: w.writeheader()
        w.writerow(row)


def average(values):
    """Reduce only detached logging values; no change to optimization loss."""
    names = sorted(values)
    x = torch.stack([values[k].detach().float() for k in names])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(x)
        x /= torch.distributed.get_world_size()
    return dict(zip(names, x.cpu().tolist()))


def callback_type(base, cfg, m, out, holders):
    class CheckedExport(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.checked_backward = False
            self.epochs = 0
            self.complete = None
            holders.append(self)

        def on_fit_start(self, trainer, module):
            if int(trainer.world_size) != m['world_size']:
                raise RuntimeError('Actual DDP world size differs from the baseline')
            check_parameters(module)
            self.visual = (digest(module.model.encoder),digest(module.model.projector))
            self.teacher = digest(module.teacher_model)
            self.initial = digest(module.model)
            if self.initial != self.teacher: raise RuntimeError('Initial student/teacher differ')
            self.started = time.monotonic()
            if trainer.is_global_zero:
                rt.dump(out/'fit_audit.json',{'world_size':int(trainer.world_size),
                    'local_batch':m['local_batch'],'global_batch':m['global_batch'],
                    'parent_state_sha256':self.initial,'frozen_visual_sha256':list(self.visual),
                    'node':os.environ.get('SLURMD_NODENAME',socket.gethostname()),
                    'teacher_forward':False,'probe_sampler':False,'precision':str(cfg.trainer.precision)})
                print('ANCHOR FIT AUDIT PASS',flush=True)

        def on_train_epoch_start(self, trainer, module):
            self.batches = 0
            self.sums = None

        def on_after_backward(self, trainer, module):
            if not self.checked_backward:
                check_parameters(module, gradients=True)
                recorded = getattr(module, '_anchor_last', {})
                if 'pc_anchor_loss' not in recorded.get('train', {}):
                    raise RuntimeError(
                        'Missing anchor training metrics after backward: '
                        f'cached_stages={list(recorded)!r}; '
                        f'last_forward_stage={getattr(module, "_anchor_last_raw_stage", None)!r}. '
                        'Expected the fit/train forward to record pc_anchor_loss. '
                        'Do not bypass this check or reuse validation-only metrics.'
                    )
                self.checked_backward = True
                if trainer.is_global_zero: print('ANCHOR FIRST BACKWARD PASS',flush=True)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            values = module._anchor_last['train']
            if any(not bool(torch.isfinite(v).all()) for v in values.values()):
                raise RuntimeError('Nonfinite direct metric')
            self.batches += 1
            if self.sums is None: self.sums = {k:v.detach().float().clone() for k,v in values.items()}
            else:
                for k, v in values.items(): self.sums[k].add_(v.detach().float())
            if batch_idx == 0 or (batch_idx+1) % 200 == 0:
                row = {'epoch':int(trainer.current_epoch)+1,'global_step':int(trainer.global_step),
                       **average(values)}  # all ranks must participate
                if trainer.is_global_zero:
                    csv_append(out/'training_progress.csv',row)
                    print('ANCHOR TRAIN '+json.dumps(row),flush=True)

        def on_train_epoch_end(self, trainer, module):
            expected = int(trainer.num_training_batches)
            if self.batches != expected:
                raise RuntimeError(f'Incomplete epoch: {self.batches}/{expected} batches')
            if (digest(module.model.encoder),digest(module.model.projector)) != self.visual:
                raise RuntimeError('Frozen visual parameters/buffers changed')
            if digest(module.teacher_model) != self.teacher:
                raise RuntimeError('Frozen teacher changed')
            check_parameters(module)
            self.epochs = int(trainer.current_epoch)+1
            metrics = average({k:v/self.batches for k,v in self.sums.items()})
            if trainer.is_global_zero:
                csv_append(out/'epoch_metrics.csv',{'epoch':self.epochs,
                    'global_step':int(trainer.global_step),'batches_per_rank':self.batches,**metrics})
                plain_model(module.model)
                digest(module.model)
                path = out/f'{cfg.output_model_name}_epoch_{self.epochs}_object.ckpt'
                if path.exists(): raise FileExistsError(path)
                temporary = path.with_name(path.name+'.partial')
                torch.save(module.model, temporary)
                temporary.replace(path)
                print('SAVED PLAIN JEPA: '+str(path),flush=True)

        def on_fit_end(self, trainer, module):
            if (getattr(trainer,'interrupted',False) or not self.checked_backward or
                    self.epochs != int(cfg.trainer.max_epochs)):
                raise RuntimeError('Trainer stopped before the required training completed')
            if m['mode']=='smoke' and int(trainer.global_step)!=2:
                raise RuntimeError('Smoke must complete exactly two optimizer steps')
            final = digest(module.model)
            if final == self.initial: raise RuntimeError('Student weights did not change')
            self.complete = {'global_step':int(trainer.global_step),'epochs':self.epochs,
                'final_state_sha256':final,'elapsed_fit_seconds':time.monotonic()-self.started}
    return CheckedExport


def execute(cfg, m, out):
    """Separately testable baseline delegation; not a reimplementation of Manager."""
    import train_pc_mh_ald as baseline
    original = (baseline.pc_mh_ald_forward,baseline.ModelObjectCallBack,baseline._load_policy_model)
    parents, holders = [], []
    def load(name):
        if str(name) != str(cfg.mh_ald.init_policy): raise RuntimeError('Unexpected parent policy')
        model = original[2](name); plain_model(model)
        state = digest(model)
        if parents and state != parents[0]: raise RuntimeError('Parent models differ')
        parents.append(state)
        return model
    baseline.pc_mh_ald_forward = anchor_only_forward
    baseline.ModelObjectCallBack = callback_type(original[1],cfg,m,out,holders)
    baseline._load_policy_model = load
    try:
        inspect.unwrap(baseline.run)(cfg)
        if len(parents)!=2 or len(holders)!=1 or holders[0].complete is None:
            raise RuntimeError('Baseline execution did not reach audited fit completion')
        return holders[0].complete
    finally:
        baseline.pc_mh_ald_forward,baseline.ModelObjectCallBack,baseline._load_policy_model = original


def wait_json(path, timeout=180):
    end = time.monotonic()+timeout
    while not path.exists():
        if time.monotonic()>end: raise RuntimeError('Timed out waiting for '+str(path))
        time.sleep(.2)
    return rt.read(path)


def run(out):
    out = Path(out).resolve()
    cfg, m = rt.verify(out)
    job = os.environ.get('SLURM_JOB_ID')
    rank = int(os.environ.get('SLURM_PROCID','0'))
    node = os.environ.get('SLURMD_NODENAME',socket.gethostname().split('.')[0])
    if not job or not torch.cuda.is_available(): raise RuntimeError('Use the provided Slurm GPU launcher')
    if node not in rt.NODES or node != m['node']:
        raise RuntimeError('Job is running on an unapproved/different node: '+node)
    if int(os.environ.get('SLURM_NTASKS','1')) != m['world_size']:
        raise RuntimeError('srun process count must match the saved GPU count')
    if rank == 0:
        # No overwrite and no Manager auto-resume into a prior checkpoint.
        with (out/'job.lock').open('x') as f: f.write(job+'\n')
        if list(out.rglob('*.ckpt')): raise RuntimeError('Fresh run contains old checkpoints')
        rt.dump(out/'started.json',{'job_id':job,'time_unix':time.time()})
    if wait_json(out/'started.json')['job_id'] != job:
        raise RuntimeError('Output belongs to another Slurm job')
    try:
        result = execute(cfg,m,out)
        # All parent ranks must return from Manager before recording completion.
        rt.dump(out/f'finished_rank{rank}.json',{'job_id':job,**result})
        if rank == 0:
            for i in range(m['world_size']):
                r = wait_json(out/f'finished_rank{i}.json')
                if (r['job_id'],r['global_step'],r['epochs'],r['final_state_sha256']) != (
                        job,result['global_step'],result['epochs'],result['final_state_sha256']):
                    raise RuntimeError('Rank completion/state mismatch')
            if list(out.glob('FAILED_rank*.json')): raise RuntimeError('A rank reported failure')
            rt.verify(out)
            if rt.sha(m['comparison_checkpoint']) != m['comparison_checkpoint_sha256']:
                raise RuntimeError('Old comparison checkpoint changed')
            final = out/f"{cfg.output_model_name}_epoch_{result['epochs']}_object.ckpt"
            if not final.is_file(): raise RuntimeError('No final model export')
            rt.dump(out/'training_complete.json',{'mode':m['mode'],'job_id':job,
                'all_ranks_finished':True,'world_size':m['world_size'],**result,
                'baseline_config_sha256':m['baseline_config_sha256'],'source_sha256':m['source_sha256'],
                'protocol_sha256':rt.sha(out/'protocol.json'),
                'final_checkpoint':str(final),'final_checkpoint_sha256':rt.sha(final),
                'policy':str(Path(cfg.subdir)/f"{cfg.output_model_name}_epoch_{result['epochs']}"),
                'note':'Execution completion, not evidence of planning success or a new method'})
            print(f"PC-ANCHOR-ONLY {m['mode'].upper()} PASS: {out}",flush=True)
    except Exception as exc:
        error = {'rank':rank,'job_id':job,'error':type(exc).__name__,
                 'message':str(exc),'traceback':traceback.format_exc()}
        rt.dump(out/f'FAILED_rank{rank}.json',error)
        if rank == 0: rt.dump(out/'BLOCKED.json',error)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command',required=True)
    s = subs.add_parser('prepare')
    s.add_argument('--base-config',required=True); s.add_argument('--output',required=True)
    s.add_argument('--mode',choices=['smoke','formal'],required=True)
    s.add_argument('--node',required=True); s.add_argument('--smoke-run')
    s = subs.add_parser('run'); s.add_argument('--output',required=True)
    s = subs.add_parser('check'); s.add_argument('--base-config',required=True)
    a = p.parse_args()
    if a.command == 'prepare': rt.prepare(a)
    elif a.command == 'run': run(a.output)
    else:
        # Exact config-first order in a FRESH process; no dataset/model/Trainer.
        rt.sources(); d = rt.config(a.base_config)
        import train_pc_mh_ald
        import train_mh_ald
        print(f"REAL CONFIG -> TRAINING IMPORT PASS: GPUs={d['trainer']['devices']}, local_batch={d['loader']['batch_size']}")
        print('IMPORT CHECK ONLY: no data/checkpoint loading, no GPU training')


if __name__ == '__main__': main()
