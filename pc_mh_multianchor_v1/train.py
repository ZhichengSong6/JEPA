#!/usr/bin/env python3
"""Original PC-MH + 0.5 * supported query MSE. Frozen visual frame / teacher.

The production data pipeline, optimizer and scheduler are executed by the
UNCHANGED repository train_pc_mh_ald entrypoint. No retrieval modules are saved
inside the inference JEPA. Same-query control changes only the target tensor.
"""
from __future__ import annotations
import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from core import load_npz,query_loss,dump,sha
import runtime as rt


class QueryPool:
    def __init__(self,prepared,seed):
        paths=sorted((Path(prepared)/'query_chunks').glob('chunk_*.npz'))
        if not paths: raise ValueError('No prepared queries')
        parts=[load_npz(p) for p in paths]
        keys=('z','u','target','old_target','mask')
        self.data={k:torch.from_numpy(np.concatenate([p[k] for p in parts])) for k in keys}
        n=len(self.data['z'])
        if not n or any(len(v)!=n for v in self.data.values()): raise ValueError('Malformed query pool')
        self.generator=torch.Generator().manual_seed(seed)

    def sample(self,n,device):
        ix=torch.randint(len(self.data['z']),(n,),generator=self.generator)
        return {k:v[ix].to(device) for k,v in self.data.items()}


def augment_forward(original,pool,query_batch,variant,last):
    """Never modify the baseline batch or consume its probe/data RNG."""
    def forward(self,batch,stage,cfg,action_mean,action_std):
        from planner_context_mh import planner_context_rollout
        output=original(self,batch,stage,cfg,action_mean,action_std)
        if stage=='train':
            q=pool.sample(query_batch,next(self.model.parameters()).device)
            pred=planner_context_rollout(self.model,q['z'],q['u'],3)
            target=q['target'] if variant=='corrected' else q['old_target']
            extra=query_loss(pred,target,q['mask'])
            base=output['loss']
            output={**output,'loss':base+.5*extra,'ma_query_loss':extra,'ma_base_loss':base}
            # Record DIRECT outputs, not callback_metrics aliases (previous CSV bug).
            last.clear();last.update({k:float(v.detach().float().cpu()) for k,v in output.items()
                                      if torch.is_tensor(v) and v.numel()==1})
            last['accepted_horizon_fraction']=float(q['mask'].float().mean().cpu())
        return output
    return forward


_ACTIVE_FORWARD = None

def ma_forward(self, batch, stage, cfg, action_mean, action_std):
    # Top-level callable remains pickleable in the baseline Lightning checkpoint.
    if _ACTIVE_FORWARD is None:
        raise RuntimeError('Use this training entrypoint; no auto-resume of a closure')
    return _ACTIVE_FORWARD(self, batch, stage, cfg, action_mean, action_std)


def rank(): return int(os.environ.get('SLURM_PROCID',os.environ.get('RANK','0')))


def start_directory(out,payload):
    marker=out/'launch.json'
    if rank()==0:
        out.mkdir(parents=True,exist_ok=False)
        dump(marker,payload)
    else:
        deadline=time.monotonic()+120
        while not marker.exists():
            if time.monotonic()>deadline: raise RuntimeError('Rank0 did not prepare launch marker')
            time.sleep(.2)
        other=json.loads(marker.read_text())
        if other['job_id']!=payload['job_id']: raise RuntimeError('Output belongs to another Slurm job')


def main():
    global _ACTIVE_FORWARD
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variant',choices=['corrected','same_queries'],default='corrected')
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    prepared=Path(args.prepared).resolve();out=Path(args.output).resolve()
    cfg,m=rt.verify(prepared)
    for stage in ('cache_complete','audit_complete','queries_complete'): rt.check_seal(prepared,stage)
    ready_path=prepared/'READY.json'
    if not args.smoke:
        if m['smoke'] or not ready_path.is_file() or not json.loads(ready_path.read_text()).get('formal_ready'):
            raise RuntimeError('Formal training forbidden: no passing full preparation READY.json')
        audit=json.loads((prepared/'audit_report.json').read_text())
        query=json.loads((prepared/'query_report.json').read_text())
        if not audit['passed'] or not query['passed']: raise RuntimeError('Independent gates rejected')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Use allocated Slurm GPUs')
    world=int(cfg.trainer.devices)
    if int(os.environ.get('SLURM_NTASKS','1'))!=world: raise RuntimeError('World size must match saved PC-MH baseline')
    root=Path(os.environ['STABLEWM_HOME']).resolve()
    if root not in out.parents or out==prepared or prepared in out.parents: raise ValueError('Separate fresh train output required')
    O=rt.omega();cfg=O.create(O.to_container(cfg,resolve=True))
    original_cfg=O.to_container(cfg,resolve=True)
    cfg.subdir=str(out.relative_to(root))
    cfg.output_model_name=str(cfg.output_model_name)+'_ma_'+args.variant+('_smoke' if args.smoke else '')
    if cfg.wandb.enabled:
        cfg.wandb.config.id=str(cfg.subdir);cfg.wandb.config.name=str(cfg.output_model_name);cfg.wandb.config.resume='never'
    if args.smoke:
        cfg.trainer.max_epochs=1;cfg.trainer.max_steps=2
        cfg.trainer.limit_train_batches=2;cfg.trainer.limit_val_batches=1
    payload={'job_id':os.environ['SLURM_JOB_ID'],'prepared':str(prepared),'variant':args.variant,
             'smoke':args.smoke,'world_size':world,'local_batch':int(cfg.loader.batch_size),
             'global_batch':world*int(cfg.loader.batch_size),'query_batch_per_rank':16,
             'lambda_query':.5,'parent_checkpoint_sha256':m['checkpoint_sha256'],
             'prepared_protocol_sha256':sha(prepared/'protocol.json'),'training_source_sha256':sha(__file__),
             'base_config':original_cfg,'no_resume':True}
    start_directory(out,payload)
    pool=QueryPool(prepared,seed=20261013+rank());last={}
    import train_pc_mh_ald as baseline
    original_forward=baseline.pc_mh_ald_forward;original_callback=baseline.ModelObjectCallBack
    original_loader=baseline._load_policy_model
    expected_parent=json.loads((prepared/'parent_state.json').read_text())['state_sha256']
    def loader(name):
        model=original_loader(name)
        if rt.digest(model)!=expected_parent: raise RuntimeError('Parent differs from cached teacher')
        return model
    class Export(original_callback):
        def on_fit_start(self,trainer,pl_module):
            if trainer.world_size!=world: raise RuntimeError('Actual DDP world size differs')
            self.visual=[rt.digest(pl_module.model.encoder),rt.digest(pl_module.model.projector)]
            self.teacher=rt.digest(pl_module.teacher_model);self.first=False;self.epochs=0
            for mod in (pl_module.model.encoder,pl_module.model.projector,pl_module.teacher_model):
                if any(p.requires_grad for p in mod.parameters()): raise RuntimeError('Frozen parameter became trainable')
            if trainer.is_global_zero: print('MA FIT AUDIT PASS',flush=True)
        def on_after_backward(self,trainer,pl_module):
            if self.first:return
            if 'ma_query_loss' not in last: raise RuntimeError('Training stage did not execute query supervision')
            for mod in (pl_module.model.encoder,pl_module.model.projector,pl_module.teacher_model):
                if any(p.grad is not None for p in mod.parameters()): raise RuntimeError('Gradient leaked to frozen modules')
            for name in ('action_encoder','predictor','pred_proj'):
                gs=[p.grad for p in getattr(pl_module.model,name).parameters() if p.grad is not None]
                if not gs or not all(bool(torch.isfinite(g).all()) for g in gs): raise RuntimeError('Bad gradient: '+name)
            self.first=True
            if trainer.is_global_zero: print('MA FIRST BACKWARD PASS',flush=True)
        def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
            if not last or not all(np.isfinite(x) for x in last.values()): raise RuntimeError('Missing/nonfinite training metrics')
            if trainer.is_global_zero and (batch_idx==0 or (batch_idx+1)%200==0):
                row={'epoch':trainer.current_epoch+1,'global_step':trainer.global_step,**last}
                path=out/'training_progress.csv';exists=path.exists()
                with path.open('a',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row))
                    if not exists:w.writeheader()
                    w.writerow(row)
                print('MA TRAIN '+json.dumps(row),flush=True)
        def on_train_epoch_end(self,trainer,pl_module):
            model=pl_module.model
            if self.visual!=[rt.digest(model.encoder),rt.digest(model.projector)] or self.teacher!=rt.digest(pl_module.teacher_model):
                raise RuntimeError('Frozen parameters or buffers changed')
            self.epochs=trainer.current_epoch+1
            if trainer.is_global_zero:
                rt.digest(model)
                for a in ('coordinate_adapter','factor_heads','inverse_dynamics_head'):
                    if getattr(model,a,None) is not None:raise RuntimeError('Unexpected exported module')
                path=out/f'{cfg.output_model_name}_epoch_{self.epochs}_object.ckpt'
                if path.exists(): raise FileExistsError(path)
                tmp=path.with_suffix('.partial');torch.save(model,tmp);tmp.replace(path)
                print('SAVED PLAIN JEPA: '+str(path),flush=True)
    _ACTIVE_FORWARD=augment_forward(original_forward,pool,16,args.variant,last)
    baseline.pc_mh_ald_forward=ma_forward
    baseline.ModelObjectCallBack=Export;baseline._load_policy_model=loader
    try:
        inspect.unwrap(baseline.run)(cfg)
        if rank()==0:
            epoch=int(cfg.trainer.max_epochs)
            path=out/f'{cfg.output_model_name}_epoch_{epoch}_object.ckpt'
            if not path.is_file():raise RuntimeError('Training returned without final model')
            dump(out/'training_complete.json',{'mode':'smoke' if args.smoke else 'formal',
                 'final_checkpoint':str(path),'final_checkpoint_sha256':sha(path),
                 'policy':str(Path(cfg.subdir)/f'{cfg.output_model_name}_epoch_{epoch}'),
                 'variant':args.variant,'note':'Execution success, not planning improvement'})
            print('MA TRAINING '+('SMOKE' if args.smoke else 'FORMAL')+' PASS',flush=True)
    except Exception as exc:
        if rank()==0:dump(out/'BLOCKED.json',{'error':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        baseline.pc_mh_ald_forward=original_forward;baseline.ModelObjectCallBack=original_callback
        baseline._load_policy_model=original_loader
        _ACTIVE_FORWARD=None


if __name__=='__main__':main()
