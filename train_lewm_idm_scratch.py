"""Stage 1: jointly train LeWM + ordinary IDM from random initialization.

The loss and inverse head are reused verbatim from the existing IDM experiment:
    prediction + 0.09 * SIGReg + 0.10 * inverse-action MSE.
No checkpoint, teacher, PC-MH objective, privileged state, or new IDM variant is
used. The default schedule is 10 epochs, followed by evaluation (not automatic
PC-MH training). Inference objects omit the head; training weights retain it.
"""
from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

from lewm_idm_scratch import (
    attach_inverse_head, build_scratch_lewm, checked_run_dir, save_inference_object,
)
from module import SIGReg
from train_lewm_idm import lewm_idm_forward
from utils import get_column_normalizer, get_img_preprocessor


MODULE_GROUPS = (
    "encoder", "projector", "action_encoder", "predictor", "pred_proj",
    "inverse_dynamics_head",
)


class ScratchIDMChecks(pl.Callback):
    """Check the actual process count and first backward, then export per epoch."""
    def __init__(self, run_dir: Path, model_name: str, expected_world_size: int):
        self.run_dir = run_dir
        self.model_name = model_name
        self.expected_world_size = expected_world_size
        self.backward_checked = False

    def on_fit_start(self, trainer, pl_module):
        if int(trainer.world_size) != self.expected_world_size:
            raise RuntimeError(
                f"Expected {self.expected_world_size} ranks; got {trainer.world_size}."
            )
        if trainer.is_global_zero:
            print(f"SCRATCH IDM FIT: world_size={trainer.world_size}", flush=True)

    def on_after_backward(self, trainer, pl_module):
        if self.backward_checked:
            return
        for group in MODULE_GROUPS:
            parameters = list(getattr(pl_module.model, group).parameters())
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not parameters or not all(p.requires_grad for p in parameters):
                raise RuntimeError(f"Module is not fully trainable: {group}")
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError(f"Missing/nonfinite first-backward gradients: {group}")
        # AdaLN-zero may legitimately yield zero-valued gradients initially.
        self.backward_checked = True
        if trainer.is_global_zero:
            print("FIRST BACKWARD OK: all six module groups receive finite gradients", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            path = self.run_dir / (
                f"{self.model_name}_epoch_{trainer.current_epoch + 1}_object.ckpt"
            )
            save_inference_object(pl_module.model, path)
            print(f"SAVED INFERENCE OBJECT (IDM head stripped): {path}", flush=True)


def _launcher_rank_zero() -> bool:
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return int(os.environ[key]) == 0
    return True


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_idm_scratch")
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)
    if cfg.initialization != "scratch" or cfg.idm.init_policy is not None:
        raise ValueError("Scratch entry requires initialization=scratch and idm.init_policy=null.")
    if not bool(cfg.idm.enabled) or float(cfg.idm.weight) < 0:
        raise ValueError("IDM must be enabled with a nonnegative weight.")
    if int(cfg.wm.history_size) != 3 or int(cfg.wm.num_preds) != 1:
        raise ValueError("This controlled experiment uses history=3 and num_preds=1.")
    if int(cfg.data.dataset.num_steps) != 4 or int(cfg.idm.max_transitions) != 3:
        raise ValueError("Expected four observation frames and three adjacent IDM transitions.")
    if set(cfg.data.dataset.keys_to_load) != {"pixels", "action", "proprio"}:
        raise ValueError("Only pixels, action, and proprio may be loaded.")
    if not os.environ.get("STABLEWM_HOME"):
        raise ValueError("Set STABLEWM_HOME to the project data directory explicitly.")
    if Path(str(cfg.output_model_name)).name != str(cfg.output_model_name):
        raise ValueError("output_model_name must be a filename stem, not a path.")

    run_dir = checked_run_dir(os.environ["STABLEWM_HOME"], str(cfg.subdir))
    expected_world_size = int(cfg.expected_world_size)
    if int(cfg.trainer.devices) * int(cfg.trainer.num_nodes) != expected_world_size:
        raise ValueError("trainer devices/nodes do not match expected_world_size.")
    if expected_world_size > 1:
        if not bool(cfg.loss.sigreg.global_batch_ddp) or not bool(cfg.trainer.sync_batchnorm):
            raise ValueError("DDP scratch runs require global SIGReg and SyncBatchNorm.")

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=int(cfg.img_size))]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False,
    )

    print("INIT_MODE=scratch; pretrained=False; no checkpoint is loaded", flush=True)
    student = build_scratch_lewm(cfg)
    action_dim = int(student.action_encoder.patch_embed.in_channels)
    attach_inverse_head(
        student, latent_dim=int(cfg.wm.embed_dim), action_dim=action_dim,
        hidden_dim=int(cfg.idm.head_hidden_dim), depth=int(cfg.idm.head_depth),
        seed=int(cfg.seed),
    )
    student.requires_grad_(True)
    student.train()
    counts = {name: sum(p.numel() for p in getattr(student, name).parameters())
              for name in MODULE_GROUPS}
    print(f"Trainable module counts: {counts}", flush=True)
    print("Objective = pred_loss + 0.09*sigreg_loss + 0.10*idm_loss (default weights)", flush=True)

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    training_module = spt.Module(
        model=student, sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lewm_idm_forward, cfg=cfg),
        optim={"model_opt": {
            "modules": "model", "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"}, "interval": "epoch",
        }},
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if _launcher_rank_zero():
        OmegaConf.save(cfg, run_dir / "config.yaml")
        manifest = {
            "initialization": "scratch", "pretrained": False, "init_checkpoint": None,
            "seed": int(cfg.seed), "epochs": int(cfg.trainer.max_epochs),
            "expected_world_size": expected_world_size,
            "local_batch": int(cfg.loader.batch_size),
            "global_batch": int(cfg.loader.batch_size) * expected_world_size,
            "trainable_modules": counts, "code_commit": os.environ.get("JEPA_GIT_HEAD"),
            "train_examples": len(train_set), "validation_examples": len(val_set),
            "inference_head_stripped": True, "pc_mh_enabled": False,
        }
        (run_dir / "scratch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[ScratchIDMChecks(run_dir, str(cfg.output_model_name), expected_world_size)],
        num_sanity_val_steps=1, logger=False, enable_checkpointing=True,
        default_root_dir=str(run_dir),
    )
    manager = spt.Manager(
        trainer=trainer, module=training_module, data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt", seed=int(cfg.seed),
    )
    manager()


if __name__ == "__main__":
    run()
