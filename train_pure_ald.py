"""Train Pure Anchored Local Dynamics (Pure-ALD) on full PushT.

Controlled ablation relative to train_ald.py:
  * student init       : official LeWM epoch 10
  * frozen visual frame: encoder + projector
  * frozen teacher     : official LeWM epoch 10
  * trainable          : action encoder + predictor + pred_proj
  * objective          : Pure-ALD ONLY

Everything except the loss composition matches the established Full-ALD run.
No privileged simulator state or counterfactual ground-truth rollout is used.
"""
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from pure_ald import pure_ald_forward
from utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


def _raw_action_stats(dataset):
    data = torch.from_numpy(np.array(dataset.get_col_data("action")))
    data = data[~torch.isnan(data).any(dim=1)]
    return data.mean(0, keepdim=True).float(), data.std(0, keepdim=True).float()


def _load_policy_model(policy_name: str):
    model = swm.policy.AutoCostModel(str(policy_name))
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"AutoCostModel({policy_name!r}) returned {type(model)}, expected nn.Module."
        )
    return model


def _freeze_visual_latent_frame(model: torch.nn.Module) -> None:
    model.encoder.requires_grad_(False)
    model.projector.requires_grad_(False)
    model.encoder.eval()
    model.projector.eval()


def _train_predictor_side(model: torch.nn.Module) -> None:
    model.action_encoder.requires_grad_(True)
    model.predictor.requires_grad_(True)
    model.pred_proj.requires_grad_(True)
    if getattr(model, "factor_heads", None) is not None:
        model.factor_heads.requires_grad_(False)


def _count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="lewm_pure_ald",
)
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    if not bool(cfg.pure_ald.enabled):
        raise ValueError("train_pure_ald.py requires pure_ald.enabled=True.")
    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError(
            "Pure-ALD must not load privileged simulator state. "
            "Use data=pusht_pure_ald."
        )

    expected_steps = int(cfg.wm.history_size) + int(cfg.pure_ald.rollout_horizon)
    if int(cfg.data.dataset.num_steps) != expected_steps:
        raise ValueError(
            "Pure-ALD sequence length mismatch. Expected "
            f"wm.history_size + pure_ald.rollout_horizon = {expected_steps}, "
            f"got data.dataset.num_steps={cfg.data.dataset.num_steps}."
        )

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    action_mean, action_std = _raw_action_stats(dataset)

    transforms = [
        get_img_preprocessor(
            source="pixels", target="pixels", img_size=cfg.img_size
        )
    ]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    print(f"Loading Pure-ALD student init: {cfg.pure_ald.init_policy}")
    student = _load_policy_model(cfg.pure_ald.init_policy)
    student.requires_grad_(True)
    _freeze_visual_latent_frame(student)
    _train_predictor_side(student)

    print(f"Loading frozen Pure-ALD teacher: {cfg.pure_ald.teacher_policy}")
    teacher = _load_policy_model(cfg.pure_ald.teacher_policy)
    teacher.eval()
    teacher.requires_grad_(False)

    if (
        student.action_encoder.patch_embed.in_channels
        != teacher.action_encoder.patch_embed.in_channels
    ):
        raise RuntimeError("Pure-ALD student/teacher action dimensions differ.")

    total_params, trainable_params = _count_parameters(student)
    visual_trainable = sum(
        p.numel()
        for module in (student.encoder, student.projector)
        for p in module.parameters()
        if p.requires_grad
    )
    if visual_trainable != 0:
        raise RuntimeError(
            f"Pure-ALD visual frame has {visual_trainable} trainable params."
        )

    print(
        "Pure-ALD parameter summary: "
        f"total={total_params:,} trainable={trainable_params:,} "
        f"frozen={total_params-trainable_params:,}"
    )
    print("Frozen: encoder + projector + teacher (+ optional factor heads)")
    print("Trainable: action_encoder + predictor + pred_proj")
    print("ONLY objective: L = L_PureALD")

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    training_module = spt.Module(
        model=student,
        # Compatibility field only. SIGReg is never called in pure_ald_forward.
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(
            pure_ald_forward,
            cfg=cfg,
            action_mean=action_mean,
            action_std=action_std,
        ),
        optim=optimizers,
    )
    training_module.teacher_model = teacher

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=training_module,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
        seed=int(cfg.seed),
    )
    manager()


if __name__ == "__main__":
    run()
