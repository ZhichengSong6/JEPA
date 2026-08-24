"""Train Stage-II-v2 APB+JVP fine-tuning on full PushT.

This entrypoint intentionally supports only the decisive pretrained-student
experiment (A-v2):

* student init  : official LeWM epoch 10
* frozen teacher: official LeWM epoch 10
* objective     : LeWM + rollout + normalized APB + direct teacher JVP

The failed odd+Gram Stage-II implementation is preserved unchanged for
comparison and reproducibility.
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
from stage2_v2_apb_jvp import stage2_v2_forward
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


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="lewm_stage2_v2",
)
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    if not bool(cfg.stage2_v2.enabled):
        raise ValueError("train_stage2_v2.py requires stage2_v2.enabled=True.")
    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError(
            "Stage-II-v2 must not load privileged simulator state. "
            "Use data=pusht_stage2_v2."
        )

    expected_steps = int(cfg.wm.history_size) + int(cfg.stage2_v2.rollout_horizon)
    if int(cfg.data.dataset.num_steps) != expected_steps:
        raise ValueError(
            "Stage-II-v2 sequence length mismatch. Expected "
            f"wm.history_size + stage2_v2.rollout_horizon = {expected_steps}, "
            f"got data.dataset.num_steps={cfg.data.dataset.num_steps}."
        )

    # ------------------------------------------------------------------
    # Dataset and identical official normalization.
    # ------------------------------------------------------------------
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

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

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

    # ------------------------------------------------------------------
    # Pretrained student + frozen teacher, both from the same official LeWM10.
    # ------------------------------------------------------------------
    print(f"Loading Stage-II-v2 student init: {cfg.stage2_v2.init_policy}")
    student = _load_policy_model(cfg.stage2_v2.init_policy)
    student.train()
    student.requires_grad_(True)

    print(f"Loading frozen JVP teacher: {cfg.stage2_v2.teacher_policy}")
    teacher = _load_policy_model(cfg.stage2_v2.teacher_policy)
    teacher.eval()
    teacher.requires_grad_(False)

    student_action_dim = student.action_encoder.patch_embed.in_channels
    teacher_action_dim = teacher.action_encoder.patch_embed.in_channels
    if student_action_dim != teacher_action_dim:
        raise RuntimeError(
            "Student/teacher packed action dimensions differ: "
            f"student={student_action_dim}, teacher={teacher_action_dim}."
        )

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
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(
            stage2_v2_forward,
            cfg=cfg,
            action_mean=action_mean,
            action_std=action_std,
        ),
        optim=optimizers,
    )

    # Registered for device movement/checkpoint visibility only. The optimizer
    # explicitly targets ``model`` and the teacher is requires_grad_(False).
    training_module.teacher_model = teacher

    # ------------------------------------------------------------------
    # Training / model-object saving compatible with AutoCostModel evaluation.
    # ------------------------------------------------------------------
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
