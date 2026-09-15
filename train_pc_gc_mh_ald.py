"""Train Planner-Context Goal-Coupled Multi-Horizon ALD on PushT.

Controlled continuation of formal MH-ALD / PC-GP-MH:
* student init       : official LeWM epoch 10
* frozen visual frame: student encoder + projector
* frozen teacher     : official LeWM epoch 10
* trainable          : action encoder + predictor + pred_proj
* objective          : TF + planner-context rollout + PC-GC-MH

No privileged simulator state, counterfactual real rollout, learned reward,
rank label, or planner modification is used.
"""
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from planner_context_goal_coupled_mh import pc_gc_mh_ald_forward
from train_mh_ald import (
    _assert_same_visual_latent_frame,
    _count_parameters,
    _freeze_visual_latent_frame,
    _load_policy_model,
    _raw_action_stats,
    _train_predictor_side,
)
from utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="lewm_pc_gc_mh_ald",
)
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    if not bool(cfg.mh_ald.enabled):
        raise ValueError("PC-GC-MH requires mh_ald.enabled=True.")
    if not bool(cfg.mh_ald.planner_context.enabled):
        raise ValueError("PC-GC-MH requires planner_context.enabled=True.")
    if not bool(cfg.mh_ald.goal_coupled.enabled):
        raise ValueError("PC-GC-MH requires goal_coupled.enabled=True.")
    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError("PC-GC-MH must not load privileged simulator state.")

    expected_steps = int(cfg.wm.history_size) + int(cfg.mh_ald.rollout_horizon)
    if int(cfg.data.dataset.num_steps) != expected_steps:
        raise ValueError(
            "PC-GC-MH sequence length mismatch: expected "
            f"{expected_steps}, got {cfg.data.dataset.num_steps}."
        )
    if int(cfg.mh_ald.planner_context.observation_history) != 1:
        raise ValueError("Official PushT planning starts from one observation.")
    if int(cfg.mh_ald.planner_context.predictor_max_history) != int(cfg.wm.history_size):
        raise ValueError(
            "Keep official predictor max history equal to wm.history_size for "
            "this controlled experiment."
        )
    if float(cfg.mh_ald.goal_coupled.weight) < 0.0:
        raise ValueError("goal_coupled.weight must be >= 0.")

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    action_mean, action_std = _raw_action_stats(dataset)

    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
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

    print(f"Loading PC-GC-MH student init: {cfg.mh_ald.init_policy}")
    student = _load_policy_model(cfg.mh_ald.init_policy)
    student.requires_grad_(True)
    _freeze_visual_latent_frame(student)
    _train_predictor_side(student)

    print(f"Loading frozen PC-GC-MH teacher: {cfg.mh_ald.teacher_policy}")
    teacher = _load_policy_model(cfg.mh_ald.teacher_policy)
    teacher.eval()
    teacher.requires_grad_(False)
    _assert_same_visual_latent_frame(student, teacher)

    student_action_dim = student.action_encoder.patch_embed.in_channels
    teacher_action_dim = teacher.action_encoder.patch_embed.in_channels
    if student_action_dim != teacher_action_dim:
        raise RuntimeError(
            "Student/teacher packed action dimensions differ: "
            f"student={student_action_dim}, teacher={teacher_action_dim}."
        )

    total_params, trainable_params = _count_parameters(student)
    visual_trainable = sum(
        p.numel()
        for module in (student.encoder, student.projector)
        for p in module.parameters()
        if p.requires_grad
    )
    if visual_trainable != 0:
        raise RuntimeError(
            f"PC-GC-MH visual latent frame is not frozen: {visual_trainable}."
        )

    print(
        "PC-GC-MH parameter summary: "
        f"total={total_params:,} trainable={trainable_params:,} "
        f"frozen={total_params - trainable_params:,}"
    )
    print("Frozen: encoder + projector (+ optional factor heads)")
    print("Trainable: action_encoder + predictor + pred_proj")
    print(
        "PC-GC-MH config: "
        f"planner_obs_history={int(cfg.mh_ald.planner_context.observation_history)} "
        f"predictor_max_history={int(cfg.mh_ald.planner_context.predictor_max_history)} "
        f"probe_radius={float(cfg.mh_ald.perturb_radius):.4f} "
        f"goal_coupled_weight={float(cfg.mh_ald.goal_coupled.weight):.2f} "
        f"directions_per_position={int(cfg.mh_ald.directions_per_position)}"
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
            pc_gc_mh_ald_forward,
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
