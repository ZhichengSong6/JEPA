"""Train Multi-Horizon Anchored Local Dynamics (MH-ALD) on full PushT.

Decisive pretrained-only experiment:

* student init       : official LeWM epoch 10
* frozen visual frame: student encoder + projector
* frozen teacher     : official LeWM epoch 10
* trainable          : action encoder + predictor + pred_proj
* objective          : one-step TF + H-step rollout + MH-ALD

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

from anchored_local_dynamics import (
    cem_aligned_mh_ald_forward,
    mh_ald_forward,
)
from module import SIGReg
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
    """Freeze the observation-side latent coordinate frame exactly."""
    model.encoder.requires_grad_(False)
    model.projector.requires_grad_(False)
    model.encoder.eval()
    model.projector.eval()


def _train_predictor_side(model: torch.nn.Module) -> None:
    """Enable exactly the action-conditioned predictor-side parameters."""
    model.action_encoder.requires_grad_(True)
    model.predictor.requires_grad_(True)
    model.pred_proj.requires_grad_(True)
    if getattr(model, "factor_heads", None) is not None:
        model.factor_heads.requires_grad_(False)


def _count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _assert_same_visual_latent_frame(student, teacher, atol: float = 1e-7) -> None:
    """Require student/teacher encoder+projector to define one fixed frame.

    MH-ALD feeds the student's frozen visual embeddings into both predictor
    rollouts.  A replacement teacher is therefore valid only when its
    observation-side encoder/projector are numerically identical to the
    student's frozen frame.
    """
    worst = 0.0
    worst_name = None
    for prefix, s_mod, t_mod in (
        ("encoder", student.encoder, teacher.encoder),
        ("projector", student.projector, teacher.projector),
    ):
        s_state = s_mod.state_dict()
        t_state = t_mod.state_dict()
        if s_state.keys() != t_state.keys():
            raise RuntimeError(
                f"Student/teacher {prefix} state_dict keys differ."
            )
        for name in s_state:
            s = s_state[name].detach().float().cpu()
            t = t_state[name].detach().float().cpu()
            if s.shape != t.shape:
                raise RuntimeError(
                    f"Student/teacher visual frame shape mismatch at "
                    f"{prefix}.{name}: {tuple(s.shape)} vs {tuple(t.shape)}"
                )
            diff = float((s - t).abs().max()) if s.numel() else 0.0
            if diff > worst:
                worst = diff
                worst_name = f"{prefix}.{name}"
    print(
        "Student/teacher frozen visual-frame check: "
        f"max_abs={worst:.3e} at {worst_name}"
    )
    if worst > float(atol):
        raise RuntimeError(
            "Replacement teacher does not share the student's frozen visual "
            f"latent frame: max_abs={worst:.3e} > atol={atol:.1e}"
        )


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="lewm_mh_ald",
)
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    if not bool(cfg.mh_ald.enabled):
        raise ValueError("train_mh_ald.py requires mh_ald.enabled=True.")
    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError(
            "MH-ALD must not load privileged simulator state. Use data=pusht_mh_ald."
        )

    expected_steps = int(cfg.wm.history_size) + int(cfg.mh_ald.rollout_horizon)
    if int(cfg.data.dataset.num_steps) != expected_steps:
        raise ValueError(
            "MH-ALD sequence length mismatch. Expected "
            f"wm.history_size + mh_ald.rollout_horizon = {expected_steps}, "
            f"got data.dataset.num_steps={cfg.data.dataset.num_steps}."
        )

    # ------------------------------------------------------------------
    # Dataset and the official LeWM action/proprio normalization.
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

    # ------------------------------------------------------------------
    # Student / teacher.
    # ------------------------------------------------------------------
    print(f"Loading MH-ALD student init: {cfg.mh_ald.init_policy}")
    student = _load_policy_model(cfg.mh_ald.init_policy)
    student.requires_grad_(True)
    _freeze_visual_latent_frame(student)
    _train_predictor_side(student)

    print(f"Loading frozen MH-ALD teacher: {cfg.mh_ald.teacher_policy}")
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

    # The teacher and student must begin in the same frozen output latent frame.
    # We use the student's frozen visual embedding for both predictor rollouts.
    # Since both are initialized from the same policy, this is exact initially
    # and remains exact because the student's visual frame is frozen.
    total_params, trainable_params = _count_parameters(student)
    visual_trainable = sum(
        p.numel()
        for module in (student.encoder, student.projector)
        for p in module.parameters()
        if p.requires_grad
    )
    if visual_trainable != 0:
        raise RuntimeError(
            f"MH-ALD visual latent frame is not frozen: {visual_trainable} trainable params."
        )

    print(
        "MH-ALD parameter summary: "
        f"total={total_params:,} trainable={trainable_params:,} "
        f"frozen={total_params - trainable_params:,}"
    )
    print("Frozen: encoder + projector (+ optional factor heads)")
    print("Trainable: action_encoder + predictor + pred_proj")

    cem_aligned_enabled = bool(
        cfg.mh_ald.get("cem_aligned", {}).get("enabled", False)
    )
    selected_forward = (
        cem_aligned_mh_ald_forward
        if cem_aligned_enabled
        else mh_ald_forward
    )
    print(
        "MH-ALD forward mode: "
        + ("CEM-aligned population-weighted" if cem_aligned_enabled else "standard")
    )
    if cem_aligned_enabled:
        ccfg = cfg.mh_ald.cem_aligned
        print(
            "CEM-aligned config: "
            f"center_radius={float(ccfg.center_radius):.4f} "
            f"probe_radius={float(cfg.mh_ald.perturb_radius):.4f} "
            f"directions_per_position={int(cfg.mh_ald.directions_per_position)} "
            f"population={2 * int(cfg.mh_ald.rollout_horizon) * int(cfg.mh_ald.directions_per_position)} "
            f"elite={float(ccfg.elite_fraction):.2f}@{float(ccfg.elite_weight):.1f}x "
            f"near={float(ccfg.near_elite_fraction):.2f}@{float(ccfg.near_elite_weight):.1f}x "
            f"base={float(ccfg.base_weight):.1f}x"
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
        # SIGReg has buffers only and is not used in mh_ald_forward. Keeping the
        # field maintains compatibility with the existing spt.Module setup.
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(
            selected_forward,
            cfg=cfg,
            action_mean=action_mean,
            action_std=action_std,
        ),
        optim=optimizers,
    )

    # Device/checkpoint registration only; teacher is frozen and not optimized.
    training_module.teacher_model = teacher

    # ------------------------------------------------------------------
    # Training and AutoCostModel-compatible object checkpoints.
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
