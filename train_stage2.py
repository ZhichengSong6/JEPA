"""Train Stage-II landscape-faithful JEPA on full PushT.

Stage II supports two controlled student initializations:

* ``pretrained``: initialize the student from official LeWM epoch 10 and use a
  frozen copy of that checkpoint as the curvature teacher;
* ``random``: initialize the same architecture from scratch while keeping the
  official LeWM checkpoint frozen as the curvature teacher.

The official train.py and Stage-I entrypoint are left untouched.
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

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from stage2_landscape_faithful import stage2_forward
from utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


def _raw_action_stats(dataset):
    """Raw 2-D action statistics matching the official training normalizer."""
    data = torch.from_numpy(np.array(dataset.get_col_data("action")))
    data = data[~torch.isnan(data).any(dim=1)]
    return data.mean(0, keepdim=True).float(), data.std(0, keepdim=True).float()


def _load_policy_model(policy_name: str):
    """Load a saved LeWM model object using the same resolver as evaluation."""
    model = swm.policy.AutoCostModel(str(policy_name))
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"AutoCostModel({policy_name!r}) returned {type(model)}, expected nn.Module."
        )
    return model


def _build_random_student(cfg):
    """Build exactly the official LeWM architecture with random parameters."""
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        factor_heads=None,
    )


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # Seed before model construction so Experiment C's random initialization is
    # reproducible. Manager receives the same seed below and re-seeds workers.
    pl.seed_everything(int(cfg.seed), workers=True)

    if not cfg.stage2.enabled:
        raise ValueError(
            "train_stage2.py requires stage2.enabled=True. "
            "Use train.py for official LeWM and train_stage1.py for Stage I."
        )
    if cfg.stage1.enabled:
        raise ValueError(
            "The first Stage-II experiment must isolate odd+curvature losses. "
            "Set stage1.enabled=False."
        )
    if cfg.factor.enabled:
        raise ValueError(
            "Stage II intentionally excludes privileged-state factor supervision; "
            "set factor.enabled=False."
        )
    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError(
            "Stage-II data must not load privileged simulator state. "
            "Use data=pusht_stage2."
        )

    expected_steps = int(cfg.wm.history_size) + int(cfg.stage2.rollout_horizon)
    if int(cfg.data.dataset.num_steps) != expected_steps:
        raise ValueError(
            "Stage-II sequence length mismatch. Expected "
            f"wm.history_size + stage2.rollout_horizon = {expected_steps}, "
            f"got data.dataset.num_steps={cfg.data.dataset.num_steps}. "
            "Launch with data=pusht_stage2."
        )

    init_mode = str(cfg.stage2.init_mode).lower()
    if init_mode not in {"pretrained", "random"}:
        raise ValueError(
            f"stage2.init_mode must be 'pretrained' or 'random', got {init_mode!r}."
        )

    #########################
    ##       dataset       ##
    #########################
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

    ##############################
    ##       model / optim      ##
    ##############################
    if init_mode == "pretrained":
        print(f"Loading Stage-II student init: {cfg.stage2.init_policy}")
        student = _load_policy_model(cfg.stage2.init_policy)
    else:
        print("Building Stage-II student from random initialization")
        student = _build_random_student(cfg)
    student.train()
    student.requires_grad_(True)

    print(f"Loading frozen curvature teacher: {cfg.stage2.teacher_policy}")
    teacher = _load_policy_model(cfg.stage2.teacher_policy)
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
            stage2_forward,
            cfg=cfg,
            action_mean=action_mean,
            action_std=action_std,
        ),
        optim=optimizers,
    )

    # Teacher is registered only for device movement/checkpoint visibility. It
    # is frozen and the optimizer pattern explicitly targets only `model`.
    training_module.teacher_model = teacher

    ##########################
    ##       training       ##
    ##########################
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
