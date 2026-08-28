"""Train strict Bias-Only calibration on full PushT LeWM.

Only a small latent offset head is optimized. The pretrained LeWM encoder,
action encoder, predictor, projector, and prediction projector are all frozen.

The training objective is exactly:

    L = || T(U_0) + B_phi(z_t, T(U_0)) - z_{t+H} ||^2

where T is the frozen official LeWM rollout started from a single observed
frame, matching planner inference semantics.
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

from bias_only_calibration import (
    BiasOnlyWorldModel,
    LatentBiasCalibrator,
    bias_only_forward,
)
from module import SIGReg
from utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


def _load_policy_model(policy_name: str):
    model = swm.policy.AutoCostModel(str(policy_name))
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"AutoCostModel({policy_name!r}) returned {type(model)}, expected nn.Module."
        )
    return model


def _count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="lewm_bias_only",
)
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    if not bool(cfg.bias_only.enabled):
        raise ValueError("train_bias_only.py requires bias_only.enabled=True.")
    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError(
            "Bias-Only must not load privileged simulator state. "
            "Use data=pusht_bias_only."
        )

    expected_steps = int(cfg.wm.history_size) + int(cfg.bias_only.rollout_horizon)
    if int(cfg.data.dataset.num_steps) != expected_steps:
        raise ValueError(
            "Bias-Only sequence length mismatch. Expected "
            f"wm.history_size + bias_only.rollout_horizon = {expected_steps}, "
            f"got data.dataset.num_steps={cfg.data.dataset.num_steps}."
        )

    # ------------------------------------------------------------------
    # Dataset and official LeWM preprocessing.
    # ------------------------------------------------------------------
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

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
    # Frozen LeWM + trainable tiny bias calibrator.
    # ------------------------------------------------------------------
    print(f"Loading frozen Bias-Only base: {cfg.bias_only.base_policy}")
    base = _load_policy_model(cfg.bias_only.base_policy)
    base.eval()
    base.requires_grad_(False)
    base.interpolate_pos_encoding = True

    latent_dim = int(cfg.wm.embed_dim)
    calibrator = LatentBiasCalibrator(
        latent_dim=latent_dim,
        hidden_dim=int(cfg.bias_only.hidden_dim),
    )
    wrapper = BiasOnlyWorldModel(
        base_model=base,
        calibrator=calibrator,
        history_size=int(cfg.wm.history_size),
    )

    # Re-enable only the calibrator after wrapper freezes the base model.
    wrapper.calibrator.requires_grad_(True)

    total_params, trainable_params = _count_parameters(wrapper)
    base_trainable = sum(
        p.numel() for p in wrapper.base_model.parameters() if p.requires_grad
    )
    if base_trainable != 0:
        raise RuntimeError(
            f"Bias-Only base LeWM has {base_trainable} trainable parameters."
        )

    print(
        "Bias-Only parameter summary: "
        f"total={total_params:,} trainable={trainable_params:,} "
        f"frozen={total_params-trainable_params:,}"
    )
    print("Frozen: entire official LeWM")
    print("Trainable: LatentBiasCalibrator only")
    print("Objective: endpoint calibration MSE only")

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
        model=wrapper,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(bias_only_forward, cfg=cfg),
        optim=optimizers,
    )

    # ------------------------------------------------------------------
    # Train and save AutoCostModel-compatible object checkpoints.
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
