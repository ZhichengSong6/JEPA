"""Experiment-B control: continue official LeWM training from epoch 10.

This control uses the SAME long offline sequence config and batch size as Stage
II Experiment A, but computes only the original one-step LeWM + SiGReg loss on
the initial official context window. It therefore controls for another 10
training epochs / batches without adding rollout, odd-symmetry, or curvature
losses.
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

import jepa  # noqa: F401
import module  # noqa: F401
from distributed_training import global_batch_sigreg
from module import SIGReg
from utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


def _load_policy_model(policy_name: str):
    model = swm.policy.AutoCostModel(str(policy_name))
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"AutoCostModel({policy_name!r}) returned {type(model)}, expected nn.Module."
        )
    return model


def lewm_continuation_forward(self, batch, stage, cfg):
    """Original LeWM loss restricted to the initial official context window."""
    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    sigreg_weight = float(cfg.loss.sigreg.weight)

    if n_preds != 1:
        raise ValueError(
            "Continuation control assumes official LeWM wm.num_preds=1, "
            f"got {n_preds}."
        )

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    if emb.shape[1] < ctx_len + n_preds:
        raise ValueError(
            f"Need at least {ctx_len + n_preds} states, got {emb.shape[1]}."
        )

    # Exactly the original LeWM one-step objective, but slice the first
    # ctx_len+n_preds states because data=pusht_stage2 contains a longer future
    # sequence so Experiment B has the same training rows/batches as A.
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_emb = emb[:, : ctx_len + n_preds]
    output["sigreg_loss"] = global_batch_sigreg(
        self.sigreg,
        sigreg_emb.transpose(0, 1),
        enabled=bool(cfg.loss.sigreg.get("global_batch_ddp", False)),
    )
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    if not cfg.continuation.enabled:
        raise ValueError(
            "train_lewm_continuation.py requires continuation.enabled=True."
        )
    if cfg.stage1.enabled or cfg.stage2.enabled or cfg.factor.enabled:
        raise ValueError(
            "Experiment B must isolate ordinary LeWM continuation: set "
            "stage1.enabled=false, stage2.enabled=false, factor.enabled=false."
        )
    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError(
            "Experiment B should use data=pusht_stage2 so A/B share the same "
            "non-privileged sequence rows and batches."
        )

    expected_steps = int(cfg.wm.history_size) + int(cfg.stage2.rollout_horizon)
    if int(cfg.data.dataset.num_steps) != expected_steps:
        raise ValueError(
            "Experiment B expects data=pusht_stage2 for matched batches; "
            f"expected num_steps={expected_steps}, got {cfg.data.dataset.num_steps}."
        )

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

    print(f"Loading LeWM continuation init: {cfg.continuation.init_policy}")
    student = _load_policy_model(cfg.continuation.init_policy)
    student.train()
    student.requires_grad_(True)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    training_module = spt.Module(
        model=student,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lewm_continuation_forward, cfg=cfg),
        optim=optimizers,
    )
    data_module = spt.data.DataModule(train=train, val=val)

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
