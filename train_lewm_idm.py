"""Pure inverse-dynamics LeWM continuation on PushT.

This experiment isolates one encoder-side intervention:

    L = L_pred + 0.09 L_SIGReg + 0.10 L_IDM
    L_IDM = || h_inv(z_t, z_{t+1}) - a_t ||^2

Student initialization is the official LeWM epoch-10 checkpoint.  Unlike
MH-ALD calibration, the complete LeWM is trainable because the purpose of IDM
is to shape the observation latent geometry itself.  The training data contain
only pixels, actions, and proprioception.

The inverse head is attached to the model during optimization so Stable
Pretraining's existing model optimizer updates it jointly with the encoder.
Object checkpoints used for AutoCostModel evaluation are saved *without* the
head; JEPA rollout/cost semantics are therefore exactly unchanged.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

import jepa  # noqa: F401  # needed when unpickling official model objects
import module  # noqa: F401
from distributed_training import global_batch_sigreg
from inverse_dynamics import InverseDynamicsHead, inverse_dynamics_objective
from module import SIGReg
from utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


def _load_policy_model(policy_name: str):
    model = swm.policy.AutoCostModel(str(policy_name))
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"AutoCostModel({policy_name!r}) returned {type(model)}, expected nn.Module."
        )
    return model


class IDMInferenceObjectCallBack(ModelObjectCallBack):
    """Save the inference model with the training-only IDM head removed."""

    def _dump_model(self, model, path):
        modules = getattr(model, "_modules", None)
        if modules is None or "inverse_dynamics_head" not in modules:
            raise RuntimeError(
                "IDM object checkpoint requested but inverse_dynamics_head "
                "is not registered on the training model."
            )

        head = modules.pop("inverse_dynamics_head")
        try:
            # The base callback calls torch.save(model, path).  While the head
            # is absent, the serialized object has exactly the ordinary JEPA
            # inference graph.  Restore immediately afterwards so training
            # checkpoints/resume state keep the auxiliary head.
            super()._dump_model(model, path)
        finally:
            model.add_module("inverse_dynamics_head", head)


def lewm_idm_forward(self, batch, stage, cfg):
    """Original LeWM objective plus one training-only inverse-dynamics loss."""
    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    sigreg_weight = float(cfg.loss.sigreg.weight)

    if n_preds != 1:
        raise ValueError(f"IDM experiment assumes wm.num_preds=1, got {n_preds}.")
    if not hasattr(self.model, "inverse_dynamics_head"):
        raise RuntimeError("Training model has no inverse_dynamics_head.")

    # Preserve the normalized-but-unmodified action tensor for the IDM target.
    # NaN rows mark invalid sequence-boundary transitions and are masked by
    # inverse_dynamics_objective().  The forward model receives the same zero
    # replacement used by official LeWM training.
    idm_action_target = batch["action"]
    batch["action"] = torch.nan_to_num(
        batch["action"], nan=0.0, posinf=0.0, neginf=0.0
    )

    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    core_len = ctx_len + n_preds
    if emb.shape[1] < core_len:
        raise ValueError(
            f"LeWM+IDM needs at least {core_len} frames, got {emb.shape[1]}."
        )

    core_emb = emb[:, :core_len]
    ctx_emb = core_emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = core_emb[:, n_preds:core_len]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = global_batch_sigreg(
        self.sigreg,
        core_emb.transpose(0, 1),
        enabled=bool(cfg.loss.sigreg.get("global_batch_ddp", False)),
    )

    inv = inverse_dynamics_objective(
        self.model.inverse_dynamics_head,
        core_emb,
        idm_action_target[:, :core_len],
        max_transitions=int(cfg.idm.max_transitions),
    )
    idm_loss = inv["loss"]

    total_loss = (
        pred_loss
        + sigreg_weight * sigreg_loss
        + float(cfg.idm.weight) * idm_loss
    )

    output.update(
        {
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "idm_loss": idm_loss,
            "loss": total_loss,
        }
    )

    losses = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k
    }
    diagnostics = {
        f"{stage}/idm_action_mae": inv["mae"].detach(),
        f"{stage}/idm_action_cosine": inv["cosine"].detach(),
        f"{stage}/idm_latent_action_norm_corr": (
            inv["latent_action_norm_corr"].detach()
        ),
        f"{stage}/idm_valid_fraction": inv["valid_fraction"].detach(),
        f"{stage}/idm_valid_count": inv["valid_count"].detach(),
    }
    self.log_dict(losses, on_step=True, sync_dist=True)
    self.log_dict(diagnostics, on_step=True, sync_dist=True)
    return output


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="lewm_idm",
)
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    if not bool(cfg.idm.enabled):
        raise ValueError("train_lewm_idm.py requires idm.enabled=true.")
    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError(
            "IDM experiment must not load simulator state. "
            "Use data=pusht_idm."
        )

    expected_steps = int(cfg.wm.history_size) + int(cfg.wm.num_preds)
    if int(cfg.data.dataset.num_steps) != expected_steps:
        raise ValueError(
            "Pure IDM uses the original four-frame LeWM clip only: "
            f"expected num_steps={expected_steps}, "
            f"got {cfg.data.dataset.num_steps}."
        )
    if int(cfg.idm.max_transitions) != expected_steps - 1:
        raise ValueError(
            "For the controlled PushT IDM experiment, max_transitions must "
            f"equal {expected_steps - 1}, got {cfg.idm.max_transitions}."
        )

    # ------------------------------------------------------------------
    # Dataset and original LeWM normalization.
    # ------------------------------------------------------------------
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=int(cfg.img_size),
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
    # Pretrained student + training-only inverse head.
    # ------------------------------------------------------------------
    print(f"Loading LeWM+IDM init: {cfg.idm.init_policy}")
    student = _load_policy_model(cfg.idm.init_policy)
    student.requires_grad_(True)
    student.train()

    if hasattr(student, "inverse_dynamics_head"):
        raise RuntimeError(
            "Initialization checkpoint unexpectedly already contains an "
            "inverse_dynamics_head."
        )

    action_dim = int(student.action_encoder.patch_embed.in_channels)
    latent_dim = int(cfg.wm.embed_dim)
    inverse_head = InverseDynamicsHead(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=int(cfg.idm.head_hidden_dim),
        depth=int(cfg.idm.head_depth),
    )
    student.add_module("inverse_dynamics_head", inverse_head)

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(
        p.numel() for p in student.parameters() if p.requires_grad
    )
    head_params = sum(p.numel() for p in inverse_head.parameters())
    if total_params != trainable_params:
        raise RuntimeError(
            "Pure LeWM+IDM should train the complete world model and inverse "
            f"head, but found total={total_params:,}, trainable={trainable_params:,}."
        )

    print(
        "LeWM+IDM parameter summary: "
        f"total/trainable={trainable_params:,}, "
        f"inverse_head={head_params:,}, "
        f"latent_dim={latent_dim}, action_dim={action_dim}"
    )
    print(
        "Objective: original LeWM prediction + SIGReg + "
        f"{float(cfg.idm.weight):.3f} * inverse dynamics"
    )

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
        forward=partial(lewm_idm_forward, cfg=cfg),
        optim=optimizers,
    )
    data_module = spt.data.DataModule(train=train, val=val)

    # ------------------------------------------------------------------
    # Logging / checkpoints.
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

    object_dump_callback = IDMInferenceObjectCallBack(
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
