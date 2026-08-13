import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg, PushTFactorHeads
from utils import (
    get_column_normalizer,
    get_img_preprocessor,
    get_pusht_factor_transform,
    get_pusht_factor_metric_scales,
    ModelObjectCallBack,
)


def _equal_factor_average(pusher_loss, block_loss, theta_loss):
    """Equal 1/3 weighting across the three 2-D physical factors."""
    return (pusher_loss + block_loss + theta_loss) / 3.0


def _factor_absolute_losses(factor_heads, latent, target, theta_norm_weight):
    """Absolute factor supervision for any [..., D] latent tensor."""
    flat_latent = latent.reshape(-1, latent.shape[-1]).float()
    flat_target = target.reshape(-1, target.shape[-1]).float()
    pred = factor_heads(flat_latent)

    pusher_loss = torch.nn.functional.mse_loss(
        pred['pusher_xy'], flat_target[:, 0:2]
    )
    block_loss = torch.nn.functional.mse_loss(
        pred['block_xy'], flat_target[:, 2:4]
    )

    theta_target = flat_target[:, 4:6]
    theta_direction_loss = (
        1.0 - (pred['theta_unit'] * theta_target).sum(dim=-1)
    ).mean()
    theta_norm_loss = (
        torch.linalg.vector_norm(pred['theta_raw'], dim=-1) - 1.0
    ).square().mean()
    theta_loss = theta_direction_loss + float(theta_norm_weight) * theta_norm_loss

    return {
        'loss': _equal_factor_average(pusher_loss, block_loss, theta_loss),
        'pusher_loss': pusher_loss,
        'block_loss': block_loss,
        'theta_loss': theta_loss,
        'theta_direction_loss': theta_direction_loss,
        'theta_norm_loss': theta_norm_loss,
    }


def _factor_distances(outputs_a, outputs_b, scales):
    return {
        'pusher': torch.linalg.vector_norm(
            outputs_a['pusher_xy'] - outputs_b['pusher_xy'], dim=-1
        ) / float(scales.pusher),
        'block': torch.linalg.vector_norm(
            outputs_a['block_xy'] - outputs_b['block_xy'], dim=-1
        ) / float(scales.block),
        'theta': torch.linalg.vector_norm(
            outputs_a['theta_unit'] - outputs_b['theta_unit'], dim=-1
        ) / float(scales.theta),
    }


def _target_factor_distances(factor_a, factor_b, scales):
    return {
        'pusher': torch.linalg.vector_norm(
            factor_a[:, 0:2] - factor_b[:, 0:2], dim=-1
        ) / float(scales.pusher),
        'block': torch.linalg.vector_norm(
            factor_a[:, 2:4] - factor_b[:, 2:4], dim=-1
        ) / float(scales.block),
        'theta': torch.linalg.vector_norm(
            factor_a[:, 4:6] - factor_b[:, 4:6], dim=-1
        ) / float(scales.theta),
    }


def _factor_metric_loss(
    factor_heads,
    first_latent,
    second_latent,
    first_factor,
    second_factor,
    scales,
    beta,
):
    first_latent = first_latent.reshape(-1, first_latent.shape[-1]).float()
    second_latent = second_latent.reshape(-1, second_latent.shape[-1]).float()
    first_factor = first_factor.reshape(-1, first_factor.shape[-1]).float()
    second_factor = second_factor.reshape(-1, second_factor.shape[-1]).float()

    pred_first = factor_heads(first_latent)
    pred_second = factor_heads(second_latent)
    pred_dist = _factor_distances(pred_first, pred_second, scales)
    tgt_dist = _target_factor_distances(first_factor, second_factor, scales)

    pusher_loss = torch.nn.functional.smooth_l1_loss(
        pred_dist['pusher'], tgt_dist['pusher'], beta=float(beta)
    )
    block_loss = torch.nn.functional.smooth_l1_loss(
        pred_dist['block'], tgt_dist['block'], beta=float(beta)
    )
    theta_loss = torch.nn.functional.smooth_l1_loss(
        pred_dist['theta'], tgt_dist['theta'], beta=float(beta)
    )
    return _equal_factor_average(pusher_loss, block_loss, theta_loss)


def _cross_pair_indices(count, device, deterministic=False):
    if count < 2:
        raise ValueError('Factor metric loss needs at least two states.')
    idx = torch.arange(count, device=device)
    if deterministic:
        return torch.roll(idx, shifts=1)
    perm = torch.randperm(count, device=device)
    # Avoid self-pairs without a rejection loop.
    if torch.any(perm == idx):
        perm = torch.roll(perm, shifts=1)
    return perm


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, and compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # prediction

    # Original LeWM objective. Keep this path unchanged when factor.enabled=False.
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    if cfg.factor.enabled:
        if self.model.factor_heads is None:
            raise RuntimeError('factor.enabled=True but model.factor_heads is None.')
        if 'factor' not in batch:
            raise KeyError(
                "factor.enabled=True but batch has no 'factor' target. "
                'The raw-state factor transform must be installed before state normalization.'
            )

        factor = batch['factor'].float()  # (B, T, 6), physical target
        if factor.shape[-1] != 6:
            raise ValueError(f'Expected factor[..., 6], got {factor.shape}.')

        # Encode loss: every observed frame must expose all three physical factors.
        enc_factor = _factor_absolute_losses(
            self.model.factor_heads,
            emb,
            factor,
            cfg.factor.theta_norm_weight,
        )

        # Predictor target alignment matches tgt_emb = emb[:, n_preds:].
        pred_factor_target = factor[:, n_preds:]
        pred_factor = _factor_absolute_losses(
            self.model.factor_heads,
            pred_emb,
            pred_factor_target,
            cfg.factor.theta_norm_weight,
        )

        # Metric loss 1: random pairs among encoded states.
        enc_latent_flat = emb.reshape(-1, emb.shape[-1])
        enc_factor_flat = factor.reshape(-1, factor.shape[-1])
        enc_perm = _cross_pair_indices(
            enc_latent_flat.shape[0],
            enc_latent_flat.device,
            deterministic=(stage != 'train'),
        )
        encoded_metric_loss = _factor_metric_loss(
            self.model.factor_heads,
            enc_latent_flat,
            enc_latent_flat[enc_perm],
            enc_factor_flat,
            enc_factor_flat[enc_perm],
            cfg.factor.metric_scales,
            cfg.factor.metric_smooth_l1_beta,
        )

        # Metric loss 2: predicted states against cross-paired encoded targets.
        # This constrains geometry of the predictor output, not only its absolute readout.
        pred_latent_flat = pred_emb.reshape(-1, pred_emb.shape[-1])
        tgt_latent_flat = tgt_emb.reshape(-1, tgt_emb.shape[-1])
        pred_target_flat = pred_factor_target.reshape(-1, pred_factor_target.shape[-1])
        pred_perm = _cross_pair_indices(
            pred_latent_flat.shape[0],
            pred_latent_flat.device,
            deterministic=(stage != 'train'),
        )
        predicted_metric_loss = _factor_metric_loss(
            self.model.factor_heads,
            pred_latent_flat,
            tgt_latent_flat[pred_perm],
            pred_target_flat,
            pred_target_flat[pred_perm],
            cfg.factor.metric_scales,
            cfg.factor.metric_smooth_l1_beta,
        )
        metric_loss = 0.5 * (encoded_metric_loss + predicted_metric_loss)

        output['encoded_factor_loss'] = enc_factor['loss']
        output['encoded_pusher_loss'] = enc_factor['pusher_loss']
        output['encoded_block_loss'] = enc_factor['block_loss']
        output['encoded_theta_loss'] = enc_factor['theta_loss']
        output['predicted_factor_loss'] = pred_factor['loss']
        output['predicted_pusher_loss'] = pred_factor['pusher_loss']
        output['predicted_block_loss'] = pred_factor['block_loss']
        output['predicted_theta_loss'] = pred_factor['theta_loss']
        output['metric_loss'] = metric_loss
        output['encoded_metric_loss'] = encoded_metric_loss
        output['predicted_metric_loss'] = predicted_metric_loss

        output['loss'] = (
            output['loss']
            + float(cfg.factor.encoded_weight) * output['encoded_factor_loss']
            + float(cfg.factor.predicted_weight) * output['predicted_factor_loss']
            + float(cfg.factor.metric_weight) * output['metric_loss']
        )

    losses_dict = {
        f'{stage}/{k}': v.detach()
        for k, v in output.items()
        if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    # Derive the physical factor from RAW state before the official state
    # z-score normalizer is appended below. This preserves theta in radians
    # for the sin/cos target. No new dataset collection is required.
    if cfg.factor.enabled:
        if 'state' not in cfg.data.dataset.keys_to_load:
            raise ValueError('Full PushT factor training requires state in keys_to_load.')
        transforms.append(
            get_pusht_factor_transform(
                source='state',
                target='factor',
                world_size=cfg.factor.world_size,
            )
        )
        metric_scales = get_pusht_factor_metric_scales(
            dataset,
            world_size=cfg.factor.world_size,
            quantile=cfg.factor.metric_scale_quantile,
            num_pairs=cfg.factor.metric_scale_num_pairs,
            seed=cfg.seed,
        )
        with open_dict(cfg):
            cfg.factor.metric_scales.pusher = metric_scales['pusher']
            cfg.factor.metric_scales.block = metric_scales['block']
            cfg.factor.metric_scales.theta = metric_scales['theta']
        print(f'PushT factor metric scales (p{100*cfg.factor.metric_scale_quantile:.0f}): {metric_scales}')

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

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

    factor_heads = None
    if cfg.factor.enabled:
        factor_heads = PushTFactorHeads(
            latent_dim=embed_dim,
            hidden_dim=cfg.factor.head_hidden_dim,
            depth=cfg.factor.head_depth,
        )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        factor_heads=factor_heads,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
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
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
