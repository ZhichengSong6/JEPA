#!/usr/bin/env python3
"""Stage-I geometry hypothesis test: fit a latent coordinate system only.

This is deliberately NOT the older bias-calibration Stage-I.

Frozen:
    official LeWM epoch-10 encoder/projector/action-encoder/predictor/pred-proj

Trainable:
    one global SPD coordinate adapter A

Training signal:
    ALD-style symmetric action probes around demonstrated actions, rolled to
    H=5 by the frozen LeWM.  A second half-radius probe checks local-response
    consistency; unreliable/contact-switching probes are excluded.

Objective:
    only penalize severely ill-conditioned *reliable active* finite-horizon
    response modes, plus a near-identity regularizer on A.

At export the adapter is inserted into JEPA itself.  The frozen predictor is
conjugated as P_y(y,a)=A P_z(A^{-1}y,a), so its underlying dynamics are exactly
unchanged while official CEM continues to use raw Euclidean distance in y.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from coordinate_geometry import SPDCoordinateAdapter
from stage1_bias_calibration import (
    _autoregressive_rollout,
    _denormalize_actions,
    _normalize_raw_actions,
    _sample_raw_action_perturbation,
)
from utils import get_column_normalizer, get_img_preprocessor


def _raw_action_stats(dataset):
    data = torch.from_numpy(np.asarray(dataset.get_col_data("action")))
    data = data[~torch.isnan(data).any(dim=1)]
    return data.mean(0, keepdim=True).float(), data.std(0, keepdim=True).float()


def _repeat_candidates(x: torch.Tensor, count: int) -> torch.Tensor:
    b = x.shape[0]
    return x[:, None].expand(b, count, *x.shape[1:]).reshape(
        b * count, *x.shape[1:]
    )


class CoordinateStage1Module(pl.LightningModule):
    def __init__(self, teacher, adapter, cfg, action_mean, action_std):
        super().__init__()
        self.teacher = teacher
        self.adapter = adapter
        self.cfg = cfg
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float())

        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Lightning recursively calls train(); force the frozen LeWM back to
        # inference semantics so dropout/BN never move.
        self.teacher.eval()
        self.adapter.train(mode)
        return self

    @torch.no_grad()
    def _encode(self, batch):
        self.teacher.eval()
        visual = {"pixels": batch["pixels"]}
        return self.teacher.encode(visual)["emb"].detach()

    @torch.no_grad()
    def _probe_responses(self, emb, normalized_actions):
        cfg = self.cfg.coord
        hs = int(self.cfg.wm.history_size)
        horizon = int(cfg.rollout_horizon)
        k = int(cfg.num_directions)

        raw = _denormalize_actions(
            normalized_actions.detach(), self.action_mean, self.action_std
        )
        responses = []
        action_dirs = []
        reliability = []
        cosines = []
        gains = []
        outer_norms = []
        radii = []

        active_start = hs - 1
        active_count = 1 if bool(cfg.perturb_first_only) else horizon
        active_end = active_start + active_count

        for _ in range(k):
            plus_raw, minus_raw, radius = _sample_raw_action_perturbation(
                raw,
                history_size=hs,
                horizon=horizon,
                radius=float(cfg.perturb_radius),
                first_only=bool(cfg.perturb_first_only),
            )

            center = 0.5 * (plus_raw + minus_raw)
            delta = 0.5 * (plus_raw - minus_raw)

            inner_scale = float(cfg.inner_radius_scale)
            inner_plus_raw = center.clone()
            inner_minus_raw = center.clone()
            inner_plus_raw[:, active_start:active_end] = (
                center[:, active_start:active_end]
                + inner_scale * delta[:, active_start:active_end]
            )
            inner_minus_raw[:, active_start:active_end] = (
                center[:, active_start:active_end]
                - inner_scale * delta[:, active_start:active_end]
            )

            candidates = torch.stack(
                [
                    _normalize_raw_actions(
                        plus_raw, self.action_mean, self.action_std
                    ),
                    _normalize_raw_actions(
                        minus_raw, self.action_mean, self.action_std
                    ),
                    _normalize_raw_actions(
                        inner_plus_raw, self.action_mean, self.action_std
                    ),
                    _normalize_raw_actions(
                        inner_minus_raw, self.action_mean, self.action_std
                    ),
                ],
                dim=1,
            )

            b = emb.shape[0]
            flat_actions = candidates.reshape(
                b * 4, *candidates.shape[2:]
            )
            flat_emb = _repeat_candidates(emb, 4)
            endpoint = _autoregressive_rollout(
                self.teacher,
                flat_emb,
                flat_actions,
                history_size=hs,
                horizon=horizon,
            )[:, -1].reshape(b, 4, -1)

            denom_outer = (2.0 * radius).clamp_min(1e-8)[:, None]
            denom_inner = (
                2.0 * inner_scale * radius
            ).clamp_min(1e-8)[:, None]
            r_outer = (endpoint[:, 0] - endpoint[:, 1]) / denom_outer
            r_inner = (endpoint[:, 2] - endpoint[:, 3]) / denom_inner

            # Multi-radius agreement is a reliability test, not a target.
            cos = F.cosine_similarity(
                r_outer, r_inner, dim=-1, eps=1e-8
            )
            n_outer = torch.linalg.vector_norm(r_outer, dim=-1)
            n_inner = torch.linalg.vector_norm(r_inner, dim=-1)
            gain = n_inner / n_outer.clamp_min(1e-8)

            reliable = (
                (radius >= float(cfg.min_effective_radius))
                & (n_outer >= float(cfg.min_response_norm))
                & (n_inner >= float(cfg.min_response_norm))
                & (cos >= float(cfg.reliability_cosine_min))
                & (gain >= float(cfg.reliability_gain_min))
                & (gain <= float(cfg.reliability_gain_max))
            )

            # Average the two consistent finite differences to reduce noise.
            response = 0.5 * (r_outer + r_inner)

            delta_active = delta[:, active_start:active_end].flatten(1)
            action_dir = (
                delta_active / radius.clamp_min(1e-8)[:, None]
            )

            responses.append(response)
            action_dirs.append(action_dir)
            reliability.append(reliable)
            cosines.append(cos)
            gains.append(gain)
            outer_norms.append(n_outer)
            radii.append(radius)

        return {
            "response": torch.stack(responses, dim=1).detach(),
            "action_dir": torch.stack(action_dirs, dim=1).detach(),
            "reliable": torch.stack(reliability, dim=1).detach(),
            "cosine": torch.stack(cosines, dim=1).detach(),
            "gain": torch.stack(gains, dim=1).detach(),
            "response_norm": torch.stack(outer_norms, dim=1).detach(),
            "radius": torch.stack(radii, dim=1).detach(),
        }

    def _condition_loss(self, probe):
        cfg = self.cfg.coord
        response = probe["response"].float()
        action_dir = probe["action_dir"].float()
        reliable = probe["reliable"]

        # One matrix exponential per batch, not per state.
        transformed = self.adapter(response)

        losses = []
        base_conditions = []
        adapted_conditions = []
        active_ranks = []
        used = 0

        for b in range(response.shape[0]):
            mask = reliable[b]
            if int(mask.sum()) < 2:
                continue

            r0 = response[b, mask]
            ry = transformed[b, mask]
            u = action_dir[b, mask]
            k = r0.shape[0]

            # Remove accidental non-orthogonality of the random input probes.
            # Generalized response spectrum:
            #   eig( (U U^T)^(-1/2) (R R^T) (U U^T)^(-1/2) )
            gu = u @ u.transpose(0, 1)
            gu = 0.5 * (gu + gu.transpose(0, 1))
            gu = gu + float(cfg.gram_eps) * torch.eye(
                k, device=gu.device, dtype=gu.dtype
            )
            eu, qu = torch.linalg.eigh(gu)
            if float(eu.min().detach()) <= float(cfg.input_eig_floor):
                continue
            invsqrt = (
                qu
                @ torch.diag(eu.clamp_min(float(cfg.gram_eps)).rsqrt())
                @ qu.transpose(0, 1)
            )

            h0 = invsqrt @ (r0 @ r0.transpose(0, 1)) @ invsqrt
            hy = invsqrt @ (ry @ ry.transpose(0, 1)) @ invsqrt
            h0 = 0.5 * (h0 + h0.transpose(0, 1))
            hy = 0.5 * (hy + hy.transpose(0, 1))

            e0 = torch.linalg.eigvalsh(h0).clamp_min(0.0)
            ey = torch.linalg.eigvalsh(hy).clamp_min(float(cfg.gram_eps))

            max0 = e0[-1]
            if float(max0.detach()) <= float(cfg.response_eig_abs_floor):
                continue

            # Only modes that are non-numerical in the frozen teacher define
            # the rank.  The relative floor is intentionally tiny: we want to
            # retain severely compressed but stable modes, while excluding
            # exact/null directions that no coordinate transform can recover.
            active = (
                (e0 >= float(cfg.response_eig_rel_floor) * max0)
                & (e0 >= float(cfg.response_eig_abs_floor))
            )
            rank = int(active.sum())
            if rank < 2:
                continue

            vals0 = e0[-rank:].clamp_min(float(cfg.gram_eps))
            valsy = ey[-rank:].clamp_min(float(cfg.gram_eps))

            base_kappa = torch.sqrt(vals0[-1] / vals0[0])
            adapted_kappa = torch.sqrt(valsy[-1] / valsy[0])

            # Conservative calibration: do NOT force isotropy.  Penalize only
            # condition numbers worse than kappa_max.
            excess = F.relu(
                torch.log(adapted_kappa)
                - torch.log(
                    torch.tensor(
                        float(cfg.kappa_max),
                        device=adapted_kappa.device,
                    )
                )
            )
            losses.append(excess.square())
            base_conditions.append(base_kappa.detach())
            adapted_conditions.append(adapted_kappa.detach())
            active_ranks.append(float(rank))
            used += 1

        if losses:
            cond_loss = torch.stack(losses).mean()
            base_condition = torch.stack(base_conditions).mean()
            adapted_condition = torch.stack(adapted_conditions).mean()
            active_rank = torch.tensor(
                active_ranks, device=response.device
            ).mean()
        else:
            # Preserve graph to adapter if the batch has no reliable rank>=2
            # local geometry.
            cond_loss = transformed.sum() * 0.0
            nan = torch.tensor(float("nan"), device=response.device)
            base_condition = nan
            adapted_condition = nan
            active_rank = torch.tensor(0.0, device=response.device)

        return {
            "condition_loss": cond_loss,
            "base_condition": base_condition,
            "adapted_condition": adapted_condition,
            "active_rank": active_rank,
            "used_fraction": torch.tensor(
                used / max(response.shape[0], 1),
                device=response.device,
                dtype=torch.float32,
            ),
        }

    def _shared_step(self, batch, stage):
        self.teacher.eval()
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        with torch.no_grad():
            emb = self._encode(batch)
            probe = self._probe_responses(emb, batch["action"])

        geom = self._condition_loss(probe)
        id_loss = self.adapter.identity_regularizer()
        total = (
            float(self.cfg.coord.condition_weight)
            * geom["condition_loss"]
            + float(self.cfg.coord.identity_weight) * id_loss
        )

        rel = probe["reliable"].float()
        metrics = {
            f"{stage}/loss": total,
            f"{stage}/condition_loss": geom["condition_loss"].detach(),
            f"{stage}/identity_loss": id_loss.detach(),
            f"{stage}/base_condition": geom["base_condition"],
            f"{stage}/adapted_condition": geom["adapted_condition"],
            f"{stage}/active_rank": geom["active_rank"],
            f"{stage}/used_state_fraction": geom["used_fraction"],
            f"{stage}/reliable_probe_fraction": rel.mean(),
            f"{stage}/probe_cosine": probe["cosine"].mean(),
            f"{stage}/probe_gain": probe["gain"].mean(),
            f"{stage}/probe_response_norm": probe["response_norm"].mean(),
            f"{stage}/effective_radius": probe["radius"].mean(),
        }
        self.log_dict(
            metrics,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.adapter.parameters(),
            lr=float(self.cfg.optimizer.lr),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="lewm_coordinate_stage1",
)
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError(
            "Coordinate Stage-I is training-side only and must not load "
            "privileged simulator state."
        )
    if not bool(cfg.coord.enabled):
        raise ValueError("coord.enabled must be true.")

    expected = int(cfg.wm.history_size) + int(cfg.coord.rollout_horizon)
    if int(cfg.data.dataset.num_steps) != expected:
        raise ValueError(
            f"Need {expected} coarse states, got {cfg.data.dataset.num_steps}."
        )

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    action_mean, action_std = _raw_action_stats(dataset)

    transforms = [
        get_img_preprocessor(
            source="pixels", target="pixels", img_size=int(cfg.img_size)
        )
    ]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    gen = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[float(cfg.train_split), 1.0 - float(cfg.train_split)],
        generator=gen,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        **cfg.val_loader,
        shuffle=False,
        drop_last=False,
    )

    teacher = swm.policy.AutoCostModel(str(cfg.coord.init_policy))
    if not isinstance(teacher, torch.nn.Module):
        raise TypeError("AutoCostModel did not return nn.Module.")
    if getattr(teacher, "coordinate_adapter", None) is not None:
        raise ValueError(
            "Stage-I must start from the unadapted official LeWM latent frame."
        )
    teacher.eval()
    teacher.requires_grad_(False)

    adapter = SPDCoordinateAdapter(int(cfg.wm.embed_dim))

    module = CoordinateStage1Module(
        teacher=teacher,
        adapter=adapter,
        cfg=cfg,
        action_mean=action_mean,
        action_std=action_std,
    )

    run_dir = Path(swm.data.utils.get_cache_dir(), str(cfg.subdir))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w") as f:
        OmegaConf.save(cfg, f)

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_loader, val_loader)

    # Export an AutoCostModel-compatible JEPA object.  Only the coordinate
    # system changes; the pretrained dynamics remain frozen and are conjugated
    # exactly by jepa.JEPA.predict().
    teacher = module.teacher.cpu().eval()
    adapter = module.adapter.cpu().eval()
    adapter.refresh_cache()
    teacher.coordinate_adapter = adapter
    teacher.eval()

    # Exact algebraic sanity check for the exported coordinate transform.
    with torch.no_grad():
        x = torch.randn(32, int(cfg.wm.embed_dim))
        roundtrip = adapter.inverse(adapter(x))
        max_roundtrip_error = float((roundtrip - x).abs().max())
    if max_roundtrip_error > float(cfg.coord.max_roundtrip_error):
        raise RuntimeError(
            f"Adapter inverse sanity check failed: {max_roundtrip_error:.3e}"
        )

    out = run_dir / f"{cfg.output_model_name}_object.ckpt"
    torch.save(teacher, out)

    callback_metrics = {}
    for key, value in trainer.callback_metrics.items():
        if torch.is_tensor(value) and value.numel() == 1:
            callback_metrics[str(key)] = float(value.detach().cpu())

    summary = {
        "status": "complete",
        "global_step": int(trainer.global_step),
        "output": str(out),
        "max_roundtrip_error": max_roundtrip_error,
        "adapter": adapter.diagnostics(),
        "metrics": callback_metrics,
        "frozen_policy": str(cfg.coord.init_policy),
        "rollout_horizon": int(cfg.coord.rollout_horizon),
        "perturb_first_only": bool(cfg.coord.perturb_first_only),
        "num_directions": int(cfg.coord.num_directions),
        "kappa_max": float(cfg.coord.kappa_max),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("=== COORDINATE STAGE-I COMPLETE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
