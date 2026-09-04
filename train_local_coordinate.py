#!/usr/bin/env python3
"""Train nonlinear local coordinates with goal-aligned controllable geometry.

This is the first method step after the bounded global-vs-local capacity test.

Frozen:
    official LeWM epoch-10 encoder/projector/action-encoder/predictor/pred-proj

Trainable:
    an exactly invertible nonlinear coordinate map Phi(z)

No privileged simulator state is loaded.

Training signal
---------------
For each offline trajectory window:

  * use the frozen LeWM to roll the demonstrated action sequence H steps;
  * apply bounded symmetric action probes around the first coarse action block;
  * retain only probes whose finite-difference terminal responses agree at two
    radii (the same contact/nonlinearity reliability gate used in Stage-I);
  * choose a later *observed* future embedding as a reachable goal;
  * learn Phi so the Euclidean terminal-to-goal residual in y=Phi(z) lies as
    much as possible in the locally action-responsive terminal subspace.

The key objective is therefore NOT controllability whitening and does NOT
penalize a large response condition number.  It only asks the Euclidean goal
direction to emphasize locally controllable directions:

    L_align = 1 - ||Proj_span(R_y) (y_goal - y_terminal)||^2
                    / ||y_goal - y_terminal||^2.

The future observation supplies a reachable goal example; there is no temporal
value label, no physical state, and no deployed metric/reward head.

At export, JEPA.predict() conjugates the frozen predictor through Phi^{-1}/Phi,
so the underlying pretrained dynamics remain unchanged while official CEM
continues to use raw Euclidean distance in the new coordinates.
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

from local_coordinate_geometry import LocalCoordinateAdapter
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


class LocalCoordinateModule(pl.LightningModule):
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
        self.teacher.eval()
        self.adapter.train(mode)
        return self

    @torch.no_grad()
    def _encode(self, batch):
        self.teacher.eval()
        return self.teacher.encode({"pixels": batch["pixels"]})["emb"].detach()

    @torch.no_grad()
    def _probe_terminal_geometry(self, emb, normalized_actions):
        cfg = self.cfg.local_coord
        hs = int(self.cfg.wm.history_size)
        horizon = int(cfg.rollout_horizon)
        k = int(cfg.num_directions)

        raw = _denormalize_actions(
            normalized_actions.detach(), self.action_mean, self.action_std
        )

        # Frozen center terminal prediction under the demonstrated H-step
        # action sequence.
        center = _autoregressive_rollout(
            self.teacher,
            emb,
            normalized_actions.detach(),
            history_size=hs,
            horizon=horizon,
        )[:, -1].detach()

        outer_plus = []
        outer_minus = []
        inner_plus = []
        inner_minus = []
        base_response = []
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

            center_raw = 0.5 * (plus_raw + minus_raw)
            delta = 0.5 * (plus_raw - minus_raw)

            inner_scale = float(cfg.inner_radius_scale)
            ip_raw = center_raw.clone()
            im_raw = center_raw.clone()
            ip_raw[:, active_start:active_end] = (
                center_raw[:, active_start:active_end]
                + inner_scale * delta[:, active_start:active_end]
            )
            im_raw[:, active_start:active_end] = (
                center_raw[:, active_start:active_end]
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
                        ip_raw, self.action_mean, self.action_std
                    ),
                    _normalize_raw_actions(
                        im_raw, self.action_mean, self.action_std
                    ),
                ],
                dim=1,
            )

            b = emb.shape[0]
            flat_actions = candidates.reshape(
                b * 4, *candidates.shape[2:]
            )
            flat_emb = _repeat_candidates(emb, 4)
            endpoints = _autoregressive_rollout(
                self.teacher,
                flat_emb,
                flat_actions,
                history_size=hs,
                horizon=horizon,
            )[:, -1].reshape(b, 4, -1)

            op, om, ip, im = [endpoints[:, i].detach() for i in range(4)]

            denom_outer = (2.0 * radius).clamp_min(1e-8)[:, None]
            denom_inner = (
                2.0 * inner_scale * radius
            ).clamp_min(1e-8)[:, None]
            r_outer = (op - om) / denom_outer
            r_inner = (ip - im) / denom_inner

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

            outer_plus.append(op)
            outer_minus.append(om)
            inner_plus.append(ip)
            inner_minus.append(im)
            base_response.append(0.5 * (r_outer + r_inner))
            reliability.append(reliable)
            cosines.append(cos)
            gains.append(gain)
            outer_norms.append(n_outer)
            radii.append(radius)

        return {
            "center": center,
            "outer_plus": torch.stack(outer_plus, dim=1),
            "outer_minus": torch.stack(outer_minus, dim=1),
            "inner_plus": torch.stack(inner_plus, dim=1),
            "inner_minus": torch.stack(inner_minus, dim=1),
            "base_response": torch.stack(base_response, dim=1),
            "reliable": torch.stack(reliability, dim=1),
            "cosine": torch.stack(cosines, dim=1),
            "gain": torch.stack(gains, dim=1),
            "response_norm": torch.stack(outer_norms, dim=1),
            "radius": torch.stack(radii, dim=1),
        }

    def _alignment_loss(self, probe, goal):
        cfg = self.cfg.local_coord
        center = probe["center"].float()
        goal = goal.float()

        # Transform actual points, not response vectors.  For nonlinear Phi,
        # the finite-difference response in y must be recomputed after Phi.
        yc = self.adapter(center)
        yg = self.adapter(goal)

        op = self.adapter(probe["outer_plus"].float())
        om = self.adapter(probe["outer_minus"].float())
        ip = self.adapter(probe["inner_plus"].float())
        im = self.adapter(probe["inner_minus"].float())

        radius = probe["radius"].float()
        inner_scale = float(cfg.inner_radius_scale)
        ry_outer = (op - om) / (
            (2.0 * radius).clamp_min(1e-8)[..., None]
        )
        ry_inner = (ip - im) / (
            (2.0 * inner_scale * radius).clamp_min(1e-8)[..., None]
        )
        ry_all = 0.5 * (ry_outer + ry_inner)

        r0_all = probe["base_response"].float()
        reliable = probe["reliable"]

        gy_all = yg - yc
        g0_all = goal - center

        losses = []
        base_alignments = []
        adapted_alignments = []
        active_ranks = []
        used = 0

        for b in range(center.shape[0]):
            mask = reliable[b]
            if int(mask.sum()) < int(cfg.min_active_rank):
                continue

            r0 = r0_all[b, mask]
            ry = ry_all[b, mask]
            g0 = g0_all[b]
            gy = gy_all[b]

            if (
                float(torch.linalg.vector_norm(g0).detach())
                < float(cfg.min_goal_residual_norm)
            ):
                continue

            # Rank is defined by the frozen teacher response, not by Phi, so
            # the adapter cannot improve the objective by numerically deleting
            # action-responsive directions.
            _, s0, vh0 = torch.linalg.svd(
                r0, full_matrices=False
            )
            if float(s0[0].detach()) <= float(cfg.response_svd_abs_floor):
                continue
            active = (
                (s0 >= float(cfg.response_svd_rel_floor) * s0[0])
                & (s0 >= float(cfg.response_svd_abs_floor))
            )
            rank = int(active.sum())
            if rank < int(cfg.min_active_rank):
                continue

            # Same algebraic rank is used in transformed coordinates.
            _, _, vhy = torch.linalg.svd(
                ry, full_matrices=False
            )
            q0 = vh0[:rank]
            qy = vhy[:rank]

            base_energy = (
                (q0 @ g0).square().sum()
                / g0.square().sum().clamp_min(float(cfg.energy_eps))
            ).clamp(0.0, 1.0)
            adapted_energy = (
                (qy @ gy).square().sum()
                / gy.square().sum().clamp_min(float(cfg.energy_eps))
            ).clamp(0.0, 1.0)

            losses.append(1.0 - adapted_energy)
            base_alignments.append(base_energy.detach())
            adapted_alignments.append(adapted_energy.detach())
            active_ranks.append(float(rank))
            used += 1

        if losses:
            align_loss = torch.stack(losses).mean()
            base_alignment = torch.stack(base_alignments).mean()
            adapted_alignment = torch.stack(adapted_alignments).mean()
            active_rank = torch.tensor(
                active_ranks, device=center.device
            ).mean()
        else:
            # Preserve an adapter-connected graph if a batch has no usable
            # local geometry.
            align_loss = (
                yc.sum() + yg.sum() + ry_all.sum()
            ) * 0.0
            nan = torch.tensor(float("nan"), device=center.device)
            base_alignment = nan
            adapted_alignment = nan
            active_rank = torch.tensor(0.0, device=center.device)

        return {
            "alignment_loss": align_loss,
            "base_alignment": base_alignment,
            "adapted_alignment": adapted_alignment,
            "active_rank": active_rank,
            "used_fraction": torch.tensor(
                used / max(center.shape[0], 1),
                device=center.device,
                dtype=torch.float32,
            ),
            "center_y": yc,
            "goal_y": yg,
            "response_y": ry_all,
        }

    def _shared_step(self, batch, stage):
        self.teacher.eval()
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        with torch.no_grad():
            emb = self._encode(batch)
            probe = self._probe_terminal_geometry(
                emb, batch["action"]
            )
            hs = int(self.cfg.wm.history_size)
            h = int(self.cfg.local_coord.rollout_horizon)
            go = int(self.cfg.local_coord.goal_offset)
            goal_index = hs + h + go - 1
            goal = emb[:, goal_index].detach()

        geom = self._alignment_loss(probe, goal)

        # Near-identity regularization is intentionally weak.  It stabilizes
        # the coordinate chart without specifying which action modes deserve
        # equal scale.
        identity_points = torch.cat(
            [probe["center"].float(), goal.float()], dim=0
        )
        id_loss = self.adapter.identity_regularizer(identity_points)

        total = (
            float(self.cfg.local_coord.alignment_weight)
            * geom["alignment_loss"]
            + float(self.cfg.local_coord.identity_weight) * id_loss
        )

        rel = probe["reliable"].float()
        disp = geom["center_y"] - probe["center"].float()
        gy = geom["goal_y"] - geom["center_y"]
        g0 = goal.float() - probe["center"].float()

        metrics = {
            f"{stage}/loss": total,
            f"{stage}/alignment_loss": geom["alignment_loss"].detach(),
            f"{stage}/identity_loss": id_loss.detach(),
            f"{stage}/base_goal_alignment": geom["base_alignment"],
            f"{stage}/adapted_goal_alignment": geom["adapted_alignment"],
            f"{stage}/goal_alignment_gain": (
                geom["adapted_alignment"] - geom["base_alignment"]
            ),
            f"{stage}/active_rank": geom["active_rank"],
            f"{stage}/used_state_fraction": geom["used_fraction"],
            f"{stage}/reliable_probe_fraction": rel.mean(),
            f"{stage}/probe_cosine": probe["cosine"].mean(),
            f"{stage}/probe_gain": probe["gain"].mean(),
            f"{stage}/probe_response_norm": probe["response_norm"].mean(),
            f"{stage}/effective_radius": probe["radius"].mean(),
            f"{stage}/coordinate_displacement_rms": (
                disp.square().mean().sqrt().detach()
            ),
            f"{stage}/goal_residual_norm_raw": (
                torch.linalg.vector_norm(g0, dim=-1).mean().detach()
            ),
            f"{stage}/goal_residual_norm_coord": (
                torch.linalg.vector_norm(gy, dim=-1).mean().detach()
            ),
            f"{stage}/response_norm_coord": (
                torch.linalg.vector_norm(
                    geom["response_y"], dim=-1
                ).mean().detach()
            ),
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
    config_name="lewm_local_coordinate",
)
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)

    if "state" in cfg.data.dataset.keys_to_load:
        raise ValueError(
            "Local-coordinate training must not load privileged simulator state."
        )
    if not bool(cfg.local_coord.enabled):
        raise ValueError("local_coord.enabled must be true.")

    expected = (
        int(cfg.wm.history_size)
        + int(cfg.local_coord.rollout_horizon)
        + int(cfg.local_coord.goal_offset)
    )
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

    teacher = swm.policy.AutoCostModel(str(cfg.local_coord.init_policy))
    if not isinstance(teacher, torch.nn.Module):
        raise TypeError("AutoCostModel did not return nn.Module.")
    if getattr(teacher, "coordinate_adapter", None) is not None:
        raise ValueError(
            "Local-coordinate training must start from unadapted official LeWM."
        )
    teacher.eval()
    teacher.requires_grad_(False)

    adapter = LocalCoordinateAdapter(
        int(cfg.wm.embed_dim),
        hidden_dim=int(cfg.local_coord.hidden_dim),
        num_blocks=int(cfg.local_coord.num_blocks),
        max_shift=float(cfg.local_coord.max_shift),
    )

    module = LocalCoordinateModule(
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

    teacher = module.teacher.cpu().eval()
    adapter = module.adapter.cpu().eval()

    with torch.no_grad():
        x = torch.randn(64, int(cfg.wm.embed_dim))
        xr = adapter.inverse(adapter(x))
        max_roundtrip_error = float((xr - x).abs().max())
    if max_roundtrip_error > float(cfg.local_coord.max_roundtrip_error):
        raise RuntimeError(
            f"Adapter inverse sanity check failed: {max_roundtrip_error:.3e}"
        )

    teacher.coordinate_adapter = adapter
    teacher.eval()

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
        "adapter": adapter.diagnostics(x),
        "metrics": callback_metrics,
        "frozen_policy": str(cfg.local_coord.init_policy),
        "rollout_horizon": int(cfg.local_coord.rollout_horizon),
        "goal_offset": int(cfg.local_coord.goal_offset),
        "perturb_first_only": bool(
            cfg.local_coord.perturb_first_only
        ),
        "num_directions": int(cfg.local_coord.num_directions),
        "objective": "goal-aligned local controllable subspace; no whitening",
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("=== LOCAL COORDINATE TRAINING COMPLETE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
