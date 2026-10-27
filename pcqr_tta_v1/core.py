"""Gradient-based adaptation using observed transitions, never goal labels.

Axes: B = observed starting states, Q = candidate plans, H = action blocks,
A = packed action dimension, D = latent dimension. Query centering is over Q,
NOT over horizons or environments. This v1 deliberately supports one live env.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
import copy
import math
import time
from typing import Any

import torch
from torch import Tensor, nn

MODES = ("frozen", "obs", "l2", "absolute", "response")


@dataclass(frozen=True)
class AdaptConfig:
    mode: str = "response"
    horizon: int = 5
    history_size: int = 3
    packed_action_dim: int = 10
    lr: float = 1e-5
    steps: int = 3
    strength: float = 1.0
    buffer_blocks: int = 64
    batch_starts: int = 16
    query_count: int = 32
    min_blocks: int = 1
    max_grad_norm: float = 1.0
    train_parts: tuple[str, ...] = ("action_encoder", "predictor", "pred_proj")
    # Uniform horizon weights; changing these is an experimental change.
    horizon_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        for k in ("horizon", "history_size", "packed_action_dim", "steps",
                  "buffer_blocks", "batch_starts", "query_count", "min_blocks"):
            value = getattr(self, k)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{k} must be a positive integer")
        if self.buffer_blocks < self.horizon or self.min_blocks > self.buffer_blocks:
            raise ValueError("Require buffer_blocks >= horizon and min_blocks <= capacity")
        for k in ("lr", "max_grad_norm"):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(f"{k} must be finite and positive")
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        if self.mode in ("absolute", "response") and self.query_count < 2:
            raise ValueError("query_count must be >= 2 for query regularization")
        allowed = {"action_encoder", "predictor", "pred_proj"}
        if not self.train_parts or set(self.train_parts) - allowed:
            raise ValueError(f"train_parts must be a nonempty subset of {allowed}")
        if len(set(self.train_parts)) != len(self.train_parts):
            raise ValueError("Duplicate train_parts")
        if self.horizon_weights is not None:
            if len(self.horizon_weights) != self.horizon:
                raise ValueError("horizon_weights must have one entry per horizon")
            if any(not math.isfinite(w) or w <= 0 for w in self.horizon_weights):
                raise ValueError("All horizon weights must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normal_clone(x: Tensor, *, device=None) -> Tensor:
    """A normal, detached tensor even when the caller uses inference_mode()."""
    with torch.inference_mode(False), torch.no_grad():
        return x.detach().to(device=device).clone()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def find_lewm(owner: nn.Module) -> tuple[str, nn.Module]:
    """Resolve actual JEPA within AutoCostModel; reject ambiguous wrappers."""
    found = []
    for name, mod in owner.named_modules():
        if (all(isinstance(getattr(mod, k, None), nn.Module)
                for k in ("encoder", "projector", "action_encoder", "predictor", "pred_proj"))
                and callable(getattr(mod, "predict", None))
                and callable(getattr(mod, "encode", None))
                and callable(getattr(mod, "rollout", None))):
            found.append((name, mod))
    if len(found) != 1:
        raise TypeError(f"Expected exactly one JEPA core, found {[x[0] for x in found]}. "
                        "Pass the intended student, not a student/teacher training container.")
    return found[0]


def rollout_latent(model: nn.Module, z0: Tensor, actions: Tensor,
                   history_size: int = 3) -> Tensor:
    """Differentiable single-frame rollout, matching JEPA.rollout history 1->2->3.

    z0: [B,1,D]; actions: [B,H,A]; returns future latents [B,H,D].
    No detaches in the autoregressive chain; image encoding happens elsewhere.
    """
    if z0.ndim != 3 or z0.shape[1] != 1 or actions.ndim != 3:
        raise ValueError("Require z0 [B,1,D] and actions [B,H,A]")
    if z0.shape[0] != actions.shape[0] or actions.shape[1] < 1 or history_size < 1:
        raise ValueError("Incompatible batch/horizon/history_size")
    if z0.device != actions.device or z0.dtype != actions.dtype:
        raise ValueError("z0/actions must have identical device and dtype")
    emb = z0
    future = []
    for h in range(actions.shape[1]):
        # Match upstream: action_encoder sees the complete available prefix.
        act_emb = model.action_encoder(actions[:, :h + 1])
        nxt = model.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
        if nxt.shape != z0.shape:
            raise RuntimeError(f"predict returned {tuple(nxt.shape)}, expected {tuple(z0.shape)}")
        future.append(nxt)
        emb = torch.cat((emb, nxt), dim=1)
    return torch.cat(future, dim=1)


def response_loss(pred: Tensor, teacher: Tensor, weights: Tensor) -> Tensor:
    """Centered-candidate MSE: [Q,H,D], population normalization (no Q-1)."""
    if pred.shape != teacher.shape or pred.ndim != 3 or pred.shape[0] < 2:
        raise ValueError("Require matching [Q,H,D], Q >= 2")
    delta = (pred - pred.mean(dim=0, keepdim=True)) - (
        teacher - teacher.mean(dim=0, keepdim=True))
    return (delta.square().mean(dim=(0, 2)) * weights).sum() / weights.sum()


def absolute_loss(pred: Tensor, teacher: Tensor, weights: Tensor) -> Tensor:
    return ((pred - teacher).square().mean(dim=(0, 2)) * weights).sum() / weights.sum()


class TransitionBuffer:
    """One episode of fully observed, contiguous, packed-action transitions.

    begin() supplies the initial observation. append() may only be called AFTER
    the complete action block executed and its endpoint was observed. No reset
    flags, goals, simulator state or future dataset frames enter this buffer.
    """
    def __init__(self, capacity: int, action_dim: int):
        if capacity < 1 or action_dim < 1:
            raise ValueError("capacity/action_dim must be positive")
        self.capacity, self.action_dim = capacity, action_dim
        self.states: deque[Tensor] = deque(maxlen=capacity + 1)
        self.actions: deque[Tensor] = deque(maxlen=capacity)
        self.total_completed = 0

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.total_completed = 0

    def _state(self, z: Tensor) -> Tensor:
        if z.ndim != 1 or not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("Observed latent must be a finite floating vector [D]")
        return normal_clone(z, device="cpu")

    def begin(self, z0: Tensor) -> None:
        if self.states:
            raise RuntimeError("Call clear/reset before starting another episode")
        self.states.append(self._state(z0))

    def append(self, executed_action: Tensor, observed_next: Tensor) -> None:
        if not self.states:
            raise RuntimeError("begin(z0) must precede append()")
        if executed_action.shape != (self.action_dim,) or not torch.isfinite(executed_action).all():
            raise ValueError(f"Expected finite packed action [{self.action_dim}]")
        next_z = self._state(observed_next)
        if next_z.shape != self.states[-1].shape or next_z.dtype != self.states[-1].dtype:
            raise ValueError("Latent shape/dtype changed during an episode")
        a = normal_clone(executed_action, device="cpu").to(dtype=next_z.dtype)
        self.actions.append(a)
        self.states.append(next_z)
        self.total_completed += 1
        if len(self.states) != len(self.actions) + 1:
            raise RuntimeError("Transition buffer lost temporal alignment")

    def __len__(self) -> int:
        return len(self.actions)

    def batch(self, horizon: int, max_starts: int, device: torch.device) -> dict[str, Tensor]:
        if not self.actions:
            raise RuntimeError("No completed action blocks")
        # Deterministic, evenly-spaced starts over the retained episode suffix.
        # Does not consume CEM's (or global) RNG; always includes newest start.
        n = len(self.actions)
        count = min(n, max_starts)
        ids = (torch.linspace(0, n - 1, count).round().long().tolist()
               if count > 1 else [n - 1])
        states, actions = list(self.states), list(self.actions)
        d = states[0].numel()
        z0 = torch.stack([states[i] for i in ids])[:, None]
        act = torch.zeros(count, horizon, self.action_dim, dtype=z0.dtype)
        target = torch.zeros(count, horizon, d, dtype=z0.dtype)
        valid = torch.zeros(count, horizon, dtype=torch.bool)
        for row, start in enumerate(ids):
            length = min(horizon, n - start)
            act[row, :length] = torch.stack(actions[start:start + length])
            target[row, :length] = torch.stack(states[start + 1:start + length + 1])
            valid[row, :length] = True
        return {"z0": z0.to(device), "actions": act.to(device),
                "targets": target.to(device), "valid": valid.to(device)}


class OnlineAdapter:
    """Mutate selected predictor modules in-place, with exact episode reset.

    Teacher is the pre-update model for THIS planning event: its query outputs
    are cached before any optimizer step. It is not an EMA and not a second
    encoder. Fixed query tensors are reused across all inner optimization steps.
    """
    def __init__(self, model: nn.Module, config: AdaptConfig):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Construct adapter/model outside inference_mode()")
        self.model, self.cfg = model, config
        self.device = next(model.parameters()).device
        self.model.eval()
        self.model.requires_grad_(False)
        frozen_ids = {id(p) for name in ("encoder", "projector")
                      for p in getattr(model, name).parameters()}
        selected_ids = set()
        for name in config.train_parts:
            mod = getattr(model, name, None)
            if not isinstance(mod, nn.Module):
                raise TypeError(f"Missing module {name}")
            for p in mod.parameters():
                if id(p) in frozen_ids:
                    raise ValueError("Trainable predictor shares a parameter with frozen encoder/projector")
                if p.dtype != torch.float32:
                    raise TypeError("v1 requires FP32 model weights; do not silently cast a checkpoint")
                p.requires_grad_(config.mode != "frozen")
                selected_ids.add(id(p))
        self.named = [(n, p) for n, p in model.named_parameters() if id(p) in selected_ids]
        if not self.named:
            raise ValueError("No parameters selected for adaptation")
        # CPU source snapshot includes buffers and all model parameters. Never written back to disk.
        self.source = {k: normal_clone(v, device="cpu") for k, v in model.state_dict().items()}
        self.buffer = TransitionBuffer(config.buffer_blocks, config.packed_action_dim)
        self.weights = torch.tensor(config.horizon_weights or (1.0,) * config.horizon,
                                    device=self.device, dtype=torch.float32)
        self.optimizer = self._optimizer()
        self.events = 0

    def _optimizer(self):
        if self.cfg.mode == "frozen":
            return None
        # Explicit zero decay; "obs" really has no hidden parameter regularizer.
        return torch.optim.Adam([p for _, p in self.named], lr=self.cfg.lr, weight_decay=0.0)

    def reset_episode(self) -> None:
        with torch.inference_mode(False), torch.no_grad():
            self.model.load_state_dict(self.source, strict=True)
        self.model.eval()
        self.buffer.clear()
        self.optimizer = self._optimizer()  # Adam moments must not cross episodes.
        self.events = 0
        for _, p in self.named:
            p.grad = None

    def assert_frozen_unchanged(self) -> None:
        chosen = {n for n, _ in self.named}
        current = self.model.state_dict()
        for name, reference in self.source.items():
            if name not in chosen and not torch.equal(current[name].detach().cpu(), reference):
                raise RuntimeError(f"Frozen parameter or buffer changed: {name}")

    def _obs_loss(self, batch: dict[str, Tensor]) -> Tensor:
        pred = rollout_latent(self.model, batch["z0"], batch["actions"], self.cfg.history_size)
        # Only completed horizons contribute. Padded targets are never labels.
        err = (pred - batch["targets"]).square().mean(dim=-1)
        weight = self.weights[None].expand_as(err)
        mask = batch["valid"]
        return (err[mask] * weight[mask]).sum() / weight[mask].sum()

    def adapt(self, current_z: Tensor, query_actions: Tensor) -> dict[str, Any]:
        """current_z [1,1,D], query_actions [Q,H,A] in normalized solver space.

        The API accepts no goal and no outcomes of query_actions. A returned
        nonfinite status means the ENTIRE update transaction was rolled back.
        """
        synchronize(self.device)
        start = time.perf_counter()
        result: dict[str, Any] = {"event": self.events, "mode": self.cfg.mode,
                                 "completed_blocks": self.buffer.total_completed,
                                 "buffer_blocks": len(self.buffer), "applied_steps": 0}
        self.events += 1
        if self.cfg.mode == "frozen" or len(self.buffer) < self.cfg.min_blocks:
            result.update(status="frozen" if self.cfg.mode == "frozen" else "warmup", adapt_ms=0.0)
            return result
        # CEM is commonly decorated with inference_mode(). Disabling no_grad
        # alone is insufficient. Clone all tensors in normal mode before use.
        with torch.inference_mode(False), torch.enable_grad():
            z = normal_clone(current_z, device=self.device).float()
            q = normal_clone(query_actions, device=self.device).float()
            if z.ndim != 3 or z.shape[:2] != (1, 1):
                raise ValueError("v1 requires one environment and one current frame")
            if q.ndim != 3 or tuple(q.shape[1:]) != (self.cfg.horizon, self.cfg.packed_action_dim):
                raise ValueError(f"Expected query_actions [Q,{self.cfg.horizon},{self.cfg.packed_action_dim}]")
            if not torch.isfinite(z).all() or not torch.isfinite(q).all():
                raise ValueError("Nonfinite query input")
            if self.cfg.mode in ("response", "absolute") and q.shape[0] < 2:
                raise ValueError("At least two candidates required")
            self.model.eval()  # eval is compatible with gradients; do not enable dropout/BN updates.
            batch = self.buffer.batch(self.cfg.horizon, self.cfg.batch_starts, self.device)
            previous = {n: normal_clone(p) for n, p in self.named}
            old_buffers = {n: normal_clone(b) for n, b in self.model.named_buffers()}
            old_optimizer = copy.deepcopy(self.optimizer.state_dict())
            query_z = z.expand(q.shape[0], -1, -1)
            with torch.no_grad():
                teacher = (rollout_latent(self.model, query_z, q, self.cfg.history_size).detach()
                           if self.cfg.mode in ("absolute", "response") else None)
            rows = []
            try:
                for step in range(self.cfg.steps):
                    self.optimizer.zero_grad(set_to_none=True)
                    obs = self._obs_loss(batch)
                    reg = obs.new_zeros(())
                    if self.cfg.mode == "l2":
                        reg = sum((p - previous[n]).square().sum() for n, p in self.named)
                        reg = reg / sum(p.numel() for _, p in self.named)
                    elif teacher is not None:
                        pred = rollout_latent(self.model, query_z, q, self.cfg.history_size)
                        reg = (response_loss(pred, teacher, self.weights) if self.cfg.mode == "response"
                               else absolute_loss(pred, teacher, self.weights))
                    loss = obs + self.cfg.strength * reg
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite_loss")
                    loss.backward()
                    grads = [p.grad for _, p in self.named if p.grad is not None]
                    if not grads:
                        raise RuntimeError("No gradients reached predictor parameters")
                    if not all(torch.isfinite(g).all() for g in grads):
                        raise FloatingPointError("nonfinite_gradient")
                    norm = torch.nn.utils.clip_grad_norm_([p for _, p in self.named], self.cfg.max_grad_norm)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite_gradient_norm")
                    self.optimizer.step()
                    if not all(torch.isfinite(p).all() for _, p in self.named):
                        raise FloatingPointError("nonfinite_parameters")
                    rows.append({"step": step, "obs": float(obs.detach()), "regularizer": float(reg.detach()),
                                 "total": float(loss.detach()), "grad_norm": float(norm)})
                with torch.no_grad():
                    final_obs = self._obs_loss(batch)
                    if not torch.isfinite(final_obs):
                        raise FloatingPointError("nonfinite_post_update_prediction")
                    after_reg = 0.0
                    if teacher is not None:
                        final_pred = rollout_latent(self.model, query_z, q, self.cfg.history_size)
                        r = (response_loss(final_pred, teacher, self.weights) if self.cfg.mode == "response"
                             else absolute_loss(final_pred, teacher, self.weights))
                        if not torch.isfinite(r):
                            raise FloatingPointError("nonfinite_post_update_queries")
                        after_reg = float(r)
                if any(not torch.equal(b, old_buffers[n]) for n, b in self.model.named_buffers()):
                    raise RuntimeError("Model mutated buffers even in eval mode")
                result.update(status="updated", applied_steps=len(rows), obs_after=float(final_obs),
                              regularizer_after=after_reg, steps=rows,
                              query_count=int(q.shape[0]) if teacher is not None else 0,
                              valid_targets=int(batch["valid"].sum()))
            except Exception as exc:
                # Failed updates must not partially contaminate subsequent control.
                with torch.no_grad():
                    for n, p in self.named:
                        p.copy_(previous[n])
                    for n, b in self.model.named_buffers():
                        b.copy_(old_buffers[n])
                self.optimizer.load_state_dict(old_optimizer)
                result.update(status="rolled_back", reason=str(exc), steps=rows)
                if not isinstance(exc, FloatingPointError):
                    raise
            finally:
                self.optimizer.zero_grad(set_to_none=True)
        synchronize(self.device)
        result["adapt_ms"] = 1000.0 * (time.perf_counter() - start)
        return result
