"""Non-invasive bridge to the repository's existing SWM policy and CEM.

No solver implementation is copied/replaced. On the first get_cost call of a
replan, cache a subset of that initial population, adapt, then score the ENTIRE
population with the updated model. All later iterations use the same weights.
"""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from .core import OnlineAdapter, normal_clone, rollout_latent, synchronize


def initial_fingerprint(info: dict[str, Any]) -> dict[str, str]:
    result = {}
    for key in ("pixels", "goal", "state", "goal_state"):
        if key not in info:
            continue
        value = info[key]
        value = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if value.dtype.kind in "OUS":
            continue
        data = np.ascontiguousarray(value)
        h = hashlib.sha256()
        h.update(str((data.dtype.str, data.shape)).encode())
        h.update(data.tobytes())
        result[key] = h.hexdigest()
    if "pixels" not in result or "goal" not in result:
        raise ValueError("Require initial pixels and goal for paired-evaluation audit")
    return result


class JSONLLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents silently appending incompatible runs.
        self._handle = self.path.open("x", encoding="utf-8")

    def __call__(self, row: dict[str, Any]) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class AdaptiveCostModel(nn.Module):
    def __init__(self, base_model: nn.Module, adapter: OnlineAdapter,
                 log: Callable[[dict[str, Any]], None] | None = None,
                 verify_rollout: bool = True):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.log = log
        self.armed = False
        self.checked = not verify_rollout
        self.cost_calls = 0
        self.replans = 0
        self.adapt_ms = 0.0
        self.applied_steps = 0

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def _query(self, info: dict[str, Any], candidates: Tensor):
        cfg, model = self.adapter.cfg, self.adapter.model
        if candidates.ndim != 4 or candidates.shape[0] != 1:
            raise ValueError("Online adaptation requires CEM candidates [1,Q,H,A]; use num_envs=1")
        if tuple(candidates.shape[2:]) != (cfg.horizon, cfg.packed_action_dim):
            raise ValueError("CEM horizon/action packing differs from adaptation configuration")
        pixels = info.get("pixels")
        if not torch.is_tensor(pixels) or pixels.ndim != 6 or pixels.shape[0] != 1 or pixels.shape[2] != 1:
            raise ValueError("Expected solver pixels [1,Q,1,C,H,W]; v1 supports single-frame planning only")
        n = candidates.shape[1]
        count = min(n, cfg.query_count)
        if cfg.mode in ("response", "absolute") and count < 2:
            raise ValueError("At least two initial CEM candidates are required")
        # Deterministic subset: no goal ranking, no elite selection, no new CEM draws.
        ids = torch.linspace(0, n - 1, count, device=candidates.device).round().long()
        with torch.inference_mode(False), torch.no_grad():
            q = normal_clone(candidates[0, ids], device=self.adapter.device).float()
            p = normal_clone(pixels[:, 0], device=self.adapter.device).float()
            z = normal_clone(model.encode({"pixels": p})["emb"])
            if self.adapter.buffer.states:
                torch.testing.assert_close(
                    z[0, 0].cpu(), self.adapter.buffer.states[-1], rtol=1e-4, atol=2e-5,
                    msg="Replan is not aligned with the last observed action-block boundary",
                )
            if not self.checked:
                # Actual installed/checkpoint implementation, not a guessed rollout.
                k = min(count, 3)
                expected = model.rollout(
                    {"pixels": p[:, None].expand(1, k, *p.shape[1:]).clone()},
                    q[None, :k].clone(), history_size=cfg.history_size,
                )["predicted_emb"][0, :, 1:]
                actual = rollout_latent(model, z.expand(k, -1, -1), q[:k], cfg.history_size)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5,
                                           msg="Latent adapter does not match installed JEPA.rollout")
                self.checked = True
        return z, q

    def get_cost(self, info_dict: dict[str, Any], action_candidates: Tensor) -> Tensor:
        self.cost_calls += 1
        if self.armed:
            self.armed = False  # No more weight changes during this CEM solve.
            self.replans += 1
            if self.adapter.cfg.mode != "frozen":
                synchronize(self.adapter.device)
                start = time.perf_counter()
                z, queries = self._query(info_dict, action_candidates)
                row = self.adapter.adapt(z, queries)
                synchronize(self.adapter.device)
                row["type"] = "adapt"
                row["replan"] = self.replans - 1
                row["hook_ms"] = 1000.0 * (time.perf_counter() - start)
                # Includes query encoding/parity check; adapt_ms is optimization only.
                self.adapt_ms += row["hook_ms"]
                self.applied_steps += row["applied_steps"]
                if self.log is not None:
                    self.log(row)
        # Delegate unchanged goal metric to the user's actual cost model.
        with torch.no_grad():
            return self.base_model.get_cost(info_dict, action_candidates)


def make_online_policy_class(base_class):
    """Subclass the INSTALLED policy, retaining its CEM, warm start and transforms.

    Instantiate one policy for one episode. Reuse across episodes is rejected;
    the supplied runner constructs a fresh policy after adapter.reset_episode().
    """
    class OnlineWorldModelPolicy(base_class):
        def __init__(self, *args, online_cost: AdaptiveCostModel, action_block: int = 5,
                     log: Callable[[dict[str, Any]], None] | None = None,
                     expected_initial: dict[str, str] | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.online_cost = online_cost
            self.online_log = log
            self.online_action_block = int(action_block)
            self.online_raw_steps = 0
            self.online_partial_actions: list[Tensor] = []
            self.online_last_action: np.ndarray | None = None
            self.online_wall_ms = 0.0
            self.expected_initial = expected_initial
            self.initial_fingerprint = None
            if self.online_action_block < 1:
                raise ValueError("action_block must be positive")
            if not callable(getattr(self, "_prepare_info", None)):
                raise TypeError("Installed WorldModelPolicy lacks _prepare_info; no unverified fallback is applied")
            if "action" not in self.process:
                raise ValueError("Require the same fitted action StandardScaler as the original evaluation")

        def set_env(self, env):
            if getattr(env, "num_envs", None) != 1:
                raise ValueError("Use one environment per adapting model; parallelize via separate processes")
            if self.online_last_action is not None:
                raise RuntimeError("Create a new OnlineWorldModelPolicy for every episode")
            return super().set_env(env)

        def _encode_latest(self, info: dict[str, Any]) -> Tensor:
            # Only pixels are provided: goal/state/proprio cannot enter adaptation.
            with torch.inference_mode(False), torch.no_grad():
                raw = info["pixels"]
                if torch.is_tensor(raw):
                    raw = normal_clone(raw)
                elif isinstance(raw, np.ndarray):
                    raw = raw.copy()
                processed = self._prepare_info({"pixels": raw})["pixels"]
                if not torch.is_tensor(processed) or processed.ndim != 5 or processed.shape[:2] != (1, 1):
                    raise ValueError("Require processed pixels [1,1,C,H,W]; world.history_size must be 1")
                pixels = normal_clone(processed, device=self.online_cost.adapter.device).float()
                encoded = self.online_cost.adapter.model.encode({"pixels": pixels})["emb"]
                if encoded.shape[:2] != (1, 1):
                    raise ValueError("Expected single current latent [1,1,D]")
                return normal_clone(encoded[0, 0])

        def _receive_previous_action(self, info: dict[str, Any]) -> Tensor:
            # World's action field must refer to the command actually applied
            # on the previous env step. Never use unexecuted plan suffixes.
            if "action" not in info:
                raise KeyError("World observation lacks previous executed 'action'; refusing guessed labels")
            x = info["action"]
            x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
            if x.ndim == 3:
                x = x[:, -1]
            if x.ndim != 2 or x.shape[0] != 1 or not np.isfinite(x).all():
                raise ValueError(f"Expected finite previous action [1,A_raw] or [1,T,A_raw], got {x.shape}")
            if x.shape != self.online_last_action.shape or not np.allclose(
                    x, self.online_last_action, rtol=2e-4, atol=2e-4):
                raise RuntimeError("World 'action' is not the last returned/executed command. "
                                   "Check action timing or wrappers before enabling adaptation.")
            normalized = self.process["action"].transform(x)
            return torch.tensor(np.array(normalized[0], copy=True), dtype=torch.float32)

        def get_action(self, info_dict: dict[str, Any], **kwargs):
            adapter = self.online_cost.adapter
            if getattr(self.env, "num_envs", None) != 1:
                raise ValueError("Online policy is single-environment only")
            synchronize(adapter.device)
            start = time.perf_counter()
            # A single case is evaluated per policy instance. Never infer episode
            # continuity merely because adjacent images look similar.
            flush = info_dict.get("_needs_flush")
            if self.online_last_action is not None and flush is not None and bool(np.asarray(flush).any()):
                raise RuntimeError("Mid-run environment reset: construct a fresh policy and reset adapter")
            if self.online_last_action is None:
                self.initial_fingerprint = initial_fingerprint(info_dict)
                if self.expected_initial is not None and self.initial_fingerprint != self.expected_initial:
                    raise RuntimeError("Paired case initial state/goal/pixels mismatch; stopping before action execution")
                if self.online_log is not None:
                    self.online_log({"type": "initial", "fingerprint": self.initial_fingerprint})
            else:
                self.online_raw_steps += 1
            if adapter.cfg.mode != "frozen":
                if self.online_last_action is None:
                    adapter.buffer.begin(self._encode_latest(info_dict))
                else:
                    self.online_partial_actions.append(self._receive_previous_action(info_dict))
                    if len(self.online_partial_actions) == self.online_action_block:
                        packed = torch.cat(self.online_partial_actions)
                        endpoint = self._encode_latest(info_dict)
                        adapter.buffer.append(packed, endpoint)
                        self.online_partial_actions.clear()
            # If base policy serves buffered actions, get_cost is never called.
            # If it replans, only its first initial-population scoring triggers adaptation.
            self.online_cost.arm()
            try:
                action = super().get_action(dict(info_dict), **kwargs)
            finally:
                self.online_cost.disarm()
            value = np.asarray(action)
            if value.ndim != 2 or value.shape[0] != 1:
                raise ValueError(f"Expected base policy action [1,A_raw], got {value.shape}")
            self.online_last_action = value.copy()
            synchronize(adapter.device)
            elapsed = 1000.0 * (time.perf_counter() - start)
            self.online_wall_ms += elapsed
            if self.online_log is not None:
                self.online_log({"type": "policy_call", "completed_raw_steps": self.online_raw_steps,
                                 "wall_ms": elapsed, "completed_blocks": adapter.buffer.total_completed})
            return action

    return OnlineWorldModelPolicy
