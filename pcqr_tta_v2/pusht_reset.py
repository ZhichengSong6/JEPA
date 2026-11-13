"""PushT reset variants and hidden physical-state audit.

The legacy mode leaves the original simulator unchanged. The experimental
fresh_dynamics mode places the dataset state without the upstream implicit
space.step(), while clearing unobserved block dynamics. This is a NEW
evaluation protocol; it is not equivalent to historical PushT evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
from typing import Any, Iterator

import numpy as np

RESET_MODES = ("legacy", "fresh_dynamics")


def _floats(value: Any, n: int, label: str) -> list[float]:
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    if x.size != n or not np.isfinite(x).all():
        raise ValueError(f"{label}: expected {n} finite numbers, got {x!r}")
    return x.tolist()


def resolve_pusht(world: Any):
    """Only support one legacy SWM PushT env. Never touch other envs."""
    try:
        envs = world.envs.unwrapped.envs
    except AttributeError as exc:
        raise RuntimeError("Expected legacy World.envs.unwrapped.envs") from exc
    if len(envs) != 1:
        raise RuntimeError("PushT reset audit requires exactly one live environment")
    env = envs[0].unwrapped
    if (type(env).__name__ != "PushT"
            or not type(env).__module__.startswith("stable_worldmodel.envs.pusht")):
        raise RuntimeError(f"Not the expected SWM PushT instance: {type(env)!r}")
    for k in ("_set_state", "_get_obs", "agent", "block", "space", "dt"):
        if not hasattr(env, k):
            raise RuntimeError(f"PushT environment missing {k!r}")
    return env


def physics_snapshot(env: Any) -> dict[str, Any]:
    """Read physical quantities without stepping or altering the simulator."""
    a, b, space = env.agent, env.block, env.space
    return {
        "agent_position": _floats(a.position, 2, "agent position"),
        "agent_velocity": _floats(a.velocity, 2, "agent velocity"),
        "block_position": _floats(b.position, 2, "block position"),
        "block_angle": float(b.angle),
        "block_velocity": _floats(b.velocity, 2, "block velocity"),
        "block_angular_velocity": float(b.angular_velocity),
        "block_force": _floats(b.force, 2, "block force"),
        "block_torque": float(b.torque),
        "space_damping": float(space.damping),
        "space_iterations": int(space.iterations),
        "dt": float(env.dt),
    }


def physics_hidden_sha256(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def fresh_set_state(env: Any, state: Any) -> None:
    """Set PushT's observed 7D state and zero hidden block dynamics.

    Unlike the upstream _set_state(), does NOT advance physics during
    placement. Called for goal/initial placement and dataset _set_state.
    """
    s = _floats(state, 7, "PushT 7D dataset state")
    env.agent.velocity = tuple(s[5:7])
    env.agent.position = tuple(s[0:2])
    env.block.angle = s[4]
    env.block.position = tuple(s[2:4])
    env.block.velocity = (0.0, 0.0)
    env.block.angular_velocity = 0.0
    env.block.force = (0.0, 0.0)
    env.block.torque = 0.0
    if hasattr(env.block, "activate"):
        env.block.activate()
    if hasattr(env.space, "reindex_shapes_for_body"):
        env.space.reindex_shapes_for_body(env.agent)
        env.space.reindex_shapes_for_body(env.block)
    observed = _floats(env._get_obs(), 7, "post-placement observed state")
    expected = list(s)
    expected[4] %= 2 * np.pi
    if not np.allclose(observed, expected, rtol=0, atol=1e-9):
        raise RuntimeError(
            "Fresh reset changed the observed dataset state; "
            "check the installed SWM/Pymunk interface before evaluating"
        )
    physical = physics_snapshot(env)
    if any(abs(x) > 1e-12 for x in (
        *physical["block_velocity"], physical["block_angular_velocity"],
    )):
        raise RuntimeError("Hidden block velocity failed to reset")


@dataclass
class ResetProbe:
    env: Any
    mode: str
    patches_applied: int = 0

    def snapshot(self) -> dict[str, Any]:
        return physics_snapshot(self.env)


@contextmanager
def scoped_pusht_reset(world: Any, mode: str = "legacy") -> Iterator[ResetProbe]:
    """Patch only the *instance* for one synchronous dataset evaluation.

    Always restore the original bound method, including on exceptions.
    This is not a package modification or a model adaptation update.
    """
    if mode not in RESET_MODES:
        raise ValueError(f"Unsupported PushT reset mode: {mode}")
    env = resolve_pusht(world)
    probe = ResetProbe(env=env, mode=mode)
    if mode == "legacy":
        yield probe
        return

    from types import MethodType
    original_entry = vars(env).get("_set_state")
    original_method = env._set_state
    if len(inspect.signature(original_method).parameters) != 1:
        raise RuntimeError("Unexpected signature of PushT._set_state")

    def patched(self, state):
        fresh_set_state(self, state)
        probe.patches_applied += 1

    env._set_state = MethodType(patched, env)
    try:
        yield probe
    finally:
        if original_entry is None:
            delattr(env, "_set_state")
        else:
            env._set_state = original_entry
        if hasattr(original_method, "__func__"):
            assert getattr(env._set_state, "__func__", None) is original_method.__func__
