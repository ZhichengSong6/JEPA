"""Recording/replay primitives. No planner, training, or task-success changes."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if hasattr(x, "detach"):
        return plain(x.detach().cpu().tolist())
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    raise TypeError(type(x).__name__)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def array_hash(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def physics(raw):
    # _get_obs omits block velocity and angular velocity. Check these too.
    vals = list(np.asarray(raw._get_obs(), dtype=float))
    for name in ("agent", "block"):
        body = getattr(raw, name)
        for key in ("position", "velocity", "force"):
            vals.extend(getattr(body, key))
        vals.extend([body.angle, body.angular_velocity, body.torque])
    return np.asarray(vals, dtype=np.float64)


def case_manifest(a, b, seed, controls=2):
    """Select ALL historical discordants, plus seeded controls per common group."""
    if a["selection"] != b["selection"]:
        raise ValueError("Source start/goal selections differ")
    x, y = a["successes"], b["successes"]
    if len(x) != len(y) or controls < 0:
        raise ValueError("Invalid manifest inputs")
    groups = {"rescued": [], "regressed": [], "both_fail": [], "both_success": []}
    for i, (u, v) in enumerate(zip(x, y)):
        g = "both_success" if u and v else "both_fail" if not u and not v else "rescued" if v else "regressed"
        groups[g].append(i)
    rng = random.Random(20260913 + int(seed))
    chosen = set(groups["rescued"] + groups["regressed"])
    for g in ("both_fail", "both_success"):
        chosen.update(rng.sample(groups[g], min(controls, len(groups[g]))))
    rows = []
    for i in sorted(chosen):
        rows.append({"seed": seed, "eval_index": i,
                     "episode_idx": a["selection"]["episodes_idx"][i],
                     "start_step": a["selection"]["start_steps"][i],
                     "historical_group": next(g for g, ids in groups.items() if i in ids)})
    return rows


class NativeRecorder:
    """Tap native reset/setters/step; replay the *actual reset recipe and prefix*.

    Installed before World.evaluate_from_dataset. Missing reset seeds are made
    explicit for this NEW diagnostic cohort, not retroactively called historical
    exact replay. Paired runs use the same dataset and reset seed. All variations
    and initial physics/images are checked separately by the driver.
    """
    def __init__(self, raw, seed):
        self.raw, self.seed = raw, int(seed)
        self.inside_reset = False
        self.reset_count = 0
        self.recipe = None
        self.steps = []
        self.segments = []
        self.originals = {}
        for name in ("reset", "_set_state", "_set_goal_state", "step"):
            self.originals[name] = (name in raw.__dict__, raw.__dict__.get(name), getattr(raw, name))
        raw.reset = self.reset
        raw._set_state = lambda *a, **kw: self.setter("_set_state", a, kw)
        raw._set_goal_state = lambda *a, **kw: self.setter("_set_goal_state", a, kw)
        raw.step = self.step

    def restore(self):
        for name, (was_local, local, _) in self.originals.items():
            if was_local:
                setattr(self.raw, name, local)
            else:
                delattr(self.raw, name)

    def reset(self, *args, **kwargs):
        if args:
            raise RuntimeError("Expected keyword-only native reset; cannot safely pin seed")
        kwargs = copy.deepcopy(kwargs)
        if kwargs.get("seed") is None:
            kwargs["seed"] = self.seed
        self.reset_count += 1
        self.recipe = {"kwargs": copy.deepcopy(kwargs), "setters": []}
        self.steps = []
        self.segments.append({"recipe": self.recipe, "steps": self.steps})
        self.inside_reset = True
        try:
            return self.originals["reset"][2](**kwargs)
        finally:
            self.inside_reset = False

    def setter(self, name, args, kwargs):
        if not self.inside_reset:
            if self.recipe is None:
                raise RuntimeError("Setter before recorded reset")
            self.recipe["setters"].append((name, copy.deepcopy(args), copy.deepcopy(kwargs)))
        return self.originals[name][2](*args, **kwargs)

    def step(self, action):
        applied = np.asarray(action).copy()
        result = self.originals["step"][2](action)
        obs, _, term, trunc, _ = result
        self.steps.append({"action": applied, "state": np.asarray(obs["state"]).copy(),
                           "physics": physics(self.raw), "term": bool(term), "trunc": bool(trunc)})
        return result

    def snapshot(self):
        if self.recipe is None:
            raise RuntimeError("No native reset captured")
        return {"recipe": copy.deepcopy(self.recipe), "prefix": copy.deepcopy(self.steps),
                "physics": physics(self.raw), "image": np.asarray(self.raw.render()).copy(),
                "goal_state": np.asarray(self.raw.goal_state).copy(),
                "reset_count": self.reset_count}


def restore_prefix(raw, snap, atol=1e-6, check_image=True):
    """Reconstruct hidden dynamics by replay, never by resetting a solve pose."""
    raw.reset(**copy.deepcopy(snap["recipe"]["kwargs"]))
    for name, args, kwargs in snap["recipe"]["setters"]:
        getattr(raw, name)(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    for j, step in enumerate(snap["prefix"]):
        _, _, term, trunc, _ = raw.step(step["action"].copy())
        if bool(term) != step["term"] or bool(trunc) != step["trunc"]:
            raise RuntimeError(f"Replay prefix termination mismatch at step {j}")
        if not np.allclose(physics(raw), step["physics"], rtol=0, atol=atol):
            raise RuntimeError(f"Replay prefix physics mismatch at step {j}")
    if not np.allclose(physics(raw), snap["physics"], rtol=0, atol=atol):
        raise RuntimeError("Replay solve-start physics mismatch")
    if check_image and not np.array_equal(np.asarray(raw.render()), snap["image"]):
        raise RuntimeError("Replay solve-start render mismatch")


def fixed_rollout(raw, actions):
    """Native fixed-H diagnostic continuation, NOT stop-on-success evaluation."""
    first, terminal = None, False
    for j, action in enumerate(actions, 1):
        _, _, terminal, trunc, _ = raw.step(action)
        if trunc:
            raise RuntimeError("Native truncation before fixed endpoint; cannot score aligned oracle")
        if terminal and first is None:
            first = j
    if not len(actions):
        raise ValueError("Empty replay actions")
    return {"executed_steps": len(actions), "first_success_step": first,
            "ever_success": first is not None, "endpoint_success": bool(terminal),
            "state": np.asarray(raw._get_obs()).copy(),
            "official_endpoint_distance": float(raw.eval_state(raw.goal_state, raw._get_obs())[1]),
            "image": np.asarray(raw.render()).copy()}


def repeat_comparison(a, b, atol=1e-6):
    """Stricter than equal success: compare recorded native trajectories too."""
    if len(a["successes"]) != len(b["successes"]):
        raise ValueError("Repeat length mismatch")
    flips, trajectory_changes = [], []
    for i, (x, y) in enumerate(zip(a["successes"], b["successes"])):
        if x != y:
            flips.append(i)
        u, v = a["native_steps"][i], b["native_steps"][i]
        same = len(u) == len(v)
        if same:
            for p, q in zip(u, v):
                if (p["term"] != q["term"] or p["trunc"] != q["trunc"]
                        or not np.allclose(p["action"], q["action"], rtol=0, atol=atol)
                        or not np.allclose(p["physics"], q["physics"], rtol=0, atol=atol)):
                    same = False
                    break
        if not same:
            trajectory_changes.append(i)
    return {"success_flips": flips, "trajectory_changes": trajectory_changes,
            "passed": not flips and not trajectory_changes}


def score_metrics(score, oracle, success, k=10):
    """Fixed-population selection metrics. Tie handling follows torch.topk."""
    import torch
    score, oracle = np.asarray(score), np.asarray(oracle)
    success = np.asarray(success, dtype=bool)
    if score.ndim != 1 or score.shape != oracle.shape or score.shape != success.shape:
        raise ValueError("Mismatched score arrays")
    if not np.isfinite(score).all() or not np.isfinite(oracle).all() or not 1 <= k < len(score):
        raise ValueError("Non-finite scores or invalid elite count")
    inds = torch.topk(torch.as_tensor(score), k=k, largest=False).indices.numpy()
    oinds = torch.topk(torch.as_tensor(oracle), k=k, largest=False).indices.numpy()
    selected = int(np.argmin(score))
    ns = int(success.sum())
    return {"selected_index": selected, "selected_ever_success": bool(success[selected]),
            "success_available": ns > 0, "success_count": ns,
            "elite_success_count": int(success[inds].sum()),
            "success_retention": float(success[inds].sum() / ns) if ns else None,
            "oracle_elite_overlap": len(set(inds) & set(oinds)) / k,
            "encoder_selection_regret": float(oracle[selected] - oracle.min()),
            "elite_boundary_gap": float(np.sort(score)[k] - np.sort(score)[k-1])}, inds
