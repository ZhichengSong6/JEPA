"""CPU-only regression tests: python -m unittest discover -s tests -p 'test_cem_mh_diagnostics.py'."""
import copy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cem_mh_diag_core import (NativeRecorder, case_manifest, fixed_rollout, physics,
                             repeat_comparison, restore_prefix, score_metrics)


class Body:
    def __init__(self):
        self.position, self.velocity, self.force = (0., 0.), (0., 0.), (0., 0.)
        self.angle = self.angular_velocity = self.torque = 0.


class Env:
    """Hidden velocity and post-success motion make pose-only/early-stop bugs visible."""
    def __init__(self):
        self.agent, self.block = Body(), Body()
        self.spec = types.SimpleNamespace(id="fake", kwargs={})
        self.goal_state = np.zeros(7)
        self.unwrapped = self

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self.agent, self.block = Body(), Body()
        self.block.velocity = (float(self.rng.uniform(.1, .3)), 0.)
        self._set_state(np.zeros(7))
        return {"state": self._get_obs()}, {}

    def _set_state(self, state):
        self.agent.position = tuple(state[:2])
        self.block.position = tuple(state[2:4])

    def _set_goal_state(self, state):
        self.goal_state = np.asarray(state).copy()

    def _get_obs(self):
        return np.array([*self.agent.position, *self.block.position, self.block.angle, *self.agent.velocity])

    def eval_state(self, goal, state):
        return bool(state[2] > .1), float(np.linalg.norm(state-goal))

    def render(self):
        return np.full((4, 4, 3), int(self.block.position[0]*10) % 255, dtype=np.uint8)

    def step(self, a):
        v = self.block.velocity[0] + float(a[0])*.1
        self.block.velocity = (v, 0.)
        self.block.position = (self.block.position[0]+v, 0.)
        return {"state": self._get_obs()}, 0., self.block.position[0] > .1, False, {}


class Tests(unittest.TestCase):
    def test_reset_recipe_and_hidden_prefix(self):
        env = Env()
        rec = NativeRecorder(env, 41000)
        try:
            env.reset()
            env._set_state(np.array([1., 2., 3., 4., 0., 0., 0.]))
            env._set_goal_state(np.arange(7))
            env.step(np.array([.2, .3]))
            snap = rec.snapshot()
            self.assertEqual(snap["recipe"]["kwargs"]["seed"], 41000)
            self.assertEqual(len(snap["recipe"]["setters"]), 2)  # internal reset setter excluded
            replay = Env()
            restore_prefix(replay, snap)
            np.testing.assert_array_equal(physics(replay), physics(env))
            snap["prefix"][0]["physics"][10] += 1
            with self.assertRaisesRegex(RuntimeError, "physics mismatch"):
                restore_prefix(replay, snap)
        finally:
            rec.restore()
        self.assertNotIn("step", env.__dict__)
        self.assertNotIn("reset", env.__dict__)

    def test_fixed_endpoint_not_first_success(self):
        env = Env()
        env.reset(seed=0)
        r = fixed_rollout(env, np.ones((25, 2)))
        self.assertEqual(r["executed_steps"], 25)
        self.assertEqual(r["first_success_step"], 1)
        self.assertGreater(r["state"][2], 25)

    def test_truncation_blocks_oracle(self):
        env = Env()
        env.reset(seed=0)
        env.step = lambda a: ({}, 0, False, True, {})
        with self.assertRaisesRegex(RuntimeError, "truncation"):
            fixed_rollout(env, np.zeros((25, 2)))

    def test_repeat_gate_detects_hidden_drift(self):
        env = Env()
        rec = NativeRecorder(env, 2)
        env.reset()
        env.step(np.ones(2))
        a = {"successes": [True], "native_steps": [copy.deepcopy(rec.steps)]}
        b = copy.deepcopy(a)
        self.assertTrue(repeat_comparison(a, b)["passed"])
        b["native_steps"][0][0]["physics"][10] += .01
        self.assertFalse(repeat_comparison(a, b)["passed"])
        rec.restore()

    def test_case_selection_no_cherry_pick(self):
        sel = {"episodes_idx": list(range(10)), "start_steps": [0]*10}
        a = {"selection": sel, "successes": [False, True]+[False]*4+[True]*4}
        b = {"selection": sel, "successes": [True, False]+[False]*4+[True]*4}
        first = case_manifest(a, b, 42, 2)
        self.assertEqual(first, case_manifest(a, b, 42, 2))
        self.assertEqual(len(first), 6)
        self.assertTrue({0, 1}.issubset({x["eval_index"] for x in first}))
        bad = copy.deepcopy(b)
        bad["selection"]["start_steps"][0] = 1
        with self.assertRaises(ValueError):
            case_manifest(a, bad, 42)

    def test_score_metrics(self):
        score = np.arange(100, dtype=np.float32)
        success = np.zeros(100, dtype=bool)
        success[[0, 10]] = True
        m, inds = score_metrics(score, score, success)
        self.assertEqual(m["elite_success_count"], 1)
        self.assertEqual(m["success_retention"], .5)
        self.assertEqual(m["oracle_elite_overlap"], 1)
        score[3] = np.nan
        with self.assertRaises(ValueError):
            score_metrics(score, score, success)

    def test_native_autoreset_keeps_old_segments(self):
        env = Env()
        rec = NativeRecorder(env, 2)
        env.reset()
        env.step(np.ones(2))
        snap = rec.snapshot()
        env.reset()
        self.assertEqual(len(rec.steps), 0)
        self.assertEqual(len(rec.segments[snap["reset_count"]-1]["steps"]), 1)
        rec.restore()

    def test_model_cross_scoring_uses_fresh_inputs(self):
        from eval_cem_mh_diagnostics import model_scores
        class Model:
            def get_cost(self, info, candidates):
                self.assertion = "mutated" not in info
                info["mutated"] = True
                return (candidates**2).sum((-2, -1))
        info = {"pixels": torch.zeros(1, 1, 3, 4, 4)}
        model = Model()
        c = np.ones((100, 5, 10), dtype=np.float32)
        a, _ = model_scores(model, info, c, "cpu")
        b, _ = model_scores(model, info, c, "cpu")
        self.assertTrue(model.assertion)
        self.assertNotIn("mutated", info)
        np.testing.assert_array_equal(a, b)


    def test_official_solver_pass_through_integration(self):
        # A 100-env / 10-iteration CPU fixture exercises the actual tap/ID map,
        # native hooks and cleanup. The fake solver intentionally has __call__
        # but no .solve attribute: no dependency on private CEM implementation.
        from omegaconf import OmegaConf
        from eval_cem_mh_diagnostics import run_closed
        class Model:
            def eval(self): return self
            def requires_grad_(self, x): return self
            def get_cost(self, info, c):
                info["mutated"] = True
                return c.square().sum((-1, -2))
        class Solver:
            def __init__(self, model):
                self.model = model
                self.g = torch.Generator().manual_seed(42)
            def __call__(self, info):
                means = []
                for i in range(100):
                    mu = torch.zeros(1, 5, 10)
                    sigma = torch.ones_like(mu)
                    for _ in range(10):
                        c = torch.randn(1, 100, 5, 10, generator=self.g)*sigma[:, None]+mu[:, None]
                        c[:, 0] = mu
                        expanded = {k: v[i:i+1, None].expand(1, 100, *v.shape[1:]) for k, v in info.items()}
                        costs = self.model.get_cost(expanded, c)
                        ii = torch.topk(costs[0], 10, largest=False).indices
                        mu = c[:, ii].mean(1)
                        sigma = c[:, ii].std(1)
                    means.append(mu[0])
                return {"actions": torch.stack(means)}
        class BasePolicy:
            def __init__(self, solver, **kwargs): self.solver = solver
            def set_env(self, env): self.env = env
            def get_action(self, info, **kwargs): return self.solver(info)
        class World:
            def __init__(self, **kwargs):
                self.envs = types.SimpleNamespace(envs=[Env() for _ in range(100)], close=lambda: None)
            def set_policy(self, p):
                self.p = p
                p.set_env(self.envs)
            def evaluate_from_dataset(self, dataset, **kwargs):
                for env in self.envs.envs:
                    env.reset()
                    env._set_state(np.zeros(7))
                    env._set_goal_state(np.ones(7))
                info = {"id": torch.arange(100)[:, None],
                        "pixels": torch.zeros(100, 1, 3, 4, 4),
                        "goal": torch.zeros(100, 1, 3, 4, 4)}
                out = self.p.get_action(info)
                for i, env in enumerate(self.envs.envs):
                    for a in out["actions"][i].reshape(25, 2).numpy(): env.step(a)
                return {"episode_successes": [True]*100, "success_rate": 100.}
        swm = types.ModuleType("stable_worldmodel")
        swm.World, swm.PlanConfig = World, lambda **kw: kw
        swm.policy = types.SimpleNamespace(WorldModelPolicy=BasePolicy)
        hydra = types.ModuleType("hydra")
        hydra.utils = types.SimpleNamespace(instantiate=lambda cfg, model: Solver(model))
        evaluation = types.ModuleType("eval")
        evaluation.img_transform = lambda cfg: None
        exact = types.ModuleType("pusht_exact_replay")
        exact.capture_live_reset_contexts = lambda env: [{"variation_count": 1} for _ in env.envs]
        cfg = OmegaConf.create({"solver": {}, "world": {}, "plan_config": {},
                               "eval": {"callables": []}, "diag_start": [0]*100,
                               "diag_episodes": list(range(100))})
        with patch.dict(sys.modules, {"stable_worldmodel": swm, "hydra": hydra,
                                     "eval": evaluation, "pusht_exact_replay": exact}):
            a, traces = run_closed(cfg, None, {}, Model(), 42, {0, 97}, {0, 5, 9}, True)
            b, empty = run_closed(cfg, None, {}, Model(), 42, {0, 97}, {0, 5, 9}, False)
        self.assertTrue(repeat_comparison(a, b)["passed"])
        self.assertEqual(len(traces), 2)
        self.assertEqual(empty, [])
        self.assertEqual([p["iteration"] for p in traces[0]["populations"]], [0, 5, 9])
        self.assertEqual(traces[1]["eval_index"], 97)
        self.assertEqual(len(traces[1]["live_steps"]), 25)
        self.assertEqual(tuple(traces[0]["returned_actions"].shape), (5, 10))


if __name__ == "__main__":
    unittest.main()
