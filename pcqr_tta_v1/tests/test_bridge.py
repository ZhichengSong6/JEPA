import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pcqr_tta_v1.core import AdaptConfig, OnlineAdapter
from pcqr_tta_v1.swm_bridge import AdaptiveCostModel, make_online_policy_class
from pcqr_tta_v1.tests.fixtures import TinyJEPA


class IdentityScaler:
    def transform(self, x):
        return x.copy()

    def inverse_transform(self, x):
        return x.copy()


class FakeSolver:
    def __init__(self, model):
        self.model = model
        self.snapshots = []
        self.generator = torch.Generator().manual_seed(91)

    @torch.inference_mode()
    def __call__(self, info):
        q = torch.randn(1, 8, 5, 10, generator=self.generator)
        p = info['pixels'][:, None].expand(1, 8, 1, 3, 2, 2)
        calls = []
        for _ in range(2):
            self.model.get_cost({'pixels': p}, q)
            calls.append([x.detach().clone() for x in self.model.parameters()])
        self.snapshots.append(calls)
        return np.array([[0.25, -0.75]], dtype=np.float32)


class FakeBasePolicy:
    def __init__(self, solver, process):
        self.solver, self.process = solver, process
        self.calls = 0

    def set_env(self, env):
        self.env = env

    def _prepare_info(self, info):
        x = torch.tensor(info['pixels']).permute(0, 1, 4, 2, 3)
        return {'pixels': x}

    def get_action(self, info):
        if self.calls % 5 == 0:
            self.current = self.solver(self._prepare_info(info))
        self.calls += 1
        return self.current.copy()


def observation(t, action):
    return {'pixels': np.full((1, 1, 2, 2, 3), t / 100, dtype=np.float32),
            'goal': np.zeros((1, 1, 2, 2, 3), dtype=np.float32),
            'action': np.asarray(action, dtype=np.float32).reshape(1, 1, 2),
            'state': np.zeros((1, 1, 7), dtype=np.float32)}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(32)
        self.core = TinyJEPA()

    def make(self, mode='response', expected=None):
        adapter = OnlineAdapter(self.core, AdaptConfig(mode=mode))
        logs = []
        cost = AdaptiveCostModel(self.core, adapter, log=logs.append)
        solver = FakeSolver(cost)
        cls = make_online_policy_class(FakeBasePolicy)
        policy = cls(solver=solver, process={'action': IdentityScaler()}, online_cost=cost,
                     log=logs.append, action_block=5, expected_initial=expected)
        policy.set_env(SimpleNamespace(num_envs=1))
        return adapter, cost, solver, policy, logs

    def test_buffers_only_completed_blocks(self):
        a, c, s, p, _ = self.make()
        action = np.zeros((1, 2), dtype=np.float32)
        for t in range(5):
            action = p.get_action(observation(t, action))
        self.assertEqual(len(a.buffer), 0)
        self.assertEqual(len(p.online_partial_actions), 4)
        action = p.get_action(observation(5, action))
        self.assertEqual(len(a.buffer), 1)
        torch.testing.assert_close(a.buffer.actions[0], torch.tensor([.25, -.75] * 5))
        self.assertEqual(c.applied_steps, 3)

    def test_exactly_one_update_per_replan_and_fixed_cem_weights(self):
        a, cost, solver, policy, logs = self.make()
        action = np.zeros((1, 2), dtype=np.float32)
        with torch.inference_mode():
            for t in range(11):
                action = policy.get_action(observation(t, action))
        self.assertEqual(cost.replans, 3)
        self.assertEqual(cost.cost_calls, 6)
        self.assertEqual(cost.applied_steps, 6)  # warmup, then two 3-step updates
        self.assertEqual([r['status'] for r in logs if r['type'] == 'adapt'],
                         ['warmup', 'updated', 'updated'])
        for first, second in solver.snapshots:
            for p, q in zip(first, second):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        a.assert_frozen_unchanged()

    def test_frozen_matches_base_policy_without_adaptation(self):
        reference = copy.deepcopy(self.core)
        expected_solver = FakeSolver(reference)
        expected = FakeBasePolicy(expected_solver, {'action': IdentityScaler()})
        expected.set_env(SimpleNamespace(num_envs=1))
        a, cost, _, policy, _ = self.make('frozen')
        actual_action = np.zeros((1, 2), dtype=np.float32)
        expected_action = actual_action.copy()
        for t in range(11):
            actual_action = policy.get_action(observation(t, actual_action))
            expected_action = expected.get_action(observation(t, expected_action))
            np.testing.assert_array_equal(actual_action, expected_action)
        self.assertEqual(cost.applied_steps, 0)
        self.assertEqual(len(a.buffer), 0)
        self.assertEqual(policy.online_raw_steps, 10)
        for name, state in self.core.state_dict().items():
            torch.testing.assert_close(state, reference.state_dict()[name], rtol=0, atol=0)

    def test_wrong_previous_action_rejected(self):
        _, _, _, p, _ = self.make()
        p.get_action(observation(0, [0, 0]))
        with self.assertRaisesRegex(RuntimeError, 'last returned'):
            p.get_action(observation(1, [100, 200]))

    def test_mid_episode_reset_rejected(self):
        _, _, _, p, _ = self.make()
        action = p.get_action(observation(0, [0, 0]))
        obs = observation(1, action)
        obs['_needs_flush'] = np.array([True])
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            p.get_action(obs)

    def test_multi_environment_rejected(self):
        _, _, _, p, _ = self.make()
        with self.assertRaises(ValueError):
            p.set_env(SimpleNamespace(num_envs=2))

    def test_initial_pairing_mismatch_rejected(self):
        _, _, _, p, _ = self.make('frozen', expected={'pixels': 'wrong'})
        with self.assertRaisesRegex(RuntimeError, 'initial state'):
            p.get_action(observation(0, [0, 0]))

    def test_initial_pairing_match_accepted(self):
        _, _, _, p, _ = self.make('frozen')
        p.get_action(observation(0, [0, 0]))
        fingerprint = p.initial_fingerprint
        _, _, _, q, _ = self.make('frozen', expected=fingerprint)
        q.get_action(observation(0, [0, 0]))

    def test_missing_goal_fails_audit_not_training(self):
        _, _, _, p, _ = self.make('frozen')
        obs = observation(0, [0, 0])
        del obs['goal']
        with self.assertRaisesRegex(ValueError, 'paired'):
            p.get_action(obs)


if __name__ == '__main__':
    unittest.main()
