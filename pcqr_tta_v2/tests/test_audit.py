"""CPU regression tests for read-only pre-update parity audit."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pcqr_tta_v2.audit_bridge import (
    _bytes_digest, action_prefix_sha256, compare_preupdate,
    make_audited_cost_class, make_audited_policy_class, parameters_sha256,
    AUDIT_KEYS,
)


class AuditTests(unittest.TestCase):
    def test_array_hash_is_dtype_and_shape_sensitive(self):
        self.assertEqual(_bytes_digest(torch.ones(2, 3)), _bytes_digest(np.ones((2, 3), dtype=np.float32)))
        self.assertNotEqual(_bytes_digest(np.ones((2, 3), dtype=np.float32)), _bytes_digest(np.ones((3, 2), dtype=np.float32)))
        self.assertNotEqual(_bytes_digest(np.ones((2, 3), dtype=np.float32)), _bytes_digest(np.ones((2, 3), dtype=np.float64)))

    def test_action_trace_order_matters(self):
        x, y = np.array([[1., 0.]]), np.array([[0., 1.]])
        self.assertNotEqual(action_prefix_sha256([x, y]), action_prefix_sha256([y, x]))
        self.assertNotEqual(action_prefix_sha256([]), action_prefix_sha256([x]))

    def test_parameter_hash_changes(self):
        p = nn.Parameter(torch.zeros(2))
        original = parameters_sha256([('p', p)])
        with torch.no_grad():
            p.add_(1)
        self.assertNotEqual(original, parameters_sha256([('p', p)]))

    def test_strict_compare_reports_specific_keys(self):
        ref = dict.fromkeys(AUDIT_KEYS, 'same')
        self.assertEqual(compare_preupdate(ref, dict(ref)), {})
        changed = dict(ref)
        changed['first_cem_population_sha256'] = 'different'
        self.assertEqual(list(compare_preupdate(ref, changed)), ['first_cem_population_sha256'])
        del changed['selected_parameters_sha256']
        self.assertIn('selected_parameters_sha256', compare_preupdate(ref, changed))

    def test_read_only_cost_wrapper_collects_first_two_plans(self):
        class FakeCost(nn.Module):
            def __init__(self, model, adapter, log=None, verify_rollout=True):
                super().__init__()
                self.armed = False
                self.replans = 0
                self.adapter = adapter
                self.log = log
                self.model = model
            def get_cost(self, info_dict, action_candidates):
                if self.armed:
                    self.replans += 1
                    self.armed = False
                return self.model.get_cost(info_dict, action_candidates)

        class Dummy(nn.Module):
            def get_cost(self, info_dict, actions):
                return actions.mean(dim=(2, 3))
        p = nn.Parameter(torch.tensor([1., 2.]))
        adapter = SimpleNamespace(named=[('p', p)])
        logs = []
        cost = make_audited_cost_class(FakeCost)(Dummy(), adapter, log=logs.append)
        info = {'pixels': torch.zeros(1, 4, 1, 3, 8, 8)}
        q = torch.randn(1, 4, 5, 10)
        cost.issued_actions.extend([np.ones((1, 2), dtype=np.float32)] * 5)
        cost.raw_steps_seen = 5
        for _ in range(3):
            cost.armed = True
            self.assertEqual(cost.get_cost(info, q).shape, (1, 4))
        self.assertEqual(cost.replans, 3)
        self.assertEqual(len(cost.replan_audits), 2)
        self.assertEqual(len(logs), 2)
        self.assertEqual(cost.replan_audits[1]['raw_steps'], 5)
        self.assertEqual(logs[1]['type'], 'preupdate_audit')

    def test_policy_wrapper_issues_actions_only_as_observed(self):
        class FakeBase:
            def __init__(self):
                self.cfg = SimpleNamespace(receding_horizon=1, action_block=5)
                self.online_cost = SimpleNamespace(issued_actions=[], raw_steps_seen=0)
                self._n = 0
            def get_action(self, info_dict, **kwargs):
                self._n += 1
                return np.full((1, 2), self._n, dtype=np.float32)
        Klass = make_audited_policy_class(FakeBase, online_factory=lambda x: x)
        policy = Klass()
        for i in range(9):
            policy.get_action({})
        self.assertEqual(policy.online_cost.raw_steps_seen, 6)
        self.assertEqual(len(policy.online_cost.issued_actions), 6)
        self.assertEqual(policy.online_cost.issued_actions[-1][0, 0], 6)


if __name__ == '__main__':
    unittest.main()
