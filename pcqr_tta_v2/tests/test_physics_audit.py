"""CPU fixture: physical-state audit remains read-only and exact."""
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pcqr_tta_v2.audit_bridge import (
    make_audited_cost_class, compare_preupdate,
)
from pcqr_tta_v2.trace_audit import save_before_update


class PhysicsAuditTests(unittest.TestCase):
    def test_physics_hidden_sha_is_recorded_and_strictly_compared(self):
        class FakeBase(nn.Module):
            def __init__(self, model, adapter, log=None):
                super().__init__()
                self.model, self.adapter, self.log = model, adapter, log
                self.armed = False
                self.replans = 0

            def get_cost(self, infos, candidates):
                if self.armed:
                    self.armed = False
                    self.replans += 1
                return candidates.square().mean(dim=(2, 3))

        model = nn.Linear(1, 1)
        adapter = SimpleNamespace(named=list(model.named_parameters()))
        events = []
        cost = make_audited_cost_class(FakeBase)(model, adapter, log=events.append)
        physical = {
            "agent_position": [1.0, 2.0],
            "block_velocity": [0.0, 0.0],
            "block_angular_velocity": 0.0,
        }
        cost.physics_probe = lambda: dict(physical)
        infos = {"pixels": torch.zeros(1, 3, 1, 3, 4, 4)}
        actions = torch.ones(1, 3, 5, 10)
        cost.armed = True
        cost.get_cost(infos, actions)
        first = cost.replan_audits[-1]
        self.assertEqual(first["physics_state"], physical)
        self.assertIn("physics_hidden_sha256", first)
        cost.armed = True
        cost.get_cost(infos, actions)
        self.assertEqual(compare_preupdate(first, cost.replan_audits[-1]), {})
        physical["block_velocity"] = [0.25, 0.0]
        cost.replans = 1
        cost.armed = True
        cost.get_cost(infos, actions)
        self.assertIn("physics_hidden_sha256",
                      compare_preupdate(first, cost.replan_audits[-1]))

    def test_trace_records_hidden_block_velocity(self):
        with tempfile.TemporaryDirectory() as root:
            path = save_before_update(
                root, "frozen_case0000", 0,
                solver_info={"pixels": torch.zeros(1, 2, 1, 3, 4, 4)},
                candidates=torch.zeros(1, 2, 5, 10),
                issued_actions=[],
                physics_snapshot={
                    "block_velocity": [0.01, -0.02],
                    "block_angular_velocity": 0.03,
                },
            )
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["physics_block_velocity"],
                                              [0.01, -0.02])
                self.assertAlmostEqual(float(z["physics_block_angular_velocity"]),
                                       0.03)


if __name__ == "__main__":
    unittest.main()
