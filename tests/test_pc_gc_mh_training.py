"""CPU regressions for planner-context goal-coupled MH training."""
import unittest

import torch

from planner_context_goal_coupled_mh import (
    combine_goal_coupled_metric,
    goal_coupled_mse,
)
from planner_context_goal_mh import planner_context_rollout


class _IdentityAction(torch.nn.Module):
    def forward(self, x):
        return x


class _DummyModel:
    def __init__(self):
        self.action_encoder = _IdentityAction()
        self.context_lengths = []

    def predict(self, emb, act_emb):
        self.context_lengths.append(int(emb.shape[1]))
        if emb.shape != act_emb.shape:
            raise AssertionError((emb.shape, act_emb.shape))
        out = emb.clone()
        out[:, -1] = emb.mean(dim=1) + 0.25 * act_emb.mean(dim=1)
        return out


class PlannerContextReuseTest(unittest.TestCase):
    def test_context_is_one_two_three_then_capped(self):
        model = _DummyModel()
        current = torch.tensor([[0.2, -0.4]])
        plans = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.2, -0.3]]])
        out = planner_context_rollout(model, current, plans, predictor_max_history=3)
        self.assertEqual(tuple(out.shape), (1, 5, 2))
        self.assertEqual(model.context_lengths, [1, 2, 3, 3, 3])


class GoalCoupledMetricTest(unittest.TestCase):
    def test_orthogonal_error_contributes_zero(self):
        error = torch.tensor([[[0.0, 3.0], [0.0, -4.0]]])
        direction = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        loss, valid, scale = goal_coupled_mse(error, direction)
        self.assertAlmostEqual(float(loss), 0.0, places=7)
        self.assertAlmostEqual(float(valid), 1.0, places=7)
        self.assertAlmostEqual(float(scale), 2.5, places=7)

    def test_candidate_gradient_preserves_residual_squared_weighting(self):
        # Same parallel error for two candidates, but second residual is 2x.
        # With one shared normalization scale, its gradient must be 4x larger.
        error = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
        direction = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        loss, valid, scale = goal_coupled_mse(error, direction)
        loss.backward()
        g = error.grad[0, :, 0]
        self.assertAlmostEqual(float(g[1] / g[0]), 4.0, places=6)
        self.assertAlmostEqual(float(valid), 1.0, places=7)
        self.assertAlmostEqual(float(scale), 2.5, places=7)

    def test_global_direction_rescaling_does_not_change_loss(self):
        error = torch.tensor([[[0.7, -0.2], [0.4, 0.3]]])
        direction = torch.tensor([[[1.0, 0.5], [2.0, -0.25]]])
        a, _, _ = goal_coupled_mse(error, direction)
        b, _, _ = goal_coupled_mse(error, 3.0 * direction)
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_zero_directions_are_safe(self):
        error = torch.tensor([[[1.0, -2.0], [0.5, 0.1]]])
        direction = torch.zeros_like(error)
        loss, valid, scale = goal_coupled_mse(error, direction)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(valid), 0.0)
        self.assertEqual(float(scale), 0.0)

    def test_weight_zero_exactly_recovers_base_mh(self):
        base = torch.tensor(1.75)
        gc = torch.tensor(0.42)
        actual = combine_goal_coupled_metric(base, gc, 0.0)
        torch.testing.assert_close(actual, base, rtol=0, atol=0)

    def test_weight_one_adds_one_gc_term(self):
        base = torch.tensor(1.75)
        gc = torch.tensor(0.42)
        actual = combine_goal_coupled_metric(base, gc, 1.0)
        torch.testing.assert_close(actual, base + gc)


if __name__ == "__main__":
    unittest.main()
