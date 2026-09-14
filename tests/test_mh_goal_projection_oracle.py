"""CPU regression tests for the MH goal-projection diagnostic."""
import unittest

import numpy as np
import torch

from eval_mh_ald_goal_projection_oracle import (
    _decompose_endpoint_error,
    _ranking_metrics,
)


class GoalProjectionDecompositionTest(unittest.TestCase):
    def test_exact_cost_identity_and_orthogonality(self):
        real = torch.tensor([
            [2.0, 0.0],
            [0.0, 3.0],
            [1.0, 1.0],
        ])
        goal = torch.tensor([0.0, 0.0])
        pred = torch.tensor([
            [3.0, 2.0],
            [-1.0, 4.0],
            [0.5, 2.0],
        ])
        d = _decompose_endpoint_error(pred, real, goal)
        torch.testing.assert_close(
            d["full_cost"] - d["exact_cost"],
            d["cross_term"] + d["error_sq"],
        )
        torch.testing.assert_close(
            d["error_sq"], d["parallel_sq"] + d["perp_sq"]
        )
        dot = torch.sum(d["residual"] * d["e_perp"], dim=-1)
        torch.testing.assert_close(dot, torch.zeros_like(dot), atol=1e-6, rtol=0)

    def test_pure_parallel_error_is_removed_by_parallel_intervention(self):
        real = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
        goal = torch.zeros(2)
        pred = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
        d = _decompose_endpoint_error(pred, real, goal)
        torch.testing.assert_close(d["perp_sq"], torch.zeros(2))
        torch.testing.assert_close(d["remove_parallel_cost"], d["exact_cost"])
        torch.testing.assert_close(d["remove_orthogonal_cost"], d["full_cost"])

    def test_pure_orthogonal_error_is_removed_by_orthogonal_intervention(self):
        real = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
        goal = torch.zeros(2)
        pred = torch.tensor([[2.0, 1.0], [2.0, 4.0]])
        d = _decompose_endpoint_error(pred, real, goal)
        torch.testing.assert_close(d["parallel_sq"], torch.zeros(2))
        torch.testing.assert_close(d["remove_parallel_cost"], d["full_cost"])
        torch.testing.assert_close(d["remove_orthogonal_cost"], d["exact_cost"])

    def test_zero_goal_residual_treats_error_as_orthogonal(self):
        real = torch.tensor([[0.0, 0.0]])
        goal = torch.tensor([0.0, 0.0])
        pred = torch.tensor([[1.0, -2.0]])
        d = _decompose_endpoint_error(pred, real, goal)
        torch.testing.assert_close(d["parallel_sq"], torch.zeros(1))
        torch.testing.assert_close(d["perp_sq"], d["error_sq"])
        torch.testing.assert_close(d["remove_orthogonal_cost"], d["exact_cost"])


class GoalProjectionRankingTest(unittest.TestCase):
    def test_perfect_score_has_perfect_ranking_metrics(self):
        exact = np.asarray([0.4, 0.1, 0.8, 0.2, 1.2], dtype=np.float64)
        physical = np.asarray([4.0, 1.0, 8.0, 2.0, 12.0], dtype=np.float64)
        m = _ranking_metrics(exact.copy(), exact, physical, 0.0)
        self.assertAlmostEqual(m["rho_exact_encoder"], 1.0)
        self.assertAlmostEqual(m["pairwise_exact_encoder"], 1.0)
        self.assertAlmostEqual(m["top10_exact_encoder_overlap"], 1.0)
        self.assertAlmostEqual(m["exact_encoder_selection_regret"], 0.0)
        self.assertEqual(m["selected_index"], 1)


if __name__ == "__main__":
    unittest.main()
