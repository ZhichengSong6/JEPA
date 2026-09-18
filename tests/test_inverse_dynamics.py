import unittest

import torch
from torch import nn

from inverse_dynamics import InverseDynamicsHead, inverse_dynamics_objective


class _DeltaHead(nn.Module):
    """Exact synthetic inverse model for a 2-D latent/action test."""

    def forward(self, z_t, z_tp1):
        return z_tp1 - z_t


class TestInverseDynamics(unittest.TestCase):
    def test_head_shape_and_architecture(self):
        head = InverseDynamicsHead(
            latent_dim=192,
            action_dim=10,
            hidden_dim=256,
            depth=2,
        )
        z_t = torch.randn(3, 4, 192)
        z_tp1 = torch.randn(3, 4, 192)
        pred = head(z_t, z_tp1)

        self.assertEqual(tuple(pred.shape), (3, 4, 10))
        linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(len(linears), 3)
        self.assertEqual(linears[0].in_features, 384)
        self.assertEqual(linears[0].out_features, 256)
        self.assertEqual(linears[1].out_features, 256)
        self.assertEqual(linears[2].out_features, 10)

    def test_transition_alignment_is_action_t_for_z_t_to_z_tp1(self):
        emb = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 2.0],
                ]
            ]
        )
        actions = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                    [-1.0, 0.0],
                    [99.0, 99.0],  # no z_{t+1}; must not be used
                ]
            ]
        )

        result = inverse_dynamics_objective(
            _DeltaHead(),
            emb,
            actions,
            max_transitions=3,
        )
        self.assertLess(float(result["loss"]), 1e-8)
        self.assertEqual(float(result["valid_count"]), 3.0)
        self.assertAlmostEqual(float(result["valid_fraction"]), 1.0, places=7)

    def test_nan_boundary_action_is_masked(self):
        emb = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [100.0, 100.0],
                    [99.0, 100.0],
                ]
            ]
        )
        actions = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [float("nan"), float("nan")],
                    [-1.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )

        result = inverse_dynamics_objective(
            _DeltaHead(),
            emb,
            actions,
            max_transitions=3,
        )
        # The two valid transitions are exact; the deliberately nonsensical
        # middle transition is excluded rather than treated as zero action.
        self.assertLess(float(result["loss"]), 1e-8)
        self.assertEqual(float(result["valid_count"]), 2.0)
        self.assertAlmostEqual(float(result["valid_fraction"]), 2.0 / 3.0, places=6)

    def test_idm_gradient_reaches_latents_and_head(self):
        torch.manual_seed(7)
        head = InverseDynamicsHead(
            latent_dim=4,
            action_dim=3,
            hidden_dim=8,
            depth=2,
        )
        emb = torch.randn(2, 4, 4, requires_grad=True)
        actions = torch.randn(2, 4, 3)

        result = inverse_dynamics_objective(
            head,
            emb,
            actions,
            max_transitions=3,
        )
        result["loss"].backward()

        self.assertIsNotNone(emb.grad)
        self.assertGreater(float(emb.grad.abs().sum()), 0.0)
        head_grad = sum(
            float(p.grad.abs().sum())
            for p in head.parameters()
            if p.grad is not None
        )
        self.assertGreater(head_grad, 0.0)


if __name__ == "__main__":
    unittest.main()
