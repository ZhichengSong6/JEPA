"""CPU regressions for pure planner-context MH training."""
import unittest

import torch

from planner_context_mh import planner_context_rollout


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


class PlannerContextMHTrainingTest(unittest.TestCase):
    def test_context_is_one_two_three_then_capped(self):
        model = _DummyModel()
        current = torch.tensor([[0.2, -0.4], [0.1, 0.5]])
        plans = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.2, -0.3]],
            [[-0.5, 0.2], [0.1, -0.4], [0.3, 0.7], [0.8, -0.2], [-0.1, 0.9]],
        ])
        out = planner_context_rollout(model, current, plans, predictor_max_history=3)
        self.assertEqual(tuple(out.shape), (2, 5, 2))
        self.assertEqual(model.context_lengths, [1, 2, 3, 3, 3])

    def test_matches_explicit_manual_recursion(self):
        model = _DummyModel()
        current = torch.tensor([[0.25, 0.75]])
        plans = torch.tensor([
            [[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.6], [0.7, -0.3]]
        ])
        actual = planner_context_rollout(model, current, plans, predictor_max_history=3)

        history = current[:, None].clone()
        expected = []
        for step in range(plans.shape[1]):
            emb = history[:, -3:]
            act = plans[:, : step + 1][:, -3:]
            nxt = emb.mean(dim=1, keepdim=True) + 0.25 * act.mean(dim=1, keepdim=True)
            expected.append(nxt)
            history = torch.cat([history, nxt], dim=1)

        torch.testing.assert_close(actual, torch.cat(expected, dim=1))

    def test_invalid_history_rejected(self):
        model = _DummyModel()
        current = torch.zeros(1, 2)
        plans = torch.zeros(1, 1, 2)
        with self.assertRaises(ValueError):
            planner_context_rollout(model, current, plans, predictor_max_history=0)


if __name__ == "__main__":
    unittest.main()
