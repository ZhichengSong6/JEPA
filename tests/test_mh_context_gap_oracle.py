"""CPU regression tests for planner-context MH response diagnostics."""

import unittest

import torch

from eval_mh_ald_context_gap_oracle import _planner_warmup_rollout


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


class PlannerWarmupTest(unittest.TestCase):
    def test_context_grows_then_caps_at_three(self):
        model = _DummyModel()
        current = torch.tensor([0.2, -0.4])
        plans = torch.tensor([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, 0.5],
                [0.2, -0.3],
            ],
            [
                [-0.5, 0.2],
                [0.1, -0.4],
                [0.3, 0.7],
                [0.8, -0.2],
                [-0.1, 0.9],
            ],
        ])
        out = _planner_warmup_rollout(
            model,
            current,
            plans,
            predictor_max_history=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(out.shape), (2, 5, 2))
        self.assertEqual(model.context_lengths, [1, 2, 3, 3, 3])

    def test_matches_explicit_manual_recursion(self):
        model = _DummyModel()
        current = torch.tensor([0.25, 0.75])
        plans = torch.tensor([[[
            0.1, 0.2
        ], [
            0.3, -0.1
        ], [
            -0.2, 0.4
        ], [
            0.5, 0.6
        ], [
            -0.4, 0.2
        ]]])
        actual = _planner_warmup_rollout(
            model,
            current,
            plans,
            predictor_max_history=3,
            device=torch.device("cpu"),
        )

        history = current[None, None].clone()
        expected = []
        for step in range(plans.shape[1]):
            emb = history[:, -3:]
            act = plans[:, : step + 1][:, -3:]
            nxt = emb.mean(dim=1, keepdim=True) + 0.25 * act.mean(
                dim=1, keepdim=True
            )
            expected.append(nxt)
            history = torch.cat([history, nxt], dim=1)
        expected = torch.cat(expected, dim=1)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
