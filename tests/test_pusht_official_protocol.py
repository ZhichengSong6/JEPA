"""Protocol regressions, without GPU/model/data dependencies."""

import unittest

import numpy as np

from pusht_official_protocol import matched_factor_accuracy, paired_outcomes, validate_protocol


class OfficialProtocolTests(unittest.TestCase):
    def formal(self):
        return dict(seed=42, batch_size=1, var_scale=1.0, history_size=1, frame_skip=1,
                    horizon=5, receding_horizon=5, action_block=5, eval_budget=50,
                    goal_offset_steps=25, img_size=224, env_name="swm/PushT-v1",
                    dataset_name="pusht_expert_train", num_samples=300, n_steps=30,
                    topk=30, num_eval=100, replay_iterations=[0, 3, 9, 19, 29])

    def test_reject_old_budget_and_changed_population(self):
        validate_protocol(self.formal(), "formal")
        for key, value in (("n_steps", 10), ("num_eval", 50), ("history_size", 3),
                           ("replay_iterations", [0, 3, 9]), ("seed", 3072)):
            spec = self.formal()
            spec[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(spec, "formal")

    def test_paired_outcomes_and_exact_test(self):
        labels, counts, summary = paired_outcomes([True, False, False, True], [True, True, False, False])
        self.assertEqual(counts, dict.fromkeys(labels, 1))
        self.assertEqual(summary["mcnemar_exact_two_sided_p"], 1.0)
        _, _, summary = paired_outcomes([False] * 8, [True] * 8)
        self.assertEqual(summary["mcnemar_exact_two_sided_p"], 2 / 256)
        _, _, summary = paired_outcomes([True] * 4, [True] * 4)
        self.assertEqual(summary["mcnemar_exact_two_sided_p"], 1.0)

    def test_both_other_factors_must_match(self):
        target = np.arange(8, dtype=float)
        controls = np.zeros((8, 2))
        self.assertEqual(matched_factor_accuracy(target, target, controls)[0], 1.0)
        self.assertEqual(matched_factor_accuracy(-target, target, controls)[0], 0.0)
        self.assertEqual(matched_factor_accuracy(np.zeros(8), target, controls)[0], 0.5)
        controls[:, 1] = target * 100
        acc, pairs = matched_factor_accuracy(target, target, controls)
        self.assertTrue(np.isnan(acc))
        self.assertEqual(pairs, 0)

    def test_constant_target_is_uninformative(self):
        acc, count = matched_factor_accuracy(np.arange(8), np.ones(8), np.zeros((8, 2)))
        self.assertTrue(np.isnan(acc))
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
