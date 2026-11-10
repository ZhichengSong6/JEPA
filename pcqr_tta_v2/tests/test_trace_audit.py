"""CPU regression tests for the optional v2 numerical audit snapshots."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pcqr_tta_v2.trace_audit import save_before_update, compare_pair


class NumericSnapshotTests(unittest.TestCase):
    def test_preupdate_snapshots_are_lossless_and_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            pixels = torch.ones(1, 4, 1, 3, 8, 8, dtype=torch.float32)
            actions = torch.arange(1 * 4 * 5 * 10, dtype=torch.float32).reshape(1, 4, 5, 10)
            raw = {
                "state": np.array([[[1., 2., 3.]]], dtype=np.float64),
                "pixels": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8),
            }
            reference = save_before_update(
                temp, "frozen_case0000", 1,
                solver_info={"pixels": pixels, "state": torch.ones(1, 4, 1, 3)},
                candidates=actions,
                issued_actions=[np.array([[0.2, 0.3]])],
                raw_info=raw,
            )
            observed = save_before_update(
                temp, "l2_case0000", 1,
                solver_info={"pixels": pixels.clone(), "state": torch.ones(1, 4, 1, 3)},
                candidates=actions.clone(),
                issued_actions=[np.array([[0.2, 0.3]])],
                raw_info={**raw, "state": raw["state"].copy()},
            )
            self.assertTrue(reference.is_file())
            with np.load(reference, allow_pickle=False) as reader:
                self.assertEqual(reader["solver_pixels"].shape, (1, 3, 8, 8))
                self.assertEqual(reader["env_state"].dtype, np.float64)
                self.assertEqual(reader["issued_action_prefix"].shape, (1, 2))
                self.assertEqual(reader["first_cem_population"].shape, (4, 5, 10))
            summary = compare_pair(reference, observed)
            self.assertTrue(all(x["status"] == "exact_equal" for x in summary.values()))
            with self.assertRaises(FileExistsError):
                save_before_update(temp, "frozen_case0000", 1,
                                   solver_info={"pixels": pixels}, candidates=actions,
                                   issued_actions=[])

    def test_numerical_differences_are_reported_without_tolerance(self):
        with tempfile.TemporaryDirectory() as temp:
            pixels = torch.zeros(1, 4, 1, 3, 4, 4, dtype=torch.float32)
            actions = torch.zeros(1, 4, 5, 10, dtype=torch.float32)
            reference = save_before_update(temp, "frozen_case0000", 1,
                                           solver_info={"pixels": pixels},
                                           candidates=actions, issued_actions=[])
            modified = pixels.clone()
            modified[0, 0, 0, 0, 0, 0] = 0.125
            other = save_before_update(temp, "l2_case0000", 1,
                                       solver_info={"pixels": modified},
                                       candidates=actions, issued_actions=[])
            stats = compare_pair(reference, other)
            self.assertEqual(stats["solver_pixels"]["status"], "different")
            self.assertEqual(stats["solver_pixels"]["num_different"], 1)
            self.assertAlmostEqual(stats["solver_pixels"]["max_abs"], 0.125)
            self.assertEqual(stats["first_cem_population"]["status"], "exact_equal")

    def test_invalid_snapshot_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_before_update(tmp, "../../escape", 0, solver_info={},
                                   candidates=None, issued_actions=[])


if __name__ == "__main__":
    unittest.main()
