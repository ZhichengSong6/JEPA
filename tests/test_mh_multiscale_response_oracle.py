"""CPU regressions for the multi-scale MH response diagnostic."""
import unittest

import numpy as np

from eval_mh_ald_multiscale_response_oracle import (
    _action_direction,
    _same_probe_structure,
)
from eval_mh_ald_teacher_response_oracle import _make_blockwise_candidates


class MultiScaleProbeTest(unittest.TestCase):
    def setUp(self):
        # Five coarse blocks, each block = five raw 2-D actions.
        self.center = np.zeros((25, 2), dtype=np.float32)
        self.kw = dict(
            future_actions=self.center,
            positions=list(range(5)),
            directions_per_position=4,
            seed=12345,
            action_block=5,
        )

    def test_same_seed_preserves_probe_identity_across_radii(self):
        c1, m1 = _make_blockwise_candidates(radius=0.08, **self.kw)
        c2, m2 = _make_blockwise_candidates(radius=0.30, **self.kw)
        self.assertTrue(_same_probe_structure(m1, m2))
        self.assertEqual(c1.shape, c2.shape)
        self.assertEqual(c1.shape[0], 41)

    def test_unbounded_case_uses_same_action_direction(self):
        c1, m1 = _make_blockwise_candidates(radius=0.08, **self.kw)
        c2, m2 = _make_blockwise_candidates(radius=0.30, **self.kw)
        for p in range(5):
            for d in range(4):
                v1 = _action_direction(c1, m1, self.center, p, d, 5)
                v2 = _action_direction(c2, m2, self.center, p, d, 5)
                self.assertAlmostEqual(float(np.dot(v1, v2)), 1.0, places=5)

    def test_radius_norms_are_exact_away_from_bounds(self):
        for radius in (0.08, 0.1565, 0.30):
            c, m = _make_blockwise_candidates(radius=radius, **self.kw)
            for i in range(1, len(c)):
                delta = np.asarray(c[i] - self.center, dtype=np.float64)
                self.assertAlmostEqual(
                    float(np.linalg.norm(delta)),
                    float(m[i]["effective_radius"]),
                    places=5,
                )


if __name__ == "__main__":
    unittest.main()
