"""CPU image-source regressions; no checkpoints, simulator, or GPU required."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cem_mh_pixel_checks import (PixelAlignmentError, check_trace_pixels,
                                dataset_start_images, policy_tensor, wrapper_pixels)


def transform():
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
                       v2.Resize(224)])


def trace(initial=True):
    rng = np.random.default_rng(0)
    native = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    dataset = np.full((224, 224, 3), 181, dtype=np.uint8)
    observed = dataset.copy() if initial else wrapper_pixels(native)
    goal = np.full((224, 224, 3), 97, dtype=np.uint8)
    tr = {"eval_index": 11, "solve_no": 0 if initial else 1,
          "snapshot": {"image": native, "prefix": [] if initial else [{}]*25},
          "info": {"pixels": policy_tensor(observed, transform())[None, None],
                   "goal": policy_tensor(goal, transform())[None, None]},
          "raw_policy_pixels": observed.copy(), "raw_policy_goal": goal.copy()}
    return tr, dataset


class PixelTests(unittest.TestCase):
    def test_dataset_origin_need_not_equal_native(self):
        tr, initial = trace()
        result = check_trace_pixels(tr, transform(), initial)
        self.assertTrue(result["passed"])
        self.assertEqual(result["expected_origin"], "dataset_start")
        self.assertGreater(result["native_vs_observation"]["max_abs"], 0)
        self.assertEqual(result["model_input"]["max_abs"], 0)

    def test_initial_native_substitution_rejected(self):
        tr, initial = trace()
        tr["info"]["pixels"] = policy_tensor(wrapper_pixels(tr["snapshot"]["image"]), transform())[None, None]
        with self.assertRaises(PixelAlignmentError):
            check_trace_pixels(tr, transform(), initial)

    def test_later_observation_must_use_wrapper(self):
        tr, initial = trace(False)
        self.assertTrue(check_trace_pixels(tr, transform(), initial)["passed"])
        # Old code resizes AFTER float conversion/normalization, not in uint8 PIL.
        old = transform()(tr["snapshot"]["image"])
        self.assertFalse(torch.allclose(old, tr["info"]["pixels"][0, -1], rtol=0, atol=1e-5))
        tr["info"]["pixels"] = old[None, None]
        with self.assertRaises(PixelAlignmentError):
            check_trace_pixels(tr, transform(), initial)

    def test_later_dataset_substitution_rejected(self):
        tr, initial = trace(False)
        tr["info"]["pixels"] = policy_tensor(initial, transform())[None, None]
        with self.assertRaises(PixelAlignmentError):
            check_trace_pixels(tr, transform(), initial)

    def test_actual_raw_input_corruption_rejected(self):
        tr, initial = trace()
        tr["raw_policy_pixels"][0, 0, 0] ^= 1
        with self.assertRaises(PixelAlignmentError):
            check_trace_pixels(tr, transform(), initial)

    def test_goal_corruption_rejected(self):
        tr, initial = trace()
        tr["info"]["goal"][0, 0, 0, 0, 0] += .01
        with self.assertRaises(PixelAlignmentError):
            check_trace_pixels(tr, transform(), initial)

    def test_legacy_trace_uses_expected_dataset_not_candidate_selection(self):
        tr, initial = trace()
        del tr["raw_policy_pixels"], tr["raw_policy_goal"]
        report = check_trace_pixels(tr, transform(), initial)
        self.assertTrue(report["passed"])
        self.assertFalse(report["raw_policy_captured"])

    def test_failure_keeps_arrays_and_numeric_report(self):
        tr, initial = trace()
        tr["info"]["pixels"][0, 0, 0, 0, 0] += .1
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "case.json"
            with self.assertRaises(PixelAlignmentError):
                check_trace_pixels(tr, transform(), initial, path)
            report = json.loads(path.read_text())
            self.assertFalse(report["passed"])
            self.assertGreater(report["model_input"]["max_abs"], .09)
            self.assertTrue(path.with_suffix(".npz").is_file())

    def test_dataset_loader_preserves_dtype_and_chw_order(self):
        arrays = [np.full((3, 224, 224), i+11, dtype=np.uint8) for i in range(2)]
        class Data:
            def load_chunk(self, episodes, starts, ends):
                self.args = (episodes, starts, ends)
                return [{"pixels": torch.from_numpy(a)[None]} for a in arrays]
        d = Data()
        cfg = types.SimpleNamespace(diag_episodes=[1, 2, 3], diag_start=[7, 8, 9])
        r = dataset_start_images(d, cfg, [{"eval_index": 0}, {"eval_index": 2}])
        self.assertEqual(list(r), [0, 2])
        self.assertEqual(r[2].dtype, np.uint8)
        self.assertEqual(r[2].shape, (224, 224, 3))
        np.testing.assert_array_equal(d.args[0], [1, 3])
        np.testing.assert_array_equal(d.args[2], [8, 10])

    def test_oracle_encoder_uses_wrapped_input(self):
        from eval_cem_mh_diagnostics import encode_images
        tr, _ = trace(False)
        class Model:
            def encode(self, info):
                self.received = info["pixels"].cpu().clone()
                return {"emb": info["pixels"].mean((-2, -1))}
        m = Model()
        encode_images(m, transform(), [tr["snapshot"]["image"]], "cpu")
        torch.testing.assert_close(m.received, tr["info"]["pixels"], rtol=0, atol=0)

    def test_float_raw_image_rejected_not_silently_rescaled(self):
        tr, _ = trace()
        with self.assertRaises(ValueError):
            policy_tensor(tr["raw_policy_pixels"].astype(np.float32), transform())

    def test_reuse_gate_rejects_changed_inputs(self):
        from eval_cem_mh_diagnostics import reusable_mh
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"seed": 42})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p/"protocol.json").write_text(json.dumps({"checkpoint_sha256": {"mh": "old"}}))
            with self.assertRaisesRegex(RuntimeError, "checkpoint_sha256"):
                reusable_mh(p, cfg, [], {"mh": "new"}, {}, [0, 5, 9], 2)


if __name__ == "__main__":
    unittest.main()
