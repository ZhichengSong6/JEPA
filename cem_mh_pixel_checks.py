"""Observation-origin checks for stable-worldmodel 0.0.6 diagnostics.

The first solver observation is a DATASET frame injected by World. Later frames
come from AddPixelsWrapper. Neither may be silently replaced by raw.render().
A dataset/native difference at time zero is recorded, not forced to zero.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors

from cem_mh_diag_core import array_hash, dump


def rgb_array(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HWC RGB; got {image.shape}, {image.dtype}")
    return image.copy()


def wrapper_pixels(image, size=(224, 224)):
    """Exact no-extra-transform AddPixelsWrapper path: PIL bilinear in uint8."""
    image = rgb_array(image)
    height, width = map(int, size)
    return np.array(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))


def policy_tensor(image, transform):
    """Match BasePolicy._prepare_info's HWC -> CHW tv_tensors.Image path."""
    image = rgb_array(image)
    return transform(tv_tensors.Image(np.transpose(image, (2, 0, 1)))).detach().cpu()


def tensor_stats(a, b):
    a, b = torch.as_tensor(a).cpu(), torch.as_tensor(b).cpu()
    result = {"expected_shape": list(a.shape), "actual_shape": list(b.shape),
              "expected_dtype": str(a.dtype), "actual_dtype": str(b.dtype),
              "expected_hash": array_hash(a.numpy()), "actual_hash": array_hash(b.numpy())}
    if a.shape == b.shape and a.numel():
        delta = (a.double() - b.double()).abs()
        result.update(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                      unequal_fraction=float((delta > 0).double().mean()))
    return result


class PixelAlignmentError(RuntimeError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def check_trace_pixels(tr, transform, initial_image, report_path=None):
    """Validate the correct origin, NOT equality of dataset and live images.

    Legacy traces can be checked against the original dataset at solve zero;
    new traces additionally retain pre-transform policy pixels and goal frames.
    Live replay integrity is independently enforced by restore_prefix.
    """
    snap = tr["snapshot"]
    initial = len(snap["prefix"]) == 0
    native_wrapped = wrapper_pixels(snap["image"])
    expected_image = rgb_array(initial_image) if initial else native_wrapped
    source = "dataset_start" if initial else "live_AddPixelsWrapper"
    expected = policy_tensor(expected_image, transform)
    actual = tr["info"]["pixels"][0, -1].detach().cpu()
    details = {"eval_index": int(tr["eval_index"]), "solve_no": int(tr["solve_no"]),
               "prefix_steps": len(snap["prefix"]), "expected_origin": source,
               "native_render_shape": list(np.asarray(snap["image"]).shape),
               "observation_shape": list(expected_image.shape),
               "model_input": tensor_stats(expected, actual),
               "native_vs_observation": tensor_stats(native_wrapped, expected_image),
               "raw_policy_captured": "raw_policy_pixels" in tr}
    # Dataset observations must match the ORIGINAL dataset, not whichever image
    # happens to be closer to the model input. Replan observations must be live.
    passed = (expected.shape == actual.shape and
              torch.allclose(expected, actual, rtol=0, atol=1e-5))
    if "raw_policy_pixels" in tr:
        raw_pixels = rgb_array(tr["raw_policy_pixels"])
        details["raw_policy_match"] = bool(np.array_equal(raw_pixels, expected_image))
        passed = passed and details["raw_policy_match"]
    if "raw_policy_goal" in tr:
        goal = policy_tensor(tr["raw_policy_goal"], transform)
        actual_goal = tr["info"]["goal"][0, -1].detach().cpu()
        details["goal_input"] = tensor_stats(goal, actual_goal)
        passed = passed and goal.shape == actual_goal.shape and torch.allclose(goal, actual_goal, rtol=0, atol=1e-5)
    details["passed"] = bool(passed)
    if report_path is not None:
        dump(report_path, details)
        if not passed:
            np.savez_compressed(Path(report_path).with_suffix(".npz"),
                                native_render=np.asarray(snap["image"]),
                                expected_observation=expected_image,
                                expected_model_input=expected.numpy(), actual_model_input=actual.numpy())
    if not passed:
        raise PixelAlignmentError(f"Policy pixel mismatch against {source}: {details['model_input']}", details)
    return details


def dataset_start_images(dataset, cfg, manifest):
    """Read untransformed dataset first frames, preserving the CHW uint8 dtype."""
    indices = [int(m["eval_index"]) for m in manifest]
    episodes = np.asarray([cfg.diag_episodes[i] for i in indices])
    starts = np.asarray([cfg.diag_start[i] for i in indices])
    chunks = dataset.load_chunk(episodes, starts, starts + 1)
    if len(chunks) != len(indices):
        raise RuntimeError("Incomplete initial dataset image load")
    result = {}
    for i, chunk in zip(indices, chunks):
        image = chunk["pixels"][0]
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[0] != 3:
            raise RuntimeError(f"Dataset pixels must be CHW (as in World.evaluate_from_dataset): {image.shape}")
        result[i] = rgb_array(np.moveaxis(image, 0, -1))
    return result


def verify_installed_pipeline(raw, transform):
    """Check our reconstruction against the INSTALLED wrapper/policy, not main."""
    import stable_worldmodel as swm
    from stable_worldmodel.wrapper import AddPixelsWrapper
    wrapper = AddPixelsWrapper(raw, pixels_shape=(224, 224))
    observed, _ = wrapper._get_pixels()
    expected = wrapper_pixels(raw.render())
    if set(observed) != {"pixels"} or not np.array_equal(expected, observed["pixels"]):
        raise RuntimeError("Installed AddPixelsWrapper differs from diagnostic pixel reconstruction")
    minimal = SimpleNamespace(process={}, transform={"pixels": transform})
    info = {"pixels": expected[None, None].copy()}
    actual = swm.policy.WorldModelPolicy._prepare_info(minimal, copy.deepcopy(info))["pixels"][0, -1]
    reconstructed = policy_tensor(expected, transform)
    if not torch.allclose(reconstructed, actual.cpu(), rtol=0, atol=1e-5):
        raise RuntimeError("Installed policy preprocessing differs from diagnostic reconstruction")
    return {"passed": True, "native_shape": list(np.asarray(raw.render()).shape),
            "observation_shape": list(expected.shape), "preprocessing": tensor_stats(reconstructed, actual.cpu())}
