"""CPU regression tests; real backbone/DDP/data integration is tested by Slurm smoke."""
import ast
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

from inverse_dynamics import InverseDynamicsHead, inverse_dynamics_objective
from lewm_idm_scratch import attach_inverse_head, checked_run_dir, save_inference_object


ROOT = Path(__file__).resolve().parents[1]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.projector = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.predictor = nn.Linear(4, 4)
        self.action_encoder = nn.Linear(2, 4)
        self.pred_proj = nn.Linear(4, 4)
        self.register_buffer("running_state", torch.arange(4.0))

    def forward(self, x):
        return self.projector(self.encoder(x))


def attach(model, seed=3072):
    return attach_inverse_head(
        model, latent_dim=4, action_dim=2, hidden_dim=8, depth=2, seed=seed,
    )


class TestScratchIDM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        parent = ROOT / "outputs" / "unit_test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def test_head_does_not_change_base_weights_or_cpu_rng(self):
        model = TinyModel()
        original = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        head = attach(model)
        self.assertIsInstance(head, InverseDynamicsHead)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in original.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in head.modules()), 3)

    def test_head_seed_is_reproducible(self):
        first, second = TinyModel(), TinyModel()
        h1 = attach(first, seed=7)
        torch.rand(100)
        h2 = attach(second, seed=7)
        for p, q in zip(h1.parameters(), h2.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_joint_loss_reaches_all_six_groups(self):
        model = TinyModel()
        attach(model)
        z = model(torch.randn(2, 4, 3))
        actions = torch.randn(2, 4, 2)
        inverse = inverse_dynamics_objective(model.inverse_dynamics_head, z, actions)
        prediction = model.pred_proj(model.predictor(z[:, :3]) + model.action_encoder(actions[:, :3]))
        loss = (prediction - z[:, 1:4]).square().mean() + 0.1 * inverse["loss"]
        loss.backward()
        for name in ("encoder", "projector", "predictor", "action_encoder", "pred_proj", "inverse_dynamics_head"):
            grads = [p.grad for p in getattr(model, name).parameters()]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, name)

    def test_export_strips_only_head_and_preserves_predictions(self):
        model = TinyModel().eval()
        head = attach(model)
        keys = list(model._modules)
        original = copy.deepcopy(model.state_dict())
        checkpoint = self.path / "epoch1_object.ckpt"
        x = torch.randn(2, 3)
        expected = model(x).detach()
        save_inference_object(model, checkpoint)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
        self.assertFalse(hasattr(restored, "inverse_dynamics_head"))
        self.assertIs(model.inverse_dynamics_head, head)
        self.assertEqual(keys, list(model._modules))
        self.assertTrue(torch.equal(expected, restored(x)))
        for key, value in original.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
            if not key.startswith("inverse_dynamics_head."):
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
        self.assertFalse(list(self.path.glob("*.partial")))

    def test_export_failure_preserves_old_checkpoint_and_training_head(self):
        model = TinyModel()
        head = attach(model)
        path = self.path / "epoch1_object.ckpt"
        path.write_bytes(b"previous checkpoint")
        with patch("lewm_idm_scratch.torch.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_inference_object(model, path)
        self.assertIs(model.inverse_dynamics_head, head)
        self.assertEqual(path.read_bytes(), b"previous checkpoint")
        self.assertFalse(list(self.path.glob("*.partial")))

    def test_double_attach_and_export_without_head_are_errors(self):
        model = TinyModel()
        with self.assertRaises(ValueError):
            save_inference_object(model, self.path / "no_head.ckpt")
        attach(model)
        with self.assertRaises(ValueError):
            attach(model)

    def test_fresh_run_guard_and_path_boundaries(self):
        fresh = checked_run_dir(self.path, "new_run")
        self.assertEqual(fresh, self.path / "new_run")
        for bad in ("", ".", "../escape", str(self.path / "absolute")):
            with self.assertRaises(ValueError):
                checked_run_dir(self.path, bad)
        fresh.mkdir()
        (fresh / "model_weights.ckpt").write_bytes(b"old")
        with self.assertRaises(FileExistsError):
            checked_run_dir(self.path, "new_run")

    def test_formal_config_is_ordinary_scratch_ten_epochs(self):
        cfg = yaml.safe_load((ROOT / "config/train/lewm_idm_scratch.yaml").read_text())
        self.assertEqual(cfg["initialization"], "scratch")
        self.assertIsNone(cfg["idm"]["init_policy"])
        self.assertEqual(cfg["trainer"]["max_epochs"], 10)
        self.assertEqual(cfg["loader"]["batch_size"] * cfg["expected_world_size"], 128)
        self.assertEqual(cfg["idm"]["weight"], 0.1)
        self.assertEqual(cfg["loss"]["sigreg"]["weight"], 0.09)
        self.assertEqual(cfg["idm"]["head_depth"], 2)
        self.assertEqual(cfg["idm"]["head_hidden_dim"], 256)
        self.assertTrue(cfg["trainer"]["sync_batchnorm"])
        self.assertTrue(cfg["loss"]["sigreg"]["global_batch_ddp"])

    def test_builder_never_loads_pretrained_policy(self):
        tree = ast.parse((ROOT / "lewm_idm_scratch.py").read_text())
        builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "build_scratch_lewm")
        calls = [node for node in ast.walk(builder) if isinstance(node, ast.Call)]
        factories = [node for node in calls if isinstance(node.func, ast.Name)
                     and node.func.id == "backbone_factory"]
        self.assertEqual(len(factories), 1)
        flags = {kw.arg: ast.literal_eval(kw.value) for kw in factories[0].keywords
                 if kw.arg in ("pretrained", "use_mask_token")}
        self.assertEqual(flags, {"pretrained": False, "use_mask_token": False})
        for node in calls:
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            self.assertNotIn(name, ("load", "load_state_dict", "AutoCostModel", "from_pretrained"))


if __name__ == "__main__":
    unittest.main()
