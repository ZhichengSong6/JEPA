#!/usr/bin/env python3
"""Prepare and run the stepwise-target PC-MH experiment.

Prepare reads the BASELINE RUN'S saved config.yaml, not repository defaults.
Run delegates to train_pc_mh_ald.run.__wrapped__ with only the forward function
and an audited export/logging callback replaced IN THIS PROCESS. No upstream
file is modified; the original data pipeline, optimizer and scheduler execute.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import operator
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASE_COMMIT = "851b60c06ee0aef6927d59f85763cb03a3aede34"
EXPECTED_GIT_BLOBS = {
    "planner_context_mh.py": "017e3c6ec5fdcd7149367097b9ef0c204b14b9b6",
    "train_pc_mh_ald.py": "89f0505266cfd3ec610851c3e93ea11d066ced0e",
    "train_mh_ald.py": "94d31e21bf3134b2c6a6a9376245b508f04f5a03",
    "anchored_local_dynamics.py": "af5f2aa992ccc686d1abdbc86720dcf36a77d787",
    "stage1_bias_calibration.py": "e1e30b3ad01bb21d2353c0073a9681b9d60801a6",
    "utils.py": "eaaf24a4074c5f29344988d6e6d6ff9f0dd082a2",
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_upstream(repo=REPO):
    result = {}
    for name, expected in EXPECTED_GIT_BLOBS.items():
        data = (Path(repo) / name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Upstream source changed: {name}; expected Git blob {expected}, got {actual}. "
                "Do not auto-reset the working tree. Reconcile the source before this experiment."
            )
        result[name] = actual
    return result


def _arithmetic(expression):
    """Small numeric-only replacement for the project's eval interpolation."""
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod}
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = walk(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        raise ValueError(f"Unsupported configuration expression: {expression!r}")
    return walk(ast.parse(str(expression), mode="eval"))


def _omega():
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", _arithmetic)
    return OmegaConf


def path_in_root(root, relative):
    relative = Path(str(relative))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Output subdir must be a nonempty relative path without '..'.")
    root = Path(root).expanduser().resolve()
    result = (root / relative).resolve()
    if result == root or root not in result.parents:
        raise ValueError("Output must be strictly inside STABLEWM_HOME.")
    return result


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for k in sorted(a.keys() | b.keys()):
            p = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                result.append({"key": p, "before": a.get(k), "after": b.get(k)})
            else:
                result.extend(_diff(a[k], b[k], p))
        return result
    if a != b:
        return [{"key": path, "before": a, "after": b}]
    return []


def derive_configuration(base_config, mode, tag):
    """Pure configuration transformation; formal science settings stay fixed."""
    OmegaConf = _omega()
    # Resolve before renaming outputs so interpolated wandb IDs can be explicitly
    # reassigned, while numeric training settings remain identical.
    original = OmegaConf.to_container(base_config, resolve=True)
    cfg = OmegaConf.create(original)
    if mode not in ("smoke", "formal"):
        raise ValueError("mode must be smoke or formal.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("tag may contain only letters, numbers, '_' and '-'.")
    if not bool(cfg.mh_ald.enabled) or not bool(cfg.mh_ald.planner_context.enabled):
        raise ValueError("Source config must be an enabled PC-MH run.")
    if str(cfg.mh_ald.init_policy) != str(cfg.mh_ald.teacher_policy):
        raise ValueError("This experiment requires identical initial student/teacher policy IDs.")
    if int(cfg.wm.num_preds) != 1 or int(cfg.wm.history_size) != 3:
        raise ValueError("This release targets the existing history=3 / num_preds=1 protocol.")
    if int(cfg.mh_ald.planner_context.observation_history) != 1:
        raise ValueError("Expected one initial observation.")
    if int(cfg.mh_ald.planner_context.predictor_max_history) != 3:
        raise ValueError("Expected predictor max history 3.")
    if int(cfg.data.dataset.num_steps) != 3 + int(cfg.mh_ald.rollout_horizon):
        raise ValueError("Source dataset length does not match history + rollout horizon.")
    if not set(cfg.data.dataset.keys_to_load) <= {"pixels", "action", "proprio"}:
        raise ValueError("Do not add state/reward/other supervision for this controlled experiment.")
    for name in ("cem_aligned", "goal_sensitive", "goal_coupled"):
        if bool(OmegaConf.select(cfg, f"mh_ald.{name}.enabled", default=False)):
            raise ValueError(f"Source is not pure PC-MH: mh_ald.{name}.enabled.")
    devices = cfg.trainer.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)
    if type(devices) is not int or devices < 1:
        raise ValueError(
            "The saved baseline config must specify trainer.devices as a positive integer. "
            "If it says 'auto', establish the old run's actual GPU count first; this script "
            "will not infer it from a filename or silently change global batch size."
        )
    if int(cfg.trainer.get("num_nodes", 1)) != 1:
        raise ValueError("This launcher supports the existing single-node experiment only.")
    if bool(cfg.trainer.get("sync_batchnorm", False)):
        raise ValueError("Expected the baseline PC-MH inference-semantics / no-SyncBN setup.")
    stem = str(cfg.output_model_name)
    if Path(stem).name != stem or not stem:
        raise ValueError("Invalid baseline output_model_name.")
    suffix = "_stepwise_" + tag + ("_smoke" if mode == "smoke" else "")
    cfg.subdir = str(cfg.subdir) + suffix
    cfg.output_model_name = stem + "_stepwise" + ("_smoke" if mode == "smoke" else "")
    if bool(cfg.wandb.enabled):
        for key in ("id", "name"):
            if key in cfg.wandb.config:
                cfg.wandb.config[key] = str(cfg.subdir if key == "id" else cfg.output_model_name)
        # Never attach this experiment to an old wandb run.
        cfg.wandb.config.resume = "never"
    if mode == "smoke":
        cfg.trainer.max_epochs = 1
        cfg.trainer.max_steps = 2
        cfg.trainer.limit_train_batches = 2
        cfg.trainer.limit_val_batches = 1
    changed = _diff(original, OmegaConf.to_container(cfg, resolve=True))
    allowed = {"subdir", "output_model_name", "wandb.config.id", "wandb.config.name",
               "wandb.config.resume"}
    if mode == "smoke":
        allowed |= {"trainer.max_epochs", "trainer.max_steps", "trainer.limit_train_batches",
                    "trainer.limit_val_batches"}
    unexpected = [r for r in changed if r["key"] not in allowed]
    if unexpected:
        raise RuntimeError(f"Unintended config changes: {unexpected}")
    return cfg, changed, devices


def _json_write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def prepare(args):
    OmegaConf = _omega()
    if not os.environ.get("STABLEWM_HOME"):
        raise ValueError("Set STABLEWM_HOME to the existing project data directory.")
    root = Path(os.environ["STABLEWM_HOME"]).resolve()
    base_path = Path(args.base_config).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Missing baseline RUN config: {base_path}. Do not use repository defaults as a substitute."
        )
    source_hashes = verify_upstream()
    base = OmegaConf.load(base_path)
    cfg, changed, devices = derive_configuration(base, args.mode, args.tag)
    run_dir = path_in_root(root, cfg.subdir)
    if run_dir == base_path.parent:
        raise ValueError("Refusing to write to the baseline run directory.")
    run_dir.mkdir(parents=True, exist_ok=False)
    # These are NEW-run artifacts, not edits to the baseline or upstream repo.
    (run_dir / "baseline_config.yaml").write_bytes(base_path.read_bytes())
    OmegaConf.save(cfg, run_dir / "launch_config.yaml")
    (run_dir / "runtime").mkdir()
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"
    payload = {
        "method": "stepwise_observation_anchored_pc_mh_v1",
        "mode": args.mode, "baseline_config": str(base_path),
        "baseline_config_sha256": sha256_file(base_path),
        "launch_config_sha256": sha256_file(run_dir / "launch_config.yaml"),
        "run_dir": str(run_dir), "repo": str(REPO), "code_commit": commit,
        "reference_commit": BASE_COMMIT, "upstream_git_blobs": source_hashes,
        "new_source_sha256": {p.name: sha256_file(p) for p in (
            HERE / "planner_context_stepwise_mh.py", Path(__file__),
        )},
        "configuration_changes": changed,
        "init_policy": str(cfg.mh_ald.init_policy),
        "teacher_policy": str(cfg.mh_ald.teacher_policy),
        "expected_world_size": devices, "num_nodes": 1,
        "local_batch": int(cfg.loader.batch_size),
        "global_batch": devices * int(cfg.loader.batch_size),
        "max_epochs": int(cfg.trainer.max_epochs),
        "num_workers_per_rank": int(cfg.loader.num_workers),
        "output_model_name": str(cfg.output_model_name),
        "training_targets": "model-derived, not simulator counterfactual ground truth",
        "inference": "unmodified JEPA / official planner / raw latent cost",
    }
    _json_write_new(run_dir / "execution.json", payload)
    print(f"Prepared {args.mode}: {run_dir}", file=sys.stderr)
    print(f"Parent={payload['init_policy']}; teacher={payload['teacher_policy']}; "
          f"GPUs={devices}; local/global batch={payload['local_batch']}/{payload['global_batch']}",
          file=sys.stderr)
    print("Config changes: " + ", ".join(x["key"] for x in changed), file=sys.stderr)
    print(run_dir / "execution.json")


def _rank():
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return int(os.environ[key])
    return 0


def _state_digest(state):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor state entry is unsupported: {name}")
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _visual_digest(model):
    return _state_digest({
        **{"encoder." + k: v for k, v in model.encoder.state_dict().items()},
        **{"projector." + k: v for k, v in model.projector.state_dict().items()},
    })


def _finite_state(model):
    import torch
    bad = [k for k, v in model.state_dict().items()
           if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex())
           and not bool(torch.isfinite(v).all())]
    if bad:
        raise RuntimeError(f"Nonfinite model tensors: {bad[:8]}")


def run_prepared(args):
    import torch
    from omegaconf import OmegaConf
    execution_path = Path(args.execution).resolve()
    metadata = json.loads(execution_path.read_text())
    run_dir = execution_path.parent
    if Path(metadata["repo"]).resolve() != REPO or Path(metadata["run_dir"]).resolve() != run_dir:
        raise RuntimeError("Prepared repo/run paths do not match this installation.")
    root = Path(os.environ["STABLEWM_HOME"]).resolve()
    if root not in run_dir.parents:
        raise ValueError("Run directory must be inside STABLEWM_HOME.")
    verify_upstream()
    for name, expected in metadata["new_source_sha256"].items():
        if sha256_file(HERE / name) != expected:
            raise RuntimeError(f"New source changed after preparation: {name}")
    config_path = run_dir / "launch_config.yaml"
    if sha256_file(config_path) != metadata["launch_config_sha256"]:
        raise RuntimeError("Prepared configuration changed after preparation.")
    if sha256_file(metadata["baseline_config"]) != metadata["baseline_config_sha256"]:
        raise RuntimeError("Baseline saved configuration changed after preparation.")
    # srun starts one Python process per GPU, as in the original MH-ALD launcher.
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use the supplied Slurm/srun launcher; CPU checks are in the test script.")
    if int(os.environ.get("SLURM_NTASKS", "1")) != metadata["expected_world_size"]:
        raise RuntimeError("SLURM_NTASKS must equal the baseline's GPU/world size.")
    marker = run_dir / "started.json"
    job_id = os.environ["SLURM_JOB_ID"]
    if _rank() == 0:
        if list(run_dir.rglob("*.ckpt")):
            raise FileExistsError("Fresh run refuses existing checkpoints; no auto-resume.")
        _json_write_new(marker, {"slurm_job_id": job_id, "start_unix": time.time()})
    else:
        deadline = time.monotonic() + 60
        while not marker.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Rank zero did not create the run-start marker.")
            time.sleep(0.1)
        # A shared filesystem may expose the name just before the short JSON write finishes.
        while True:
            try:
                started = json.loads(marker.read_text())
                break
            except json.JSONDecodeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        if started["slurm_job_id"] != job_id:
            raise RuntimeError("Another job already started this output directory.")
    cfg = OmegaConf.load(config_path)
    import train_pc_mh_ald as baseline
    from planner_context_stepwise_mh import pc_stepwise_mh_forward

    source_loader = baseline._load_policy_model
    loaded = []
    def audited_loader(policy):
        model = source_loader(policy)
        for attr in ("coordinate_adapter", "factor_heads", "inverse_dynamics_head"):
            if getattr(model, attr, None) is not None:
                raise RuntimeError(f"Unexpected extra inference module: {attr}")
        _finite_state(model)
        digest = _state_digest(model.state_dict())
        if loaded and digest != loaded[0]:
            raise RuntimeError("Initial student and teacher parameters/buffers differ.")
        loaded.append(digest)
        if _rank() == 0:
            print(f"PARENT STATE CHECK: {policy} sha256={digest}", flush=True)
        return model

    callback_base = baseline.ModelObjectCallBack
    class CheckedExport(callback_base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.checked_backward = False
            self.visual_start = None
            self.teacher_start = None
            self.trainer_seen = None
            self.last_model = None

        def on_fit_start(self, trainer, pl_module):
            self.trainer_seen = trainer
            self.last_model = pl_module.model
            if int(trainer.world_size) != metadata["expected_world_size"]:
                raise RuntimeError("Actual world size differs from the original PC-MH run.")
            for group in ("encoder", "projector"):
                if any(p.requires_grad for p in getattr(pl_module.model, group).parameters()):
                    raise RuntimeError(f"Visual module is trainable: {group}")
            if any(p.requires_grad for p in pl_module.teacher_model.parameters()):
                raise RuntimeError("Teacher must be fully frozen.")
            self.visual_start = _visual_digest(pl_module.model)
            self.teacher_start = _state_digest(pl_module.teacher_model.state_dict())
            if trainer.is_global_zero:
                _json_write_new(run_dir / "fit_audit.json", {
                    "world_size": int(trainer.world_size), "local_batch": metadata["local_batch"],
                    "global_batch": metadata["global_batch"], "parent_state_sha256": loaded[0],
                    "visual_state_sha256": self.visual_start, "teacher_state_sha256": self.teacher_start,
                    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                    "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
                    "precision": str(cfg.trainer.precision),
                })
                print("FIT AUDIT PASS: fixed visual frame; identical frozen teacher; "
                      f"world_size={trainer.world_size}", flush=True)

        def on_after_backward(self, trainer, pl_module):
            if self.checked_backward:
                return
            for name in ("action_encoder", "predictor", "pred_proj"):
                ps = list(getattr(pl_module.model, name).parameters())
                gs = [p.grad for p in ps if p.grad is not None]
                if not ps or not all(p.requires_grad for p in ps) or not gs:
                    raise RuntimeError(f"Missing predictor-side gradient: {name}")
                if not all(bool(torch.isfinite(g).all()) for g in gs):
                    raise RuntimeError(f"Nonfinite gradient: {name}")
            for frozen in (pl_module.model.encoder, pl_module.model.projector, pl_module.teacher_model):
                if any(p.grad is not None for p in frozen.parameters()):
                    raise RuntimeError("Gradient leaked into visual frame or teacher.")
            self.checked_backward = True
            if trainer.is_global_zero:
                print("FIRST BACKWARD PASS: predictor-side gradients only", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss_value = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if torch.is_tensor(loss_value) and not bool(torch.isfinite(loss_value).all()):
                raise RuntimeError("Nonfinite loss; abort instead of saving a corrupt model.")
            if trainer.is_global_zero and (batch_idx == 0 or (batch_idx + 1) % 200 == 0):
                keys = ("train/loss", "train/tf_loss", "train/pc_stepwise_rollout_loss",
                        "train/pc_stepwise_mh_loss", "train/pc_stepwise_half_response_loss")
                row = {"epoch": int(trainer.current_epoch) + 1, "global_step": int(trainer.global_step)}
                for k in keys:
                    v = trainer.callback_metrics.get(k)
                    row[k] = float(v.detach().cpu()) if torch.is_tensor(v) and v.numel() == 1 else ""
                path = run_dir / "training_progress.csv"
                exists = path.exists()
                with path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
                print("STEPWISE TRAIN: " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)

        def on_train_epoch_end(self, trainer, pl_module):
            if _visual_digest(pl_module.model) != self.visual_start:
                raise RuntimeError("Frozen visual frame changed.")
            if _state_digest(pl_module.teacher_model.state_dict()) != self.teacher_start:
                raise RuntimeError("Frozen teacher changed.")
            # The upstream callback may save twice at its final epoch. Export
            # once here, atomically, and propagate all save failures.
            if trainer.is_global_zero:
                epoch = int(trainer.current_epoch) + 1
                if epoch % self.epoch_interval == 0 or epoch == int(trainer.max_epochs):
                    self._dump_model(
                        pl_module.model,
                        self.dirpath / f"{self.filename}_epoch_{epoch}_object.ckpt",
                    )

        def _dump_model(self, model, path):
            _finite_state(model)
            for attr in ("coordinate_adapter", "factor_heads", "inverse_dynamics_head"):
                if getattr(model, attr, None) is not None:
                    raise RuntimeError(f"Unexpected module in inference export: {attr}")
            path = Path(path)
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite an inference object: {path}")
            partial = path.with_name(path.name + f".{os.getpid()}.partial")
            try:
                torch.save(model, partial)
                os.replace(partial, path)
            finally:
                if partial.exists():
                    partial.unlink()
            print(f"SAVED PLAIN JEPA: {path}", flush=True)

    original_forward, original_callback = baseline.pc_mh_ald_forward, baseline.ModelObjectCallBack
    baseline.pc_mh_ald_forward = pc_stepwise_mh_forward
    baseline.ModelObjectCallBack = CheckedExport
    baseline._load_policy_model = audited_loader
    try:
        inspect.unwrap(baseline.run)(cfg)
        if _rank() == 0:
            expected_epoch = int(cfg.trainer.max_epochs)
            final_path = run_dir / f"{cfg.output_model_name}_epoch_{expected_epoch}_object.ckpt"
            if not final_path.is_file():
                raise RuntimeError(f"Training returned without expected final inference object: {final_path}")
            _json_write_new(run_dir / "training_complete.json", {
                "mode": metadata["mode"], "final_checkpoint": str(final_path),
                "policy": str(Path(str(cfg.subdir)) / f"{cfg.output_model_name}_epoch_{expected_epoch}"),
                "final_checkpoint_sha256": sha256_file(final_path), "finish_unix": time.time(),
                "expected_world_size": metadata["expected_world_size"],
                "note": "Execution success, not evidence of planning improvement.",
            })
            print(f"STEPWISE PC-MH {metadata['mode'].upper()} PASS: {final_path}", flush=True)
    finally:
        baseline.pc_mh_ald_forward = original_forward
        baseline.ModelObjectCallBack = original_callback
        baseline._load_policy_model = source_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare", help="Derive exact settings from a saved baseline RUN config.")
    p.add_argument("--base-config", required=True)
    p.add_argument("--mode", choices=("smoke", "formal"), required=True)
    p.add_argument("--tag", default="v1")
    r = subs.add_parser("run", help="Run inside the allocated Slurm job.")
    r.add_argument("--execution", required=True)
    args = parser.parse_args()
    try:
        prepare(args) if args.command == "prepare" else run_prepared(args)
    except Exception as exc:
        print(f"STEPWISE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
