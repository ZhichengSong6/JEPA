"""Construction/export helpers for scratch LeWM + ordinary inverse dynamics.

No pretrained policy is loaded. The backbone and predictor construction order
matches train.py. The auxiliary head is added only after the base world model
has been constructed and does not consume its subsequent CPU RNG stream.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from inverse_dynamics import InverseDynamicsHead


def build_scratch_lewm(cfg, *, backbone_factory=None):
    """Build the ordinary LeWM inference graph with a random visual backbone."""
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    if backbone_factory is None:
        import stable_pretraining as spt
        backbone_factory = spt.backbone.utils.vit_hf

    encoder = backbone_factory(
        cfg.encoder_scale,
        patch_size=int(cfg.patch_size),
        image_size=int(cfg.img_size),
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = int(encoder.config.hidden_size)
    embed_dim = int(cfg.wm.embed_dim)
    action_dim = int(cfg.data.dataset.frameskip) * int(cfg.wm.action_dim)
    predictor = ARPredictor(
        num_frames=int(cfg.wm.history_size),
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=nn.BatchNorm1d,
    )
    return JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=pred_proj, factor_heads=None,
    )


def attach_inverse_head(model: nn.Module, *, latent_dim: int, action_dim: int,
                        hidden_dim: int, depth: int, seed: int) -> nn.Module:
    """Attach the existing ordinary MLP head without changing base weights/RNG.

    Construction happens on CPU, before Lightning transfers the model to GPUs.
    This is not a new IDM objective or a bounded/normalized inverse head.
    """
    if hasattr(model, "inverse_dynamics_head"):
        raise ValueError("An inverse_dynamics_head is already attached.")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Attach the IDM head before moving the model off CPU.")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        head = InverseDynamicsHead(
            latent_dim=int(latent_dim), action_dim=int(action_dim),
            hidden_dim=int(hidden_dim), depth=int(depth),
        )
    model.add_module("inverse_dynamics_head", head)
    return head


def checked_run_dir(cache_root: str | Path, subdir: str) -> Path:
    """Reject escapes from STABLEWM_HOME and accidental checkpoint resumes."""
    root = Path(cache_root).expanduser().resolve()
    relative = Path(str(subdir))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("subdir must be a nonempty relative path without '..'.")
    run_dir = (root / relative).resolve()
    if run_dir == root or root not in run_dir.parents:
        raise ValueError("Run directory must be strictly inside STABLEWM_HOME.")
    if run_dir.exists():
        if not run_dir.is_dir():
            raise ValueError(f"Run path is not a directory: {run_dir}")
        previous = list(run_dir.glob("*.ckpt")) + list(run_dir.glob("*.partial"))
        if previous:
            raise FileExistsError(
                f"Scratch run refuses existing checkpoints in {run_dir}. "
                "Use a new subdir; no automatic resume is performed."
            )
    return run_dir


def save_inference_object(model: nn.Module, path: str | Path) -> None:
    """Atomically save a plain inference object; restore training head on error.

    Both the temporary file and final file live in the model's project data
    directory. Save failures propagate to Slurm instead of reporting success.
    Only call this on rank zero between optimization steps.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "inverse_dynamics_head" not in model._modules:
        raise ValueError("Training model is missing its registered inverse head.")
    partial_path = path.with_name(path.name + f".{os.getpid()}.partial")
    head = model._modules.pop("inverse_dynamics_head")
    try:
        torch.save(model, partial_path)
        os.replace(partial_path, path)
    finally:
        model.add_module("inverse_dynamics_head", head)
        if partial_path.exists():
            partial_path.unlink()


def validate_policy(policy: str, *, device: str = "cpu", img_size: int = 224) -> dict:
    """Reload a trusted project checkpoint and exercise its encoder/predictor."""
    import stable_worldmodel as swm

    if not os.environ.get("STABLEWM_HOME"):
        raise ValueError("Set STABLEWM_HOME before checkpoint validation.")
    model = swm.policy.AutoCostModel(policy).eval()
    if hasattr(model, "inverse_dynamics_head"):
        raise RuntimeError("The training-only IDM head leaked into the inference object.")
    for optional in ("factor_heads", "coordinate_adapter"):
        if getattr(model, optional, None) is not None:
            raise RuntimeError(f"Unexpected inference module: {optional}")
    invalid = [name for name, value in model.state_dict().items()
               if value.is_floating_point() and not bool(torch.isfinite(value).all())]
    if invalid:
        raise RuntimeError(f"Nonfinite model parameters/buffers: {invalid}")
    model = model.to(device)
    action_dim = int(model.action_encoder.patch_embed.in_channels)
    with torch.no_grad():
        info = {
            "pixels": torch.zeros(2, 4, 3, img_size, img_size, device=device),
            "action": torch.zeros(2, 4, action_dim, device=device),
        }
        encoded = model.encode(info)
        predicted = model.predict(encoded["emb"][:, :3], encoded["act_emb"][:, :3])
        if predicted.shape != encoded["emb"][:, 1:4].shape:
            raise RuntimeError("Prediction shape does not match the three next-latent targets.")
        if not bool(torch.isfinite(encoded["emb"]).all() & torch.isfinite(predicted).all()):
            raise RuntimeError("Nonfinite outputs in checkpoint forward check.")
    return {
        "policy": policy, "model_type": type(model).__module__ + "." + type(model).__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "idm_head_present": False, "finite_state": True, "finite_forward": True,
        "action_dim": action_dim, "latent_dim": int(predicted.shape[-1]),
        "device": device,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate a scratch LeWM+IDM inference checkpoint.")
    parser.add_argument("--validate-policy", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()
    result = validate_policy(args.validate_policy, device=args.device, img_size=args.img_size)
    print(json.dumps(result, indent=2))
    print("INFERENCE VALIDATION PASSED")
