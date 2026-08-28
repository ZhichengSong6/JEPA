#!/usr/bin/env python3
"""Offline held-out endpoint evaluation for strict Bias-Only calibration."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from bias_only_calibration import (
    BiasOnlyWorldModel,
    frozen_visual_encode,
    planner_style_rollout_from_single,
)
from utils import get_column_normalizer, get_img_preprocessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-dir",
        default="outputs/bias_only_offline_eval",
    )
    return p.parse_args()


def _summary(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else None,
        "median": float(np.median(x)) if len(x) else None,
        "p10": float(np.percentile(x, 10)) if len(x) else None,
        "p90": float(np.percentile(x, 90)) if len(x) else None,
    }


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    dataset = swm.data.HDF5Dataset(
        num_steps=8,
        frameskip=5,
        name="pusht_expert_train",
        keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action", "proprio"],
        transform=None,
    )
    transforms = [
        get_img_preprocessor(
            source="pixels", target="pixels", img_size=a.img_size
        )
    ]
    for col in ["action", "proprio"]:
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd = torch.Generator().manual_seed(a.seed)
    _, val_set = spt.data.random_split(
        dataset,
        lengths=[0.9, 0.1],
        generator=rnd,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        persistent_workers=(a.num_workers > 0),
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device(a.device)
    model = swm.policy.AutoCostModel(a.policy).to(device).eval()
    model.requires_grad_(False)
    if not isinstance(model, BiasOnlyWorldModel):
        raise TypeError(
            f"Expected BiasOnlyWorldModel, got {type(model)} for {a.policy}"
        )

    rows = []
    with torch.inference_mode():
        for bi, batch in enumerate(val):
            if bi >= a.num_batches:
                break
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

            emb = frozen_visual_encode(model.base_model, batch["pixels"])
            current_index = model.history_size - 1
            horizon = 5
            goal_index = current_index + horizon

            current = emb[:, current_index]
            goal = emb[:, goal_index]
            actions = batch["action"][
                :, current_index : current_index + horizon
            ]
            teacher = planner_style_rollout_from_single(
                model.base_model,
                current,
                actions,
                history_size=model.history_size,
                horizon=horizon,
            )[:, -1]
            pred_bias = model.bias(current, teacher)
            corrected = teacher + pred_bias
            target_bias = goal - teacher

            teacher_mse = (teacher - goal).pow(2).mean(dim=-1)
            corrected_mse = (corrected - goal).pow(2).mean(dim=-1)
            pred_norm = torch.linalg.vector_norm(pred_bias, dim=-1)
            target_norm = torch.linalg.vector_norm(target_bias, dim=-1)
            cosine = torch.nn.functional.cosine_similarity(
                pred_bias, target_bias, dim=-1, eps=1e-8
            )

            for j in range(len(teacher_mse)):
                rows.append(
                    {
                        "teacher_endpoint_mse": float(
                            teacher_mse[j].cpu()
                        ),
                        "corrected_endpoint_mse": float(
                            corrected_mse[j].cpu()
                        ),
                        "mse_ratio": float(
                            corrected_mse[j].cpu()
                            / teacher_mse[j].clamp_min(1e-12).cpu()
                        ),
                        "pred_bias_norm": float(pred_norm[j].cpu()),
                        "target_bias_norm": float(target_norm[j].cpu()),
                        "bias_cosine": float(cosine[j].cpu()),
                    }
                )
            print(f"batch {bi+1}/{a.num_batches} n={len(rows)}")

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "policy": a.policy,
        "seed": a.seed,
        "num_samples": len(rows),
        "teacher_endpoint_mse": _summary(
            [r["teacher_endpoint_mse"] for r in rows]
        ),
        "corrected_endpoint_mse": _summary(
            [r["corrected_endpoint_mse"] for r in rows]
        ),
        "mse_ratio": _summary([r["mse_ratio"] for r in rows]),
        "pred_bias_norm": _summary(
            [r["pred_bias_norm"] for r in rows]
        ),
        "target_bias_norm": _summary(
            [r["target_bias_norm"] for r in rows]
        ),
        "bias_cosine": _summary([r["bias_cosine"] for r in rows]),
        "improved_fraction": float(
            np.mean(
                [
                    r["corrected_endpoint_mse"]
                    < r["teacher_endpoint_mse"]
                    for r in rows
                ]
            )
        ),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
