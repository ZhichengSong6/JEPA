import json
import os
import time
from pathlib import Path
from types import MethodType

os.environ["MUJOCO_GL"] = "egl"

import hydra
import numpy as np
import stable_pretraining as spt  # noqa: F401 - keeps environment identical to eval.py
import stable_worldmodel as swm
import torch  # noqa: F401 - checkpoint/runtime dependency
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

from eval import get_dataset, get_episodes_length, img_transform


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _patch_pusht_eval_state(world, theta_threshold_rad: float, pos_threshold_px: float = 20.0):
    """Patch only PushT's success/termination criterion.

    Planning, model cost, CEM, dynamics, reward-distance definition, and all
    other environment behavior are unchanged.  PushT.step() calls eval_state(),
    so replacing this method changes the actual termination criterion during
    rollout instead of merely re-scoring the final state afterward.
    """

    env_list = getattr(getattr(world.envs, "unwrapped", None), "envs", None)
    if env_list is None:
        env_list = getattr(world.envs, "envs", None)
    if env_list is None:
        raise RuntimeError("Could not locate individual environments inside world.envs")

    patched = 0
    for wrapped_env in env_list:
        env = wrapped_env.unwrapped
        if not hasattr(env, "eval_state"):
            raise RuntimeError(f"Environment {type(env)} has no eval_state() method")

        def strict_eval_state(self, goal_state, cur_state):
            goal_state = np.asarray(goal_state)
            cur_state = np.asarray(cur_state)

            # Keep the official PushT position criterion: one L2 norm over
            # [pusher_x, pusher_y, block_x, block_y].
            pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])

            # Circular angle difference in [0, pi].
            delta = goal_state[4] - cur_state[4]
            angle_diff = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))

            success = bool(
                (pos_diff < pos_threshold_px)
                and (angle_diff < theta_threshold_rad)
            )

            # Preserve the original reward-distance definition.
            state_dist = np.linalg.norm(goal_state - cur_state)
            return success, state_dist

        env.eval_state = MethodType(strict_eval_state, env)
        patched += 1

    return patched


def _prepare_eval(cfg, world, dataset):
    """Copy the official eval.py sampling logic exactly."""
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }
    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )
    world.set_policy(policy)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    # IMPORTANT: preserve the exact slightly-unusual sampling expression from
    # the official eval.py so seed=42 reproduces the same 50 starts.
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1,
        size=cfg.eval.num_eval,
        replace=False,
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    print(random_episode_indices)

    rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = rows[col_name]
    eval_start_idx = rows["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    return random_episode_indices, eval_episodes, eval_start_idx


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    if cfg.policy == "random":
        raise ValueError("This diagnostic expects a trained world-model policy, not random.")

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    theta_deg = float(cfg.get("strict_theta_deg", 10.0))
    theta_rad = np.deg2rad(theta_deg)
    pos_threshold_px = float(cfg.get("strict_pos_threshold_px", 20.0))
    save_video = bool(cfg.get("save_video", False))

    # Same world construction as official eval.py.
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    patched = _patch_pusht_eval_state(
        world,
        theta_threshold_rad=theta_rad,
        pos_threshold_px=pos_threshold_px,
    )
    print(
        f"Patched {patched} PushT envs: pos < {pos_threshold_px:.3f}px, "
        f"theta < {theta_deg:.3f}deg ({theta_rad:.6f} rad)"
    )

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    random_rows, eval_episodes, eval_start_idx = _prepare_eval(cfg, world, dataset)

    results_root = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    theta_label = (f"{theta_deg:g}").replace(".", "p")
    video_path = results_root / f"strict_theta_{theta_label}deg_videos"

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        save_video=save_video,
        video_path=video_path,
    )
    elapsed = time.time() - start_time

    print(metrics)
    print(f"Strict-theta evaluation time: {elapsed:.3f} seconds")

    payload = {
        "policy": str(cfg.policy),
        "theta_threshold_deg": theta_deg,
        "theta_threshold_rad": theta_rad,
        "position_threshold_px": pos_threshold_px,
        "seed": int(cfg.seed),
        "dataset_rows": random_rows,
        "episode_idx": eval_episodes,
        "start_step": eval_start_idx,
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        "eval_budget": int(cfg.eval.eval_budget),
        "metrics": metrics,
        "evaluation_time_seconds": elapsed,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    json_path = results_root / f"pusht_strict_theta_{theta_label}deg.json"
    txt_path = results_root / f"pusht_strict_theta_{theta_label}deg.txt"
    results_root.mkdir(parents=True, exist_ok=True)

    with json_path.open("w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    with txt_path.open("w") as f:
        f.write("==== STRICT PUSHT EVALUATION ====\n")
        f.write(f"policy: {cfg.policy}\n")
        f.write(f"position_threshold_px: {pos_threshold_px}\n")
        f.write(f"theta_threshold_deg: {theta_deg}\n")
        f.write(f"theta_threshold_rad: {theta_rad}\n")
        f.write(f"dataset_rows: {random_rows.tolist()}\n")
        f.write(f"episode_idx: {np.asarray(eval_episodes).tolist()}\n")
        f.write(f"start_step: {np.asarray(eval_start_idx).tolist()}\n")
        f.write(f"metrics: {_to_jsonable(metrics)}\n")
        f.write(f"evaluation_time_seconds: {elapsed}\n")
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    run()
