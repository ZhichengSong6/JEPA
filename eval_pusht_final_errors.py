import csv
import json
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import hydra
import numpy as np
import stable_pretraining as spt  # noqa: F401 - keeps environment identical to eval.py
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

from eval import get_dataset, get_episodes_length, img_transform


def _numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _latest_state(state_array):
    """Convert World state info to shape [N, 7]."""
    x = _numpy(state_array)
    if x.ndim == 3:
        x = x[:, -1]
    if x.ndim != 2 or x.shape[-1] < 5:
        raise RuntimeError(f"Unexpected state shape: {x.shape}")
    return np.asarray(x, dtype=np.float64)


def _angle_error_rad(a, b):
    delta = np.asarray(a) - np.asarray(b)
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def _error_components(states, goals):
    states = np.asarray(states, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    return {
        "pusher_xy": np.linalg.norm(states[:, :2] - goals[:, :2], axis=1),
        "block_xy": np.linalg.norm(states[:, 2:4] - goals[:, 2:4], axis=1),
        # This is the position term used by the official PushT success test.
        "joint_pos": np.linalg.norm(states[:, :4] - goals[:, :4], axis=1),
        "theta_rad": _angle_error_rad(states[:, 4], goals[:, 4]),
    }


def _summary(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }


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


def _prepare_eval(cfg, world, dataset):
    """Copy the official eval.py sampling/model setup exactly."""
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

    return col_name, random_episode_indices, eval_episodes, eval_start_idx


def _load_start_goal_states(dataset, eval_episodes, eval_start_idx, goal_offset_steps):
    ep_idx_arr = np.asarray(eval_episodes)
    start_arr = np.asarray(eval_start_idx)
    end_arr = start_arr + int(goal_offset_steps)
    chunks = dataset.load_chunk(ep_idx_arr, start_arr, end_arr)

    starts = []
    goals = []
    for ep in chunks:
        s = _numpy(ep["state"])
        starts.append(np.asarray(s[0], dtype=np.float64))
        goals.append(np.asarray(s[-1], dtype=np.float64))
    return np.stack(starts), np.stack(goals)


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    if cfg.policy == "random":
        raise ValueError("This diagnostic expects a trained world-model policy, not random.")

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    save_video = bool(cfg.get("save_video", False))

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    col_name, random_rows, eval_episodes, eval_start_idx = _prepare_eval(
        cfg, world, dataset
    )
    start_states, goal_states = _load_start_goal_states(
        dataset,
        eval_episodes,
        eval_start_idx,
        cfg.eval.goal_offset_steps,
    )

    n = len(eval_episodes)
    first_success_step = np.full(n, -1, dtype=np.int64)
    first_success_states = np.full((n, goal_states.shape[1]), np.nan, dtype=np.float64)

    # Hook World.step() only to observe raw simulator states.  It does not
    # change the action, planner, CEM cost, or environment transition.
    original_step = world.step
    step_counter = 0

    def recording_step():
        nonlocal step_counter
        original_step()
        step_counter += 1
        states = _latest_state(world.infos["state"])
        terminated = np.asarray(world.terminateds, dtype=bool)
        newly_successful = terminated & (first_success_step < 0)
        if np.any(newly_successful):
            first_success_step[newly_successful] = step_counter
            first_success_states[newly_successful] = states[newly_successful]

    world.step = recording_step

    results_root = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    video_path = results_root / "final_error_eval_videos"

    start_time = time.time()
    try:
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
    finally:
        world.step = original_step
    elapsed = time.time() - start_time

    final_states = _latest_state(world.infos["state"])
    final_err = _error_components(final_states, goal_states)
    first_err = _error_components(first_success_states, goal_states)

    theta_deg = np.rad2deg(final_err["theta_rad"])
    first_theta_deg = np.rad2deg(first_err["theta_rad"])
    official_success = np.asarray(metrics["episode_successes"], dtype=bool)

    # These two are explicitly FINAL-STATE criteria, not rollout success.
    final_within_official = (final_err["joint_pos"] < 20.0) & (
        final_err["theta_rad"] < np.pi / 9
    )
    final_within_10deg = (final_err["joint_pos"] < 20.0) & (
        final_err["theta_rad"] < np.pi / 18
    )

    rows = []
    for i in range(n):
        row = {
            "eval_index": i + 1,
            "dataset_row": int(random_rows[i]),
            "episode_idx": int(eval_episodes[i]),
            "start_step": int(eval_start_idx[i]),
            "official_rollout_success": bool(official_success[i]),
            "first_success_step": int(first_success_step[i]),
            "final_pusher_xy_error_px": float(final_err["pusher_xy"][i]),
            "final_block_xy_error_px": float(final_err["block_xy"][i]),
            "final_joint_position_error_px": float(final_err["joint_pos"][i]),
            "final_theta_error_deg": float(theta_deg[i]),
            "final_within_official_20deg": bool(final_within_official[i]),
            "final_within_10deg": bool(final_within_10deg[i]),
            "first_success_pusher_xy_error_px": float(first_err["pusher_xy"][i]),
            "first_success_block_xy_error_px": float(first_err["block_xy"][i]),
            "first_success_joint_position_error_px": float(first_err["joint_pos"][i]),
            "first_success_theta_error_deg": float(first_theta_deg[i]),
            "start_pusher_x": float(start_states[i, 0]),
            "start_pusher_y": float(start_states[i, 1]),
            "start_block_x": float(start_states[i, 2]),
            "start_block_y": float(start_states[i, 3]),
            "start_theta_rad": float(start_states[i, 4]),
            "goal_pusher_x": float(goal_states[i, 0]),
            "goal_pusher_y": float(goal_states[i, 1]),
            "goal_block_x": float(goal_states[i, 2]),
            "goal_block_y": float(goal_states[i, 3]),
            "goal_theta_rad": float(goal_states[i, 4]),
            "final_pusher_x": float(final_states[i, 0]),
            "final_pusher_y": float(final_states[i, 1]),
            "final_block_x": float(final_states[i, 2]),
            "final_block_y": float(final_states[i, 3]),
            "final_theta_rad": float(final_states[i, 4]),
        }
        rows.append(row)

    summary = {
        "policy": str(cfg.policy),
        "num_eval": n,
        "official_rollout_success_rate": float(metrics["success_rate"]),
        "final_within_official_20deg_rate": float(np.mean(final_within_official) * 100.0),
        "final_within_10deg_rate": float(np.mean(final_within_10deg) * 100.0),
        "final_pusher_xy_error_px": _summary(final_err["pusher_xy"]),
        "final_block_xy_error_px": _summary(final_err["block_xy"]),
        "final_joint_position_error_px": _summary(final_err["joint_pos"]),
        "final_theta_error_deg": _summary(theta_deg),
        "first_success_pusher_xy_error_px": _summary(first_err["pusher_xy"]),
        "first_success_block_xy_error_px": _summary(first_err["block_xy"]),
        "first_success_joint_position_error_px": _summary(first_err["joint_pos"]),
        "first_success_theta_error_deg": _summary(first_theta_deg),
        "evaluation_time_seconds": elapsed,
        "note": (
            "official_rollout_success is the environment's ever-terminated metric. "
            "final_* metrics are measured at the end of the full eval budget. "
            "final_within_10deg is only a final-state diagnostic and is NOT the strict-rollout metric."
        ),
    }

    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "pusht_final_errors.csv"
    json_path = results_root / "pusht_final_errors.json"
    txt_path = results_root / "pusht_final_errors.txt"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "summary": summary,
        "metrics": metrics,
        "dataset_rows": random_rows,
        "episode_idx": eval_episodes,
        "start_step": eval_start_idx,
        "episodes": rows,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with json_path.open("w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    with txt_path.open("w") as f:
        f.write("==== PUSHT FINAL PHYSICAL ERROR DIAGNOSTIC ====\n")
        f.write(json.dumps(_to_jsonable(summary), indent=2))
        f.write("\n\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))

    print(metrics)
    print(json.dumps(_to_jsonable(summary), indent=2))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    run()
