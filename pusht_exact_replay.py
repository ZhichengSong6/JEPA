#!/usr/bin/env python3
"""Exact dataset-conditioned PushT replay helpers for diagnostics.

The official stable-worldmodel dataset evaluation reconstructs each episode from
its dataset seed and any stored variation.* columns. Candidate replay must use
that same reset context for every candidate in a case; otherwise candidate
ranking is contaminated by changing physics/visual factors of variation.
"""

from __future__ import annotations

import numpy as np
import torch
import stable_worldmodel as swm


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _scalar(x):
    a = _np(x)
    if a.size == 1:
        return a.reshape(-1)[0].item()
    return a.copy()


def load_dataset_reset_contexts(dataset, eval_rows):
    """Return one immutable reset context per official evaluation row."""
    rows = dataset.get_row_data(np.asarray(eval_rows))
    columns = list(dataset.column_names)

    seeds = rows.get("seed", None)
    variation_cols = [c for c in columns if str(c).startswith("variation.")]

    contexts = []
    for i in range(len(eval_rows)):
        seed = None
        if seeds is not None:
            seed = int(_scalar(seeds[i]))

        values = {}
        for col in variation_cols:
            key = str(col).removeprefix("variation.")
            values[key] = _scalar(rows[col][i])

        if seed is None and not values:
            raise RuntimeError(
                "Exact replay impossible: dataset exposes neither a 'seed' "
                "column nor any 'variation.*' columns."
            )

        contexts.append({
            "seed": seed,
            "variation_names": list(values.keys()),
            "variation_values": values,
            "seed_available": seed is not None,
            "variation_count": len(values),
        })
    return contexts


def context_variations(context):
    return (
        list(context.get("variation_names", [])),
        dict(context.get("variation_values", {})),
    )


def reset_physical_exact(env, state, goal, context):
    """Reset to the dataset episode context, then overwrite state/goal."""
    names, vals = context_variations(context)
    options = {}
    if names:
        options["variation"] = names
        options["variation_values"] = vals

    # Match dataset-conditioned evaluation seed whenever it exists. If every
    # variation is explicitly pinned but the dataset has no seed, use a fixed
    # deterministic seed only for any residual simulator RNG.
    seed = context.get("seed", None)
    seed_arg = int(seed) if seed is not None else 0

    env.reset(seed=seed_arg, options=options if options else None)
    raw = env.unwrapped
    raw._set_goal_state(np.asarray(goal, dtype=np.float64))
    raw._set_state(np.asarray(state, dtype=np.float64))


def load_goal_images(dataset, episodes, start_steps, goal_offset):
    """Load the exact raw dataset goal frame used by dataset evaluation."""
    starts = np.asarray(start_steps)
    chunks = dataset.load_chunk(
        np.asarray(episodes),
        starts,
        starts + int(goal_offset) + 1,
    )
    images = []
    for ep in chunks:
        px = _np(ep["pixels"][-1])
        if px.ndim == 3 and px.shape[0] in (1, 3, 4):
            px = np.moveaxis(px, 0, -1)
        images.append(np.asarray(px))
    return images


def capture_live_reset_contexts(vector_env):
    """Snapshot the ACTUAL current variation values from live official envs.

    This is the authoritative fallback when the evaluation dataset has neither
    seed nor variation.* columns. It must be called only after the World has
    already reset to the episode(s), i.e. from the first solver call.
    """
    envs = getattr(vector_env, "envs", None)
    if envs is None:
        base = getattr(vector_env, "unwrapped", vector_env)
        envs = getattr(base, "envs", None)
    if envs is None:
        raise RuntimeError(
            "Cannot access live sub-environments for exact variation snapshot."
        )

    contexts = []
    for i, env in enumerate(envs):
        raw = env.unwrapped
        vspace = getattr(raw, "variation_space", None)
        if vspace is None:
            contexts.append({
                "seed": None,
                "variation_names": [],
                "variation_values": {},
                "seed_available": False,
                "variation_count": 0,
                "source": "live_env_no_variation_space",
            })
            continue

        names = list(vspace.names())
        values = {}
        for name in names:
            subspace = swm.utils.get_in(vspace, name.split("."))
            val = getattr(subspace, "value", None)
            if val is None:
                raise RuntimeError(
                    f"Live variation '{name}' for env {i} has no current value."
                )
            values[name] = _scalar(val)

        contexts.append({
            "seed": None,
            "variation_names": names,
            "variation_values": values,
            "seed_available": False,
            "variation_count": len(values),
            "source": "live_env_snapshot",
        })

    return contexts


def load_dataset_reset_contexts_optional(dataset, eval_rows):
    """Use dataset reset metadata when available, otherwise return None.

    For this PushT dataset, official evaluation may expose neither seed nor
    variation.* columns. In that case exact replay must snapshot the live env
    after official reset rather than inventing a deterministic seed.
    """
    rows = dataset.get_row_data(np.asarray(eval_rows))
    columns = list(dataset.column_names)
    has_seed = "seed" in rows
    variation_cols = [c for c in columns if str(c).startswith("variation.")]
    if not has_seed and not variation_cols:
        return None
    return load_dataset_reset_contexts(dataset, eval_rows)
