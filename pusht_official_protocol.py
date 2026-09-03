"""Small, NumPy-only protocol checks for the official PushT diagnostic."""

import math

import numpy as np


def validate_protocol(spec, mode):
    """A formal run cannot silently inherit the historical B=3000 protocol."""
    fixed = {
        "seed": 42, "batch_size": 1, "var_scale": 1.0,
        "history_size": 1, "frame_skip": 1, "horizon": 5,
        "receding_horizon": 5, "action_block": 5,
        "eval_budget": 50, "goal_offset_steps": 25, "img_size": 224,
        "env_name": "swm/PushT-v1", "dataset_name": "pusht_expert_train",
    }
    if mode == "formal":
        fixed.update(num_samples=300, n_steps=30, topk=30, num_eval=100)
        expected_iterations = [0, 3, 9, 19, 29]
    elif mode == "smoke":
        fixed.update(num_samples=30, n_steps=2, topk=3, num_eval=6)
        expected_iterations = [0, 1]
    else:
        raise ValueError("mode must be smoke or formal")
    errors = [f"{k}={spec.get(k)!r}, required {v!r}"
              for k, v in fixed.items() if spec.get(k) != v]
    if list(spec.get("replay_iterations", [])) != expected_iterations:
        errors.append(f"replay_iterations must be {expected_iterations}")
    if errors:
        raise ValueError("Protocol mismatch: " + "; ".join(errors))


def paired_outcomes(lewm, ald):
    lewm = np.asarray(lewm, dtype=bool)
    ald = np.asarray(ald, dtype=bool)
    if lewm.ndim != 1 or ald.shape != lewm.shape or not len(lewm):
        raise ValueError("Paired outcomes need equal, nonempty 1-D arrays")
    labels = [
        "both_success" if a and b else
        "lewm_fail_ald_success" if b else
        "lewm_success_ald_fail" if a else "both_fail"
        for a, b in zip(lewm, ald)
    ]
    counts = {k: labels.count(k) for k in (
        "both_success", "lewm_fail_ald_success", "both_fail", "lewm_success_ald_fail"
    )}
    rescue, regression = counts["lewm_fail_ald_success"], counts["lewm_success_ald_fail"]
    discordant = rescue + regression
    # Exact two-sided paired test; populations are not independent trials.
    p = min(1.0, 2 * sum(math.comb(discordant, i)
                         for i in range(min(rescue, regression) + 1)) / 2**discordant)
    return labels, counts, {
        "lewm_success_count": int(lewm.sum()),
        "ald_tf_success_count": int(ald.sum()),
        "num_eval": len(lewm),
        "lewm_success_percent": float(100 * lewm.mean()),
        "ald_tf_success_percent": float(100 * ald.mean()),
        "ald_minus_lewm_percentage_points": float(100 * (ald.mean() - lewm.mean())),
        "mcnemar_exact_two_sided_p": p,
    }


def matched_factor_accuracy(score, target, controls, match_frac=0.10, diff_frac=0.10):
    """Rank one factor while matching *both* other factor costs.

    This is an observational matched-cost check, not a causal intervention.
    Zero target spread is uninformative; it must not be counted as random.
    """
    score, target = np.asarray(score), np.asarray(target)
    controls = np.asarray(controls)
    if controls.ndim == 1:
        controls = controls[:, None]
    mask = np.isfinite(score) & np.isfinite(target) & np.isfinite(controls).all(axis=1)
    score, target, controls = score[mask], target[mask], controls[mask]
    if len(target) < 3:
        return float("nan"), 0
    spread = float(np.diff(np.percentile(target, [25, 75]))[0])
    if spread <= 1e-8:
        return float("nan"), 0
    control_spread = np.diff(np.percentile(controls, [25, 75], axis=0), axis=0)[0]
    i, j = np.triu_indices(len(score), 1)
    matched = (np.abs(controls[i] - controls[j]) <=
               np.maximum(match_frac * control_spread, 1e-8)).all(axis=1)
    dt, ds = target[i] - target[j], score[i] - score[j]
    matched &= np.abs(dt) > max(diff_frac * spread, 1e-8)
    dt, ds = dt[matched], ds[matched]
    if not len(dt):
        return float("nan"), 0
    correct = np.where(np.abs(ds) <= 1e-12, 0.5, np.sign(ds) == np.sign(dt))
    return float(correct.mean()), len(dt)


def finite_summary(values):
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return {"count": len(x), "mean": float(x.mean()) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None}
