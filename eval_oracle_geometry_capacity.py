#!/usr/bin/env python3
"""One-shot oracle geometry-capacity test for fixed LeWM latents.

This is intentionally a bounded diagnostic, not a new training pipeline.

Question
--------
Is the failure of raw Euclidean planning primarily:

  A) the *supervision* used to learn one global metric, or
  B) the *metric family* itself (one global quadratic geometry)?

We freeze the official LeWM representation and the exact authoritative CEM
populations.  Physical cost is used ONLY as an oracle diagnostic target.

We compare four capacities:

  1. Identity: current raw Euclidean metric.
  2. Global diagonal metric, learned on all OTHER cases and evaluated on the
     held-out case (leave-one-case-out; LOCO).
  3. Global full PSD metric, same LOCO protocol.
  4. Case-local full PSD metric: for each hard case, fit on a deterministic
     subset of that case's populations and evaluate on held-out populations
     from the SAME case.  This is a conservative test for state/case dependence.

If global full PSD works out-of-case, the global coordinate family is viable
and the previous controllability supervision was the problem.
If global full PSD fails but case-local full PSD works, state dependence is
strongly implicated.
If even case-local full PSD fails on held-out populations, a symmetric
quadratic geometry is probably not enough and we should stop adding more
Mahalanobis-style diagnostics.

No metric here is used for planning or claimed as a final method.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import DictConfig

from eval import get_dataset, img_transform
from eval_lowbudget_failure_autopsy import _build_process
from eval_b3000_paired_failure_analysis import (
    _extract_variations,
    _normalized_to_raw,
    _slice_info,
)
from eval_b3000_critical_jepa_mechanism import (
    _find_trace_for_env,
    _goal_embedding_from_solver,
)
from eval_pusht_action_ranking import _encode_images
from eval_pusht_official_diagnostic import (
    load_model,
    load_recording,
    replay,
    score_metrics,
    write_csv,
    write_json,
)

POP_RE = re.compile(
    r"^eval(?P<case>\d+)_(?P<source>lewm|ald_tf)_solve(?P<solve>\d+)_iter(?P<it>\d+)\.npz$"
)


def _bool_csv(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def _read_manifest(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _sha256_model_without_adapter(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("coordinate_adapter."):
            continue
        h.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        h.update(raw.numpy().tobytes())
    return h.hexdigest()


class IdentityMetric(nn.Module):
    def forward(self, d):
        return d.square().sum(dim=-1)


class DiagonalMetric(nn.Module):
    """Positive diagonal Mahalanobis metric with unit geometric-mean weight."""

    def __init__(self, dim: int):
        super().__init__()
        self.log_w = nn.Parameter(torch.zeros(dim))

    def weights(self):
        x = self.log_w - self.log_w.mean()
        return torch.exp(x.clamp(-8.0, 8.0))

    def forward(self, d):
        return (d.square() * self.weights()).sum(dim=-1)

    @torch.no_grad()
    def diagnostics(self):
        w = self.weights()
        return {
            "weight_min": float(w.min()),
            "weight_max": float(w.max()),
            "condition": float(w.max() / w.min().clamp_min(1e-12)),
        }


class FullPSDMetric(nn.Module):
    """Full PSD Mahalanobis metric M=A^T A with fixed Frobenius scale."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.raw = nn.Parameter(torch.eye(dim))

    def matrix_factor(self):
        # Remove the irrelevant global metric scale.
        a = self.raw
        target = float(self.dim) ** 0.5
        return a * (target / torch.linalg.vector_norm(a).clamp_min(1e-8))

    def forward(self, d):
        a = self.matrix_factor()
        y = F.linear(d, a)
        return y.square().sum(dim=-1)

    def regularizer(self):
        a = self.matrix_factor()
        eye = torch.eye(self.dim, device=a.device, dtype=a.dtype)
        return (a - eye).square().mean()

    @torch.no_grad()
    def diagnostics(self):
        a = self.matrix_factor().float()
        s = torch.linalg.svdvals(a)
        metric_eigs = s.square()
        return {
            "metric_eig_min": float(metric_eigs.min()),
            "metric_eig_max": float(metric_eigs.max()),
            "metric_condition": float(
                metric_eigs.max() / metric_eigs.min().clamp_min(1e-12)
            ),
            "factor_fro_from_identity": float(
                torch.linalg.vector_norm(
                    a - torch.eye(self.dim, device=a.device)
                )
            ),
        }


def _population_key(case, source, solve, it):
    return f"c{case}_{source}_s{solve}_i{it}"


def _safe_mean(xs):
    x = np.asarray(list(xs), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def _score_summary(score, components, topk):
    out = score_metrics(
        np.asarray(score, dtype=np.float64), components, int(topk)
    )
    return {
        "physical_rho": out["physical_rho"],
        "object_rho": out["object_rho"],
        "theta_rho": out["theta_rho"],
        "block_rho": out["block_rho"],
        "pusher_rho": out["pusher_rho"],
        "physical_selected_percentile":
            out["physical_selected_target_percentile"],
        "object_selected_percentile":
            out["object_selected_target_percentile"],
        "physical_top10_recall": out["physical_top10_recall"],
        "object_top10_recall": out["object_top10_recall"],
        "theta_matched_accuracy": out["theta_matched_accuracy"],
        "block_matched_accuracy": out["block_matched_accuracy"],
        "pusher_matched_accuracy": out["pusher_matched_accuracy"],
    }


def _build_pair_bank(populations, margin_frac, pairs_per_population, seed):
    """Pre-sample deterministic within-population oracle ranking pairs."""
    rng = np.random.default_rng(int(seed))
    banks = {}
    for p in populations:
        phys = np.asarray(p["components"]["official_cost"], dtype=np.float64)
        q25, q75 = np.percentile(phys, [25, 75])
        margin = float(margin_frac) * max(float(q75 - q25), 1e-12)
        n = len(phys)

        ii, jj, yy = [], [], []
        attempts = 0
        target = int(pairs_per_population)
        while len(ii) < target and attempts < target * 50:
            m = min((target - len(ii)) * 3, 16384)
            i = rng.integers(0, n, size=m)
            j = rng.integers(0, n, size=m)
            valid = (i != j) & (np.abs(phys[j] - phys[i]) > margin)
            i, j = i[valid], j[valid]
            if len(i):
                y = np.sign(phys[j] - phys[i]).astype(np.float32)
                take = min(target - len(ii), len(i))
                ii.extend(i[:take].tolist())
                jj.extend(j[:take].tolist())
                yy.extend(y[:take].tolist())
            attempts += m

        if len(ii) < max(128, target // 4):
            raise RuntimeError(
                f"Not enough informative oracle pairs for {p['key']}: {len(ii)}"
            )

        identity = np.asarray(p["identity_score"], dtype=np.float64)
        q25s, q75s = np.percentile(identity, [25, 75])
        score_scale = max(float(q75s - q25s), 1e-6)

        banks[p["key"]] = {
            "i": torch.as_tensor(ii, dtype=torch.long),
            "j": torch.as_tensor(jj, dtype=torch.long),
            "y": torch.as_tensor(yy, dtype=torch.float32),
            "score_scale": score_scale,
        }
    return banks


def _fit_metric(
    kind,
    train_pops,
    pair_banks,
    dim,
    device,
    *,
    steps,
    lr,
    populations_per_step,
    pairs_per_pop_step,
    temperature,
    full_identity_reg,
    seed,
):
    if kind == "diag":
        metric = DiagonalMetric(dim).to(device)
    elif kind == "full":
        metric = FullPSDMetric(dim).to(device)
    else:
        raise ValueError(kind)

    opt = torch.optim.AdamW(metric.parameters(), lr=float(lr), weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    train_pops = list(train_pops)
    last = {}
    for step in range(int(steps)):
        if len(train_pops) <= int(populations_per_step):
            chosen = train_pops
        else:
            ids = rng.choice(
                len(train_pops), size=int(populations_per_step), replace=False
            )
            chosen = [train_pops[int(i)] for i in ids]

        losses = []
        correct = []
        for p in chosen:
            bank = pair_banks[p["key"]]
            total = len(bank["i"])
            m = min(int(pairs_per_pop_step), total)
            pick = torch.randint(
                total, (m,), generator=gen, dtype=torch.long
            )
            i = bank["i"][pick].to(device)
            j = bank["j"][pick].to(device)
            y = bank["y"][pick].to(device)

            d = p["delta"].to(device)
            # Score only unique candidates touched by this pair mini-batch.
            idx = torch.unique(torch.cat([i, j]))
            score_unique = metric(d.index_select(0, idx))
            remap = torch.full(
                (len(d),), -1, device=device, dtype=torch.long
            )
            remap[idx] = torch.arange(len(idx), device=device)
            si = score_unique.index_select(0, remap[i])
            sj = score_unique.index_select(0, remap[j])

            normalized_diff = (
                (sj - si) / float(bank["score_scale"])
            )
            logits = y * normalized_diff / float(temperature)
            losses.append(F.softplus(-logits).mean())
            correct.append((logits > 0).float().mean().detach())

        loss = torch.stack(losses).mean()
        if kind == "full" and float(full_identity_reg) > 0:
            loss = loss + float(full_identity_reg) * metric.regularizer()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(metric.parameters(), 10.0)
        opt.step()

        if step in {0, int(steps)//4, int(steps)//2, 3*int(steps)//4, int(steps)-1}:
            last = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "pair_accuracy": float(torch.stack(correct).mean()),
            }

    metric.eval()
    return metric, last


@torch.no_grad()
def _eval_metric(metric, pops, device, topk, label, split):
    rows = []
    for p in pops:
        score = metric(p["delta"].to(device)).detach().cpu().numpy()
        stats = _score_summary(score, p["components"], topk)
        base = _score_summary(
            p["identity_score"], p["components"], topk
        )
        row = {
            "metric": label,
            "split": split,
            "eval_index": p["case"],
            "case_type": p["case_type"],
            "is_control": p["is_control"],
            "source": p["source"],
            "solve": p["solve"],
            "cem_iteration": p["iteration"],
            **{f"identity_{k}": v for k, v in base.items()},
            **{f"metric_{k}": v for k, v in stats.items()},
        }
        for k in stats:
            a, b = stats[k], base[k]
            try:
                row[f"delta_{k}"] = float(a) - float(b)
            except Exception:
                row[f"delta_{k}"] = float("nan")
        rows.append(row)
    return rows


def _aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["metric"], r["split"], "all", "all")].append(r)
        groups[(r["metric"], r["split"], "case", str(r["eval_index"]))].append(r)
        groups[(
            r["metric"], r["split"], "cohort",
            "control" if r["is_control"] else "hard"
        )].append(r)

    out = []
    keys = [
        "delta_physical_rho",
        "delta_object_rho",
        "delta_theta_rho",
        "delta_block_rho",
        "delta_physical_selected_percentile",
        "delta_physical_top10_recall",
    ]
    for (metric, split, gkind, gname), rr in sorted(groups.items()):
        row = {
            "metric": metric,
            "split": split,
            "group_kind": gkind,
            "group": gname,
            "population_count": len(rr),
        }
        for k in keys:
            row[f"mean_{k}"] = _safe_mean(r[k] for r in rr)
        dphys = np.asarray(
            [r["delta_physical_rho"] for r in rr], dtype=np.float64
        )
        dsel = np.asarray(
            [r["delta_physical_selected_percentile"] for r in rr],
            dtype=np.float64,
        )
        m = np.isfinite(dphys)
        row["physical_rho_better_fraction"] = (
            float(np.mean(dphys[m] > 0)) if m.any() else None
        )
        m = np.isfinite(dsel)
        row["selected_physical_better_fraction"] = (
            float(np.mean(dsel[m] < 0)) if m.any() else None
        )
        out.append(row)
    return out


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    dc = cfg.get("diagnostic", {})
    run_dir = Path(str(dc.get("run_dir", ""))).expanduser().resolve()
    if not (run_dir / "run_identity.json").is_file():
        raise ValueError("Set +diagnostic.run_dir to the authoritative formal run")

    out = Path(
        str(dc.get(
            "output_dir",
            "outputs/oracle_geometry_capacity",
        ))
    ).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    primary_cases = [int(x) for x in dc.get("cases", [7, 27, 53, 77])]
    n_controls = int(dc.get("num_controls", 2))
    sources = [str(x) for x in dc.get("sources", ["lewm", "ald_tf"])]
    solves = [int(x) for x in dc.get("solves", [0, 1])]
    iterations = [int(x) for x in dc.get("iterations", [0, 3, 9, 19, 29])]
    model_batch = int(dc.get("model_batch_size", 64))
    seed = int(dc.get("seed", 3072))

    # Fit budget is deliberately fixed and modest: this is one bounded
    # capacity test, not a hyperparameter-search loop.
    steps_diag = int(dc.get("steps_diag", 300))
    steps_full = int(dc.get("steps_full", 500))
    lr_diag = float(dc.get("lr_diag", 3e-2))
    lr_full = float(dc.get("lr_full", 3e-3))
    margin_frac = float(dc.get("pair_margin_frac", 0.02))
    pair_bank_size = int(dc.get("pair_bank_size", 2048))
    populations_per_step = int(dc.get("populations_per_step", 8))
    pairs_per_pop_step = int(dc.get("pairs_per_pop_step", 256))
    temperature = float(dc.get("temperature", 0.25))
    full_identity_reg = float(dc.get("full_identity_reg", 1e-4))
    replay_tol = float(dc.get("replay_state_tolerance", 2e-4))

    identity = json.loads((run_dir / "run_identity.json").read_text())
    protocol = identity["protocol"]
    topk = int(protocol["topk"])
    action_block = int(protocol["action_block"])
    device = torch.device(str(cfg.solver.device))

    manifest = _read_manifest(run_dir / "paired_manifest.csv")
    by_case = {int(r["eval_index"]): r for r in manifest}
    controls = [
        int(r["eval_index"])
        for r in manifest
        if _bool_csv(r.get("is_matched_control", "false"))
        and r.get("case_type") == "both_success"
    ][:max(0, n_controls)]
    cases = list(dict.fromkeys(primary_cases + controls))

    missing_cases = [c for c in cases if c not in by_case]
    if missing_cases:
        raise ValueError(f"Missing cases in manifest: {missing_cases}")

    model = load_model("lewm_epoch_10", device)
    model_digest = _sha256_model_without_adapter(model)
    dim = int(getattr(model, "embed_dim", 192))

    runs = {
        src: load_recording(run_dir / "recordings" / f"{src}.pt", device)
        for src in sources
    }
    dataset = get_dataset(cfg, str(protocol["dataset_name"]))
    process = _build_process(cfg, dataset)
    transform = img_transform(cfg)

    available = {}
    for path in (run_dir / "populations").glob("*.npz"):
        m = POP_RE.match(path.name)
        if m:
            available[(
                int(m.group("case")),
                m.group("source"),
                int(m.group("solve")),
                int(m.group("it")),
            )] = path

    requested = [
        (case, src, solve, it)
        for case in cases
        for src in sources
        for solve in solves
        for it in iterations
    ]
    missing = [k for k in requested if k not in available]
    if missing:
        raise FileNotFoundError(f"Missing requested populations: {missing[:8]}")

    print(f"Preparing fixed latent dataset for {len(requested)} populations...", flush=True)
    pops = []
    env = gym.make(str(protocol["env_name"]), render_mode="rgb_array")
    t0 = time.time()
    try:
        for n, (case, src, solve, it) in enumerate(requested, 1):
            path = available[(case, src, solve, it)]
            with np.load(path, allow_pickle=False) as npz:
                candidates = np.asarray(npz["candidates_normalized"])
                saved_final = np.asarray(npz["final_state"], dtype=np.float64)
                goal_state = np.asarray(npz["goal_state"], dtype=np.float64)
                start_state = np.asarray(npz["start_state"], dtype=np.float64)
                components = {
                    name: np.asarray(npz[name], dtype=np.float64)
                    for name in (
                        "pusher_cost", "block_cost", "theta_cost",
                        "joint_xy_cost", "object_task_cost", "official_cost",
                    )
                }

            run_src = runs[src]
            tr, li = _find_trace_for_env(run_src["solver"].trace, case, solve)
            if li is None:
                raise RuntimeError(
                    f"Missing recording case={case} src={src} solve={solve}"
                )
            info = _slice_info(tr["solver_info"], li)
            variations = _extract_variations(info)
            raw = _normalized_to_raw(
                candidates, process["action"], action_block
            )
            final, images, _, _ = replay(
                env, start_state, goal_state, raw, variations, 42
            )
            err = float(np.max(np.abs(final - saved_final)))
            if err > replay_tol:
                raise RuntimeError(
                    f"Replay mismatch {path.name}: max_abs={err:.3e}"
                )

            z = _encode_images(
                model, transform, images, device, model_batch
            ).float()
            zg = _goal_embedding_from_solver(model, info, device).float()
            delta = (z - zg[None]).detach().cpu()
            identity_score = delta.square().sum(-1).numpy()

            pops.append({
                "key": _population_key(case, src, solve, it),
                "case": case,
                "case_type": by_case[case]["case_type"],
                "is_control": case in controls,
                "source": src,
                "solve": solve,
                "iteration": it,
                "delta": delta,
                "identity_score": identity_score,
                "components": components,
            })
            print(
                f"[{n:03d}/{len(requested):03d}] case={case} src={src} "
                f"solve={solve} iter={it}",
                flush=True,
            )
    finally:
        env.close()

    latent_cache = out / "fixed_latent_populations.npz"
    np.savez_compressed(
        latent_cache,
        keys=np.asarray([p["key"] for p in pops]),
        case=np.asarray([p["case"] for p in pops], dtype=np.int32),
        source=np.asarray([p["source"] for p in pops]),
        solve=np.asarray([p["solve"] for p in pops], dtype=np.int32),
        iteration=np.asarray([p["iteration"] for p in pops], dtype=np.int32),
        delta=np.stack([p["delta"].numpy() for p in pops]).astype(np.float32),
        physical=np.stack(
            [p["components"]["official_cost"] for p in pops]
        ).astype(np.float32),
    )

    pair_banks = _build_pair_bank(
        pops, margin_frac, pair_bank_size, seed
    )

    result_rows = []
    fit_rows = []

    # ------------------------------------------------------------------
    # 1) Leave-one-case-out global diagonal/full metrics.
    # ------------------------------------------------------------------
    for held_case in cases:
        train_pops = [p for p in pops if p["case"] != held_case]
        test_pops = [p for p in pops if p["case"] == held_case]

        for kind in ("diag", "full"):
            steps = steps_diag if kind == "diag" else steps_full
            lr = lr_diag if kind == "diag" else lr_full
            print(
                f"FIT global-{kind} LOCO holdout case={held_case}: "
                f"train_pops={len(train_pops)} test_pops={len(test_pops)}",
                flush=True,
            )
            metric, train_last = _fit_metric(
                kind,
                train_pops,
                pair_banks,
                dim,
                device,
                steps=steps,
                lr=lr,
                populations_per_step=populations_per_step,
                pairs_per_pop_step=pairs_per_pop_step,
                temperature=temperature,
                full_identity_reg=full_identity_reg,
                seed=seed + held_case * 17 + (0 if kind == "diag" else 1),
            )
            label = f"global_{kind}_loco"
            rows = _eval_metric(
                metric, test_pops, device, topk, label, f"holdout_case_{held_case}"
            )
            result_rows.extend(rows)
            diag = (
                metric.diagnostics()
                if hasattr(metric, "diagnostics") else {}
            )
            fit_rows.append({
                "metric": label,
                "held_out_case": held_case,
                "train_population_count": len(train_pops),
                "test_population_count": len(test_pops),
                **train_last,
                **diag,
            })

    # ------------------------------------------------------------------
    # 2) Case-local full PSD on hard cases only.
    # Deterministic population split: 70/30 by shuffled population key.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    for case in primary_cases:
        local = [p for p in pops if p["case"] == case]
        order = rng.permutation(len(local))
        ntrain = max(8, int(round(0.7 * len(local))))
        train_pops = [local[int(i)] for i in order[:ntrain]]
        test_pops = [local[int(i)] for i in order[ntrain:]]
        if not test_pops:
            raise RuntimeError(f"No local holdout populations for case {case}")

        print(
            f"FIT case-local-full case={case}: "
            f"train_pops={len(train_pops)} test_pops={len(test_pops)}",
            flush=True,
        )
        metric, train_last = _fit_metric(
            "full",
            train_pops,
            pair_banks,
            dim,
            device,
            steps=steps_full,
            lr=lr_full,
            populations_per_step=min(populations_per_step, len(train_pops)),
            pairs_per_pop_step=pairs_per_pop_step,
            temperature=temperature,
            full_identity_reg=full_identity_reg,
            seed=seed + case * 101,
        )
        label = "case_local_full"
        rows = _eval_metric(
            metric, test_pops, device, topk, label, f"case_{case}_local_holdout"
        )
        result_rows.extend(rows)
        fit_rows.append({
            "metric": label,
            "held_out_case": case,
            "train_population_count": len(train_pops),
            "test_population_count": len(test_pops),
            **train_last,
            **metric.diagnostics(),
        })

    aggregate = _aggregate(result_rows)
    write_csv(out / "population_results.csv", result_rows)
    write_csv(out / "aggregate_summary.csv", aggregate)
    write_csv(out / "fit_diagnostics.csv", fit_rows)

    hard_global = [
        r for r in aggregate
        if r["group_kind"] == "cohort"
        and r["group"] == "hard"
        and r["metric"] in {"global_diag_loco", "global_full_loco"}
    ]
    local_cases = [
        r for r in aggregate
        if r["metric"] == "case_local_full"
        and r["group_kind"] == "case"
    ]

    summary = {
        "status": "complete",
        "scientific_question": (
            "Is one global symmetric quadratic latent geometry sufficient "
            "when given oracle physical ranking supervision, or does useful "
            "geometry need to vary by state/case?"
        ),
        "scope": (
            "One bounded geometry-family test. No further metric-capacity "
            "sweeps are intended after this result."
        ),
        "official_run": str(run_dir),
        "base_policy": "lewm_epoch_10",
        "base_model_sha256": model_digest,
        "primary_hard_cases": primary_cases,
        "controls": controls,
        "cases_in_loco": cases,
        "num_populations": len(pops),
        "candidate_count_per_population": int(len(pops[0]["delta"])),
        "oracle_target": "official physical diagnostic cost; never used for planning",
        "fit_protocol": {
            "global": (
                "Leave one entire case out. Fit on all other cases and evaluate "
                "only on the held-out case."
            ),
            "case_local": (
                "Hard cases only; deterministic 70/30 population split within "
                "each case. Fit/evaluate populations are disjoint."
            ),
            "pair_margin_frac": margin_frac,
            "pair_bank_size_per_population": pair_bank_size,
            "steps_diag": steps_diag,
            "steps_full": steps_full,
            "full_identity_reg": full_identity_reg,
        },
        "global_hard_cohort_rows": hard_global,
        "case_local_rows": local_cases,
        "decision_logic": {
            "global_full_strong": (
                "If global_full_loco improves physical/object ranking and "
                "selection on held-out hard cases, keep a global coordinate "
                "family and replace controllability supervision."
            ),
            "local_only_strong": (
                "If global_full_loco is weak but case_local_full is clearly "
                "strong on held-out populations, move to state-dependent G(z)."
            ),
            "both_weak": (
                "If even case_local_full is weak, stop symmetric quadratic "
                "geometry work and move to directed/action-conditioned or "
                "nonlinear planning geometry."
            ),
        },
        "elapsed_seconds": float(time.time() - t0),
    }
    write_json(out / "summary.json", summary)

    print("\n===== ORACLE GEOMETRY CAPACITY TEST =====")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {out}")


if __name__ == "__main__":
    run()
