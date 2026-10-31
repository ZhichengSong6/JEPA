"""Run a NEW, sequential, paired PushT TTA pilot with the existing environment.

Not a reproduction/overwrite of the historical 500-case benchmark. Every mode
uses the same case list, checkpoint, CEM settings and per-case seed. One live
environment/model only. No friction/physics perturbations are silently added.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.util
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from .core import AdaptConfig, MODES, OnlineAdapter, find_lewm, synchronize
from .swm_bridge import AdaptiveCostModel, JSONLLogger, make_online_policy_class


def jsonable(x: Any):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if torch.is_tensor(x):
        return jsonable(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    return repr(x)


def dump(path: Path, data):
    # Each path is new in an exclusively created run directory.
    with path.open("x", encoding="utf-8") as f:
        json.dump(jsonable(data), f, ensure_ascii=False, indent=2, allow_nan=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_policy(policy: str | None, policy_dir: str | None, home: Path):
    if policy is not None:
        path = Path(policy).expanduser()
        path = path if path.is_absolute() else home / path
        if str(path).endswith('_object.ckpt'):
            path = Path(str(path)[:-len('_object.ckpt')])
        obj = Path(str(path) + '_object.ckpt')
    else:
        directory = Path(policy_dir).expanduser().resolve()
        matches = sorted(directory.rglob('*epoch_10_object.ckpt'))
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one epoch_10_object.ckpt under {directory}; "
                               f"found {len(matches)}: {[str(p) for p in matches]}. Use --policy explicitly.")
        obj = matches[0]
        path = Path(str(obj)[:-len('_object.ckpt')])
    if not obj.is_file():
        raise FileNotFoundError(obj)
    path = path.resolve()
    try:
        relative = str(path.relative_to(home.resolve()))
    except ValueError as exc:
        raise ValueError("Checkpoint must be under STABLEWM_HOME for AutoCostModel") from exc
    return relative, obj.resolve()


def load_cases(path: str | None, dataset, eval_api, cfg, count: int, seed: int):
    if path:
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or doc.get('schema') != 'pcqr-cases-v1':
            raise ValueError("--cases requires schema pcqr-cases-v1. Do not guess/overwrite the old cohort format.")
        if doc.get('dataset') != cfg.eval.dataset_name or doc.get('goal_offset_steps') != cfg.eval.goal_offset_steps:
            raise ValueError("Case file dataset/goal_offset differs from the evaluation config")
        cases = doc.get('cases')
        if not isinstance(cases, list) or not cases:
            raise ValueError("Case file must contain a nonempty cases list")
        for c in cases:
            if set(c) != {'episode', 'start'} or not all(type(v) is int and v >= 0 for v in c.values()):
                raise ValueError("Each case must be {'episode': nonnegative int, 'start': nonnegative int}")
        if len({(c['episode'], c['start']) for c in cases}) != len(cases):
            raise ValueError("Duplicate case")
        return doc
    key = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    eps = np.asarray(dataset.get_col_data(key)).reshape(-1)
    steps = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    unique = np.unique(eps)
    lengths = eval_api.get_episodes_length(dataset, unique)
    limits = dict(zip(unique.tolist(), (lengths - cfg.eval.goal_offset_steps - 1).tolist()))
    valid = np.flatnonzero(steps <= np.array([limits[e] for e in eps]))
    # This is an explicitly NEW pilot cohort; do not reuse its results as old benchmark numbers.
    if len(valid) < count:
        raise ValueError(f"Only {len(valid)} valid starts for {count} requested cases")
    ids = np.sort(np.random.default_rng(seed).choice(valid, count, replace=False))
    return {'schema': 'pcqr-cases-v1', 'dataset': str(cfg.eval.dataset_name),
            'goal_offset_steps': int(cfg.eval.goal_offset_steps), 'selection_seed': seed,
            'protocol': 'new_sequential_tta_pilot_not_historical_benchmark',
            'cases': [{'episode': int(eps[i]), 'start': int(steps[i])} for i in ids]}



def evaluate_case(world: Any, dataset: Any, *, case: dict[str, int],
                  goal_offset_steps: int, eval_budget: int, callables: Any,
                  out_dir: Path, mode: str, case_index: int) -> Any:
    """Call the existing SWM evaluator without writing rollout videos.

    Legacy SWM defaults save_video=True and constructs Path(video_path).
    None is not a disable-video flag. Supply both the explicit switch and a
    real, case-specific path so no branch of the evaluator receives None.
    Metrics and all control arguments are forwarded without modification.
    """
    if mode not in MODES or type(case_index) is not int or case_index < 0:
        raise ValueError("Invalid mode/case_index for evaluation output path")
    video_dir = Path(out_dir).resolve() / 'videos' / f'{mode}_case{case_index:04d}'
    return world.evaluate_from_dataset(
        dataset, start_steps=[case['start']], episodes_idx=[case['episode']],
        goal_offset_steps=goal_offset_steps, eval_budget=eval_budget,
        callables=callables, save_video=False, video_path=str(video_dir),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--policy', help='Checkpoint prefix relative to STABLEWM_HOME, no suffix')
    source.add_argument('--policy-dir', help='Directory containing exactly one *epoch_10_object.ckpt')
    ap.add_argument('--repo', type=Path, default=Path.cwd())
    ap.add_argument('--out', type=Path, required=True, help='Must not already exist')
    ap.add_argument('--cases', help='Existing pcqr-cases-v1 JSON; never changes that file')
    ap.add_argument('--num-cases', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1729)
    ap.add_argument('--budget', type=int, choices=[1000, 3000], default=1000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--modes', nargs='+', choices=MODES, default=['frozen', 'obs', 'response'])
    ap.add_argument('--adapt-config', type=Path, default=Path(__file__).parent / 'configs/default.json')
    ap.add_argument('--lr', type=float)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--strength', type=float)
    ap.add_argument('--eval-budget', type=int, default=50, help='Raw environment steps, original default 50')
    args = ap.parse_args()
    if args.num_cases < 1 or args.eval_budget < 25 or len(set(args.modes)) != len(args.modes):
        ap.error('Require positive case count, eval-budget >=25 and no duplicate modes')
    if 'STABLEWM_HOME' not in os.environ:
        ap.error('Set STABLEWM_HOME explicitly; do not guess the checkpoint/data root')
    repo = args.repo.expanduser().resolve()
    if not (repo / 'eval.py').is_file() or not (repo / 'config/eval/pusht.yaml').is_file():
        ap.error('--repo must point at the existing JEPA/LeWM_official checkout')
    if args.out.exists():
        ap.error('--out already exists; choose a new run directory')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        ap.error('CUDA requested but unavailable. Unit tests run on CPU; PushT GPU smoke needs your server.')
    home = Path(os.environ['STABLEWM_HOME']).expanduser().resolve()
    policy_name, object_path = resolve_policy(args.policy, args.policy_dir, home)

    # Preserve the original repository's dependency import order/resolvers.
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('_pcqr_existing_eval', repo / 'eval.py')
    eval_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_api)
    import stable_worldmodel as swm
    import hydra
    from omegaconf import OmegaConf
    from sklearn.preprocessing import StandardScaler
    if not hasattr(swm.policy, 'AutoCostModel'):
        raise RuntimeError('Installed SWM lacks the repository AutoCostModel API. Use the existing lewm env; do not upgrade.')

    cfg = OmegaConf.load(repo / 'config/eval/pusht.yaml')
    cfg.solver = OmegaConf.load(repo / 'config/eval/solver/cem.yaml')
    cfg.seed = args.seed
    cfg.policy = policy_name
    cfg.world.num_envs = 1
    cfg.world.history_size = 1
    cfg.world.max_episode_steps = 2 * args.eval_budget
    cfg.eval.eval_budget = args.eval_budget
    cfg.solver.batch_size = 1
    cfg.solver.num_samples = 100 if args.budget == 1000 else 300
    cfg.solver.n_steps = 10
    cfg.solver.topk = 10 if args.budget == 1000 else 30
    cfg.solver.device = args.device
    if int(cfg.world.get('frame_skip', 1)) != 1:
        raise ValueError('v1 requires world.frame_skip=1 to count executed raw actions correctly')
    with args.adapt_config.open(encoding='utf-8') as f:
        settings = json.load(f)
    for name in ('lr', 'steps', 'strength'):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    settings['train_parts'] = tuple(settings['train_parts'])
    if settings.get('horizon_weights') is not None:
        settings['horizon_weights'] = tuple(settings['horizon_weights'])
    adapt_cfg = AdaptConfig(**settings)
    if int(cfg.plan_config.horizon) != adapt_cfg.horizon or int(cfg.plan_config.action_block) != 5:
        raise ValueError('This PushT runner requires H=5, action_block=5; core supports explicit alternatives')
    if adapt_cfg.packed_action_dim != 10 or adapt_cfg.history_size != 3:
        raise ValueError('PushT bridge requires packed_action_dim=10 and upstream history_size=3')
    if int(cfg.plan_config.get('history_len', 1)) != 1:
        raise ValueError('PlanConfig.history_len must be 1 for this v1 adapter')

    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / 'REQUEST.json', vars(args))
    try:
        dataset = eval_api.get_dataset(cfg, cfg.eval.dataset_name)
        cases_doc = load_cases(args.cases, dataset, eval_api, cfg, args.num_cases, args.seed)
        dump(args.out / 'cases.json', cases_doc)
        process = {}
        for col in cfg.dataset.keys_to_cache:
            if col == 'pixels':
                continue
            values = np.asarray(dataset.get_col_data(col))
            values = values[~np.isnan(values).any(axis=1)]
            processor = StandardScaler().fit(values)
            process[col] = processor
            if col != 'action':
                process[f'goal_{col}'] = processor
        transform = {'pixels': eval_api.img_transform(cfg), 'goal': eval_api.img_transform(cfg)}
        model = swm.policy.AutoCostModel(policy_name).to(args.device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        core_path, core = find_lewm(model)
        versions = {}
        for name in ('torch', 'stable-worldmodel', 'stable-pretraining', 'numpy', 'hydra-core'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        manifest = {
            'protocol': cases_doc['protocol'] if 'protocol' in cases_doc else 'supplied_pcqr_case_list',
            'checkpoint': str(object_path), 'checkpoint_sha256': sha256(object_path),
            'policy': policy_name, 'core_module_path': core_path,
            'config': OmegaConf.to_container(cfg, resolve=True), 'adaptation': adapt_cfg.to_dict(),
            'versions': versions,
            'video_recording': {'save_video': False, 'video_path_policy': 'per_mode_per_case_non_null'},
            'source_sha256': {str(p.relative_to(repo)): sha256(p) for p in [repo / 'jepa.py', repo / 'eval.py',
                             repo / 'config/eval/pusht.yaml', repo / 'config/eval/solver/cem.yaml'] if p.exists()},
            'policy_source_file': inspect.getfile(swm.policy.WorldModelPolicy),
            'warning': 'New sequential TTA pilot; no effect claim; no automatic dynamics shift; no server prevalidation',
        }
        dump(args.out / 'MANIFEST.json', manifest)
        print(json.dumps({'policy': policy_name, 'modes': args.modes,
                          'cases': len(cases_doc['cases']), 'out': str(args.out)}, ensure_ascii=False), flush=True)
        policy_cls = make_online_policy_class(swm.policy.WorldModelPolicy)
        fingerprints: dict[int, dict[str, str]] = {}
        rows = []
        for mode in args.modes:
            adapter = OnlineAdapter(core, replace(adapt_cfg, mode=mode))
            for index, case in enumerate(cases_doc['cases']):
                case_seed = args.seed + 104729 * index
                seed_all(case_seed)
                cfg.solver.seed = case_seed
                synchronize(adapter.device)
                started = time.perf_counter()
                adapter.reset_episode()
                logger = JSONLLogger(args.out / f'{mode}_case{index:04d}.jsonl')
                world = None
                try:
                    cost = AdaptiveCostModel(model, adapter, log=logger)
                    solver = hydra.utils.instantiate(cfg.solver, model=cost)
                    policy = policy_cls(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                                        process=process, transform=transform, online_cost=cost,
                                        action_block=5, log=logger, expected_initial=fingerprints.get(index))
                    # Fresh world per case and mode avoids stateful RNG carryover.
                    world = swm.World(**cfg.world, image_shape=(224, 224))
                    world.set_policy(policy)
                    metrics = evaluate_case(
                        world, dataset, case=case,
                        goal_offset_steps=int(cfg.eval.goal_offset_steps), eval_budget=args.eval_budget,
                        callables=OmegaConf.to_container(cfg.eval.get('callables'), resolve=True),
                        out_dir=args.out, mode=mode, case_index=index)
                    synchronize(adapter.device)
                    elapsed = time.perf_counter() - started
                    adapter.assert_frozen_unchanged()
                    if policy.initial_fingerprint is None or cost.replans < 1:
                        raise RuntimeError('Evaluation returned without a policy call/replan')
                    fingerprints.setdefault(index, policy.initial_fingerprint)
                    row = {'mode': mode, 'case_index': index, **case, 'case_seed': case_seed,
                           'episode_wall_s': elapsed, 'policy_wall_ms': policy.online_wall_ms,
                           'adaptation_hook_ms': cost.adapt_ms, 'applied_optimizer_steps': cost.applied_steps,
                           'replans': cost.replans, 'cost_calls': cost.cost_calls,
                           'completed_blocks': adapter.buffer.total_completed,
                           'initial_fingerprint': policy.initial_fingerprint, 'metrics': jsonable(metrics)}
                    dump(args.out / f'{mode}_case{index:04d}_result.json', row)
                    rows.append(row)
                    print(json.dumps(jsonable(row), ensure_ascii=False), flush=True)
                finally:
                    logger.close()
                    adapter.reset_episode()
                    if world is not None:
                        close = getattr(world, 'close', None)
                        if callable(close):
                            close()
                        elif hasattr(world, 'envs') and callable(getattr(world.envs, 'close', None)):
                            world.envs.close()
            adapter.reset_episode()
        with (args.out / 'results.csv').open('x', newline='', encoding='utf-8') as f:
            fields = ['mode', 'case_index', 'episode', 'start', 'case_seed', 'episode_wall_s', 'policy_wall_ms',
                      'adaptation_hook_ms', 'applied_optimizer_steps', 'replans', 'cost_calls', 'completed_blocks', 'metrics']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                result = {k: row[k] for k in fields}
                result['metrics'] = json.dumps(result['metrics'], ensure_ascii=False)
                writer.writerow(result)
        dump(args.out / 'COMPLETE.json', {'status': 'completed', 'case_results': len(rows),
                                          'paired_initials_verified': True, 'modes': args.modes,
                                          'note': 'Raw metrics preserved without guessing success-rate units.'})
    except Exception as exc:
        dump(args.out / 'FAILED.json', {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
