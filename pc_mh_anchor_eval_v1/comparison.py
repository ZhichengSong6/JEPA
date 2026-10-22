"""Pure paired-evaluation bookkeeping. No models, simulator or training imports."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

SEEDS = (142, 143, 144, 145, 146)
BUDGETS = {1000: (100, 10, 10), 3000: (300, 10, 30)}
LABELS = ('pcmh', 'anchor')
N = 100


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refuse empty result table')
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def sealed_write(path, payload):
    if Path(path).exists():
        raise FileExistsError(path)
    dump(path, {'payload': payload, 'payload_sha256': object_hash(payload)})


def sealed_read(path):
    obj = read(path)
    if object_hash(obj['payload']) != obj['payload_sha256']:
        raise RuntimeError(f'Result checksum mismatch: {path}')
    return obj['payload']


def excluded_episodes(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'episode_idx' not in rows[0]:
        raise ValueError('Need the existing selected_cases.csv with episode_idx')
    return sorted({int(r['episode_idx']) for r in rows})


def select_cohort(episodes, steps, excluded, seeds=SEEDS, n=N, offset=25):
    """Uniform valid-row sampling, excluding full diagnostic episodes.

    One immutable list per seed, shared by both models and both budgets.
    Sample without row replacement across seeds; repeated episodes are allowed
    and accounted for by the descriptive episode-cluster bootstrap.
    """
    e, s = np.asarray(episodes).reshape(-1), np.asarray(steps).reshape(-1)
    if e.shape != s.shape or not len(e) or not np.isfinite(e).all() or not np.isfinite(s).all():
        raise ValueError('Invalid episode/step columns')
    if np.any(e != e.astype('int64')) or np.any(s != s.astype('int64')) or np.any(s < 0):
        raise ValueError('Episode and step columns must be integer valued')
    unique, inverse = np.unique(e, return_inverse=True)
    last = np.full(len(unique), -1, dtype='int64')
    np.maximum.at(last, inverse, s.astype('int64'))
    valid = np.flatnonzero((s <= last[inverse] - offset) & ~np.isin(e, excluded))
    if len(set(seeds)) != len(seeds) or n < 1 or len(valid) < n*len(seeds):
        raise ValueError('Invalid seeds or insufficient eligible starts')
    records = []
    for seed in seeds:
        chosen = np.sort(np.random.default_rng(seed).choice(valid, size=n, replace=False))
        valid = valid[~np.isin(valid, chosen)]
        for i, row in enumerate(chosen):
            records.append({'seed': int(seed), 'eval_index': i, 'dataset_row': int(row),
                            'episode_idx': int(e[row]), 'start_step': int(s[row]),
                            'goal_step': int(s[row]) + offset})
    return records


def successes(result, n=N):
    values = result.get('successes')
    if not isinstance(values, list) or len(values) != n or any(type(x) is not bool for x in values):
        raise ValueError('Incomplete/non-boolean episode_successes; never infer from success_rate')
    return np.asarray(values, dtype=bool)


def paired_rows(cases, a, b, budget):
    x, y = successes(a, len(cases)), successes(b, len(cases))
    return [{**c, 'budget': int(budget), 'pcmh_success': int(u), 'anchor_success': int(v),
             'pcmh_minus_anchor': int(u)-int(v)} for c, u, v in zip(cases, x, y)]


def summary(rows, seeds=SEEDS, n=N):
    expected = {(b, seed, i) for b in BUDGETS for seed in seeds for i in range(n)}
    keys = [(r['budget'], r['seed'], r['eval_index']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError('Result matrix is incomplete/duplicated; no partial overall mean')
    tables, totals = [], []
    for budget in BUDGETS:
        r = [x for x in rows if x['budget'] == budget]
        for seed in seeds:
            part = [x for x in r if x['seed'] == seed]
            a, b = [sum(x[k] for x in part) for k in ('pcmh_success','anchor_success')]
            tables.append({'budget': budget, 'seed': seed, 'n': n,
                           'pcmh_success_pct': 100*a/n, 'anchor_success_pct': 100*b/n,
                           'pcmh_minus_anchor_pp': 100*(a-b)/n})
        a = np.array([x['pcmh_success'] for x in r]); b = np.array([x['anchor_success'] for x in r])
        ep = np.array([x['episode_idx'] for x in r]); ids, ix = np.unique(ep, return_inverse=True)
        counts = np.bincount(ix); sums = np.bincount(ix, weights=a-b)
        rng = np.random.default_rng(20261022 + budget)
        draws = rng.integers(0, len(ids), (5000, len(ids)))
        boot = 100*sums[draws].sum(1)/counts[draws].sum(1)
        lo, hi = np.quantile(boot, [.025, .975])
        q = [x for x in tables if x['budget'] == budget]
        totals.append({'budget':budget, 'n':len(r), 'unique_dataset_episodes':len(ids),
            'pcmh_mean_pct':float(100*a.mean()), 'anchor_mean_pct':float(100*b.mean()),
            'pcmh_seed_sd_pp':float(np.std([x['pcmh_success_pct'] for x in q], ddof=1)) if len(q)>1 else None,
            'anchor_seed_sd_pp':float(np.std([x['anchor_success_pct'] for x in q], ddof=1)) if len(q)>1 else None,
            'pcmh_minus_anchor_pp':float(100*(a-b).mean()),
            'pcmh_only':int(((a==1)&(b==0)).sum()), 'anchor_only':int(((a==0)&(b==1)).sum()),
            'both_success':int(((a==1)&(b==1)).sum()), 'both_fail':int(((a==0)&(b==0)).sum()),
            'episode_cluster_ci95_low_pp':float(lo), 'episode_cluster_ci95_high_pp':float(hi)})
    return tables, totals
