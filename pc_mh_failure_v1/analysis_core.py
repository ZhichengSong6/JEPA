"""Pure diagnostic math. No policy changes, learned losses or simulator imports."""
from __future__ import annotations
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np

LABELS = ('pcmh', 'stepwise')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, 'detach'):
        return plain(x.detach().cpu().tolist())
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        raise ValueError('Refusing nonfinite diagnostic output')
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    raise TypeError(type(x).__name__)


def dump(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(plain(obj), indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def write_csv(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if fields:
            w.writeheader()
            w.writerows(rows)


def group(a, b):
    return ('both_success' if a and b else 'both_fail' if not a and not b
            else 'rescued' if b else 'regressed')


def choose_cases(rows, limit=12, seed=20261012):
    """Fixed quotas, no hand-picked videos: 4 common failures, 3/3 discordants, 2 controls."""
    if limit < 1:
        raise ValueError('limit must be positive')
    keys = [(r['seed'], r['eval_index']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate case identities')
    rng = random.Random(seed)
    pools = {}
    for g in ('both_fail', 'regressed', 'rescued', 'both_success'):
        pools[g] = sorted([r.copy() for r in rows if r['group'] == g],
                          key=lambda r: (r['seed'], r['eval_index']))
        rng.shuffle(pools[g])
    selected = []
    for g, quota in [('both_fail', 4), ('regressed', 3), ('rescued', 3), ('both_success', 2)]:
        n = min(quota, len(pools[g]), limit - len(selected))
        selected.extend(pools[g][:n]); pools[g] = pools[g][n:]
    for g in ('both_fail', 'regressed', 'rescued', 'both_success'):
        n = min(len(pools[g]), limit - len(selected))
        selected.extend(pools[g][:n])
    return sorted(selected, key=lambda r: (r['seed'], r['eval_index']))


def cost_stats(predicted_cost, real_cost):
    """Within-one-population ranking; ties are excluded, not arbitrarily scored as errors."""
    a, b = [np.asarray(x, dtype=np.float64) for x in (predicted_cost, real_cost)]
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite([a, b]).all():
        raise ValueError('Invalid candidate costs')
    i, j = np.triu_indices(len(a), k=1)
    da, db = a[i] - a[j], b[i] - b[j]
    tol = 1e-6 + 1e-6 * max(float(np.abs(b).max()), 1.0)
    valid = (np.abs(da) > tol) & (np.abs(db) > tol)
    selected = int(np.argmin(a))
    return {'cost_mse': float(np.square(a-b).mean()),
            'selected_index': selected,
            'encoder_regret': float(b[selected]-b.min()),
            'rank_pairs': int(valid.sum()),
            'inversion_rate': float((da[valid]*db[valid] < 0).mean()) if valid.any() else None,
            'tie_tolerance': tol}


def decomposition(student, target, real):
    s, y, r = [np.asarray(x, dtype=np.float64) for x in (student, target, real)]
    if s.shape != y.shape or s.shape != r.shape or s.ndim < 2:
        raise ValueError('Prediction/target/replay shapes differ')
    if not all(np.isfinite(x).all() for x in (s,y,r)):
        raise ValueError('Nonfinite latent in decomposition')
    efit, etarget = s-y, y-r
    total = float(np.square(s-r).mean())
    fit = float(np.square(efit).mean())
    bias = float(np.square(etarget).mean())
    cross = float(2*(efit*etarget).mean())
    return {'prediction_mse': total, 'reference_fit_mse': fit,
            'reference_target_mse': bias, 'cross_term': cross,
            'identity_error': abs(total-fit-bias-cross)}


def flags(scores, real_cost, ever_success, elite, mean_ever_success):
    """Overlapping local flags, NOT an exclusive episode-level causal classifier."""
    s, r = [np.asarray(x) for x in (scores, real_cost)]
    ok = np.asarray(ever_success, dtype=bool)
    if s.shape != r.shape or s.shape != ok.shape or s.ndim != 1 or not len(s):
        raise ValueError('Mismatched outcome arrays')
    si, oi = int(np.argmin(s)), int(np.argmin(r))
    ei = np.asarray(elite, dtype=int)
    if not len(ei) or ei.min() < 0 or ei.max() >= len(s):
        raise ValueError('Invalid elite indices')
    return {'success_available': bool(ok.any()), 'successful_candidates': int(ok.sum()),
            'source_best_ever_success': bool(ok[si]), 'encoder_best_ever_success': bool(ok[oi]),
            'predictor_top1_miss': bool(ok.any() and ok[oi] and not ok[si]),
            'encoder_top1_task_mismatch': bool(ok.any() and not ok[oi]),
            'no_success_in_recorded_population': bool(not ok.any()),
            'source_elite_successes': int(ok[ei].sum()),
            'elite_mean_failure': bool(ok[ei].any() and not mean_ever_success),
            'mean_ever_success': bool(mean_ever_success)}


def make_joint_probes(center, seed, radius=0.1565, pairs=((0,1),(0,4),(2,4))):
    """Symmetric bounded probes in RAW [-1,1] actions, with matched-total-norm singles.

    For every pair: 4 small singles, 4 signed joint corners, 4 larger singles.
    The joint norm sqrt(2)*component_radius equals each larger single's norm.
    Shared radius is reduced symmetrically if bounds require it. No clipping of
    an individual probe, which would invalidate the factorial contrast.
    """
    c = np.asarray(center, dtype=np.float64)
    if c.shape != (5,10) or not np.isfinite(c).all() or (np.abs(c)>1+1e-6).any() or radius <= 0:
        raise ValueError('Need finite raw [5,10] center in [-1,1] and positive radius')
    c = np.clip(c, -1, 1)  # only <=1e-6 numerical boundary excess is accepted
    rng = np.random.default_rng(seed)
    probes, specs = [c.copy()], []
    for p,q in pairs:
        if not 0 <= p < q < 5:
            raise ValueError('Require 0 <= p < q < 5')
        vectors=[]; caps=[]
        for block in (p,q):
            slack=np.maximum(1-np.abs(c[block]),0)
            v=rng.normal(size=10); v[slack < 1e-9]=0
            norm=np.linalg.norm(v)
            if norm < 1e-12:
                vectors.append(v); caps.append(0.0); continue
            v/=norm
            active=np.abs(v)>1e-12
            cap=np.min(slack[active]/np.abs(v[active]))
            vectors.append(v); caps.append(float(cap))
        r=min(radius/np.sqrt(2), min(caps)*0.999/np.sqrt(2))
        rec={'p':p,'q':q,'component_radius':float(r),'joint_radius':float(np.sqrt(2)*r),
             'requested_total_radius':float(radius), 'skipped':bool(r<1e-8), 'indices':{}}
        if r>=1e-8:
            dp,dq=np.zeros_like(c),np.zeros_like(c)
            dp[p]=r*vectors[0]; dq[q]=r*vectors[1]
            for name,d in [('p+',dp),('p-',-dp),('q+',dq),('q-',-dq),
                           ('++',dp+dq),('+-',dp-dq),('-+',-dp+dq),('--',-dp-dq),
                           ('P+',np.sqrt(2)*dp),('P-',-np.sqrt(2)*dp),
                           ('Q+',np.sqrt(2)*dq),('Q-',-np.sqrt(2)*dq)]:
                rec['indices'][name]=len(probes); probes.append(c+d)
        specs.append(rec)
    result=np.stack(probes).astype(np.float32)
    if np.abs(result).max()>1+1e-6:
        raise RuntimeError('Symmetric probe construction exceeded raw bounds')
    return result,specs


def interaction_rows(outputs, specs, real_label='replay'):
    """Finite mixed contrast, not an exact environmental Hessian."""
    arrays={k:np.asarray(v,dtype=np.float64) for k,v in outputs.items()}
    shape=arrays[real_label].shape
    if len(shape)!=3 or any(v.shape!=shape or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Need equally shaped finite [candidate,horizon,latent] outputs')
    rows=[]
    for spec in specs:
        if spec['skipped']:
            rows.append({'p':spec['p'],'q':spec['q'],'skipped':True}); continue
        ix=spec['indices']; rad=spec['component_radius']
        joint=[ix[k] for k in ('++','+-','-+','--')]
        small=[ix[k] for k in ('p+','p-','q+','q-')]
        matched=[ix[k] for k in ('P+','P-','Q+','Q-')]
        mixed={k:(v[ix['++']]-v[ix['+-']]-v[ix['-+']]+v[ix['--']])/4 for k,v in arrays.items()}
        for h in range(shape[1]):
            truth=mixed[real_label][h]
            signal=float(np.mean(truth**2))
            effect=float(np.mean((arrays[real_label][joint,h]-arrays[real_label][0,h])**2))
            for label,v in arrays.items():
                if label==real_label: continue
                err=float(np.mean((mixed[label][h]-truth)**2))
                error_by_set={name:float(np.mean((v[indices,h]-arrays[real_label][indices,h])**2))
                              for name,indices in [('joint',joint),('small_single',small),('matched_single',matched)]}
                rows.append({'p':spec['p'],'q':spec['q'],'horizon':h+1,'model':label,
                             'skipped':False,'causally_joint_affected':h>=spec['q'],
                             'component_radius':rad,'joint_radius':spec['joint_radius'],
                             'real_interaction_mse':signal,'real_joint_effect_mse':effect,
                             'interaction_error_mse':err,
                             'interaction_relative_rmse':float(np.sqrt(err/signal)) if signal>1e-10 else None,
                             'interaction_signal_fraction':signal/effect if effect>1e-10 else None,
                             **{k+'_prediction_mse':val for k,val in error_by_set.items()}})
    return rows
