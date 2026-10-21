"""Multi-anchor residual differences. No simulator or task-success labels.

All distances use only z0 and the causal action prefix. The residual memory,
feature scales, gates and estimator are frozen before the student is trained.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.with_suffix(p.suffix + '.partial').open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    p.with_suffix(p.suffix + '.partial').replace(p)


def load_npz(path):
    # Explicitly close both readers before any directory cleanup (including NFS).
    with Path(path).open('rb') as f, np.load(f, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def save_npz(path, **arrays):
    p = Path(path)
    if p.exists():
        raise FileExistsError(p)
    with p.with_suffix('.partial').open('wb') as f:
        np.savez_compressed(f, **arrays)
    p.with_suffix('.partial').replace(p)


def finite(x, name):
    if not np.isfinite(np.asarray(x)).all():
        raise ValueError(f'Nonfinite {name}; no silent sanitization for new supervision')


def episode_partition(episode_ids, seed):
    """60/20/20 by complete episode within the ORIGINAL training subset."""
    ids = np.unique(np.asarray(episode_ids, dtype=np.int64))
    if len(ids) < 15:
        raise ValueError('Too few training episodes for bank/calibration/audit isolation')
    ids = np.random.default_rng(seed).permutation(ids)
    a, b = int(.6 * len(ids)), int(.8 * len(ids))
    return dict(bank=ids[:a], calibration=ids[a:b], audit=ids[b:])


def features(z, u, h, scales):
    z, u = np.asarray(z), np.asarray(u)
    if z.ndim != 2 or u.ndim != 3 or len(z) != len(u) or not 1 <= h <= u.shape[1]:
        raise ValueError('Need z [N,D], u [N,H,A], 1<=h<=H')
    finite(z, 'z'); finite(u[:, :h], 'causal action prefix')
    fz = ((z-scales['z_mean'])/scales['z_std']) / np.sqrt(z.shape[1])
    fu = ((u[:, :h]-scales['u_mean'])/scales['u_std']).reshape(len(u), -1)
    fu /= np.sqrt(h*u.shape[-1])
    return fz.astype('float32'), fu.astype('float32')


def fit_scales(bank):
    z, u = bank['z'], bank['u'].reshape(-1, bank['u'].shape[-1])
    finite(z, 'memory z'); finite(u, 'memory actions')
    return {'z_mean': z.mean(0), 'z_std': np.maximum(z.std(0), 1e-4),
            'u_mean': u.mean(0), 'u_std': np.maximum(u.std(0), 1e-4)}


class Memory:
    """Exact GPU/CPU KNN, one nearest window per distinct episode.

    No outcome-dependent neighbor selection. Exclude the reference episode for
    both U and U0. Stable tie-breaking uses the first memory row. No FAISS needed.
    """
    def __init__(self, bank, device='cpu', k=8, scales=None, chunk=64):
        self.bank, self.device = bank, torch.device(device)
        self.k, self.chunk = int(k), int(chunk)
        self.scales = fit_scales(bank) if scales is None else scales
        if self.k < 2 or self.chunk < 1:
            raise ValueError('k>=2 and positive query chunk required')
        n, h, d = bank['residual'].shape
        if bank['z'].shape != (n,d) or bank['u'].shape[:2] != (n,h):
            raise ValueError('Malformed residual memory')
        if len(np.unique(bank['episode'])) <= self.k:
            raise ValueError('Insufficient distinct memory episodes after exclusion')
        finite(bank['residual'], 'memory residual')
        self.ep = torch.as_tensor(bank['episode'], device=self.device)
        self._prepared = {}

    @torch.no_grad()
    def estimate(self, z, u, h, exclude_episode):
        if h not in self._prepared:
            bz, ba = features(self.bank['z'], self.bank['u'], h, self.scales)
            b = torch.as_tensor(np.concatenate([bz, ba], 1), device=self.device)
            bz = torch.as_tensor(bz, device=self.device)
            ba = torch.as_tensor(ba, device=self.device)
            r = torch.as_tensor(self.bank['residual'][:,h-1], device=self.device)
            self._prepared[h] = (b, bz, ba, r, b.square().sum(1))
        b, bz, ba, r, bnorm = self._prepared[h]
        qz, qa = features(z, u, h, self.scales)
        exclude = np.asarray(exclude_episode).reshape(-1)
        if len(exclude) != len(qz):
            raise ValueError('Need one excluded reference episode per query')
        estimates, stats, neighbors = [], [], []
        for start in range(0,len(qz),self.chunk):
            end = min(start+self.chunk,len(qz))
            zz = torch.as_tensor(qz[start:end],device=self.device)
            aa = torch.as_tensor(qa[start:end],device=self.device)
            q = torch.cat([zz,aa],1)
            dist = (q.square().sum(1)[:,None]+bnorm[None]-2*q@b.T).clamp_min_(0)
            ex = torch.as_tensor(exclude[start:end],device=self.device)
            dist.masked_fill_(self.ep[None] == ex[:,None],float('inf'))
            picks, ds = [], []
            for _ in range(self.k):
                value, idx = dist.min(1)
                if not torch.isfinite(value).all():
                    raise ValueError('Cannot find K distinct supported episodes')
                picks.append(idx); ds.append(value)
                dist.masked_fill_(self.ep[None] == self.ep[idx,None],float('inf'))
            ix = torch.stack(picks,1); distance = torch.stack(ds,1)
            # Adaptive bandwidth fixed by the K-th INPUT neighbor distance.
            weight = torch.softmax(-distance/distance[:,-1:].clamp_min(1e-8),1)
            residuals = r[ix]
            avg = (residuals*weight[:,:,None]).sum(1)
            spread = ((residuals-avg[:,None]).square().mean(-1)*weight).sum(1)
            dz = (bz[ix]-zz[:,None]).square().sum(-1).max(1).values
            da = (ba[ix]-aa[:,None]).square().sum(-1).max(1).values
            estimates.append(avg.cpu().numpy())
            stats.append(torch.stack([dz,da,spread],1).cpu().numpy())
            neighbors.append(ix.cpu().numpy())
        return np.concatenate(estimates), np.concatenate(stats), np.concatenate(neighbors)


def pair_records(memory, heldout, h, batch=64):
    """Choose nearest *input* reference i in heldout, not in its own episode.

    Neither i nor j is in the memory. j's actual future appears ONLY in the
    evaluation error; never in either correction estimate or in pair selection.
    """
    z, u, ep = heldout['z'], heldout['u'], heldout['episode']
    zz, aa = features(z,u,h,memory.scales)
    x = torch.as_tensor(np.concatenate([zz,aa],1),device=memory.device)
    eps = torch.as_tensor(ep,device=memory.device)
    ids = []
    for s in range(0,len(x),batch):
        q = x[s:s+batch]
        d = (q.square().sum(1)[:,None]+x.square().sum(1)[None]-2*q@x.T).clamp_min_(0)
        d.masked_fill_(eps[None] == eps[s:s+len(q),None],float('inf'))
        v, idx = d.min(1)
        if not torch.isfinite(v).all(): raise ValueError('No different-episode reference')
        ids.append(idx.cpu().numpy())
    ref = np.concatenate(ids)
    rhat, stats, _ = memory.estimate(z,u,h,ep)
    truth = heldout['residual'][:,h-1]
    before = np.mean((truth[ref]-truth)**2,1)
    after = np.mean((truth[ref]+rhat-rhat[ref]-truth)**2,1)
    pair_z = np.sum((zz-zz[ref])**2,1)
    pair_a = np.sum((aa-aa[ref])**2,1)
    gate_stats = np.column_stack([np.maximum(stats,stats[ref]),pair_z,pair_a])
    return dict(before=before, after=after, stats=gate_stats, ref=ref, episode=ep)


def masked_metrics(rec, mask):
    mask = np.asarray(mask,dtype=bool)
    n = int(mask.sum())
    if not n: return dict(n=0,coverage=0.,before=None,after=None,gain=None)
    b, a = float(rec['before'][mask].mean()),float(rec['after'][mask].mean())
    return dict(n=n,coverage=float(mask.mean()),before=b,after=a,
                gain=1-a/b if b>1e-12 else None)


def select_gate(rec, min_rows=64):
    """Calibrate on a fixed small grid; choose highest coverage passing >=10%."""
    options=[]
    for quantile in (.50,.75,.90):
        t = np.quantile(rec['stats'],quantile,axis=0)
        mask = np.all(rec['stats']<=t,axis=1)
        metric = masked_metrics(rec,mask)
        options.append(dict(quantile=quantile, thresholds=t.tolist(), **metric))
    feasible=[x for x in options if x['n']>=min_rows and x['coverage']>=.20
              and x['gain'] is not None and x['gain']>=.10]
    best=max(feasible,key=lambda x:x['coverage']) if feasible else None
    return {'enabled':best is not None,'selected':best,'trials':options}


def accepted(stats, gate):
    if not gate['enabled']: return np.zeros(len(stats),dtype=bool)
    return np.all(stats<=np.asarray(gate['selected']['thresholds']),axis=1)


def correct_queries(memory, z, u0, uq, ep, gate_list, old_target, raw_uq, raw_u0):
    """Exact reference cancellation. Rejected horizons get no new loss.

    raw bounds are enforced per prefix; never clip official CEM plans.
    Targets remain estimated, not counterfactual ground truth.
    """
    n,h,d=old_target.shape
    if uq.shape[:2]!=(n,h) or z.shape!=(n,d): raise ValueError('Query shapes differ')
    finite(old_target,'old query target'); finite(uq,'query actions')
    finite(raw_uq,'raw query actions'); finite(raw_u0,'raw reference actions')
    target=old_target.copy(); mask=np.zeros((n,h),dtype=bool)
    all_stats=[]
    for t in range(1,h+1):
        rz, sz, _=memory.estimate(z,u0,t,ep)
        rq, sq, _=memory.estimate(z,uq,t,ep)
        zz,aa=features(z,u0,t,memory.scales)
        _,ab=features(z,uq,t,memory.scales)
        stat=np.column_stack([np.maximum(sz,sq),np.zeros(n),np.sum((aa-ab)**2,1)])
        domain=(np.abs(raw_uq[:,:t])<=1+1e-6).all((1,2)) & (np.abs(raw_u0[:,:t])<=1+1e-6).all((1,2))
        mask[:,t-1]=accepted(stat,gate_list[t-1]) & domain
        target[:,t-1]+=rq-rz
        all_stats.append(stat)
    finite(target,'corrected targets')
    return target,mask,np.stack(all_stats,1)


def query_loss(pred, target, mask):
    """Global masked mean under DDP; all ranks participate, including zero masks."""
    if pred.shape!=target.shape or mask.shape!=pred.shape[:2]:
        raise ValueError('Query loss shapes differ')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite query prediction/target')
    per=(pred.float()-target.detach().float()).square().mean(-1)
    count=mask.detach().to(per).sum()
    world=1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world=torch.distributed.get_world_size()
        torch.distributed.all_reduce(count)
    return (per*mask.detach().to(per)).sum()*world/count.clamp_min(1)
