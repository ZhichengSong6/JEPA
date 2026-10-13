"""Official installed CEM generates training queries, never outcome labels.

The cost adapter consumes cached frozen latents. No World/env is constructed;
no candidate is executed in a simulator. No gradients through search.
"""
from __future__ import annotations
import numpy as np
import torch


class LatentCost(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.teacher=model; self.calls=[]

    @torch.inference_mode()
    def get_cost(self, info, candidates):
        from planner_context_mh import planner_context_rollout
        if tuple(candidates.shape)!=(1,100,5,10):
            raise RuntimeError(f'Official CEM shape changed: {tuple(candidates.shape)}')
        z=info['z0'][:,0,-1].to(candidates.device)
        goal=info['zg'][:,0,-1].to(candidates.device)
        pred=planner_context_rollout(self.teacher,z.expand(100,-1),candidates[0],3)
        cost=(pred[:,-1]-goal).square().sum(-1)[None]
        if not torch.isfinite(cost).all(): raise RuntimeError('Nonfinite teacher CEM cost')
        self.calls.append((candidates[0].detach().cpu().clone(),cost[0].detach().cpu().clone()))
        return cost


def select_queries(population,cost,returned,seed):
    """Mean, best distinct elite, first distinct non-elite, independent random.
    Stable fallback order fills duplicates, without injecting the factual action.
    """
    p=torch.as_tensor(population);c=torch.as_tensor(cost);ret=torch.as_tensor(returned)
    if p.shape!=(100,5,10) or c.shape!=(100,) or ret.shape!=(5,10): raise ValueError('CEM export shapes')
    if not all(bool(torch.isfinite(t).all()) for t in (p,c,ret)): raise ValueError('Nonfinite candidates')
    order=torch.argsort(c,stable=True).tolist()
    rng=np.random.default_rng(seed)
    chosen=[ret.clone()]; kinds=['returned_mean']
    pools=[('elite',order[:10]+order[10:]),
           ('boundary',order[10:]+order[:10]),
           ('random',rng.permutation(100).tolist())]
    for kind,indices in pools:
        for i in indices:
            if not any(torch.equal(p[i],v) for v in chosen):
                chosen.append(p[i].clone());kinds.append(kind);break
        else: raise RuntimeError('Collapsed population cannot provide four distinct queries')
    return torch.stack(chosen).numpy(),kinds


@torch.inference_mode()
def generate(model,z0,goal,seed,device='cuda:0'):
    import gymnasium as gym
    import stable_worldmodel as swm
    adapter=LatentCost(model)
    solver=swm.solver.CEMSolver(model=adapter,batch_size=1,num_samples=100,
        var_scale=1.,n_steps=10,topk=10,device=device,seed=int(seed))
    solver.configure(action_space=gym.spaces.Box(-1.,1.,shape=(1,2),dtype=np.float32),
        n_envs=1,config=swm.PlanConfig(horizon=5,receding_horizon=5,action_block=5))
    info={'z0':torch.as_tensor(z0,dtype=torch.float32)[None,None],
          'zg':torch.as_tensor(goal,dtype=torch.float32)[None,None]}
    result=solver(info)
    if len(adapter.calls)!=10: raise RuntimeError('Official CEM iteration count differs')
    p,c=adapter.calls[-1]
    returned=torch.as_tensor(result['actions']).detach().cpu()[0]
    # Same device float32 arithmetic as the native solver, including topk ties.
    gp=p.to(device); gc=c.to(device)
    expected=gp[torch.topk(gc,10,largest=False).indices].mean(0).cpu()
    if not torch.allclose(returned,expected,rtol=0,atol=1e-6):
        raise RuntimeError('Returned action differs from final official elite mean')
    query,kinds=select_queries(p,c,returned,seed+700001)
    return query,kinds,p.numpy(),c.numpy()


@torch.inference_mode()
def validate_pixel_cost(model,item,plans,z0,goal,device):
    """Real pixel get_cost vs cached-latent adapter before generating full pool."""
    from planner_context_mh import planner_context_rollout
    a=torch.as_tensor(plans,dtype=torch.float32,device=device)[None]
    n=a.shape[1]
    pix=torch.as_tensor(item['pixels']).float()
    info={'pixels':pix[2:3][None,None].expand(1,n,-1,-1,-1,-1).clone(),
          'goal':pix[7:8][None,None].expand(1,n,-1,-1,-1,-1).clone(),
          'action':torch.zeros(1,n,1,10)}
    actual=model.get_cost(info,a)[0]
    z=torch.as_tensor(z0,dtype=torch.float32,device=device)[None].expand(n,-1)
    g=torch.as_tensor(goal,dtype=torch.float32,device=device)
    direct=(planner_context_rollout(model,z,a[0],3)[:,-1]-g).square().sum(-1)
    error=float((actual-direct).abs().max())
    if not torch.allclose(actual,direct,rtol=1e-4,atol=1e-4):
        raise RuntimeError(f'Cached-latent and actual JEPA goal costs differ: max_abs={error}')
    return error
