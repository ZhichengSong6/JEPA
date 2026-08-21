"""Small distributed-training helpers for controlled LeWM experiments.

The original SIGReg implementation estimates its statistic across the batch
axis. Plain DDP would silently change that objective because each rank would
see only its local mini-batch.  ``global_batch_sigreg`` restores the single-GPU
semantics by differentiably gathering the embedding batch across ranks before
computing the statistic.

The random projection matrix is broadcast from rank 0 so every rank evaluates
exactly the same global SIGReg objective.  Default/single-GPU training remains
unchanged when ``enabled=False`` or torch.distributed is not initialized.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as differentiable_all_gather


def distributed_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size())


def global_batch_sigreg(sigreg, proj: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    """Evaluate SIGReg on the global DDP batch with correct autograd.

    Args:
        sigreg: the existing ``module.SIGReg`` instance.
        proj: tensor shaped ``(T, B_local, D)``.
        enabled: when False, call the original single-rank implementation.

    With W ranks, every rank builds the same ``(T, W*B_local, D)`` tensor and
    computes the same scalar objective using the same random projections.
    ``torch.distributed.nn.functional.all_gather`` supplies the backward
    reduce-scatter, while DDP's parameter-gradient averaging yields the same
    global-batch gradient scale as a single process operating on the gathered
    batch.
    """
    if proj.ndim != 3:
        raise ValueError(f"SIGReg expects (T,B,D), got {tuple(proj.shape)}")

    world_size = distributed_world_size()
    if not enabled or world_size == 1:
        return sigreg(proj)

    gathered = differentiable_all_gather(proj)
    global_proj = torch.cat(tuple(gathered), dim=1)

    # Mirror SIGReg.forward exactly, except that A is shared by all ranks.  All
    # ranks sample once so their RNG streams advance consistently; broadcast
    # then guarantees identical projections even if rank RNG states differ.
    A = torch.randn(
        global_proj.size(-1),
        int(sigreg.num_proj),
        device=global_proj.device,
        dtype=torch.float32,
    )
    dist.broadcast(A, src=0)
    A = A.div_(A.norm(p=2, dim=0).clamp_min_(1e-12))

    x_t = (global_proj @ A).unsqueeze(-1) * sigreg.t
    err = (
        (x_t.cos().mean(-3) - sigreg.phi).square()
        + x_t.sin().mean(-3).square()
    )
    statistic = (err @ sigreg.weights) * global_proj.size(-2)
    return statistic.mean()
