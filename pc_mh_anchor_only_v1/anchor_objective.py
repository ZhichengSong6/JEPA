"""PC-anchor-only: a factual-center control, NOT a new target method.

Reuses the original TF, planner-context rollout, causal mask and reduction.
No sampler, teacher forward, residual memory, query pool or simulator is used.
"""
from __future__ import annotations
import math
import torch


def center_mask_loss(error: torch.Tensor, directions: int = 1) -> torch.Tensor:
    """Zero-probe specialization of the original MH reduction, with gradients.

    error: [B,H,D]. Position p contributes only to h >= p (zero-based).
    For equal directions at every position this is sum_h h*MSE_h/sum_h h.
    expand is a view: no repeated predictor computation is needed.
    """
    from anchored_local_dynamics import _mh_causal_mask, _mh_masked_mse
    if error.ndim != 3 or min(error.shape) < 1:
        raise ValueError('Expected nonempty error [B,H,D]')
    if type(directions) is not int or directions < 1:
        raise ValueError('directions must be a positive integer')
    h = error.shape[1]
    positions = torch.arange(h, device=error.device).repeat_interleave(directions)
    mask = _mh_causal_mask(positions, h, dtype=error.dtype, device=error.device)
    return _mh_masked_mse(error[:, None].expand(-1, len(positions), -1, -1), mask)


def anchor_only_forward(self, batch, stage, cfg, action_mean, action_std):
    """Same signature as pc_mh_ald_forward; top-level/pickleable callable."""
    from anchored_local_dynamics import _frozen_visual_encode
    from planner_context_mh import planner_context_rollout
    student = self.model
    ctx, horizon = int(cfg.wm.history_size), int(cfg.mh_ald.rollout_horizon)
    pc = cfg.mh_ald.planner_context
    if (ctx, int(cfg.wm.num_preds), int(pc.observation_history),
            int(pc.predictor_max_history)) != (3, 1, 1, 3):
        raise ValueError('Expected original context 3 / num_preds 1 / planner 1->2->3')
    if not cfg.mh_ald.enabled or not pc.enabled or horizon < 1:
        raise ValueError('Expected enabled baseline MH/planner_context configuration')
    student.eval()  # autograd remains enabled; preserve baseline BN/dropout semantics
    if hasattr(self, 'teacher_model'):
        self.teacher_model.eval()  # retained by the unchanged initializer; never called
    student.encoder.eval()
    student.projector.eval()
    # Preserve baseline invalid-action behavior, NOT a new data cleaning rule.
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    if batch['pixels'].shape[1] < ctx + horizon:
        raise ValueError('Insufficient observation frames')
    if batch['action'].shape[-1] != student.action_encoder.patch_embed.in_channels:
        raise ValueError('Packed action dimension mismatch')
    emb = _frozen_visual_encode(student, batch)
    target = emb[:, ctx:ctx+horizon].detach()
    current = emb[:, ctx-1].detach()
    plan = batch['action'][:, ctx-1:ctx-1+horizon]
    if plan.shape[1] != horizon:
        raise ValueError('Insufficient recorded action blocks')
    # Identical TF branch to the pinned PC-MH implementation.
    pred = student.predict(emb[:, :ctx], student.action_encoder(batch['action'][:, :ctx]))
    tf = (pred - emb[:, 1:1+ctx].detach()).pow(2).mean()
    rollout = planner_context_rollout(student, current, plan, int(pc.predictor_max_history))
    error = rollout - target
    roll = error.pow(2).mean()
    anchor = center_mask_loss(error, int(cfg.mh_ald.directions_per_position))
    weights = [float(cfg.mh_ald[k]) for k in ('tf_weight', 'rollout_weight', 'weight')]
    if not all(math.isfinite(w) and w >= 0 for w in weights):
        raise ValueError('Invalid loss weights')
    result = {'loss': weights[0]*tf + weights[1]*roll + weights[2]*anchor,
              'tf_loss': tf, 'pc_rollout_loss': roll, 'pc_anchor_loss': anchor}
    result.update({f'h{h+1}_mse': error[:, h].pow(2).mean() for h in range(horizon)})
    if not all(bool(torch.isfinite(v).all()) for v in result.values()):
        raise RuntimeError('Nonfinite loss: do not export a corrupt model')
    # Direct outputs, not callback_metrics aliases: prevents empty CSV loss fields.
    if not hasattr(self, '_anchor_last'):
        self._anchor_last = {}
    # stable_pretraining.Module.training_step passes 'fit', whereas our
    # training callbacks consume 'train'. Normalize this PRIVATE cache only;
    # preserve the framework's original stage in log_dict keys below.
    cache_stage = 'train' if stage in ('fit', 'train') else stage
    self._anchor_last[cache_stage] = {k: v.detach() for k, v in result.items()}
    self._anchor_last_raw_stage = stage
    self.log_dict({f'{stage}/{k}': v.detach() for k, v in result.items()},
                  on_step=True, sync_dist=True)
    return result
