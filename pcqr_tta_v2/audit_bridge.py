"""Audited, matched-first-feedback adapter around PCQR-TTA v1.

Pure read-only fingerprints; model updates remain delegated to v1. The audit
is run before the first CEM population gets adapted/scored, for every mode.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn

AUDIT_KEYS = (
    "raw_steps",
    "issued_action_prefix_sha256",
    "current_replan_pixels_sha256",
    "first_cem_population_sha256",
    "selected_parameters_sha256",
)


def _bytes_digest(x: Any) -> str:
    """Hash the exact dtype, shape, and bytes; no tolerance-based equivalence."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    v = np.ascontiguousarray(np.asarray(x))
    if v.dtype.kind not in 'biufc':
        raise TypeError(f"Unexpected array dtype for audit: {v.dtype}")
    h = hashlib.sha256()
    h.update(v.dtype.str.encode('ascii'))
    h.update(str(v.shape).encode('ascii'))
    h.update(v.tobytes(order='C'))
    return h.hexdigest()


def action_prefix_sha256(actions: list[Any]) -> str:
    """Digest of ordered *issued* raw actions, before the active planner call."""
    h = hashlib.sha256()
    h.update(str(len(actions)).encode('ascii'))
    for a in actions:
        h.update(_bytes_digest(a).encode('ascii'))
    return h.hexdigest()


def parameters_sha256(named_parameters) -> str:
    """Hash selected current weights; detect silent cross-mode carryover."""
    h = hashlib.sha256()
    for name, parameter in named_parameters:
        h.update(name.encode('utf-8'))
        h.update(_bytes_digest(parameter).encode('ascii'))
    return h.hexdigest()


def compare_preupdate(expected: dict, observed: dict) -> dict[str, tuple[Any, Any]]:
    """Exact audit difference; caller must fail closed for the strict protocol."""
    return {key: (expected.get(key), observed.get(key))
            for key in AUDIT_KEYS if key not in expected or key not in observed
            or expected[key] != observed[key]}


def make_audited_cost_class(base_cost_class=None):
    """Extend the installed v1 bridge without changing CEM or adaptation."""
    if base_cost_class is None:
        from pcqr_tta_v1.swm_bridge import AdaptiveCostModel as base_cost_class

    class AuditedCostModel(base_cost_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.issued_actions: list[Any] = []
            self.raw_steps_seen = 0
            self.replan_audits: list[dict] = []
            self.trace_dir = None  # optional, diagnostic; no effect on CEM/model
            self.trace_prefix = None
            self.current_raw_info = None

        def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
            # self.armed is True once per solver invocation, exactly before
            # the first scoring call. Never change, sample or rank candidates.
            if self.armed and self.replans <= 1:
                if not torch.is_tensor(info_dict.get('pixels')):
                    raise TypeError('Audit requires solver pixels tensor')
                p = info_dict['pixels']
                if p.ndim != 6 or p.shape[0] != 1 or p.shape[1] < 1:
                    raise ValueError(f'Unexpected solver pixels {tuple(p.shape)}')
                if action_candidates.ndim != 4 or action_candidates.shape[0] != 1:
                    raise ValueError('Audit requires [1,Q,H,A] candidate tensor')
                audit = {
                    'replan': self.replans,
                    'raw_steps': self.raw_steps_seen,
                    'issued_action_prefix_sha256': action_prefix_sha256(self.issued_actions),
                    'current_replan_pixels_sha256': _bytes_digest(p[0, 0]),
                    'first_cem_population_sha256': _bytes_digest(action_candidates[0]),
                    'selected_parameters_sha256': parameters_sha256(self.adapter.named),
                }
                if self.trace_dir is not None:
                    from .trace_audit import save_before_update
                    trace = save_before_update(
                        self.trace_dir, self.trace_prefix, self.replans,
                        solver_info=info_dict,
                        candidates=action_candidates,
                        issued_actions=self.issued_actions,
                        raw_info=self.current_raw_info,
                    )
                    audit['numeric_trace_file'] = str(trace)
                self.replan_audits.append(audit)
                if self.log is not None:
                    self.log({'type': 'preupdate_audit', **audit})
            return super().get_cost(info_dict, action_candidates)

    return AuditedCostModel


def make_audited_policy_class(base_policy_class, online_factory=None):
    """Capture raw issued actions, including frozen, without altering policy."""
    if online_factory is None:
        from pcqr_tta_v1.swm_bridge import make_online_policy_class as online_factory
    base = online_factory(base_policy_class)

    class AuditedWorldModelPolicy(base):
        def get_action(self, info_dict: dict, **kwargs):
            # The next get_cost() may be the first CEM call at a replan.
            # At that point, all actions here were already returned by policy.
            self.online_cost.raw_steps_seen = len(self.online_cost.issued_actions)
            # A read-only reference valid until the synchronous CEM call; the
            # numerical probe clones it before this environment step advances.
            if self.online_cost.trace_dir is not None:
                self.online_cost.current_raw_info = info_dict
            action = super().get_action(info_dict, **kwargs)
            # Only need the prefix through the first feedback-driven replan.
            n_keep = int(self.cfg.receding_horizon) * int(self.cfg.action_block) + 1
            if len(self.online_cost.issued_actions) < n_keep:
                v = np.array(action, copy=True)
                if not np.isfinite(v).all():
                    raise ValueError('Nonfinite issued action prevents audit')
                self.online_cost.issued_actions.append(v)
            return action

    return AuditedWorldModelPolicy
