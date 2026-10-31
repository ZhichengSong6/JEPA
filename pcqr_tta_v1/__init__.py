"""Planning-query response-preserving test-time adaptation for LeWM.

This is an experimental implementation, not a claim of improved task success.
"""
from .core import AdaptConfig, OnlineAdapter, TransitionBuffer, rollout_latent

__all__ = ["AdaptConfig", "OnlineAdapter", "TransitionBuffer", "rollout_latent"]
__version__ = "0.1.1"
