"""
Backward-compatible re-exports for scheduling and sampling helpers.

This module is a compatibility surface. New ownership for scheduler building
and sampling-loop helpers lives under `src.scheduling`.
"""

from __future__ import annotations

from scheduling import (
    SCHEDULER_REGISTRY,
    _prepare_attention_context,
    build_scheduler,
    normalize_latent_conditioning,
    resolve_conditioning_mode,
    resolve_scheduler_override,
    sample_with_scheduler,
    sync_if_cuda,
)

__all__ = [
    "SCHEDULER_REGISTRY",
    "resolve_conditioning_mode",
    "build_scheduler",
    "resolve_scheduler_override",
    "normalize_latent_conditioning",
    "_prepare_attention_context",
    "sample_with_scheduler",
    "sync_if_cuda",
]
