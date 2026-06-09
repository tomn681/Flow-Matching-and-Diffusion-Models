"""
Compatibility re-export layer for diffusion/flow model helpers.

Canonical ownership now lives in:
- `utils.model_utils.diffusion_loading`
- `utils.model_utils.diffusion_runtime`
"""

from __future__ import annotations

from .diffusion_loading import build_diffusion_model, warn_attention_conditioning_shape
from .diffusion_runtime import decode_diffusion_batch, encode_diffusion_batch, prepare_diffusion_visual_batch

__all__ = [
    "build_diffusion_model",
    "warn_attention_conditioning_shape",
    "encode_diffusion_batch",
    "decode_diffusion_batch",
    "prepare_diffusion_visual_batch",
]
