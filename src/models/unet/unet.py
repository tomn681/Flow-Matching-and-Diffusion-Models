"""
Deprecated compatibility module.

Prefer importing from `models.unet.efficient`.
"""

from .efficient import EfficientUNetND, TimestepEmbedSequential

__all__ = ["EfficientUNetND", "TimestepEmbedSequential"]

