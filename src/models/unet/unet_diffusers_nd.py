"""
Deprecated compatibility module.

Prefer importing from `models.unet.diffusers`.
"""

from .diffusers import UNetDiffusersND, UNetExactND

__all__ = ["UNetDiffusersND", "UNetExactND"]

