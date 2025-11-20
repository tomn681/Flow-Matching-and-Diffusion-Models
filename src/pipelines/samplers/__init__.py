"""
Sampling entrypoints for trained models.

Use via `python -m src.sample --sampler <name> ...`.
"""

from .vae import sample as sample_vae

__all__ = ["sample_vae"]
