"""
Training entrypoints for individual models.
"""

from .vae_lib import train as train_vae

__all__ = ["train_vae"]
