"""
Training entrypoints for individual models.
"""

from compat.legacy_training import (
    train_diffusion,
    train_flow_matching,
    train_vae,
)

__all__ = ["train_vae", "train_flow_matching", "train_diffusion"]
