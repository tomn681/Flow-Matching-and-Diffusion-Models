"""
Training and inference pipelines for the diffusion / VAE components.
"""

from .train_vae import main as train_vae

__all__ = ["train_vae"]
