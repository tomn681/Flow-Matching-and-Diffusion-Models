"""
Training and inference pipelines for the diffusion / VAE components.
"""

from .sample_vae import main as sample_vae

__all__ = ["sample_vae"]
