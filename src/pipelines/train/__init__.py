"""
Training entrypoints for individual models.
"""

from .vae_lib import train as train_vae
from .flow_matching_lib import train as train_flow_matching
from .diffusion_lib import train as train_diffusion

__all__ = ["train_vae", "train_flow_matching", "train_diffusion"]
