"""
Backward-compatible wrappers for legacy training/sampling/config APIs.
"""

from .legacy_config import adapt_legacy_config_v1
from .legacy_samplers import DiffusionHandler, FlowMatchingHandler, ModelHandler, VAEHandler
from .legacy_training import train_diffusion, train_flow_matching, train_vae

__all__ = [
    "adapt_legacy_config_v1",
    "ModelHandler",
    "DiffusionHandler",
    "FlowMatchingHandler",
    "VAEHandler",
    "train_diffusion",
    "train_flow_matching",
    "train_vae",
]

