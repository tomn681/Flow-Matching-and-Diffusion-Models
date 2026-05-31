"""
Higher-level neural modules grouped by task (e.g., VAE components).
"""

from . import vae
from .lora import LoRALinear

__all__ = ["vae", "LoRALinear"]
