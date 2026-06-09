"""
Trainer-facing composable loss components and the loss registry.

Boundary:
- `src.losses` owns components, composition, and registration.
- `src.nn.losses` owns the tensor-level math and small discriminator
  primitives these components are built on.
"""

from .assembler import LossAssembler
from .base import BaseLossComponent
from .adversarial import GANDiscriminatorLoss, GANGeneratorLoss
from .perceptual import PerceptualLossComponent
from .regularization import KLLoss, VQLoss
from .registry import LOSS_REGISTRY

# Import reconstruction module for registry side effects.
from . import reconstruction as _reconstruction  # noqa: F401

__all__ = [
    "BaseLossComponent",
    "GANDiscriminatorLoss",
    "GANGeneratorLoss",
    "KLLoss",
    "LossAssembler",
    "LOSS_REGISTRY",
    "PerceptualLossComponent",
    "VQLoss",
]
