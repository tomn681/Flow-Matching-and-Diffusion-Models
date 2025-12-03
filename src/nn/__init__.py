"""
Core neural network components shared across the diffusion / flow-matching
library. The package is organised into reusable blocks and low-level ops, plus
assembled architectures such as the Efficient UNet.
"""

from . import blocks, ops, losses
from .blocks import (
    QKVAttention,
    LinearQKVAttention,
    SpatialSelfAttention,
    TimestepBlock,
    zero_module,
    ResBlockND,
)
from .ops import (
    ConvND,
    ConvTransposeND,
    PoolND,
    AvgPoolND,
    MaxPoolND,
    UnPoolND,
    UpsampleND,
    DownsampleND,
    timestep_embedding,
)
from .losses import (
    PerceptualLoss,
    PatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
    vq_regularizer,
)
__all__ = [
    "blocks",
    "ops",
    "losses",
    "QKVAttention",
    "LinearQKVAttention",
    "SpatialSelfAttention",
    "TimestepBlock",
    "zero_module",
    "ResBlockND",
    "ConvND",
    "ConvTransposeND",
    "PoolND",
    "AvgPoolND",
    "MaxPoolND",
    "UnPoolND",
    "UpsampleND",
    "DownsampleND",
    "timestep_embedding",
    "PerceptualLoss",
    "PatchDiscriminator",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "vq_regularizer",
]
