"""
Core neural network components shared across the diffusion / flow-matching
library. The package is organised into reusable blocks and low-level ops, plus
assembled architectures such as the Efficient UNet.
"""

from . import blocks, ops
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
__all__ = [
    "blocks",
    "ops",
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
]
