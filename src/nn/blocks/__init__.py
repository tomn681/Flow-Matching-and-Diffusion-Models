"""
Reusable building blocks (attention, residual units, timestep-aware modules).
"""

from .attention import (
    DiffusersAttentionND,
    LegacyQKVSpatialCrossAttention,
    LegacyQKVSpatialSelfAttention,
    LinearQKVAttention,
    QKVAttention,
    SpatialCrossAttention,
    SpatialSelfAttention,
)
from .transformer import BasicTransformerBlock, Transformer2DModelND
from .common import zero_module
from .residual import (
    ResBlockND,
    build_resblock_gn_silu,
    build_resblock_gn_swish,
    build_resblock_rmsnorm_silu,
    build_resblock_rmsnorm_swish,
)
from .timestep import TimestepBlock
from .legacy_unet import (
    BLOCK_REGISTRY,
    CrossAttnDownBlock2DCompat,
    CrossAttnUpBlock2DCompat,
    DownBlock2DCompat,
    UNetMidBlock2DCrossAttnCompat,
    UNetMidBlock2DCompat,
    UpBlock2DCompat,
)

__all__ = [
    "QKVAttention",
    "LinearQKVAttention",
    "LegacyQKVSpatialSelfAttention",
    "LegacyQKVSpatialCrossAttention",
    "SpatialSelfAttention",
    "SpatialCrossAttention",
    "DiffusersAttentionND",
    "BasicTransformerBlock",
    "Transformer2DModelND",
    "zero_module",
    "ResBlockND",
    "build_resblock_gn_silu",
    "build_resblock_gn_swish",
    "build_resblock_rmsnorm_silu",
    "build_resblock_rmsnorm_swish",
    "TimestepBlock",
    "BLOCK_REGISTRY",
    "DownBlock2DCompat",
    "CrossAttnDownBlock2DCompat",
    "UpBlock2DCompat",
    "CrossAttnUpBlock2DCompat",
    "UNetMidBlock2DCompat",
    "UNetMidBlock2DCrossAttnCompat",
]
