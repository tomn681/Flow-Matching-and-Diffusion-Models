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
from .common import zero_module
from .residual import (
    ResBlockND,
    build_resblock_gn_silu,
    build_resblock_gn_swish,
    build_resblock_rmsnorm_silu,
    build_resblock_rmsnorm_swish,
)
from .timestep import TimestepBlock
from .legacy_unet import DownBlock2DCompat, UpBlock2DCompat, UNetMidBlock2DCompat

__all__ = [
    "QKVAttention",
    "LinearQKVAttention",
    "LegacyQKVSpatialSelfAttention",
    "LegacyQKVSpatialCrossAttention",
    "SpatialSelfAttention",
    "SpatialCrossAttention",
    "DiffusersAttentionND",
    "zero_module",
    "ResBlockND",
    "build_resblock_gn_silu",
    "build_resblock_gn_swish",
    "build_resblock_rmsnorm_silu",
    "build_resblock_rmsnorm_swish",
    "TimestepBlock",
    "DownBlock2DCompat",
    "UpBlock2DCompat",
    "UNetMidBlock2DCompat",
]
