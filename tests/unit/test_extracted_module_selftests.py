from __future__ import annotations

import pytest
import torch

from core.types import ModelOutput
from nn.blocks.attention import SpatialCrossAttention, SpatialSelfAttention
from nn.blocks.residual import ResBlockND
from nn.ops.pooling import PoolND, UnPoolND
from nn.ops.upsampling import DownsampleND, UpsampleND
from utils.model_utils.vae_utils import decode_vae_batch, encode_vae_batch, reconstruct_vae_batch


def test_attention_blocks_shape_smoke() -> None:
    x = torch.randn(2, 16, 8, 8)
    y = SpatialSelfAttention(channels=16, spatial_dims=2)(x)
    assert y.shape == x.shape

    y_multi = SpatialSelfAttention(channels=16, num_heads=4, spatial_dims=2)(x)
    assert y_multi.shape == x.shape

    ctx_map = torch.randn(2, 4, 8, 8)
    z = SpatialCrossAttention(dim=16, context_dim=4, spatial_dims=2)(x, ctx_map)
    assert z.shape == x.shape


def test_spatial_self_attention_rejects_invalid_head_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SpatialSelfAttention(channels=10, num_heads=4, spatial_dims=2)


def test_resblock_nd_shape_smoke() -> None:
    x = torch.randn(2, 32, 8, 8)
    block = ResBlockND(spatial_dims=2, channels=32, emb_channels=64, out_channels=32, dropout=0.0)
    emb = torch.randn(2, 64)
    y = block(x, emb)
    assert y.shape == x.shape


def test_pool_unpool_roundtrip_shapes() -> None:
    x = torch.randn(2, 3, 16, 16)
    pool = PoolND(spatial_dims=2, in_channels=3, out_channels=8, pool_factor=2)
    unpool = UnPoolND(spatial_dims=2, in_channels=8, out_channels=3, pool_factor=2)
    h = pool(x)
    y = unpool(h)
    assert h.shape == (2, 8, 8, 8)
    assert y.shape == x.shape


def test_up_downsample_shapes() -> None:
    x = torch.randn(2, 8, 16, 16)
    down = DownsampleND(spatial_dims=2, channels=8, use_conv=True)
    up = UpsampleND(spatial_dims=2, channels=8, use_conv=True)
    h = down(x)
    y = up(h)
    assert h.shape == (2, 8, 8, 8)
    assert y.shape == x.shape


def test_vae_utils_encode_decode_reconstruct_contract() -> None:
    class _Posterior:
        def __init__(self, t: torch.Tensor):
            self._t = t

        def mode(self) -> torch.Tensor:
            return self._t

    class _Dummy:
        def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def raw_output_to_image(self, x: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
            return x

        def encode(self, x: torch.Tensor, normalize: bool = False):
            return _Posterior(x)

        def decode(self, z: torch.Tensor, denorm: bool = False) -> torch.Tensor:
            return z

        def __call__(self, x: torch.Tensor, sample_posterior: bool = False):
            return ModelOutput(reconstruction=x)

    model = _Dummy()
    x = torch.randn(2, 1, 8, 8)
    assert encode_vae_batch(model, x).shape == x.shape
    assert decode_vae_batch(model, x).shape == x.shape
    assert reconstruct_vae_batch(model, x).shape == x.shape
