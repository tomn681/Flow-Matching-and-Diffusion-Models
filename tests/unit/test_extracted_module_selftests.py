from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from core.types import ModelOutput
from nn.blocks.attention import LegacyQKVSpatialSelfAttention, QKVAttention, QKVSpatialSelfAttention, SpatialCrossAttention, SpatialSelfAttention
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


def test_corrected_qkv_spatial_self_attention_matches_sdpa_reference() -> None:
    torch.manual_seed(0)
    layer = QKVSpatialSelfAttention(dim=8, heads=2, dim_head=4, use_linear=False, use_efficient_attn=True)
    x = torch.randn(2, 8, 4, 4)
    with torch.no_grad():
        torch.nn.init.normal_(layer.qkv.weight)
        torch.nn.init.normal_(layer.qkv.bias)
        torch.nn.init.normal_(layer.proj_out.weight)
        torch.nn.init.normal_(layer.proj_out.bias)

    y = layer(x)

    b, c, *spatial = x.shape
    residual = x.reshape(b, c, -1)
    h = layer.norm(x).reshape(b, c, -1)
    qkv = layer.qkv(h)
    q, k, v = qkv.chunk(3, dim=1)
    q = q.reshape(b, layer.heads, layer.dim_head, -1).transpose(-2, -1)
    k = k.reshape(b, layer.heads, layer.dim_head, -1).transpose(-2, -1)
    v = v.reshape(b, layer.heads, layer.dim_head, -1).transpose(-2, -1)
    attn = F.scaled_dot_product_attention(q, k, v)
    h_ref = attn.transpose(-2, -1).reshape(b, layer.inner_dim, -1)
    h_ref = layer.proj_out(h_ref)
    y_ref = (residual + h_ref).reshape(b, c, *spatial)

    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-6)


def test_legacy_qkv_spatial_self_attention_is_not_equivalent_to_corrected_layout() -> None:
    torch.manual_seed(0)
    fixed = QKVSpatialSelfAttention(dim=8, heads=2, dim_head=4, use_linear=False, use_efficient_attn=False)
    legacy = LegacyQKVSpatialSelfAttention(dim=8, heads=2, dim_head=4, use_linear=False, use_efficient_attn=False)
    x = torch.randn(2, 8, 4, 4)
    with torch.no_grad():
        legacy.norm.weight.copy_(fixed.norm.weight)
        legacy.norm.bias.copy_(fixed.norm.bias)
        legacy.qkv.weight.copy_(fixed.qkv.weight)
        legacy.qkv.bias.copy_(fixed.qkv.bias)
        legacy.proj_out.weight.copy_(fixed.proj_out.weight)
        legacy.proj_out.bias.copy_(fixed.proj_out.bias)
        torch.nn.init.normal_(fixed.qkv.weight)
        torch.nn.init.normal_(fixed.qkv.bias)
        torch.nn.init.normal_(fixed.proj_out.weight)
        torch.nn.init.normal_(fixed.proj_out.bias)
        legacy.qkv.weight.copy_(fixed.qkv.weight)
        legacy.qkv.bias.copy_(fixed.qkv.bias)
        legacy.proj_out.weight.copy_(fixed.proj_out.weight)
        legacy.proj_out.bias.copy_(fixed.proj_out.bias)

    assert isinstance(fixed.attention, QKVAttention)
    assert isinstance(legacy.attention, QKVAttention)
    assert not torch.allclose(fixed(x), legacy(x))


def test_resblock_nd_shape_smoke() -> None:
    x = torch.randn(2, 32, 8, 8)
    block = ResBlockND(
        spatial_dims=2,
        channels=32,
        emb_channels=64,
        out_channels=32,
        dropout=0.0,
        add_embedding_to_hidden=True,
    )
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
