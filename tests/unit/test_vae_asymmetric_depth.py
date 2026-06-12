from __future__ import annotations

import torch

from nn.blocks.attention import LegacyQKVSpatialSelfAttention, QKVSpatialSelfAttention, SpatialSelfAttention
from nn.modules.vae.attention_factory import build_vae_attention_layer
from models.vae.kl import AutoencoderKL
from models.vae.vq import VQVAE


def test_kl_vae_supports_asymmetric_decoder_depth() -> None:
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        resolution=16,
        base_ch=32,
        ch_mult=(1, 2),
        num_res_blocks={"encoder": 1, "decoder": 3},
        attn_resolutions=(),
        z_channels=4,
        embed_dim=4,
        use_attention=False,
        spatial_dims=2,
    )
    encoder_blocks = [len(stage.blocks) for stage in model.encoder.downs]
    decoder_blocks = [len(stage.blocks) for stage in model.decoder.ups]
    assert all(n == 1 for n in encoder_blocks)
    assert all(n == 4 for n in decoder_blocks)


def test_vq_vae_supports_asymmetric_decoder_depth() -> None:
    model = VQVAE(
        in_channels=1,
        out_channels=1,
        resolution=16,
        base_ch=32,
        ch_mult=(1, 2),
        num_res_blocks={"encoder": 1, "decoder": 3},
        attn_resolutions=(),
        z_channels=4,
        embed_dim=4,
        use_attention=False,
        spatial_dims=2,
        codebook_size=16,
        quantizer_type="classic",
    )
    encoder_blocks = [len(stage.blocks) for stage in model.encoder.downs]
    decoder_blocks = [len(stage.blocks) for stage in model.decoder.ups]
    assert all(n == 1 for n in encoder_blocks)
    assert all(n == 4 for n in decoder_blocks)


def test_spatial_vae_attention_factory_uses_attn_heads() -> None:
    layer = build_vae_attention_layer(
        channels=32,
        attention_impl="spatial",
        spatial_dims=2,
        norm_eps=1e-6,
        zero_init_attn_out=True,
        attn_heads=4,
        attn_dim_head=64,
    )
    assert isinstance(layer, SpatialSelfAttention)
    assert layer.attn.num_heads == 4
    assert layer.attn.out_proj.weight.abs().sum().item() == 0.0


def test_qkv_vae_attention_factory_uses_corrected_qkv_attention() -> None:
    layer = build_vae_attention_layer(
        channels=32,
        attention_impl="qkv",
        spatial_dims=2,
        norm_eps=1e-6,
        zero_init_attn_out=True,
        attn_heads=4,
        attn_dim_head=8,
    )
    assert isinstance(layer, QKVSpatialSelfAttention)


def test_legacy_qkv_vae_attention_factory_preserves_legacy_attention() -> None:
    layer = build_vae_attention_layer(
        channels=32,
        attention_impl="legacy_qkv",
        spatial_dims=2,
        norm_eps=1e-6,
        zero_init_attn_out=True,
        attn_heads=4,
        attn_dim_head=8,
    )
    assert isinstance(layer, LegacyQKVSpatialSelfAttention)


def test_kl_vae_spatial_attention_receives_configured_head_count() -> None:
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        resolution=16,
        base_ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(8,),
        z_channels=4,
        embed_dim=4,
        use_attention=True,
        attn_heads=4,
        attention_impl="spatial",
        spatial_dims=2,
    )
    stage_attn = model.encoder.downs[1].attns[0]
    assert isinstance(stage_attn, SpatialSelfAttention)
    assert stage_attn.attn.num_heads == 4


def test_kl_vae_zero_inits_spatial_attention_output_by_default() -> None:
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        resolution=16,
        base_ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(8,),
        z_channels=4,
        embed_dim=4,
        use_attention=True,
        attention_impl="spatial",
        spatial_dims=2,
    )
    stage_attn = model.encoder.downs[1].attns[0]
    assert isinstance(stage_attn, SpatialSelfAttention)
    assert stage_attn.attn.out_proj.weight.abs().sum().item() == 0.0


def test_kl_vae_applies_latent_dropout_only_while_training(monkeypatch) -> None:
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        resolution=16,
        base_ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(),
        z_channels=4,
        embed_dim=4,
        use_attention=False,
        spatial_dims=2,
        latent_dropout=0.1,
    )
    x = torch.randn(2, 1, 16, 16)
    calls: list[tuple[float, bool]] = []

    def _fake_dropout2d(z: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
        calls.append((p, training))
        return z

    monkeypatch.setattr("models.vae.kl.F.dropout2d", _fake_dropout2d)

    model.train()
    _ = model(x)
    assert calls == [(0.1, True)]

    calls.clear()
    model.eval()
    _ = model(x)
    assert calls == []
