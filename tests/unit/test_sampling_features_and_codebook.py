from __future__ import annotations

import torch

from scheduling.sampling_loop import _apply_cfg_rescale
from nn.modules.vae.codebook import VectorQuantizer, VectorQuantizerEMA
from models.vae.vq import VQVAE


def test_cfg_rescale_adjusts_guided_prediction_std() -> None:
    pred_cond = torch.randn(2, 1, 8, 8)
    pred = pred_cond * 4.0
    adjusted = _apply_cfg_rescale(pred, pred_cond=pred_cond, guidance_scale=7.5, cfg_rescale=0.7)
    assert adjusted.shape == pred.shape
    assert not torch.allclose(adjusted, pred)


def test_vector_quantizer_uses_small_uniform_init() -> None:
    quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=4)
    assert float(quantizer.embedding.abs().max().item()) <= (1.0 / 16.0) + 1e-6


def test_vector_quantizer_ema_tracks_usage_and_revives_codes() -> None:
    quantizer = VectorQuantizerEMA(
        num_embeddings=8,
        embedding_dim=4,
        decay=0.9,
        dead_code_threshold=10.0,
        revive_dead_codes=True,
        track_usage=True,
    )
    before = quantizer.embedding.clone()
    x = torch.randn(2, 4, 4, 4)
    quantizer.train()
    _, _, _, _ = quantizer(x)
    assert "usage_fraction" in quantizer.last_telemetry
    assert not torch.allclose(before, quantizer.embedding)


def test_vector_quantizer_l2_normalized_codes_stay_unit_norm() -> None:
    quantizer = VectorQuantizerEMA(
        num_embeddings=8,
        embedding_dim=4,
        l2_normalize_codes=True,
    )
    x = torch.randn(2, 4, 4, 4)
    quantizer.train()
    quantizer(x)
    norms = quantizer.embedding.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_vqvae_exposes_codebook_usage_telemetry() -> None:
    model = VQVAE(
        in_channels=1,
        out_channels=1,
        resolution=8,
        base_ch=32,
        ch_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        z_channels=4,
        embed_dim=4,
        use_attention=False,
        spatial_dims=2,
        codebook_size=8,
        quantizer_type="ema",
        track_code_usage=True,
    )
    x = torch.randn(2, 1, 8, 8)
    output = model(x)
    assert output.auxiliary is not None
    assert "usage_fraction" in output.auxiliary
    assert "dead_code_fraction" in output.auxiliary
