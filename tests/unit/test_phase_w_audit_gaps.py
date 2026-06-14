from __future__ import annotations

import warnings

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from losses import LOSS_REGISTRY
from models.factory import ModelFactory
from nn.blocks.residual import ResBlockND
from nn.modules.vae.codebook import VectorQuantizerEMA
from nn.modules.vae.reparameterizer import DiagonalGaussian
from nn.ops.normalization import RMSNormND, make_group_norm
from noise import FlowMatchingNoise


def test_resblock_rejects_unused_embedding_path() -> None:
    with pytest.raises(ValueError, match="no embedding-consumption path"):
        ResBlockND(channels=8, emb_channels=16, dropout=0.0)


def test_group_norm_warns_when_group_count_mutates() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        norm = make_group_norm(10, groups=8)
    assert norm.num_groups == 5
    assert any("falling back" in str(w.message) for w in caught)


def test_rmsnorm_normalizes_channels_only() -> None:
    norm = RMSNormND(2)
    x = torch.tensor([[[[3.0, 3.0]], [[4.0, 4.0]]]])
    out = norm(x)
    expected_rms = torch.sqrt(torch.tensor([[(3.0**2 + 4.0**2) / 2.0]])).view(1, 1, 1, 1)
    expected = x / expected_rms
    assert torch.allclose(out, expected, atol=1e-6)


def test_diagonal_gaussian_kl_runs_in_fp32_and_returns_half_dtype() -> None:
    params = torch.randn(2, 8, 4, 4, dtype=torch.float16)
    posterior = DiagonalGaussian(params)
    out = posterior.kl()
    assert out.dtype == torch.float16
    assert out.shape == (2,)


def test_model_factory_converts_absolute_attention_resolutions_to_factors() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ModelFactory._build_efficient_unet(
            {
                "sample_size": 32,
                "model_channels": 32,
                "channel_mult": [1, 2, 4],
                "num_res_blocks": 1,
                "attention_resolutions": [16, 8],
                "in_channels": 4,
                "out_channels": 4,
            },
            cond_mode="",
            channels=4,
        )
    assert tuple(model.attention_resolutions) == (2, 4)
    assert any("downsample-factor units" in str(w.message) for w in caught)


def test_flow_matching_noise_supports_logit_normal_shift_sampling() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.5)
    process = FlowMatchingNoise(
        scheduler,
        timestep_sampling="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
    )
    clean = torch.randn(8, 1, 4, 4)
    out = process(clean, clean.device)
    assert out.timesteps.shape == (8,)
    assert torch.all(out.timesteps > 0)
    assert torch.all(out.timesteps < scheduler.config.num_train_timesteps - 1)


def test_denoising_loss_registered() -> None:
    assert "denoising_mse" in LOSS_REGISTRY


def test_vq_ema_all_reduces_stats_when_distributed(monkeypatch) -> None:
    calls = {"count": 0}

    monkeypatch.setattr("nn.modules.vae.codebook.utils.is_distributed", lambda: True)

    def _reduce(x):
        calls["count"] += 1
        return x

    monkeypatch.setattr("nn.modules.vae.codebook.utils.all_reduce_tensor", _reduce)
    quantizer = VectorQuantizerEMA(num_embeddings=8, embedding_dim=4)
    quantizer.train()
    _ = quantizer(torch.randn(2, 4, 8, 8))
    assert calls["count"] == 2
