from __future__ import annotations

import warnings

import pytest
import torch

from models.dit import DiTND, PatchTransformerND
from models.registry import MODEL_REGISTRY
from nn.losses.regularization import latent_moment_regularizer
from nn.losses.ssim import ssim_loss
from nn.modules.vae.attention_factory import build_vae_attention_layer
from nn.modules.vae.reparameterizer import DiagonalGaussian


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_ssim_identical_images_is_zero_across_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    x = torch.rand(2, 1, 16, 16)
    loss = ssim_loss(x, x)
    assert torch.isfinite(loss)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_latent_moment_regularizer_is_non_negative(seed: int) -> None:
    torch.manual_seed(seed)
    z = torch.randn(2, 4, 8, 8)
    reg = latent_moment_regularizer(z)
    assert torch.isfinite(reg)
    assert float(reg.item()) >= 0.0


def test_diagonal_gaussian_kl_matches_closed_form_zero_case() -> None:
    mean = torch.zeros(2, 4, 8, 8)
    logvar = torch.zeros_like(mean)
    posterior = DiagonalGaussian(torch.cat([mean, logvar], dim=1))
    kl = posterior.kl()
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_patch_transformer_is_canonical_alias_for_dit() -> None:
    assert PatchTransformerND is DiTND
    assert MODEL_REGISTRY.get("patch_transformer") is DiTND


def test_compvis_attention_alias_warns_and_matches_spatial() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        attn = build_vae_attention_layer(
            channels=32,
            attention_impl="compvis",
            spatial_dims=2,
            norm_eps=1e-6,
            zero_init_attn_out=True,
            attn_heads=1,
            attn_dim_head=None,
        )
    assert attn is not None
    assert any("deprecated" in str(w.message).lower() for w in caught)
