from __future__ import annotations

import torch
import torch.nn as nn

from scheduling import CONDITIONING_ADAPTER_REGISTRY, LatentAttentionAdapter, resolve_conditioning_adapter


def test_conditioning_adapter_registry_entries() -> None:
    assert CONDITIONING_ADAPTER_REGISTRY.list() == ["attention", "concatenate", "latent_attention", "none"]


def test_none_adapter_returns_input_and_no_context() -> None:
    adapter = resolve_conditioning_adapter("none")
    x = torch.randn(2, 1, 8, 8)
    out_x, ctx = adapter(x, None, None)
    assert torch.equal(out_x, x)
    assert ctx is None


def test_concatenate_adapter_concatenates_channels() -> None:
    adapter = resolve_conditioning_adapter("concatenate")
    x = torch.randn(2, 1, 8, 8)
    cond = torch.randn(2, 1, 8, 8)
    out_x, ctx = adapter(x, cond, None)
    assert out_x.shape[1] == 2
    assert ctx is None


def test_attention_adapter_returns_context_tensor() -> None:
    adapter = resolve_conditioning_adapter("attention")
    x = torch.randn(2, 1, 8, 8)
    cond = torch.randn(2, 1, 8, 8)
    out_x, ctx = adapter(x, cond, None)
    assert torch.equal(out_x, x)
    assert ctx is not None
    assert ctx.shape == cond.shape


def test_latent_attention_adapter_returns_context_tensor() -> None:
    adapter = resolve_conditioning_adapter("latent_attention")
    x = torch.randn(2, 1, 8, 8)
    cond = torch.randn(2, 1, 8, 8)
    out_x, ctx = adapter(x, cond, "standardize")
    assert torch.equal(out_x, x)
    assert ctx is not None
    assert ctx.shape == cond.shape


def test_latent_attention_adapter_encodes_via_configured_vae() -> None:
    class _DummyVAE(nn.Module):
        def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def encode(self, x: torch.Tensor, normalize: bool = False):
            return x * 0.5 if normalize else x * 0.5

    adapter = LatentAttentionAdapter.create(_DummyVAE())
    x = torch.zeros(1, 1, 4, 4)
    cond = torch.ones(1, 1, 4, 4)
    _out_x, ctx = adapter(x, cond, None)
    assert ctx is not None
    assert torch.allclose(ctx, cond * 0.5)
