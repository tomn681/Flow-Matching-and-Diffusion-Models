from __future__ import annotations

import torch

from scheduling import CONDITIONING_ADAPTER_REGISTRY, resolve_conditioning_adapter


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
