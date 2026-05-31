from __future__ import annotations

import torch
import torch.nn as nn

from scheduling import CONDITIONING_ADAPTER_REGISTRY, LatentAttentionAdapter, resolve_conditioning_adapter
from scheduling.conditioning_chain import ChainAdapterSpec, ConditioningChain


def test_conditioning_adapter_registry_entries() -> None:
    assert CONDITIONING_ADAPTER_REGISTRY.list() == [
        "attention",
        "chain",
        "concatenate",
        "depth",
        "inpainting",
        "latent_attention",
        "none",
        "super_resolution",
    ]


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


def test_conditioning_chain_empty_is_identity() -> None:
    chain = ConditioningChain([])
    x = torch.randn(2, 1, 8, 8)
    out_x, ctx = chain(x, None, None)
    assert torch.equal(out_x, x)
    assert ctx is None


def test_conditioning_chain_combines_context_from_multiple_attention_adapters() -> None:
    attention = resolve_conditioning_adapter("attention")
    chain = ConditioningChain(
        [
            ChainAdapterSpec("a", attention),
            ChainAdapterSpec("b", attention),
        ]
    )
    x = torch.randn(2, 1, 8, 8)
    cond_a = torch.randn(2, 1, 8, 8)
    cond_b = torch.randn(2, 1, 8, 8)
    out_x, ctx = chain(x, {"a": cond_a, "b": cond_b}, None)
    assert torch.equal(out_x, x)
    assert ctx is not None
    assert ctx.shape == (2, 2, 8, 8)


def test_conditioning_chain_concat_plus_attention_shapes() -> None:
    chain = resolve_conditioning_adapter("chain")
    x = torch.randn(2, 1, 8, 8)
    concat_cond = torch.randn(2, 1, 8, 8)
    attn_cond = torch.randn(2, 3, 8, 8)
    out_x, ctx = chain(x, {"concatenate": concat_cond, "attention": attn_cond}, None)
    assert out_x.shape == (2, 2, 8, 8)
    assert ctx is not None
    assert ctx.shape == attn_cond.shape


def test_inpainting_adapter_output_channels_and_mask_preserved() -> None:
    adapter = resolve_conditioning_adapter("inpainting")
    x = torch.randn(2, 3, 8, 8)
    mask = torch.randint(0, 2, (2, 1, 8, 8), dtype=torch.float32)
    original = torch.randn(2, 3, 8, 8)
    out_x, ctx = adapter(x, {"mask": mask, "original": original}, None)
    assert out_x.shape == (2, 7, 8, 8)
    assert ctx is None
    assert torch.allclose(out_x[:, 3:4, ...], mask)


def test_inpainting_adapter_masked_original_matches_formula() -> None:
    adapter = resolve_conditioning_adapter("inpainting")
    x = torch.zeros(1, 2, 2, 2)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    original = torch.tensor([[[[3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0]]]])
    out_x, _ctx = adapter(x, {"mask": mask, "original": original}, None)
    masked_original = out_x[:, 3:, ...]
    expected = original * (1.0 - mask)
    assert torch.allclose(masked_original, expected)


def test_super_resolution_adapter_upsamples_and_concatenates() -> None:
    adapter = resolve_conditioning_adapter("super_resolution")
    x = torch.randn(2, 1, 256, 256)
    cond = torch.randn(2, 1, 64, 64)
    out_x, ctx = adapter(x, cond, None)
    assert out_x.shape == (2, 2, 256, 256)
    assert ctx is None


def test_depth_adapter_unsqueezes_rank3_and_concatenates() -> None:
    adapter = resolve_conditioning_adapter("depth")
    x = torch.randn(2, 1, 64, 64)
    cond = torch.randn(2, 64, 64)
    out_x, ctx = adapter(x, cond, None)
    assert out_x.shape == (2, 2, 64, 64)
    assert ctx is None
