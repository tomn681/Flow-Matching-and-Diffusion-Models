from __future__ import annotations

import torch
import pytest

from nn.modules import TemporalAttentionND


def test_temporal_attention_preserves_2d_video_shape() -> None:
    module = TemporalAttentionND(channels=32, num_heads=8)
    x = torch.randn(2, 32, 4, 8, 8)
    y = module(x)
    assert y.shape == x.shape


def test_temporal_attention_preserves_3d_video_shape() -> None:
    module = TemporalAttentionND(channels=16, num_heads=4)
    x = torch.randn(2, 16, 3, 4, 4, 4)
    y = module(x)
    assert y.shape == x.shape


def test_temporal_attention_backward_runs() -> None:
    module = TemporalAttentionND(channels=24, num_heads=6)
    x = torch.randn(2, 24, 5, 4, 4, requires_grad=True)
    y = module(x)
    y.mean().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_temporal_attention_causal_flag_runs() -> None:
    module = TemporalAttentionND(channels=24, num_heads=6, causal=True)
    x = torch.randn(2, 24, 5, 4, 4)
    y = module(x)
    assert y.shape == x.shape


def test_temporal_attention_rejects_invalid_rank() -> None:
    module = TemporalAttentionND(channels=8, num_heads=2)
    x = torch.randn(2, 8, 5)
    with pytest.raises(ValueError, match="expects input shaped as"):
        module(x)


def test_temporal_attention_requires_divisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        TemporalAttentionND(channels=10, num_heads=3)
