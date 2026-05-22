from __future__ import annotations

import pytest
import torch

from scheduling.sampling_loop import _prepare_attention_context, normalize_latent_conditioning, sample_with_scheduler


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.arange(4, -1, -1)

    def set_timesteps(self, n: int) -> None:
        self.timesteps = torch.arange(int(n) - 1, -1, -1)

    def step(self, pred: torch.Tensor, t, current: torch.Tensor):
        class _Step:
            def __init__(self, prev_sample: torch.Tensor):
                self.prev_sample = prev_sample

        return _Step(current - 0.1 * pred)


class _FakeModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor, context_ca=None):
        return torch.zeros_like(inputs)


def test_normalize_latent_conditioning_standardize() -> None:
    x = torch.randn(2, 3, 4, 4)
    y = normalize_latent_conditioning(x, "standardize")
    assert y.shape == x.shape
    means = y.mean(dim=(2, 3))
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)


def test_normalize_latent_conditioning_minmax() -> None:
    x = torch.randn(2, 3, 4, 4)
    y = normalize_latent_conditioning(x, "minmax")
    assert y.shape == x.shape
    assert torch.all(y >= -1e-6)
    assert torch.all(y <= 1.0 + 1e-6)


def test_prepare_attention_context_raises_on_invalid_rank() -> None:
    with pytest.raises(ValueError, match="Unsupported conditioning shape"):
        _prepare_attention_context(torch.randn(2))


def test_sample_with_scheduler_runs() -> None:
    model = _FakeModel()
    scheduler = _FakeScheduler()
    out = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=3,
        sample_shape=(2, 1, 8, 8),
        device=torch.device("cpu"),
    )
    assert out.shape == (2, 1, 8, 8)


def test_sample_with_scheduler_invalid_last_n_steps_raises() -> None:
    model = _FakeModel()
    scheduler = _FakeScheduler()
    with pytest.raises(ValueError, match="last_n_steps must be > 0"):
        sample_with_scheduler(
            model=model,
            scheduler=scheduler,
            num_inference_steps=3,
            sample_shape=(2, 1, 8, 8),
            device=torch.device("cpu"),
            last_n_steps=0,
        )

