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

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        max_t = max(int(self.timesteps.max().item()), 1)
        alpha = timesteps.to(original.device, dtype=original.dtype).view(-1, 1, 1, 1) / float(max_t)
        return (1.0 - alpha) * original + alpha * noise


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor, context_ca=None):
        self.calls += 1
        return torch.zeros_like(inputs)


class _GuidanceSensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor, context_ca=None):
        self.calls += 1
        base = 0.5 * inputs + 0.01 * timesteps.view(-1, 1, 1, 1).to(inputs.dtype)
        if context_ca is None:
            return base
        ctx = context_ca
        if ctx.dim() > 2:
            ctx = ctx.flatten(1)
        strength = ctx.mean(dim=1, keepdim=True).view(-1, 1, 1, 1).to(inputs.dtype)
        return base + 0.1 * strength


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


def test_sample_with_scheduler_cfg_attention_calls_model_twice_per_step() -> None:
    model = _FakeModel()
    scheduler = _FakeScheduler()
    out = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=4,
        sample_shape=(2, 1, 8, 8),
        device=torch.device("cpu"),
        conditioning_mode="attention",
        conditioning_batch=torch.randn(2, 1, 8, 8),
        guidance_scale=3.0,
    )
    assert out.shape == (2, 1, 8, 8)
    assert model.calls == 8


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


def test_sample_with_scheduler_cfg_scale_one_matches_baseline() -> None:
    seed = 1234
    context = torch.randn(2, 1, 8, 8)
    model = _GuidanceSensitiveModel()
    scheduler = _FakeScheduler()
    torch.manual_seed(seed)
    out_baseline = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=4,
        sample_shape=(2, 1, 8, 8),
        device=torch.device("cpu"),
        conditioning_mode="attention",
        conditioning_batch=context,
        guidance_scale=1.0,
    )
    model = _GuidanceSensitiveModel()
    scheduler = _FakeScheduler()
    torch.manual_seed(seed)
    out_cfg_one = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=4,
        sample_shape=(2, 1, 8, 8),
        device=torch.device("cpu"),
        conditioning_mode="attention",
        conditioning_batch=context,
        guidance_scale=1.0,
    )
    assert out_cfg_one.shape == out_baseline.shape == (2, 1, 8, 8)
    assert torch.allclose(out_cfg_one, out_baseline)


def test_sample_with_scheduler_cfg_scale_changes_output() -> None:
    seed = 5678
    context = torch.randn(2, 1, 8, 8)
    model = _GuidanceSensitiveModel()
    scheduler = _FakeScheduler()
    torch.manual_seed(seed)
    out_scale_one = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=4,
        sample_shape=(2, 1, 8, 8),
        device=torch.device("cpu"),
        conditioning_mode="attention",
        conditioning_batch=context,
        guidance_scale=1.0,
    )
    model = _GuidanceSensitiveModel()
    scheduler = _FakeScheduler()
    torch.manual_seed(seed)
    out_scale_high = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=4,
        sample_shape=(2, 1, 8, 8),
        device=torch.device("cpu"),
        conditioning_mode="attention",
        conditioning_batch=context,
        guidance_scale=7.5,
    )
    assert out_scale_high.shape == out_scale_one.shape == (2, 1, 8, 8)
    assert not torch.allclose(out_scale_high, out_scale_one)


def test_sample_with_scheduler_img2img_strength_one_matches_noise_init() -> None:
    seed = 123
    scheduler = _FakeScheduler()
    model = _FakeModel()
    init_image = torch.ones(2, 1, 4, 4)
    torch.manual_seed(seed)
    out_noise = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=5,
        sample_shape=(2, 1, 4, 4),
        device=torch.device("cpu"),
    )
    scheduler = _FakeScheduler()
    model = _FakeModel()
    torch.manual_seed(seed)
    out_img2img = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=5,
        sample_shape=(2, 1, 4, 4),
        device=torch.device("cpu"),
        init_image=init_image,
        strength=1.0,
    )
    assert torch.allclose(out_noise, out_img2img)


def test_sample_with_scheduler_img2img_strength_zero_returns_input() -> None:
    scheduler = _FakeScheduler()
    model = _FakeModel()
    init_image = torch.randn(2, 1, 4, 4)
    out = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=5,
        sample_shape=(2, 1, 4, 4),
        device=torch.device("cpu"),
        init_image=init_image,
        strength=0.0,
    )
    assert torch.allclose(out, init_image)


def test_sample_with_scheduler_img2img_intermediate_strength_between_input_and_noise() -> None:
    seed = 321
    scheduler = _FakeScheduler()
    model = _FakeModel()
    init_image = torch.full((2, 1, 4, 4), 0.75)
    num_steps = 6
    strength = 0.5
    torch.manual_seed(seed)
    out = sample_with_scheduler(
        model=model,
        scheduler=scheduler,
        num_inference_steps=num_steps,
        sample_shape=(2, 1, 4, 4),
        device=torch.device("cpu"),
        init_image=init_image,
        strength=strength,
    )
    scheduler.set_timesteps(num_steps)
    start_idx = int(scheduler.timesteps.numel() * (1.0 - strength))
    start_idx = min(max(start_idx, 0), scheduler.timesteps.numel() - 1)
    t_start = scheduler.timesteps[start_idx]
    max_t = max(int(scheduler.timesteps.max().item()), 1)
    alpha = float(t_start.item()) / float(max_t)
    torch.manual_seed(seed)
    noise = torch.randn_like(init_image)
    lo = torch.minimum(init_image, noise)
    hi = torch.maximum(init_image, noise)
    assert torch.all(out >= lo - 1e-6)
    assert torch.all(out <= hi + 1e-6)
    assert 0.0 < alpha < 1.0
