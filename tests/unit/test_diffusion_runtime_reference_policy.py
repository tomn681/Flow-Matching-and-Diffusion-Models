from __future__ import annotations

import pytest
import torch

import utils.model_utils.diffusion_runtime as diffusion_runtime


class _FakeNoisingScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.arange(3, -1, -1)
        self.last_original: torch.Tensor | None = None

    def set_timesteps(self, n: int) -> None:
        self.timesteps = torch.arange(int(n) - 1, -1, -1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del noise, timesteps
        self.last_original = original_samples.detach().clone()
        return original_samples + 123.0


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, scheduler: _FakeNoisingScheduler) -> dict:
    captured: dict = {}

    monkeypatch.setattr(
        diffusion_runtime,
        "build_scheduler",
        lambda *_args, **_kwargs: (scheduler, 4),
    )

    def _fake_sample_with_scheduler(*args, **kwargs):
        del args
        captured.update(kwargs)
        return kwargs["init_sample"]

    monkeypatch.setattr(diffusion_runtime, "sample_with_scheduler", _fake_sample_with_scheduler)
    return captured


def test_inference_reference_initialization_uses_conditioning_not_target(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _FakeNoisingScheduler()
    _patch_runtime(monkeypatch, scheduler)
    target = torch.zeros(2, 1, 4, 4)
    conditioning = torch.ones(2, 1, 4, 4)

    out = diffusion_runtime.decode_diffusion_batch(
        model=torch.nn.Identity(),
        training_cfg={"conditioning": "concatenate"},
        model_cfg={"model_type": "diffusion", "scheduler": {}},
        device=torch.device("cpu"),
        batch_shape=tuple(target.shape),
        conditioning_batch=conditioning,
        target_batch=target,
        reference_batch=target,
        init_from_reference=True,
        last_n_steps=1,
    )

    assert torch.allclose(scheduler.last_original, conditioning)
    assert torch.allclose(out, conditioning + 123.0)


def test_training_reference_initialization_can_use_target(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _FakeNoisingScheduler()
    _patch_runtime(monkeypatch, scheduler)
    target = torch.full((2, 1, 4, 4), 0.25)

    diffusion_runtime.decode_diffusion_batch(
        model=torch.nn.Identity(),
        training_cfg={"conditioning": "concatenate"},
        model_cfg={"model_type": "diffusion", "scheduler": {}},
        device=torch.device("cpu"),
        batch_shape=tuple(target.shape),
        conditioning_batch=None,
        target_batch=target,
        init_from_reference=True,
        last_n_steps=1,
        runtime_mode="train",
    )

    assert torch.allclose(scheduler.last_original, target)


def test_inference_reference_initialization_requires_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _FakeNoisingScheduler()
    _patch_runtime(monkeypatch, scheduler)
    target = torch.zeros(2, 1, 4, 4)

    with pytest.raises(ValueError, match="requires tensor-valued conditioning"):
        diffusion_runtime.decode_diffusion_batch(
            model=torch.nn.Identity(),
            training_cfg={"conditioning": "concatenate"},
            model_cfg={"model_type": "diffusion", "scheduler": {}},
            device=torch.device("cpu"),
            batch_shape=tuple(target.shape),
            conditioning_batch=None,
            target_batch=target,
            init_from_reference=True,
            last_n_steps=1,
        )


def test_inference_reference_initialization_validates_conditioning_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _FakeNoisingScheduler()
    _patch_runtime(monkeypatch, scheduler)
    target = torch.zeros(2, 1, 4, 4)
    conditioning = torch.ones(2, 1, 8, 8)

    with pytest.raises(ValueError, match="does not match requested batch_shape"):
        diffusion_runtime.decode_diffusion_batch(
            model=torch.nn.Identity(),
            training_cfg={"conditioning": "concatenate"},
            model_cfg={"model_type": "diffusion", "scheduler": {}},
            device=torch.device("cpu"),
            batch_shape=tuple(target.shape),
            conditioning_batch=conditioning,
            target_batch=target,
            init_from_reference=True,
            last_n_steps=1,
        )
