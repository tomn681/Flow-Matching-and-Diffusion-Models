from __future__ import annotations

import inspect
from pathlib import Path

import torch

import src
from core import Discriminatable, NoisingScheduler
from core.types import ModelOutput
from scheduling.sampling_loop import _forward_model, sample_with_scheduler


class _ReturnsModelOutput(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, context_ca=None) -> ModelOutput:
        _ = t, context_ca
        return ModelOutput(reconstruction=x + 1.0)


class _NoisingScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([], dtype=torch.long)

    def set_timesteps(self, n: int) -> None:
        self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.long)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        _ = timesteps
        return original_samples + 0.1 * noise

    def step(self, pred: torch.Tensor, t: torch.Tensor, sample: torch.Tensor):
        _ = pred, t
        return type("Out", (), {"prev_sample": sample})()


class _NonNoisingScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([], dtype=torch.long)

    def set_timesteps(self, n: int) -> None:
        self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.long)

    def step(self, pred: torch.Tensor, t: torch.Tensor, sample: torch.Tensor):
        _ = pred, t
        return type("Out", (), {"prev_sample": sample})()


class _DiscriminatableModel:
    def make_discriminator(self) -> torch.nn.Module | None:
        return torch.nn.Linear(1, 1)


def test_public_version_is_0_9_0() -> None:
    assert src.__version__ == "1.0.0"


def test_changelog_exists_for_0_9_release() -> None:
    root = Path(__file__).resolve().parents[2]
    changelog = root / "CHANGELOG.md"
    assert changelog.exists()
    text = changelog.read_text(encoding="utf-8")
    assert "## 1.0.0" in text


def test_forward_model_unwraps_modeloutput_sample() -> None:
    x = torch.randn(2, 1, 4, 4)
    t = torch.randint(0, 10, (2,), dtype=torch.long)
    out = _forward_model(_ReturnsModelOutput(), x, t)
    assert torch.allclose(out, x + 1.0)


def test_img2img_requires_noising_scheduler_protocol() -> None:
    model = torch.nn.Identity()
    init_image = torch.randn(1, 1, 4, 4)
    scheduler = _NonNoisingScheduler()
    try:
        sample_with_scheduler(
            model=model,  # type: ignore[arg-type]
            scheduler=scheduler,
            num_inference_steps=2,
            sample_shape=(1, 1, 4, 4),
            device=torch.device("cpu"),
            conditioning_mode="none",
            init_image=init_image,
            strength=0.5,
        )
        raise AssertionError("Expected ValueError for scheduler without NoisingScheduler capability.")
    except ValueError as exc:
        assert "add_noise" in str(exc)


def test_noising_scheduler_protocol_runtime_check() -> None:
    assert isinstance(_NoisingScheduler(), NoisingScheduler)


def test_discriminatable_protocol_runtime_check() -> None:
    assert isinstance(_DiscriminatableModel(), Discriminatable)


def test_generative_trainer_no_longer_uses_hasattr_for_discriminator_hook() -> None:
    from training.generative_trainer import GenerativeTrainer

    source = inspect.getsource(GenerativeTrainer._build_discriminator)
    assert "hasattr(" not in source
    assert "Discriminatable" in source


def test_validate_refactor_covers_phase_q_audit_cases() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "validate_refactor.sh").read_text(encoding="utf-8")
    for marker in [
        "36. VideoUNetND: forward pass smoke",
        "37. Medical3DDataset: channel_order handling",
        "38. TemporalAttentionND: causal flag smoke",
        "39. Sampler registry includes video_unet and distillation",
        "40. SCHEDULER_REGISTRY uses Registry[T]",
        "41. LOSS_REGISTRY includes WGAN-GP and R1 components",
        "42. Package version is 1.0.0 and CHANGELOG exists",
    ]:
        assert marker in text
