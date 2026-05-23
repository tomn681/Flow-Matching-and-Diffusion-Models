import torch

from noise import (
    ConsistencyNoise,
    DDPMNoise,
    EDMNoise,
    FlowMatchingNoise,
    NOISE_REGISTRY,
    RectifiedFlowNoise,
)


class _DummySchedulerConfig:
    num_train_timesteps = 1000


class _DummyScheduler:
    def __init__(self) -> None:
        self.config = _DummySchedulerConfig()

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        scale = (timesteps.float() / max(1, self.config.num_train_timesteps - 1)).view(-1, 1, 1, 1)
        return clean + scale * noise


def test_noise_registry_entries() -> None:
    assert NOISE_REGISTRY.list() == ["consistency", "ddpm", "edm", "flow_matching", "rectified_flow"]


def test_ddpm_noise_shapes() -> None:
    scheduler = _DummyScheduler()
    process = DDPMNoise(scheduler)

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)


def test_flow_matching_noise_shapes() -> None:
    scheduler = _DummyScheduler()
    process = FlowMatchingNoise(scheduler)

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)


def test_consistency_noise_shapes() -> None:
    scheduler = _DummyScheduler()
    process = ConsistencyNoise(scheduler)

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)


def test_edm_noise_shapes() -> None:
    scheduler = _DummyScheduler()
    process = EDMNoise(scheduler)

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)


def test_rectified_flow_noise_shapes() -> None:
    scheduler = _DummyScheduler()
    process = RectifiedFlowNoise(scheduler)

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)
