import torch
from pathlib import Path
from diffusers import FlowMatchEulerDiscreteScheduler

from noise import (
    DDPMNoise,
    EDMNoise,
    FlowMatchingNoise,
    NOISE_REGISTRY,
    ReflowNoise,
    RectifiedFlowNoise,
    X0DenoisingNoise,
    generate_reflow_pairs,
)


class _DummySchedulerConfig:
    num_train_timesteps = 1000
    prediction_type = "epsilon"


class _DummyScheduler:
    def __init__(self) -> None:
        self.config = _DummySchedulerConfig()

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        scale = (timesteps.float() / max(1, self.config.num_train_timesteps - 1)).view(-1, 1, 1, 1)
        return clean + scale * noise


class _DummyX0SchedulerConfig:
    num_train_timesteps = 1000
    prediction_type = "sample"


class _DummyX0Scheduler(_DummyScheduler):
    def __init__(self) -> None:
        self.config = _DummyX0SchedulerConfig()


def _flow_scheduler() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)


def test_noise_registry_entries() -> None:
    assert NOISE_REGISTRY.list() == ["consistency", "ddpm", "edm", "flow_matching", "rectified_flow", "reflow", "x0_denoising"]


def test_ddpm_noise_shapes() -> None:
    scheduler = _DummyScheduler()
    process = DDPMNoise(scheduler)

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)


def test_flow_matching_noise_shapes() -> None:
    scheduler = _flow_scheduler()
    process = FlowMatchingNoise(scheduler)

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)


def test_consistency_noise_shapes() -> None:
    scheduler = _DummyX0Scheduler()
    process = X0DenoisingNoise(scheduler)

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)


def test_edm_noise_is_disabled() -> None:
    scheduler = _DummyScheduler()
    process = EDMNoise(scheduler)
    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)
    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)
    assert "sigmas" in out.extra


def test_rectified_flow_noise_shapes() -> None:
    scheduler = _flow_scheduler()
    process = RectifiedFlowNoise(scheduler)

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)


def test_reflow_noise_shapes(tmp_path: Path) -> None:
    scheduler = _flow_scheduler()
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        torch.save({"z0": torch.randn(1, 8, 8), "z1": torch.randn(1, 8, 8)}, pairs_dir / f"{i:03d}.pt")

    process = ReflowNoise(scheduler, pairs_dir=str(pairs_dir))

    clean = torch.randn(4, 1, 8, 8)
    out = process(clean, clean.device)

    assert out.noisy.shape == clean.shape
    assert out.target.shape == clean.shape
    assert out.timesteps.shape == (clean.size(0),)


def test_reflow_noise_lazy_loads_only_batch_pairs(tmp_path: Path, monkeypatch) -> None:
    scheduler = _flow_scheduler()
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(16):
        torch.save({"z0": torch.randn(1, 8, 8), "z1": torch.randn(1, 8, 8)}, pairs_dir / f"{i:03d}.pt")

    load_calls = {"count": 0}
    original_torch_load = torch.load

    def _counting_load(*args, **kwargs):
        load_calls["count"] += 1
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _counting_load)
    process = ReflowNoise(scheduler, pairs_dir=str(pairs_dir))
    clean = torch.randn(4, 1, 8, 8)
    _ = process(clean, clean.device)
    assert load_calls["count"] == clean.size(0)


def test_generate_reflow_pairs_writes_files(tmp_path: Path) -> None:
    class _FakeModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor, context_ca=None) -> torch.Tensor:
            _ = t, context_ca
            return torch.zeros_like(x)

    out_dir = tmp_path / "pairs_out"
    generate_reflow_pairs(
        model=_FakeModel(),
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=20),
        num_pairs=5,
        sample_shape=(1, 8, 8),
        device=torch.device("cpu"),
        output_dir=out_dir,
        num_inference_steps=4,
        batch_size=2,
    )
    files = sorted(out_dir.glob("*.pt"))
    assert len(files) == 5
    payload = torch.load(files[0], map_location="cpu")
    assert set(payload.keys()) == {"z0", "z1"}
