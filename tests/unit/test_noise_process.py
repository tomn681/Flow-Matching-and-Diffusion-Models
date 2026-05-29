import torch
from pathlib import Path

from noise import (
    ConsistencyNoise,
    DDPMNoise,
    EDMNoise,
    FlowMatchingNoise,
    NOISE_REGISTRY,
    ReflowNoise,
    RectifiedFlowNoise,
    generate_reflow_pairs,
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
    assert NOISE_REGISTRY.list() == ["consistency", "ddpm", "edm", "flow_matching", "rectified_flow", "reflow"]


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


def test_reflow_noise_shapes(tmp_path: Path) -> None:
    scheduler = _DummyScheduler()
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


def test_generate_reflow_pairs_writes_files(tmp_path: Path) -> None:
    class _FakeScheduler:
        def __init__(self) -> None:
            self.timesteps = torch.tensor([], dtype=torch.long)

        class _Cfg:
            num_train_timesteps = 20

        config = _Cfg()

        def set_timesteps(self, n: int) -> None:
            self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.long)

        def step(self, pred: torch.Tensor, t: torch.Tensor, sample: torch.Tensor):
            _ = t
            return type("Out", (), {"prev_sample": sample - 0.1 * pred})()

    class _FakeModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor, context_ca=None) -> torch.Tensor:
            _ = t, context_ca
            return torch.zeros_like(x)

    out_dir = tmp_path / "pairs_out"
    generate_reflow_pairs(
        model=_FakeModel(),
        scheduler=_FakeScheduler(),
        num_pairs=5,
        sample_shape=(2, 1, 8, 8),
        device=torch.device("cpu"),
        output_dir=out_dir,
        num_inference_steps=4,
        batch_size=2,
    )
    files = sorted(out_dir.glob("*.pt"))
    assert len(files) == 5
    payload = torch.load(files[0], map_location="cpu")
    assert set(payload.keys()) == {"z0", "z1"}
