from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training import GANTrainer, TRAINER_REGISTRY


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        _ = idx
        x = torch.randn(1, 8, 8)
        return {"target": x}


class _TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _TinyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TinyGeneratorWithTimestep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scale = (t.float().view(-1, 1, 1, 1) + 1.0) / 1000.0
        return self.conv(x) + scale


def test_trainer_registry_contains_gan_key() -> None:
    assert "gan" in TRAINER_REGISTRY.list()
    assert TRAINER_REGISTRY.get("gan") is GANTrainer


def test_gan_trainer_smoke(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "disc_lr": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_gan"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {"model_type": "gan"},
    }
    trainer = GANTrainer(
        cfg,
        model_override=_TinyGenerator(),
        discriminator_override=_TinyDiscriminator(),
    )
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "gan_last.pt").exists()
    assert (out / "gan_best.pt").exists()
    csv_header = (out / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert csv_header == "epoch,loss,g_gan,d_gan"


def test_gan_trainer_smoke_with_timestep_generator(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "disc_lr": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_gan_t"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {"model_type": "gan"},
    }
    trainer = GANTrainer(
        cfg,
        model_override=_TinyGeneratorWithTimestep(),
        discriminator_override=_TinyDiscriminator(),
    )
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "gan_last.pt").exists()
