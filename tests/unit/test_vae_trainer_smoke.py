from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from core.types import ModelOutput
from training import TRAINER_REGISTRY, VAETrainer


class _DummyPosterior:
    def __init__(self, z: torch.Tensor) -> None:
        self._z = z

    def kl(self) -> torch.Tensor:
        return torch.zeros(self._z.size(0), device=self._z.device)


class _DummyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def raw_output_to_image(self, x: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> ModelOutput:
        rec = x * self.weight
        return ModelOutput(reconstruction=rec, posterior=_DummyPosterior(rec), codebook_loss=torch.tensor(0.0, device=x.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def make_discriminator(self) -> nn.Module:
        return nn.Sequential(nn.Conv2d(1, 1, kernel_size=1))


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        x = torch.zeros(1, 8, 8)
        return {"target": x, "image": x}


def test_trainer_registry_contains_vae() -> None:
    assert "vae" in TRAINER_REGISTRY.list()


def test_vae_trainer_fit_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_vae_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyVAE().to(device)
        if set_eval:
            model.eval()
        return model

    monkeypatch.setattr("training.vae_trainer.build_vae_model", _fake_build_vae_model)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts"),
            "save_images": False,
            "save_images_every": 1,
            "visual_samples": 4,
            "recon_type": "l1",
            "kl_weight": 0.0,
            "codebook_weight": 0.0,
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "latent_type": "kl",
            "embed_dim": 1,
            "resolution": 8,
            "ch_mult": [1],
            "spatial_dims": 2,
        },
    }

    trainer = VAETrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "vae_last.pt").exists()
    assert (out / "vae_best.pt").exists()
    assert (out / "metrics.csv").exists()


def test_vae_trainer_fit_smoke_with_gan_and_perceptual(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_vae_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyVAE().to(device)
        if set_eval:
            model.eval()
        return model

    class _DummyPerceptual:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def to(self, device: torch.device):
            return self

        def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean(torch.abs(pred - target))

    monkeypatch.setattr("training.vae_trainer.build_vae_model", _fake_build_vae_model)
    monkeypatch.setattr("losses.perceptual.PerceptualLoss", _DummyPerceptual)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_gan"),
            "save_images": False,
            "save_images_every": 1,
            "visual_samples": 4,
            "recon_type": "l1",
            "kl_weight": 0.0,
            "codebook_weight": 0.0,
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
            "gan_weight": 0.5,
            "gan_start": 0,
            "perceptual_weight": 0.2,
        },
        "model": {
            "latent_type": "kl",
            "embed_dim": 1,
            "resolution": 8,
            "ch_mult": [1],
            "spatial_dims": 2,
        },
    }

    trainer = VAETrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "vae_last.pt").exists()
    assert (out / "vae_best.pt").exists()
    assert (out / "metrics.csv").exists()
