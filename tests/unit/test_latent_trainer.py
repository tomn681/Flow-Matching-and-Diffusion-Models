from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from datasets import LatentCacheDataset
from training import (
    LatentDiffusionTrainer,
    LatentFlowMatchingTrainer,
    LatentRectifiedFlowTrainer,
    TRAINER_REGISTRY,
)


class _DummyScheduler:
    class _Cfg:
        num_train_timesteps = 1000

    config = _Cfg()

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        scale = timesteps.float().view(-1, *([1] * (clean.dim() - 1))) / max(1, self.config.num_train_timesteps - 1)
        return clean + scale * noise


class _DummyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context_ca=None):
        return x * self.weight


class _DummyPosterior:
    def __init__(self, x: torch.Tensor) -> None:
        self._x = x

    def mode(self) -> torch.Tensor:
        return self._x


class _DummyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.ones(1))

    def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def encode(self, x: torch.Tensor, normalize: bool = False):
        if normalize:
            return x * 0.5
        return _DummyPosterior(x * 0.5)


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        x = torch.zeros(1, 8, 8)
        return {"target": x, "image": x}


def _base_cfg(tmp_path: Path, model_type: str, *, use_presaved_latents: bool) -> dict:
    return {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / f"ckpts_{model_type}"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": model_type,
            "scheduler": {},
            "conditioning": "none",
            "use_presaved_latents": use_presaved_latents,
            "vae_checkpoint": str(tmp_path / "dummy_vae.pt"),
            "vae": {"latent_type": "kl"},
            "unet": {},
        },
    }


def test_trainer_registry_contains_latent_keys() -> None:
    keys = set(TRAINER_REGISTRY.list())
    assert {"latent_diffusion", "latent_flow_matching", "latent_rectified_flow"}.issubset(keys)
    assert TRAINER_REGISTRY.get("latent_diffusion") is LatentDiffusionTrainer
    assert TRAINER_REGISTRY.get("latent_flow_matching") is LatentFlowMatchingTrainer
    assert TRAINER_REGISTRY.get("latent_rectified_flow") is LatentRectifiedFlowTrainer


def test_latent_diffusion_trainer_online_encoding_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict):
        return _DummyScheduler(), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.latent_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.latent_trainer.build_scheduler", _fake_build_scheduler)
    def _fake_load_frozen_vae(self):
        vae = _DummyVAE()
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        return vae

    monkeypatch.setattr("training.latent_trainer.LatentGenerativeTrainer._load_frozen_vae", _fake_load_frozen_vae)

    cfg = _base_cfg(tmp_path, "latent_diffusion", use_presaved_latents=False)
    trainer = LatentDiffusionTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "latent_diff_last.pt").exists()
    assert (out / "latent_diff_best.pt").exists()
    assert (out / "metrics.csv").exists()
    assert trainer.vae_model is not None
    assert all(not p.requires_grad for p in trainer.vae_model.parameters())


def test_latent_flow_matching_trainer_presaved_latents_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict):
        return _DummyScheduler(), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.latent_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.latent_trainer.build_scheduler", _fake_build_scheduler)

    latent_dir = tmp_path / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        torch.save({"target": torch.zeros(1, 8, 8), "image": torch.zeros(1, 8, 8)}, latent_dir / f"{i:03d}.pt")

    cfg = _base_cfg(tmp_path, "latent_flow_matching", use_presaved_latents=True)
    trainer = LatentFlowMatchingTrainer(cfg)
    ds = LatentCacheDataset(latent_dir)
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "latent_flow_last.pt").exists()
    assert (out / "latent_flow_best.pt").exists()
    assert (out / "metrics.csv").exists()


def test_latent_cache_dataset_supports_split_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "latents"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir(parents=True)
    torch.save({"target": torch.zeros(1, 8, 8)}, root / "train" / "000.pt")
    torch.save({"target": torch.ones(1, 8, 8)}, root / "val" / "000.pt")

    train_ds = LatentCacheDataset(root, split="train")
    val_ds = LatentCacheDataset(root, split="val")
    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert torch.equal(train_ds[0]["target"], torch.zeros(1, 8, 8))
    assert torch.equal(val_ds[0]["target"], torch.ones(1, 8, 8))
