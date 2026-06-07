from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from core.types import ModelOutput
from pipelines.train.vae_lib import train as legacy_vae_train
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


class _CaptureNormalizeVAE(_DummyVAE):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("VAETrainer should not call model.image_to_model_range when input_normalize is configured.")

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> ModelOutput:
        self.last_input = x.detach().clone()
        return super().forward(x, sample_posterior=sample_posterior)


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        x = torch.zeros(1, 8, 8)
        return {"target": x, "image": x}


class _InputTargetDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict:
        _ = idx
        return {"image": torch.zeros(1, 8, 8), "target": torch.ones(1, 8, 8)}


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


def test_vae_trainer_matches_legacy_loss_fixed_seed(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_vae_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyVAE().to(device)
        if set_eval:
            model.eval()
        return model

    monkeypatch.setattr("training.vae_trainer.build_vae_model", _fake_build_vae_model)
    monkeypatch.setattr("pipelines.train.vae_lib.build_vae_model", _fake_build_vae_model)
    monkeypatch.setattr("utils.allocate_run_dir", lambda base: Path(base))
    monkeypatch.setattr("utils.summarize_model", lambda *args, **kwargs: None)

    ds = _TinyDataset()
    common_training = {
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "save_images": False,
        "save_images_every": 1,
        "visual_samples": 4,
        "recon_type": "l1",
        "kl_weight": 0.0,
        "codebook_weight": 0.0,
        "use_amp": False,
        "manual_device": "cpu",
        "seed": 123,
    }
    model_cfg = {
        "latent_type": "kl",
        "embed_dim": 1,
        "resolution": 8,
        "ch_mult": [1],
        "spatial_dims": 2,
    }

    legacy_cfg = {"training": {**common_training, "output_dir": str(tmp_path / "legacy_out")}, "model": model_cfg}
    new_cfg = {"training": {**common_training, "output_dir": str(tmp_path / "new_out")}, "model": model_cfg}

    legacy_cfg_path = tmp_path / "legacy_cfg.json"
    new_cfg_path = tmp_path / "new_cfg.json"
    legacy_cfg_path.write_text(json.dumps(legacy_cfg))
    new_cfg_path.write_text(json.dumps(new_cfg))

    legacy_vae_train(ds, legacy_cfg_path, val_dataset=ds)
    trainer = VAETrainer.from_config(new_cfg_path)
    trainer.fit(ds, val_dataset=ds)

    with (tmp_path / "legacy_out" / "metrics.csv").open(newline="") as fh:
        legacy_rows = list(csv.DictReader(fh))
    with (Path(trainer.output_dir) / "metrics.csv").open(newline="") as fh:
        new_rows = list(csv.DictReader(fh))

    assert legacy_rows, "Legacy metrics.csv has no rows."
    assert new_rows, "New trainer metrics.csv has no rows."

    legacy_loss = float(legacy_rows[-1]["loss"])
    new_loss = float(new_rows[-1]["loss"])
    assert abs(legacy_loss - new_loss) <= 1e-5


def test_vae_trainer_scheduler_steps(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_vae_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyVAE().to(device)
        if set_eval:
            model.eval()
        return model

    monkeypatch.setattr("training.vae_trainer.build_vae_model", _fake_build_vae_model)

    cfg = {
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_sched"),
            "save_images": False,
            "save_images_every": 1,
            "visual_samples": 4,
            "recon_type": "l1",
            "kl_weight": 0.0,
            "codebook_weight": 0.0,
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
            "scheduler": {
                "name": "StepLR",
                "params": {"step_size": 1, "gamma": 0.5},
            },
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

    assert trainer.optimizer is not None
    final_lr = trainer.optimizer.param_groups[0]["lr"]
    assert final_lr < 1e-3


def test_vae_trainer_uses_image_as_input_and_target_as_reconstruction_target(monkeypatch, tmp_path: Path) -> None:
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
            "output_dir": str(tmp_path / "ckpts_input_target"),
            "save_images": False,
            "save_images_every": 1,
            "visual_samples": 2,
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
    ds = _InputTargetDataset()
    trainer.fit(ds, val_dataset=ds)

    metrics_path = Path(trainer.output_dir) / "metrics.csv"
    rows = list(csv.DictReader(metrics_path.open()))
    assert rows, "Expected at least one metrics row."
    # If trainer incorrectly uses target as input, reconstruction would exactly match target (loss ~0).
    assert float(rows[-1]["loss"]) > 0.5


def test_vae_trainer_positive_input_normalize_uses_raw_unit_interval(monkeypatch, tmp_path: Path) -> None:
    model = _CaptureNormalizeVAE()

    def _fake_build_vae_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        built = model.to(device)
        if set_eval:
            built.eval()
        return built

    monkeypatch.setattr("training.vae_trainer.build_vae_model", _fake_build_vae_model)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_positive"),
            "save_images": False,
            "save_images_every": 1,
            "visual_samples": 2,
            "recon_type": "bce_focal",
            "input_normalize": "positive",
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

    class _PositiveDataset:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int) -> dict:
            _ = idx
            x = torch.full((1, 8, 8), 0.25)
            return {"target": x, "image": x}

    trainer = VAETrainer(cfg)
    ds = _PositiveDataset()
    trainer.fit(ds, val_dataset=ds)

    assert model.last_input is not None
    assert torch.allclose(model.last_input, torch.full_like(model.last_input, 0.25))
    assert getattr(trainer.model, "input_range") == "zero_to_one"


def test_vae_trainer_metrics_only_include_active_losses(monkeypatch, tmp_path: Path) -> None:
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
            "output_dir": str(tmp_path / "ckpts_metrics"),
            "save_images": False,
            "save_images_every": 1,
            "visual_samples": 2,
            "recon_type": "bce_focal",
            "kl_weight": 1e-4,
            "kl_anneal_steps": 0,
            "codebook_weight": 0.0,
            "gan_weight": 0.0,
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

    metrics_path = Path(trainer.output_dir) / "metrics.csv"
    header = metrics_path.read_text().splitlines()[0]
    assert "recon_bce_focal" in header
    assert "kl" in header
    assert "vq" not in header
    assert "d_gan" not in header
