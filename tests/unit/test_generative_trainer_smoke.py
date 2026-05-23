from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training import (
    ConsistencyTrainer,
    DiffusionTrainer,
    EDMTrainer,
    FlowMatchingTrainer,
    RectifiedFlowTrainer,
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

    def make_discriminator(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        x = torch.zeros(1, 8, 8)
        return {"target": x, "image": x}


def test_trainer_registry_contains_generative_keys() -> None:
    keys = set(TRAINER_REGISTRY.list())
    assert {"diffusion", "flow_matching", "consistency", "edm", "rectified_flow"}.issubset(keys)
    assert TRAINER_REGISTRY.get("diffusion") is DiffusionTrainer
    assert TRAINER_REGISTRY.get("flow_matching") is FlowMatchingTrainer
    assert TRAINER_REGISTRY.get("consistency") is ConsistencyTrainer
    assert TRAINER_REGISTRY.get("edm") is EDMTrainer
    assert TRAINER_REGISTRY.get("rectified_flow") is RectifiedFlowTrainer


def test_generative_trainer_diffusion_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict):
        return _DummyScheduler(), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_diff"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "diffusion",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = DiffusionTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "diff_last.pt").exists()
    assert (out / "diff_best.pt").exists()
    assert (out / "metrics.csv").exists()


def test_generative_trainer_flow_matching_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict):
        return _DummyScheduler(), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_flow"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "flow_matching",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = FlowMatchingTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    assert (out / "flow_last.pt").exists()
    assert (out / "flow_best.pt").exists()
    assert (out / "metrics.csv").exists()


def test_generative_trainer_diffusion_with_gan_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict):
        return _DummyScheduler(), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "disc_lr": 1e-3,
            "gan_weight": 0.5,
            "gan_space": "prediction",
            "gan_start": 0,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_diff_gan"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "diffusion",
            "scheduler": {},
            "conditioning": "none",
            "out_channels": 1,
        },
    }

    trainer = DiffusionTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)

    out = Path(trainer.output_dir)
    csv_lines = (out / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines
    assert "d_gan" in csv_lines[0]


def test_generative_trainer_consistency_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict):
        return _DummyScheduler(), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_consistency"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "consistency",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = ConsistencyTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)
    out = Path(trainer.output_dir)
    assert (out / "consistency_last.pt").exists()
    assert (out / "consistency_best.pt").exists()


def test_generative_trainer_edm_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict):
        return _DummyScheduler(), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_edm"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "edm",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = EDMTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)
    out = Path(trainer.output_dir)
    assert (out / "edm_last.pt").exists()
    assert (out / "edm_best.pt").exists()


def test_generative_trainer_rectified_flow_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict):
        return _DummyScheduler(), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_rectified"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "rectified_flow",
            "scheduler": {},
            "conditioning": "none",
        },
    }

    trainer = RectifiedFlowTrainer(cfg)
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds)
    out = Path(trainer.output_dir)
    assert (out / "rectified_flow_last.pt").exists()
    assert (out / "rectified_flow_best.pt").exists()


def test_generative_trainer_sets_discriminator_eval_during_validation(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
        model = _DummyUNet().to(device)
        if set_eval:
            model.eval()
        return model

    def _fake_build_scheduler(scheduler_cfg: dict, training_cfg: dict):
        return _DummyScheduler(), int(training_cfg.get("num_inference_steps", 50))

    monkeypatch.setattr("training.generative_trainer.build_diffusion_model", _fake_build_diffusion_model)
    monkeypatch.setattr("training.generative_trainer.build_scheduler", _fake_build_scheduler)

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "disc_lr": 1e-3,
            "gan_weight": 0.5,
            "gan_space": "prediction",
            "gan_start": 0,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_disc_eval"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": "diffusion",
            "scheduler": {},
            "conditioning": "none",
            "out_channels": 1,
        },
    }

    trainer = DiffusionTrainer(cfg)
    ds = _TinyDataset()
    trainer._setup(ds, val_dataset=ds, resume=None)
    assert trainer.discriminator is not None
    sample = ds[0]
    batch = {"target": sample["target"].unsqueeze(0), "image": sample["image"].unsqueeze(0)}
    trainer._run_step(batch, epoch=1, train=True)
    assert trainer.discriminator.training is True
    trainer._run_step(batch, epoch=1, train=False)
    assert trainer.discriminator.training is False
