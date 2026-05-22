from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training import DiffusionTrainer, FlowMatchingTrainer, TRAINER_REGISTRY, GenerativeTrainer


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


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        x = torch.zeros(1, 8, 8)
        return {"target": x, "image": x}


def test_trainer_registry_contains_generative_keys() -> None:
    keys = set(TRAINER_REGISTRY.list())
    assert {"diffusion", "flow_matching"}.issubset(keys)
    assert TRAINER_REGISTRY.get("diffusion") is DiffusionTrainer
    assert TRAINER_REGISTRY.get("flow_matching") is FlowMatchingTrainer


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
