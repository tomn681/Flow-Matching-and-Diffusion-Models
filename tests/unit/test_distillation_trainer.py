from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training import DistillationTrainer, TRAINER_REGISTRY


class _DummyScheduler:
    class _Cfg:
        num_train_timesteps = 1000

    config = _Cfg()

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        scale = timesteps.float().view(-1, *([1] * (clean.dim() - 1))) / max(1, self.config.num_train_timesteps - 1)
        return clean + scale * noise


class _TinyUNet(nn.Module):
    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return x * self.gain


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        _ = idx
        return {"target": torch.zeros(1, 8, 8)}


def _base_cfg(tmp_path: Path) -> dict:
    return {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_distill"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
            "save_every": 1,
            "validate": True,
        },
        "model": {
            "model_type": "distillation",
            "student_model_type": "diffusion",
            "teacher_steps": 128,
            "student_steps": 64,
            "scheduler": {},
        },
    }


def test_distillation_trainer_registry_key_present() -> None:
    assert TRAINER_REGISTRY.get("distillation") is DistillationTrainer


def test_distillation_trainer_requires_teacher_checkpoint_without_override(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    trainer = DistillationTrainer(config=cfg, callbacks=[], model_override=_TinyUNet(0.5))
    ds = _TinyDataset()
    try:
        trainer.fit(ds, val_dataset=ds, resume=None)
        raise AssertionError("Expected ValueError when teacher checkpoint is missing.")
    except ValueError as exc:
        assert "teacher_checkpoint" in str(exc)


def test_distillation_trainer_smoke_fit_with_overrides(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    trainer = DistillationTrainer(
        config=cfg,
        callbacks=[],
        model_override=_TinyUNet(0.5),
        teacher_override=_TinyUNet(1.0),
        scheduler_override=_DummyScheduler(),
    )
    ds = _TinyDataset()
    trainer.fit(ds, val_dataset=ds, resume=None)
    assert trainer.global_step > 0

