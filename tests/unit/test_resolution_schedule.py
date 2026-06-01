from __future__ import annotations

from pathlib import Path

import torch

from training.base import BaseTrainer
from training.resolution_schedule import ResolutionSchedule


class _TinyDataset:
    def __init__(self, size: int = 4, img_size: tuple[int, int] = (8, 8)) -> None:
        self.size = size
        self.img_size = img_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        _ = idx
        return {"target": torch.zeros(1, self.img_size[0], self.img_size[1])}


class _NoopTrainer(BaseTrainer):
    def _build_model(self):
        return torch.nn.Conv2d(1, 1, kernel_size=1)

    def _training_step(self, batch, *, epoch):
        _ = batch, epoch
        return {"loss": 0.0}


def test_resolution_schedule_resolves_progressive_epochs() -> None:
    schedule = ResolutionSchedule({1: 64, 5: 128, 10: 256})
    assert schedule.resolution_for_epoch(1) == 64
    assert schedule.resolution_for_epoch(4) == 64
    assert schedule.resolution_for_epoch(5) == 128
    assert schedule.resolution_for_epoch(9) == 128
    assert schedule.resolution_for_epoch(10) == 256


def test_base_trainer_applies_resolution_schedule_and_rebuilds_dataloaders(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts"),
            "use_amp": False,
            "manual_device": "cpu",
            "resolution_schedule": {1: 16},
        },
        "model": {"model_type": "unet"},
    }
    trainer = _NoopTrainer(config=cfg, callbacks=[])
    train_ds = _TinyDataset()
    val_ds = _TinyDataset()
    trainer._setup(train_ds, val_dataset=val_ds, resume=None)

    old_train_loader = trainer.train_loader
    old_val_loader = trainer.val_loader
    trainer._apply_resolution_schedule(epoch=1)

    assert train_ds.img_size == (16, 16)
    assert val_ds.img_size == (16, 16)
    assert trainer.current_resolution == 16
    assert trainer.train_loader is not old_train_loader
    assert trainer.val_loader is not old_val_loader

