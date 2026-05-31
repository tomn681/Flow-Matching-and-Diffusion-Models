from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training import TRAINER_REGISTRY, UNetTrainer


class _TinyDataset:
    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        base = torch.full((1, 8, 8), float(idx))
        return {"image": base, "target": base}


class _TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return self.conv(x)


def _base_cfg(tmp_path: Path) -> dict:
    return {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_unet"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
            "save_every": 1,
            "validate": True,
        },
        "model": {
            "model_type": "unet",
            "unet": {
                "in_channels": 1,
                "out_channels": 1,
                "spatial_dims": 2,
            },
        },
    }


def test_unet_trainer_registry_key_present() -> None:
    assert TRAINER_REGISTRY.get("unet") is UNetTrainer


def test_unet_trainer_smoke_fit(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    ds = _TinyDataset(n=4)
    model = _TinyUNet()
    trainer = UNetTrainer(config=cfg, callbacks=[], model_override=model)
    trainer.fit(ds, val_dataset=ds, resume=None)
    assert trainer.global_step > 0

