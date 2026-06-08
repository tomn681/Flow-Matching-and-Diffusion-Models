from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from losses import LossAssembler
from training import DiffusionTrainer, TrainerBuilder, TrainingEventBus, VAETrainer


def _base_cfg(tmp_path: Path, model_type: str = "diffusion") -> dict:
    return {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_builder"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {
            "model_type": model_type,
            "scheduler": {},
            "conditioning": "none",
        },
    }


def test_trainer_builder_method_chaining(tmp_path: Path) -> None:
    events = []
    listener = lambda **_: events.append("x")
    model = nn.Identity()
    cfg = _base_cfg(tmp_path)

    trainer = (
        TrainerBuilder()
        .with_config(cfg)
        .with_model(model)
        .with_callbacks([])
        .with_event_listener("epoch_end", listener)
        .with_ema(decay=0.95)
        .build()
    )

    assert isinstance(trainer, DiffusionTrainer)
    assert trainer._build_model() is model
    assert trainer.training_cfg.get("ema_decay") == 0.95
    assert isinstance(trainer.event_bus, TrainingEventBus)


def test_trainer_builder_rejects_invalid_combinations(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, model_type="vae")
    builder = TrainerBuilder().with_config(cfg)

    try:
        builder.with_noise(object()).build()
        raise AssertionError("Expected ValueError for with_noise on non-generative trainer.")
    except ValueError as exc:
        assert "with_noise" in str(exc)


def test_trainer_builder_with_frozen_vae_updates_config(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, model_type="latent_diffusion")
    ckpt = tmp_path / "vae.pt"
    builder = TrainerBuilder().with_config(cfg).with_frozen_vae(ckpt)

    assert builder._config is not None
    assert builder._config["model"]["vae_checkpoint"] == str(ckpt)


def test_trainer_builder_uses_explicit_losses_override_capability(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, model_type="vae")
    losses = LossAssembler([])
    trainer = TrainerBuilder().with_config(cfg).with_losses(losses).build()
    assert isinstance(trainer, VAETrainer)
    assert trainer._losses_override is losses
