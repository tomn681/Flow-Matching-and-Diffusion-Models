from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training import BaseTrainer, TrainingEventBus


class _TinyDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        x = torch.zeros(1, 4, 4)
        return {"target": x}


class _EventTrainer(BaseTrainer):
    def _build_model(self) -> torch.nn.Module:
        return nn.Identity()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD([nn.Parameter(torch.tensor(0.0))], lr=0.1)

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return {"loss": 1.0}


def test_event_bus_on_emit_remove() -> None:
    bus = TrainingEventBus()
    payload: list[int] = []

    def _listener(value: int) -> None:
        payload.append(value)

    bus.on("x", _listener)
    bus.emit("x", value=7)
    bus.remove("x", _listener)
    bus.emit("x", value=9)

    assert payload == [7]


def test_base_trainer_emits_lifecycle_events(tmp_path: Path) -> None:
    events: list[str] = []
    epoch_end_payload: list[dict] = []

    bus = TrainingEventBus()
    bus.on("train_start", lambda trainer: events.append("train_start"))
    bus.on("epoch_start", lambda epoch, trainer: events.append(f"epoch_start:{epoch}"))
    bus.on("step_end", lambda epoch, step, global_step, metrics, trainer: events.append(f"step_end:{step}"))
    bus.on("validation_end", lambda epoch, metrics, trainer: events.append(f"validation_end:{epoch}"))
    bus.on("checkpoint_saved", lambda epoch, metrics, state, trainer: events.append(f"checkpoint_saved:{epoch}"))
    bus.on(
        "epoch_end",
        lambda epoch, metrics, state, trainer: epoch_end_payload.append({"epoch": epoch, "metrics": dict(metrics)}),
    )
    bus.on("train_end", lambda trainer: events.append("train_end"))

    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_events"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {},
    }
    ds = _TinyDataset()
    trainer = _EventTrainer(cfg, event_bus=bus)
    trainer.fit(ds, val_dataset=ds)

    assert "train_start" in events
    assert "epoch_start:1" in events
    assert "step_end:1" in events
    assert "validation_end:1" in events
    assert "checkpoint_saved:1" in events
    assert "train_end" in events
    assert len(epoch_end_payload) == 1
    assert epoch_end_payload[0]["epoch"] == 1
    assert "loss" in epoch_end_payload[0]["metrics"]

