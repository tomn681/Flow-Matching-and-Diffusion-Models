from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from training import BaseTrainer, TrainingEventBus
from training.callbacks import StepMetricsCallback


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


class _WeightedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_idx = 0

    def _build_model(self) -> torch.nn.Module:
        return nn.Identity()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD([nn.Parameter(torch.tensor(0.0))], lr=0.1)

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        _ = epoch
        self._call_idx += 1
        weights = [3, 1]
        losses = [1.0, 10.0]
        idx = min(self._call_idx - 1, len(weights) - 1)
        return {"loss": losses[idx], "__num_samples__": weights[idx]}

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        _ = epoch
        return {"loss": float(torch.rand(1).item())}


def test_event_bus_on_emit_remove() -> None:
    bus = TrainingEventBus()
    values: list[int] = []

    def _listener(value: int, payload=None) -> None:
        assert payload["value"] == value
        values.append(value)

    bus.on("x", _listener)
    bus.emit("x", value=7)
    bus.remove("x", _listener)
    bus.emit("x", value=9)

    assert values == [7]


def test_base_trainer_emits_lifecycle_events(tmp_path: Path) -> None:
    events: list[str] = []
    epoch_end_payload: list[dict] = []

    bus = TrainingEventBus()
    bus.on("train_start", lambda trainer, payload=None: events.append("train_start"))
    bus.on("epoch_start", lambda epoch, trainer, payload=None: events.append(f"epoch_start:{epoch}"))
    bus.on("step_end", lambda epoch, step, global_step, metrics, trainer, payload=None: events.append(f"step_end:{step}"))
    bus.on("validation_end", lambda epoch, metrics, trainer, payload=None: events.append(f"validation_end:{epoch}"))
    bus.on(
        "epoch_end",
        lambda epoch, metrics, state, trainer, payload=None: epoch_end_payload.append({"epoch": epoch, "metrics": dict(metrics), "payload": payload}),
    )
    bus.on("train_end", lambda trainer, payload=None: events.append("train_end"))

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
    assert "train_end" in events
    assert len(epoch_end_payload) == 1
    assert epoch_end_payload[0]["epoch"] == 1
    assert "loss" in epoch_end_payload[0]["metrics"]
    assert epoch_end_payload[0]["payload"].epoch == 1


def test_base_trainer_uses_sample_weighted_epoch_means(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_weighted"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 0,
        },
        "model": {},
    }
    trainer = _WeightedTrainer(cfg)
    trainer.model = trainer._build_model()
    trainer.optimizer = torch.optim.SGD([nn.Parameter(torch.tensor(0.0))], lr=0.1)
    trainer.train_loader = [
        {"target": torch.zeros(2, 1, 4, 4)},
        {"target": torch.zeros(2, 1, 4, 4)},
    ]
    metrics = trainer._train_epoch(epoch=1)
    assert metrics["loss"] == pytest.approx((1.0 * 3.0 + 10.0 * 1.0) / 4.0)


def test_base_trainer_validation_is_deterministic_for_fixed_dataset(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "output_dir": str(tmp_path / "ckpts_det_val"),
            "use_amp": False,
            "manual_device": "cpu",
            "seed": 123,
            "deterministic_validation": True,
        },
        "model": {},
    }
    trainer = _WeightedTrainer(cfg)
    ds = _TinyDataset()
    trainer._setup(ds, val_dataset=ds, resume=None)
    metrics_a = trainer._validate_epoch(epoch=1)
    metrics_b = trainer._validate_epoch(epoch=1)
    assert metrics_a == metrics_b


def test_step_metrics_callback_writes_telemetry_csv(tmp_path: Path) -> None:
    callback = StepMetricsCallback(every_n_steps=1)

    param = nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([param], lr=0.5)
    param.grad = torch.tensor(2.0)

    class _FakeTrainer:
        output_dir = tmp_path
        optimizer = opt
        disc_optimizer = None
        scaler = None
        disc_scaler = None
        _last_step_batch_size = 4
        _last_step_seconds = 2.0

    callback.on_step_end(
        epoch=1,
        step=1,
        global_step=1,
        metrics={"loss": 0.25},
        trainer=_FakeTrainer(),
    )
    path = tmp_path / "step_metrics.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("epoch,step,global_step,loss,lr,grad_norm")
    assert "2.0" in rows[1]


def test_event_payloads_are_immutable() -> None:
    bus = TrainingEventBus()
    seen = {}

    def _listener(epoch: int, trainer, payload=None) -> None:
        seen["payload"] = payload

    bus.on("epoch_start", _listener)
    bus.emit("epoch_start", epoch=1, trainer=object())
    with pytest.raises(TypeError):
        seen["payload"].metrics["x"] = 1
