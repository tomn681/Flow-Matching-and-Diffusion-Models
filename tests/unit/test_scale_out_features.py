from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from training.base import BaseTrainer
from training.integrations import MLflowCallback, ThroughputCallback, WandBCallback


class _TinyTrainer(BaseTrainer):
    def _build_model(self):
        return nn.Linear(1, 1)

    def _training_step(self, batch, *, epoch):
        x = batch["target"].to(self.device)
        pred = self.model(x)
        loss = pred.mean()
        self._backward(loss)
        self._step_optimizers(self.optimizer)
        return {"loss": float(loss.detach().item())}


class _CheckpointableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.enabled = True

    def forward(self, x):
        return self.linear(x)


class _FakeDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        del idx
        return {"target": torch.ones(1)}


def test_trainer_enables_gradient_checkpointing_flag() -> None:
    trainer = _TinyTrainer(config={"training": {"gradient_checkpointing": True}, "model": {}})
    trainer.model = _CheckpointableModel()
    trainer._maybe_enable_gradient_checkpointing()
    assert trainer.model.enabled is True


def test_trainer_uses_fsdp_strategy_wrapper(monkeypatch) -> None:
    trainer = _TinyTrainer(config={"training": {"distributed_strategy": "fsdp"}, "model": {}})
    model = nn.Linear(1, 1)
    trainer.model = model
    trainer.distributed = True
    trainer.device = torch.device("cpu")
    wrapped = object()
    monkeypatch.setattr("training.base.utils.wrap_fsdp", lambda module, device=None: wrapped)
    trainer._maybe_wrap_distributed_model()
    assert trainer.model is wrapped


def test_throughput_callback_writes_csv(tmp_path: Path) -> None:
    callback = ThroughputCallback(every_n_steps=1)
    trainer = type(
        "T",
        (),
        {
            "is_main_process": True,
            "output_dir": tmp_path,
            "_last_step_batch_size": 4,
            "_last_step_seconds": 0.5,
        },
    )()
    callback.on_step_end(epoch=1, step=1, global_step=1, metrics={"loss": 1.0}, trainer=trainer)
    path = tmp_path / "throughput.csv"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "imgs_per_sec" in text
    assert "8.0" in text


def test_wandb_callback_logs_and_finishes(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class _Run:
        def log(self, payload, step=None):
            calls.append(("log", dict(payload)))

        def finish(self):
            calls.append(("finish", None))

    class _FakeWandB:
        def init(self, **kwargs):
            calls.append(("init", dict(kwargs)))
            return _Run()

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandB())
    cb = WandBCallback(project="proj", run_name="run", config={"a": 1})
    trainer = type("T", (), {"is_main_process": True, "global_step": 7})()
    cb.on_epoch_end(epoch=1, metrics={"loss": 0.5}, state={}, trainer=trainer)
    cb.on_train_end(trainer=trainer)
    assert calls[0][0] == "init"
    assert calls[1][0] == "log"
    assert calls[-1][0] == "finish"


def test_mlflow_callback_logs_and_ends(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class _FakeMLflow:
        def set_experiment(self, name):
            calls.append(("experiment", name))

        def start_run(self, run_name=None):
            calls.append(("start", run_name))

        def log_metrics(self, payload, step=None):
            calls.append(("metrics", dict(payload)))

        def log_metric(self, key, value, step=None):
            calls.append(("metric", (key, value, step)))

        def end_run(self):
            calls.append(("end", None))

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMLflow())
    cb = MLflowCallback(experiment_name="exp", run_name="run")
    trainer = type("T", (), {"is_main_process": True, "global_step": 3})()
    cb.on_epoch_end(epoch=2, metrics={"loss": 1.25}, state={}, trainer=trainer)
    cb.on_train_end(trainer=trainer)
    assert calls[0] == ("experiment", "exp")
    assert calls[1] == ("start", "run")
    assert calls[-1][0] == "end"


def test_throughput_callback_is_auto_registered_and_populated(tmp_path: Path) -> None:
    cfg = {
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "output_dir": str(tmp_path),
            "manual_device": "cpu",
            "throughput_every": 1,
        },
        "model": {},
    }
    trainer = _TinyTrainer(config=cfg)
    ds = _FakeDataset()
    trainer.fit(ds, val_dataset=None)
    assert (Path(trainer.output_dir) / "throughput.csv").exists()
