from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from training.base import BaseTrainer


class _MinimalTrainer(BaseTrainer):
    def _build_model(self):
        return nn.Linear(1, 1)

    def _training_step(self, batch, *, epoch):
        return {"loss": 0.0}


def test_ensure_device_same_device() -> None:
    t = torch.randn(2, 3)
    result = BaseTrainer._ensure_device(t, torch.device("cpu"))
    assert result is t


def test_ensure_device_returns_tensor_on_target() -> None:
    t = torch.randn(2, 3)
    result = BaseTrainer._ensure_device(t, torch.device("cpu"))
    assert result.device == torch.device("cpu")


def test_backward_without_scaler() -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = None
    param = torch.tensor(2.0, requires_grad=True)
    loss = param * 3.0
    trainer._backward(loss)
    assert param.grad is not None
    assert param.grad.item() == pytest.approx(3.0)


def test_backward_with_disabled_scaler() -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
    param = torch.tensor(2.0, requires_grad=True)
    loss = param * 3.0
    trainer._backward(loss)
    assert param.grad is not None


def test_step_optimizers_plain() -> None:
    param = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([param], lr=0.1)
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
    param.grad = torch.tensor(1.0)
    trainer._step_optimizers(opt)
    assert param.item() == pytest.approx(0.9)


def test_step_optimizers_skips_none() -> None:
    param = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([param], lr=0.1)
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
    param.grad = torch.tensor(1.0)
    trainer._step_optimizers(opt, None)
    assert param.item() == pytest.approx(0.9)


def test_build_checkpoint_dict_contains_expected_fields() -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.best_metric = 0.123
    # build a minimal TrainingState directly to avoid model/optimizer setup dependency
    from core.types import TrainingState

    ts = TrainingState(
        epoch=2,
        global_step=10,
        model_state={"w": torch.tensor(1.0)},
        optimizer_state={"state": {}},
        metrics={"loss": 1.0},
        extra={"scheduler": None, "scaler": None},
    )
    payload = trainer._build_checkpoint_dict(ts)
    assert payload["model"] == ts.model_state
    assert payload["optimizer"] == ts.optimizer_state
    assert payload["epoch"] == 2
    assert payload["global_step"] == 10
    assert payload["best_metric"] == pytest.approx(0.123)


def test_resume_from_payload_hook_is_noop() -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    assert trainer._resume_from_payload({"x": 1}) is None
