from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from training.base import BaseTrainer


class _MinimalTrainer(BaseTrainer):
    def _build_model(self):
        return nn.Linear(1, 1)

    def _training_step(self, batch, *, epoch):
        return {"loss": 0.0}


class _TinyAttnModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.to_v = nn.Linear(8, 8)
        self.to_out = nn.ModuleList([nn.Linear(8, 8), nn.Dropout(0.0)])


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


def test_base_trainer_validates_config_on_construction() -> None:
    with pytest.raises(ValueError, match="epochs"):
        _MinimalTrainer(config={"training": {"epochs": 0}, "model": {}})


def test_backward_with_disabled_scaler() -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    param = torch.tensor(2.0, requires_grad=True)
    loss = param * 3.0
    trainer._backward(loss)
    assert param.grad is not None


def test_step_optimizers_plain() -> None:
    param = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([param], lr=0.1)
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    param.grad = torch.tensor(1.0)
    trainer._step_optimizers(opt)
    assert param.item() == pytest.approx(0.9)


def test_step_optimizers_skips_none() -> None:
    param = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([param], lr=0.1)
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    param.grad = torch.tensor(1.0)
    trainer._step_optimizers(opt, None)
    assert param.item() == pytest.approx(0.9)


def test_step_optimizers_with_enabled_scaler_skips_optimizers_without_grads() -> None:
    class _FakeScaler:
        def __init__(self) -> None:
            self.stepped: list[torch.optim.Optimizer] = []
            self.updated = False

        def is_enabled(self) -> bool:
            return True

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            self.stepped.append(optimizer)

        def update(self) -> None:
            self.updated = True

    param_a = torch.tensor(1.0, requires_grad=True)
    param_b = torch.tensor(2.0, requires_grad=True)
    opt_a = torch.optim.SGD([param_a], lr=0.1)
    opt_b = torch.optim.SGD([param_b], lr=0.1)
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = _FakeScaler()
    param_a.grad = torch.tensor(1.0)
    param_b.grad = None
    trainer._step_optimizers(opt_a, opt_b)
    assert trainer.scaler.stepped == [opt_a]
    assert trainer.scaler.updated is True


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
        extra={"scheduler": None, "scaler": None, "resolution_stage": 2},
    )
    payload = trainer._build_checkpoint_dict(ts)
    assert payload["model"] == ts.model_state
    assert payload["optimizer"] == ts.optimizer_state
    assert payload["epoch"] == 2
    assert payload["global_step"] == 10
    assert payload["best_metric"] == pytest.approx(0.123)
    assert payload["extra"]["resolution_stage"] == 2


def test_resume_from_payload_hook_is_noop() -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    assert trainer._resume_from_payload({"x": 1}) is None


def test_resolve_resume_epoch_from_primary_key() -> None:
    assert BaseTrainer._resolve_resume_epoch({"epoch": 8}) == 8


def test_resolve_resume_epoch_from_legacy_key() -> None:
    assert BaseTrainer._resolve_resume_epoch({"current_epoch": 11}) == 11


def test_resolve_resume_epoch_from_checkpoint_path() -> None:
    ckpt = Path("/tmp/run/epoch0009/epoch.pt")
    assert BaseTrainer._resolve_resume_epoch({}, ckpt_path=ckpt) == 9


def test_build_optimizer_uses_only_trainable_params() -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.model = nn.Linear(4, 4)
    for p in trainer.model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="No trainable parameters"):
        trainer._build_optimizer()


def test_maybe_apply_lora_from_training_config() -> None:
    trainer = _MinimalTrainer(
        config={
            "training": {"lora": {"enabled": True, "rank": 2, "alpha": 1.0}},
            "model": {},
        }
    )
    trainer.model = _TinyAttnModel()
    trainer._maybe_apply_lora()
    trainable = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(("lora_A" in name or "lora_B" in name) for name in trainable)
