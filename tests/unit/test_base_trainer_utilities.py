from __future__ import annotations

import os
from pathlib import Path
import warnings

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from scheduling.lr import build_lr_scheduler
from training.base import BaseTrainer
from training.callbacks import CheckpointCallback


class _MinimalTrainer(BaseTrainer):
    def _build_model(self):
        return nn.Linear(1, 1)

    def _training_step(self, batch, *, epoch):
        return {"loss": 0.0}


class _InterruptOnEpochTrainer(_MinimalTrainer):
    def __init__(self, *args, interrupt_epoch: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._interrupt_epoch = int(interrupt_epoch)

    def _training_step(self, batch, *, epoch):
        if int(epoch) == self._interrupt_epoch:
            raise KeyboardInterrupt
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


def test_training_value_prefers_validated_and_normalized_training_config() -> None:
    trainer = _MinimalTrainer(config={"training": {"train_batch_size": 9}, "model": {}})
    assert trainer.training.batch_size == 9
    assert trainer._training_value("batch_size") == 9


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


def test_step_optimizers_clips_grad_norm() -> None:
    param = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([param], lr=1.0)
    trainer = _MinimalTrainer(config={"training": {"max_grad_norm": 0.1}, "model": {}})
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    param.grad = torch.tensor(10.0)
    trainer._step_optimizers(opt)
    assert param.item() == pytest.approx(0.9, abs=1e-5)


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


def test_step_optimizers_with_enabled_scaler_marks_optimizer_step_for_scheduler() -> None:
    class _FakeScaler:
        def __init__(self) -> None:
            self.stepped: list[torch.optim.Optimizer] = []

        def is_enabled(self) -> bool:
            return True

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            self.stepped.append(optimizer)

        def update(self) -> None:
            return None

    param = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([param], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = _FakeScaler()
    trainer.lr_scheduler = scheduler
    param.grad = torch.tensor(1.0)

    trainer._step_optimizers(opt)

    assert trainer._optimizer_stepped_since_scheduler is True
    assert trainer._main_optimizer_step_count == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scheduler.step()
    assert not any("lr_scheduler.step() before optimizer.step()" in str(w.message) for w in caught)


def test_step_scheduler_advances_during_optimizer_step() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([param], lr=1.0)
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    trainer._scheduler_step_unit = "step"
    param.grad = torch.tensor(1.0)
    trainer._step_optimizers(opt)
    assert trainer.lr_scheduler.last_epoch == 1


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
    assert payload["format_version"] >= 1
    assert payload["best_metric"] == pytest.approx(0.123)
    assert payload["resolved_config"] == trainer.raw_config
    assert payload["extra"]["resolution_stage"] == 2


def test_resume_from_payload_hook_is_noop() -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    assert trainer._resume_from_payload({"x": 1}) is None


def test_resolve_resume_epoch_from_primary_key() -> None:
    assert BaseTrainer._resolve_resume_epoch({"epoch": 8}) == 8


def test_resolve_resume_epoch_from_legacy_key() -> None:
    assert BaseTrainer._resolve_resume_epoch({"current_epoch": 11}) == 11


def test_resolve_resume_epoch_without_explicit_epoch_returns_zero() -> None:
    ckpt = Path("/tmp/run/epoch66_loss0123.pt")
    assert BaseTrainer._resolve_resume_epoch({}, ckpt_path=ckpt) == 0


def test_setup_restores_global_step_and_callback_best_metric_from_resume(tmp_path: Path) -> None:
    trainer = _MinimalTrainer(
        config={
            "training": {
                "manual_device": "cpu",
                "output_dir": str(tmp_path),
                "batch_size": 1,
                "num_workers": 0,
            },
            "model": {},
        },
        callbacks=[CheckpointCallback(filename_prefix="model", monitor="loss", mode="min")],
    )
    ckpt = tmp_path / "resume.pt"
    torch.save(
        {
            "model": trainer._build_model().state_dict(),
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
            "epoch": 3,
            "global_step": 77,
            "best_metric": 0.25,
        },
        ckpt,
    )
    trainer._setup([{"target": torch.zeros(1, 1, 1)}], val_dataset=None, resume=str(ckpt))
    assert trainer.global_step == 77
    callback = trainer.callbacks[0]
    assert isinstance(callback, CheckpointCallback)
    assert callback.best_metric == pytest.approx(0.25)


def test_setup_restores_scheduler_state_by_default(tmp_path: Path) -> None:
    class _SchedulerTrainer(_MinimalTrainer):
        def _build_lr_scheduler(self):
            if self.optimizer is None:
                raise RuntimeError("optimizer not initialized")
            return StepLR(self.optimizer, step_size=1, gamma=0.1)

    trainer = _SchedulerTrainer(
        config={
            "training": {
                "manual_device": "cpu",
                "output_dir": str(tmp_path),
                "batch_size": 1,
                "num_workers": 0,
                "learning_rate": 1e-3,
            },
            "model": {},
        }
    )
    seed_model = trainer._build_model()
    seed_optimizer = torch.optim.AdamW(seed_model.parameters(), lr=1e-3)
    seed_scheduler = StepLR(seed_optimizer, step_size=1, gamma=0.1)
    seed_optimizer.step()
    seed_scheduler.step()
    ckpt = tmp_path / "resume_sched.pt"
    torch.save(
        {
            "model": seed_model.state_dict(),
            "optimizer": seed_optimizer.state_dict(),
            "scheduler": seed_scheduler.state_dict(),
            "scaler": None,
            "epoch": 3,
        },
        ckpt,
    )

    trainer._setup([{"target": torch.zeros(1, 1, 1)}], val_dataset=None, resume=str(ckpt))

    assert trainer.lr_scheduler is not None
    assert trainer.lr_scheduler.last_epoch == seed_scheduler.last_epoch


def test_setup_continue_scheduler_mode_falls_back_to_resumed_lr_when_progress_cannot_be_inferred(tmp_path: Path) -> None:
    class _SchedulerTrainer(_MinimalTrainer):
        def _build_lr_scheduler(self):
            if self.optimizer is None:
                raise RuntimeError("optimizer not initialized")
            return StepLR(self.optimizer, step_size=1, gamma=0.1)

    trainer = _SchedulerTrainer(
        config={
            "training": {
                "manual_device": "cpu",
                "output_dir": str(tmp_path),
                "batch_size": 1,
                "num_workers": 0,
                "learning_rate": 1e-3,
            },
            "model": {},
        }
    )
    trainer._resume_scheduler_mode_override = "continue"
    seed_model = trainer._build_model()
    seed_optimizer = torch.optim.AdamW(seed_model.parameters(), lr=1e-3)
    seed_scheduler = StepLR(seed_optimizer, step_size=1, gamma=0.1)
    seed_optimizer.step()
    seed_scheduler.step()
    ckpt = tmp_path / "resume_sched_skip.pt"
    torch.save(
        {
            "model": seed_model.state_dict(),
            "optimizer": seed_optimizer.state_dict(),
            "scheduler": seed_scheduler.state_dict(),
            "scaler": None,
            "epoch": 3,
        },
        ckpt,
    )

    trainer._setup([{"target": torch.zeros(1, 1, 1)}], val_dataset=None, resume=str(ckpt))

    assert trainer.lr_scheduler is not None
    assert trainer.lr_scheduler.last_epoch == 0
    resumed_lr = float(seed_optimizer.param_groups[0]["lr"])
    assert float(trainer.optimizer.param_groups[0]["lr"]) == pytest.approx(resumed_lr)
    assert float(trainer.lr_scheduler.base_lrs[0]) == pytest.approx(resumed_lr)


def test_setup_reset_scheduler_mode_starts_from_scheduler_base_lr(tmp_path: Path) -> None:
    class _SchedulerTrainer(_MinimalTrainer):
        def _build_lr_scheduler(self):
            if self.optimizer is None:
                raise RuntimeError("optimizer not initialized")
            return StepLR(self.optimizer, step_size=1, gamma=0.1)

    trainer = _SchedulerTrainer(
        config={
            "training": {
                "manual_device": "cpu",
                "output_dir": str(tmp_path),
                "batch_size": 1,
                "num_workers": 0,
                "learning_rate": 1e-3,
            },
            "model": {},
        }
    )
    trainer._resume_scheduler_mode_override = "reset"
    seed_model = trainer._build_model()
    seed_optimizer = torch.optim.AdamW(seed_model.parameters(), lr=1e-3)
    seed_scheduler = StepLR(seed_optimizer, step_size=1, gamma=0.1)
    seed_optimizer.step()
    seed_scheduler.step()
    ckpt = tmp_path / "resume_sched_reset.pt"
    torch.save(
        {
            "model": seed_model.state_dict(),
            "optimizer": seed_optimizer.state_dict(),
            "scheduler": seed_scheduler.state_dict(),
            "scaler": None,
            "epoch": 3,
        },
        ckpt,
    )

    trainer._setup([{"target": torch.zeros(1, 1, 1)}], val_dataset=None, resume=str(ckpt))

    assert trainer.lr_scheduler is not None
    assert trainer.lr_scheduler.last_epoch == 0
    assert float(trainer.optimizer.param_groups[0]["lr"]) == pytest.approx(1e-3)
    assert float(trainer.lr_scheduler.base_lrs[0]) == pytest.approx(1e-3)


def test_setup_continue_scheduler_mode_remaps_warmup_cosine_progress_to_new_horizon(tmp_path: Path) -> None:
    class _WarmupTrainer(_MinimalTrainer):
        def _build_lr_scheduler(self):
            if self.optimizer is None:
                raise RuntimeError("optimizer not initialized")
            return build_lr_scheduler(self.optimizer, self._lr_scheduler_config())

    old_cfg = {
        "training": {
            "manual_device": "cpu",
            "output_dir": str(tmp_path),
            "batch_size": 1,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "epochs": 50,
            "lr_warmup_steps": 0,
            "lr_scheduler": {"name": "warmup_cosine"},
        },
        "model": {},
    }
    new_cfg = {
        "training": {
            "manual_device": "cpu",
            "output_dir": str(tmp_path),
            "batch_size": 1,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "epochs": 200,
            "lr_warmup_steps": 0,
            "lr_scheduler": {"name": "warmup_cosine", "params": {"epochs": 10}},
        },
        "model": {},
    }

    seed_trainer = _WarmupTrainer(config=old_cfg)
    seed_trainer.model = seed_trainer._build_model()
    seed_trainer.optimizer = seed_trainer._build_optimizer()
    seed_trainer.train_dataset = [{"target": torch.zeros(1, 1, 1)}]
    seed_trainer._rebuild_dataloaders(target_resolution=None, train_dataset=seed_trainer.train_dataset, val_dataset=None)
    seed_trainer.lr_scheduler = seed_trainer._build_lr_scheduler()
    for param in seed_trainer.model.parameters():
        param.grad = torch.zeros_like(param)
    for _ in range(20):
        seed_trainer.optimizer.step()
        seed_trainer.lr_scheduler.step()

    ckpt = tmp_path / "resume_warmup_cosine.pt"
    torch.save(
        {
            "model": seed_trainer.model.state_dict(),
            "optimizer": seed_trainer.optimizer.state_dict(),
            "scheduler": seed_trainer.lr_scheduler.state_dict(),
            "scaler": None,
            "epoch": 20,
            "resolved_config": old_cfg,
        },
        ckpt,
    )

    trainer = _WarmupTrainer(config=new_cfg)
    trainer._resume_scheduler_mode_override = "continue"
    trainer._setup([{"target": torch.zeros(1, 1, 1)}], val_dataset=None, resume=str(ckpt))

    assert trainer.lr_scheduler is not None
    assert trainer.lr_scheduler.last_epoch == 4
    expected_lr = 1e-3 * 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * 0.4))).item()
    assert float(trainer.optimizer.param_groups[0]["lr"]) == pytest.approx(expected_lr)


def test_interrupt_checkpoint_is_saved_at_epoch_start_and_not_overwritten_mid_epoch(tmp_path: Path) -> None:
    trainer = _InterruptOnEpochTrainer(
        config={
            "training": {
                "manual_device": "cpu",
                "output_dir": str(tmp_path),
                "batch_size": 1,
                "num_workers": 0,
                "epochs": 3,
            },
            "model": {},
        },
        interrupt_epoch=2,
    )
    dataset = [{"target": torch.zeros(1, 1, 1)}]

    with pytest.raises(KeyboardInterrupt):
        trainer.fit(dataset, val_dataset=None)

    ckpt = Path(trainer.output_dir) / "interrupt_last.pt"
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 1
    assert payload.get("extra", {}).get("interrupt_epoch_start") is True

    resumed = _MinimalTrainer(
        config={
            "training": {
                "manual_device": "cpu",
                "output_dir": str(tmp_path),
                "batch_size": 1,
                "num_workers": 0,
                "epochs": 3,
            },
            "model": {},
        }
    )
    resumed._setup(dataset, val_dataset=None, resume=str(ckpt))
    assert resumed.start_epoch == 2


def test_build_optimizer_uses_only_trainable_params() -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.model = nn.Linear(4, 4)
    for p in trainer.model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="No trainable parameters"):
        trainer._build_optimizer()


def test_rebuild_dataloaders_uses_distributed_sampler_when_distributed() -> None:
    trainer = _MinimalTrainer(config={"training": {"batch_size": 2, "num_workers": 0}, "model": {}})
    trainer.distributed = True
    trainer.rank = 0
    trainer.world_size = 2
    dataset = [{"target": torch.zeros(1, 1, 1)} for _ in range(4)]
    trainer.train_dataset = dataset
    trainer.val_dataset = dataset
    trainer._rebuild_dataloaders(target_resolution=None)
    assert trainer.train_loader is not None
    assert trainer.val_loader is not None
    assert trainer.train_loader.sampler.__class__.__name__ == "DistributedSampler"
    assert trainer.val_loader.sampler.__class__.__name__ == "DistributedSampler"


def test_reduce_metrics_all_reduces_across_ranks(monkeypatch) -> None:
    trainer = _MinimalTrainer(config={"training": {}, "model": {}})
    trainer.distributed = True
    trainer.device = torch.device("cpu")

    def _fake_reduce(tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 2.0

    monkeypatch.setattr("training.base.utils.all_reduce_tensor", _fake_reduce)
    reduced = trainer._reduce_metrics({"loss": 4.0, "aux": 2.0}, 2.0)
    assert reduced["loss"] == pytest.approx(2.0)
    assert reduced["aux"] == pytest.approx(1.0)


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


def test_format_epoch_summary_uses_zero_padded_width_and_single_line(monkeypatch) -> None:
    trainer = _MinimalTrainer(config={"training": {"epochs": 500}, "model": {}})
    monkeypatch.setattr("training.base.shutil.get_terminal_size", lambda fallback=(120, 24): os.terminal_size((200, 24)))

    summary = trainer._format_epoch_summary(
        epoch=9,
        epochs=500,
        train_items=[("loss", 0.123456), ("denoise_mse", 0.123456)],
        val_items=[("loss", 0.023456), ("denoise_mse", 0.023456)],
    )

    assert summary == (
        "Ep 009/500 | train loss 0.123456 mse 0.123456 | "
        "val loss 0.023456 mse 0.023456"
    )


def test_format_epoch_summary_falls_back_to_grouped_two_line_layout(monkeypatch) -> None:
    trainer = _MinimalTrainer(config={"training": {"epochs": 50}, "model": {}})
    monkeypatch.setattr("training.base.shutil.get_terminal_size", lambda fallback=(120, 24): os.terminal_size((50, 24)))
    train_items = [
        ("loss", 0.123456),
        ("denoise_mse", 0.123456),
        ("long_metric_name", 0.333333),
    ]
    val_items = [
        ("loss", 0.023456),
        ("denoise_mse", 0.023456),
        ("long_metric_name", 0.111111),
    ]

    summary = trainer._format_epoch_summary(
        epoch=1,
        epochs=50,
        train_items=train_items,
        val_items=val_items,
    )

    assert summary == (
        "Ep 001/050 | train loss 0.123456 mse 0.123456 long_metric_name 0.333333\n"
        "           | val loss 0.023456 mse 0.023456 long_metric_name 0.111111"
    )
