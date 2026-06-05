from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from core import NoisingScheduler
from training import DistillationTrainer, TRAINER_REGISTRY


class _DummyScheduler:
    class _Cfg:
        num_train_timesteps = 1000
        prediction_type = "epsilon"

    config = _Cfg()
    alphas_cumprod = torch.linspace(0.999, 0.001, 1000)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        scale = timesteps.float().view(-1, *([1] * (clean.dim() - 1))) / max(1, self.config.num_train_timesteps - 1)
        return clean + scale * noise

    def step(self, pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        _ = timestep

        class _Out:
            def __init__(self, prev_sample: torch.Tensor) -> None:
                self.prev_sample = prev_sample

        return _Out(sample - 0.1 * pred)


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


def test_distillation_trainer_uses_honest_step_budget_fields(tmp_path: Path) -> None:
    trainer = DistillationTrainer(config=_base_cfg(tmp_path), callbacks=[])
    assert trainer.teacher_step_budget == 128
    assert trainer.student_step_budget == 64


def test_dummy_scheduler_satisfies_noising_scheduler_protocol() -> None:
    assert isinstance(_DummyScheduler(), NoisingScheduler)


def test_progressive_distillation_mode_set_correctly(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["training"]["distillation_mode"] = "progressive"
    trainer = DistillationTrainer(config=cfg, callbacks=[])
    assert trainer.distillation_mode == "progressive"


def test_distillation_mode_invalid_raises(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["training"]["distillation_mode"] = "typo"
    try:
        DistillationTrainer(config=cfg, callbacks=[])
    except ValueError as exc:
        assert "distillation_mode" in str(exc)
    else:
        raise AssertionError("Expected invalid distillation_mode to raise ValueError.")


def test_progressive_distillation_uses_single_t_per_batch(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["training"]["distillation_mode"] = "progressive"
    trainer = DistillationTrainer(
        config=cfg,
        callbacks=[],
        model_override=_TinyUNet(0.5),
        teacher_override=_TinyUNet(1.0),
        scheduler_override=_DummyScheduler(),
    )
    trainer._setup(_TinyDataset(), val_dataset=None, resume=None)
    noisy, timesteps = trainer._sample_noisy_single_t(torch.zeros(3, 1, 8, 8))
    assert noisy.shape == (3, 1, 8, 8)
    assert len(set(timesteps.tolist())) == 1


def test_progressive_distillation_teacher_called_twice_per_step(tmp_path: Path) -> None:
    class _CountingTeacher(_TinyUNet):
        def __init__(self) -> None:
            super().__init__(1.0)
            self.calls = 0

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return super().forward(x, t)

    cfg = _base_cfg(tmp_path)
    cfg["training"]["distillation_mode"] = "progressive"
    teacher = _CountingTeacher()
    trainer = DistillationTrainer(
        config=cfg,
        callbacks=[],
        model_override=_TinyUNet(0.5),
        teacher_override=teacher,
        scheduler_override=_DummyScheduler(),
    )
    batch = {"target": torch.zeros(2, 1, 8, 8)}
    trainer._setup(_TinyDataset(), val_dataset=None, resume=None)
    trainer._run_step(batch, train=False)
    assert teacher.calls == 2


def test_progressive_distillation_target_shape_matches_model_output(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["training"]["distillation_mode"] = "progressive"
    trainer = DistillationTrainer(
        config=cfg,
        callbacks=[],
        model_override=_TinyUNet(0.5),
        teacher_override=_TinyUNet(1.0),
        scheduler_override=_DummyScheduler(),
    )
    trainer._setup(_TinyDataset(), val_dataset=None, resume=None)
    noisy = torch.randn(2, 1, 8, 8)
    timesteps = torch.full((2,), 10, dtype=torch.long)
    target = trainer._teacher_two_step_target_in_epsilon_space(noisy, timesteps)
    student = trainer.model(noisy, timesteps)
    assert target.shape == student.shape


def test_progressive_distillation_alpha_bar_clamping_at_t_zero(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["training"]["distillation_mode"] = "progressive"
    trainer = DistillationTrainer(
        config=cfg,
        callbacks=[],
        model_override=_TinyUNet(0.5),
        teacher_override=_TinyUNet(1.0),
        scheduler_override=_DummyScheduler(),
    )
    trainer._setup(_TinyDataset(), val_dataset=None, resume=None)
    noisy = torch.randn(2, 1, 8, 8)
    timesteps = torch.zeros(2, dtype=torch.long)
    target = trainer._teacher_two_step_target_in_epsilon_space(noisy, timesteps)
    assert torch.isfinite(target).all()


def test_progressive_distillation_non_ddpm_scheduler_raises_in_setup(tmp_path: Path) -> None:
    class _BadScheduler:
        class _Cfg:
            num_train_timesteps = 1000
            prediction_type = "epsilon"

        config = _Cfg()

        def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            _ = timesteps
            return clean + noise

    cfg = _base_cfg(tmp_path)
    cfg["training"]["distillation_mode"] = "progressive"
    trainer = DistillationTrainer(
        config=cfg,
        callbacks=[],
        model_override=_TinyUNet(0.5),
        teacher_override=_TinyUNet(1.0),
        scheduler_override=_BadScheduler(),
    )
    try:
        trainer._setup(_TinyDataset(), val_dataset=None, resume=None)
    except ValueError as exc:
        assert "alphas_cumprod" in str(exc)
    else:
        raise AssertionError("Expected non-DDPM scheduler to raise for progressive mode.")


def test_feature_matching_mode_output_unchanged(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    trainer = DistillationTrainer(
        config=cfg,
        callbacks=[],
        model_override=_TinyUNet(0.5),
        teacher_override=_TinyUNet(1.0),
        scheduler_override=_DummyScheduler(),
    )
    trainer._setup(_TinyDataset(), val_dataset=None, resume=None)
    batch = {"target": torch.zeros(2, 1, 8, 8)}
    metrics = trainer._run_step(batch, train=False)
    assert "loss" in metrics
    assert metrics["loss"] >= 0.0


def test_progressive_distillation_step_budget_less_than_2_raises(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["training"]["distillation_mode"] = "progressive"
    cfg["model"]["teacher_steps"] = 1
    cfg["model"]["student_steps"] = 0
    trainer = DistillationTrainer(
        config=cfg,
        callbacks=[],
        model_override=_TinyUNet(0.5),
        teacher_override=_TinyUNet(1.0),
        scheduler_override=_DummyScheduler(),
    )
    try:
        trainer._setup(_TinyDataset(), val_dataset=None, resume=None)
    except ValueError as exc:
        assert "must be > 0" in str(exc)
    else:
        raise AssertionError("Expected invalid progressive step budget to raise ValueError.")
