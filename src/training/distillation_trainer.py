from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from core.types import unwrap_model_prediction
from scheduling.builder import build_scheduler
from scheduling.lr import build_lr_scheduler
from utils.model_utils.diffusion_utils import build_diffusion_model
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback, TensorBoardCallback
from .registry import TRAINER_REGISTRY
import utils


@TRAINER_REGISTRY.register("distillation")
class DistillationTrainer(BaseTrainer):
    """Teacher-student distillation trainer for diffusion-family denoisers."""

    checkpoint_prefix = "distill"

    def __init__(
        self,
        config: dict,
        callbacks: list[Any] | None = None,
        event_bus=None,
        model_override: torch.nn.Module | None = None,
        teacher_override: torch.nn.Module | None = None,
        scheduler_override: Any = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks, event_bus=event_bus)
        self._model_override = model_override
        self._teacher_override = teacher_override
        self._scheduler_override = scheduler_override
        self.teacher: torch.nn.Module | None = None
        self.teacher_scheduler = None
        self.teacher_steps = int(self.model_cfg.get("teacher_steps", 128))
        self.student_steps = int(self.model_cfg.get("student_steps", 64))

        if callbacks is None:
            self.callbacks = [
                CheckpointCallback(
                    filename_prefix=self.checkpoint_prefix,
                    monitor="val_loss" if self.training_cfg.get("validate", True) else "loss",
                    mode="min",
                    save_every=int(self.training_cfg.get("save_every", 0)),
                ),
                MetricsCSVCallback(),
                TensorBoardCallback(),
            ]

    @classmethod
    def from_config(cls, path_or_dict: str | Path | dict) -> "DistillationTrainer":
        if isinstance(path_or_dict, (str, Path)):
            cfg = utils.load_json_config(path_or_dict)
        else:
            cfg = path_or_dict
        return cls(config=cfg)

    def _effective_student_model_type(self) -> str:
        student_type = str(self.model_cfg.get("student_model_type", "diffusion")).strip().lower()
        if student_type not in {"diffusion", "flow_matching", "rectified_flow", "consistency", "edm"}:
            raise ValueError(
                f"Unsupported model.student_model_type '{student_type}' for distillation trainer."
            )
        return student_type

    def _model_build_config(self) -> dict:
        cfg = dict(self.raw_config)
        model_cfg = dict(cfg.get("model", {}))
        model_cfg["model_type"] = self._effective_student_model_type()
        cfg["model"] = model_cfg
        return cfg

    def _build_model(self) -> torch.nn.Module:
        if self._model_override is not None:
            return self._model_override.to(self.device)
        return build_diffusion_model(self._model_build_config(), self.device, ckpt_path=None, set_eval=False)

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.optimizer is None:
            raise RuntimeError("DistillationTrainer._build_lr_scheduler called before optimizer initialization.")
        return build_lr_scheduler(self.optimizer, self.training_cfg)

    def _load_teacher(self, ckpt_path: str | Path) -> torch.nn.Module:
        return build_diffusion_model(
            self._model_build_config(),
            self.device,
            ckpt_path=str(ckpt_path),
            set_eval=True,
        )

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)
        if self.student_steps <= 0 or self.teacher_steps <= 0:
            raise ValueError("model.student_steps and model.teacher_steps must be > 0.")
        if self.student_steps >= self.teacher_steps:
            raise ValueError("Distillation requires model.student_steps < model.teacher_steps.")

        if self._teacher_override is not None:
            self.teacher = self._teacher_override.to(self.device)
            self.teacher.eval()
        else:
            teacher_ckpt = self.model_cfg.get("teacher_checkpoint")
            if not teacher_ckpt:
                raise ValueError("Distillation requires 'model.teacher_checkpoint'.")
            self.teacher = self._load_teacher(str(teacher_ckpt))

        for param in self.teacher.parameters():
            param.requires_grad_(False)

        if self._scheduler_override is not None:
            self.teacher_scheduler = self._scheduler_override
        else:
            self.teacher_scheduler, _ = build_scheduler(self.model_cfg.get("scheduler", {}), self.training_cfg)

    def _sample_noisy(self, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.teacher_scheduler is None:
            raise RuntimeError("Teacher scheduler not initialized.")
        num_train_timesteps = int(getattr(self.teacher_scheduler.config, "num_train_timesteps", 1000))
        timesteps = torch.randint(0, num_train_timesteps, (clean.size(0),), device=self.device).long()
        noise = torch.randn_like(clean)
        if hasattr(self.teacher_scheduler, "add_noise"):
            noisy = self.teacher_scheduler.add_noise(clean, noise, timesteps)
        else:
            scale = timesteps.float().view(-1, *([1] * (clean.dim() - 1))) / max(1, num_train_timesteps - 1)
            noisy = clean + scale * noise
        return noisy, timesteps

    def _run_step(self, batch: dict, *, train: bool) -> dict[str, float]:
        if self.model is None or self.teacher is None:
            raise RuntimeError("DistillationTrainer called before initialization.")
        if self.optimizer is None:
            raise RuntimeError("DistillationTrainer optimizer not initialized.")

        clean = batch["target"].to(self.device)
        noisy, timesteps = self._sample_noisy(clean)
        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_pred = self.teacher(noisy, timesteps)
            teacher_pred = unwrap_model_prediction(teacher_pred)

        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            student_pred = self.model(noisy, timesteps)
            student_pred = unwrap_model_prediction(student_pred)
            loss = F.mse_loss(student_pred, teacher_pred)

        if train:
            self._backward(loss)
            self._step_optimizers(self.optimizer)

        return {"loss": float(loss.detach().item())}

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        del epoch
        return self._run_step(batch, train=True)

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        del epoch
        with torch.no_grad():
            return self._run_step(batch, train=False)

