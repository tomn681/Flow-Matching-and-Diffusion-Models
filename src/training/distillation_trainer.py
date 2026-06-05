from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from core import NoisingScheduler
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
    """Teacher-student feature-matching distillation for diffusion-family denoisers.

    The student is trained to match the teacher prediction at the same noisy input
    and timestep. `teacher_steps` / `student_steps` are experiment bookkeeping for
    downstream sampling budgets; they do not change the loss function.
    """

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
        self.teacher_step_budget = int(self.model_cfg.get("teacher_steps", 128))
        self.student_step_budget = int(self.model_cfg.get("student_steps", 64))
        self.distillation_mode = str(self.training_cfg.get("distillation_mode", "feature_matching")).strip().lower()
        if self.distillation_mode not in {"feature_matching", "progressive"}:
            raise ValueError(
                f"distillation_mode must be 'feature_matching' or 'progressive', got '{self.distillation_mode}'."
            )

    def _build_default_callbacks(self) -> list[Any]:
        return [
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
        if self.student_step_budget <= 0 or self.teacher_step_budget <= 0:
            raise ValueError("model.student_steps and model.teacher_steps must be > 0.")
        if self.student_step_budget >= self.teacher_step_budget:
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
        if self.distillation_mode == "progressive":
            sched = self.teacher_scheduler
            if not hasattr(sched, "alphas_cumprod"):
                raise ValueError(
                    "Progressive distillation requires a DDPM-family scheduler with alphas_cumprod."
                )
            pred_type = getattr(getattr(sched, "config", None), "prediction_type", None)
            if pred_type != "epsilon":
                raise ValueError(
                    "Progressive distillation requires prediction_type='epsilon'. "
                    f"Got: {pred_type!r}"
                )
            if self.teacher_step_budget < 2:
                raise ValueError("Progressive distillation requires teacher_step_budget >= 2.")

    def _sample_noisy(self, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.teacher_scheduler is None:
            raise RuntimeError("Teacher scheduler not initialized.")
        num_train_timesteps = int(getattr(self.teacher_scheduler.config, "num_train_timesteps", 1000))
        timesteps = torch.randint(0, num_train_timesteps, (clean.size(0),), device=self.device).long()
        noise = torch.randn_like(clean)
        if isinstance(self.teacher_scheduler, NoisingScheduler):
            noisy = self.teacher_scheduler.add_noise(clean, noise, timesteps)
        else:
            scale = timesteps.float().view(-1, *([1] * (clean.dim() - 1))) / max(1, num_train_timesteps - 1)
            noisy = clean + scale * noise
        return noisy, timesteps

    def _sample_noisy_single_t(self, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.teacher_scheduler is None:
            raise RuntimeError("Teacher scheduler not initialized.")
        if not isinstance(self.teacher_scheduler, NoisingScheduler):
            raise ValueError("Progressive distillation requires a scheduler with add_noise support.")
        num_train_timesteps = int(getattr(self.teacher_scheduler.config, "num_train_timesteps", 1000))
        t_scalar = torch.randint(0, num_train_timesteps, (1,), device=self.device).long()
        timesteps = t_scalar.expand(clean.size(0))
        noise = torch.randn_like(clean)
        noisy = self.teacher_scheduler.add_noise(clean, noise, timesteps)
        return noisy, timesteps

    def _teacher_two_step_target_in_epsilon_space(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if self.teacher is None or self.teacher_scheduler is None:
            raise RuntimeError("Progressive target requested before teacher/scheduler initialization.")
        stride = max(1, int(getattr(self.teacher_scheduler.config, "num_train_timesteps", 1000)) // self.teacher_step_budget)
        t_int = int(timesteps[0].item())

        with torch.no_grad():
            eps_1 = unwrap_model_prediction(self.teacher(x_t, timesteps))
            out_1 = self.teacher_scheduler.step(eps_1, t_int, x_t)
            x_t_minus_1 = out_1.prev_sample

            t_prev_int = max(t_int - stride, 0)
            t_prev = torch.full_like(timesteps, t_prev_int)
            eps_2 = unwrap_model_prediction(self.teacher(x_t_minus_1, t_prev))
            out_2 = self.teacher_scheduler.step(eps_2, t_prev_int, x_t_minus_1)
            x_t_minus_2 = out_2.prev_sample

        alpha_bar_t = self.teacher_scheduler.alphas_cumprod[t_int].to(x_t.device, dtype=x_t.dtype)
        view_shape = (1,) + (1,) * (x_t.dim() - 1)
        sqrt_alpha = alpha_bar_t.sqrt().view(*view_shape)
        sqrt_one_minus = (1.0 - alpha_bar_t).sqrt().clamp(min=1e-8).view(*view_shape)
        epsilon_target = (x_t - sqrt_alpha * x_t_minus_2) / sqrt_one_minus
        return epsilon_target

    def _run_step(self, batch: dict, *, train: bool) -> dict[str, float]:
        if self.model is None or self.teacher is None:
            raise RuntimeError("DistillationTrainer called before initialization.")
        if self.optimizer is None:
            raise RuntimeError("DistillationTrainer optimizer not initialized.")

        clean = batch["target"].to(self.device)
        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        if self.distillation_mode == "progressive":
            noisy, timesteps = self._sample_noisy_single_t(clean)
            epsilon_target = self._teacher_two_step_target_in_epsilon_space(noisy, timesteps)
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                student_pred = self.model(noisy, timesteps)
                student_pred = unwrap_model_prediction(student_pred)
                loss = F.mse_loss(student_pred, epsilon_target.detach())
        else:
            noisy, timesteps = self._sample_noisy(clean)
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
