from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from noise import NOISE_REGISTRY
from scheduling import (
    resolve_conditioning_mode,
    resolve_conditioning_adapter,
)
from scheduling.builder import build_scheduler
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback
from .registry import TRAINER_REGISTRY
from utils.model_utils.diffusion_utils import build_diffusion_model
import utils


@TRAINER_REGISTRY.register("flow_matching")
@TRAINER_REGISTRY.register("diffusion")
class GenerativeTrainer(BaseTrainer):
    """Tier-1 trainer for diffusion and flow-matching UNet models."""

    def __init__(self, config: dict, callbacks: list[Any] | None = None) -> None:
        super().__init__(config=config, callbacks=callbacks)
        model_type = str(self.model_cfg.get("model_type", "diffusion")).lower()
        if model_type not in {"diffusion", "flow_matching"}:
            raise ValueError(f"Unsupported model_type '{model_type}' for GenerativeTrainer.")
        self.model_type = model_type
        self.grad_accum = max(1, int(self.training_cfg.get("gradient_accumulation_steps", 1)))
        self.latent_norm = self.training_cfg.get("latent_norm")
        self.conditioning_mode = resolve_conditioning_mode(
            self.training_cfg.get("conditioning") or self.model_cfg.get("conditioning")
        )
        self.conditioning_adapter = resolve_conditioning_adapter(self.conditioning_mode)
        self.noise_process = None

        if callbacks is None:
            prefix = "diff" if self.model_type == "diffusion" else "flow"
            self.callbacks = [
                CheckpointCallback(
                    filename_prefix=prefix,
                    monitor="val_loss" if self.training_cfg.get("validate", True) else "loss",
                    mode="min",
                    save_every=int(self.training_cfg.get("save_every", 0)),
                ),
                MetricsCSVCallback(),
            ]

    @classmethod
    def from_config(cls, path_or_dict: str | Path | dict) -> "GenerativeTrainer":
        if isinstance(path_or_dict, (str, Path)):
            cfg = utils.load_json_config(path_or_dict)
        else:
            cfg = path_or_dict
        return cls(config=cfg)

    def _build_model(self) -> torch.nn.Module:
        return build_diffusion_model(self.raw_config, self.device, ckpt_path=None, set_eval=False)

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)
        scheduler_cfg = self.model_cfg.get("scheduler", {})
        train_scheduler, _ = build_scheduler(scheduler_cfg, self.training_cfg)
        noise_key = "ddpm" if self.model_type == "diffusion" else "flow_matching"
        self.noise_process = NOISE_REGISTRY.build(noise_key, scheduler=train_scheduler)

    def _run_step(self, batch: dict, *, epoch: int, train: bool) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("GenerativeTrainer._run_step called before model initialization.")
        if self.optimizer is None:
            raise RuntimeError("GenerativeTrainer._run_step called before optimizer initialization.")
        if self.scaler is None:
            raise RuntimeError("GenerativeTrainer._run_step called before scaler initialization.")
        if self.noise_process is None:
            raise RuntimeError("GenerativeTrainer._run_step called before noise process initialization.")

        clean = batch["target"].to(self.device)
        cond = batch.get("image")
        cond = cond.to(self.device) if cond is not None else None

        bs = clean.size(0)
        chunk_size = max(1, (bs + self.grad_accum - 1) // self.grad_accum)
        clean_chunks = clean.split(chunk_size)
        cond_chunks = cond.split(chunk_size) if cond is not None else [None] * len(clean_chunks)
        accum_steps = len(clean_chunks)
        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_samples = 0

        for clean_chunk, cond_chunk in zip(clean_chunks, cond_chunks):
            noisy_batch = self.noise_process(clean_chunk, self.device)
            model_input = noisy_batch.noisy
            model_input, context = self.conditioning_adapter(model_input, cond_chunk, self.latent_norm)

            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                pred = (
                    self.model(model_input, noisy_batch.timesteps, context_ca=context)
                    if context is not None
                    else self.model(model_input, noisy_batch.timesteps)
                )
                pred = pred[0] if isinstance(pred, (list, tuple)) else getattr(pred, "sample", pred)
                loss = F.mse_loss(pred, noisy_batch.target)

            if train:
                if self.scaler.is_enabled():
                    self.scaler.scale(loss / accum_steps).backward()
                else:
                    (loss / accum_steps).backward()

            chunk_bs = clean_chunk.size(0)
            total_loss += float(loss.detach().item()) * chunk_bs
            total_samples += chunk_bs

        if train:
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        denom = max(1, total_samples)
        return {"loss": total_loss / denom}

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return self._run_step(batch, epoch=epoch, train=True)

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return self._run_step(batch, epoch=epoch, train=False)
