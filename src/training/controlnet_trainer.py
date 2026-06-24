from __future__ import annotations

from pathlib import Path
import sys as _sys
from typing import Any

import torch
import torch.nn.functional as F

import utils
from core.types import unwrap_model_prediction
from models.controlnet import initialize_controlnet_from_unet, load_frozen_base_unet
from models.factory import ModelFactory
from noise import NOISE_REGISTRY
from scheduling.builder import build_scheduler
from scheduling.lr import build_lr_scheduler
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback, TensorBoardCallback
from .registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("controlnet")
class ControlNetTrainer(BaseTrainer):
    """Train a ControlNet adapter against a frozen base UNet using DDPM noise."""

    checkpoint_prefix = "controlnet"
    supports_model_override = True
    supports_noise_override = True

    def __init__(
        self,
        config: dict,
        callbacks: list[Any] | None = None,
        event_bus=None,
        model_override: torch.nn.Module | None = None,
        base_unet_override: torch.nn.Module | None = None,
        noise_override: Any = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks, event_bus=event_bus)
        self._model_override = model_override
        self._base_unet_override = base_unet_override
        self._noise_override = noise_override
        self.base_unet: torch.nn.Module | None = None
        self.noise_process = None
        self.noise_scheduler = None

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
    def from_config(cls, path_or_dict: str | Path | dict) -> "ControlNetTrainer":
        if isinstance(path_or_dict, (str, Path)):
            cfg = utils.load_json_config(path_or_dict)
        else:
            cfg = path_or_dict
        return cls(config=cfg)

    def _build_model(self) -> torch.nn.Module:
        base_ckpt = self.model_cfg.get("base_unet_checkpoint")
        if self._base_unet_override is not None:
            self.base_unet = self._base_unet_override.to(self.device)
            self.base_unet.requires_grad_(False)
            self.base_unet.eval()
        else:
            if not base_ckpt:
                raise ValueError("ControlNetTrainer requires model.base_unet_checkpoint in config.")
            self.base_unet = load_frozen_base_unet(base_ckpt, self.device)

        if self._model_override is not None:
            model = self._model_override.to(self.device)
        else:
            model = ModelFactory.build(self.raw_config).to(self.device)
            initialize_controlnet_from_unet(model, self.base_unet)
        return model

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.optimizer is None:
            raise RuntimeError("ControlNetTrainer._build_lr_scheduler called before optimizer initialization.")
        lr_spec = self.training_cfg.get("lr_scheduler")
        if lr_spec is None:
            return None
        cfg = self._lr_scheduler_config()
        cfg["lr_scheduler"] = lr_spec
        return build_lr_scheduler(self.optimizer, cfg)

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)
        scheduler_cfg = self.model_cfg.get("scheduler", {})
        self.noise_scheduler, _ = build_scheduler(scheduler_cfg, self.training_cfg)
        self.noise_process = (
            self._noise_override
            if self._noise_override is not None
            else NOISE_REGISTRY.build("ddpm", scheduler=self.noise_scheduler)
        )

    def _extract_context(self, batch: dict):
        context = batch.get("attn_cond")
        if torch.is_tensor(context):
            return context.to(self.device)
        return None

    def _run_step(self, batch: dict, *, train: bool) -> dict[str, float]:
        if self.model is None or self.optimizer is None or self.scaler is None or self.base_unet is None or self.noise_process is None:
            raise RuntimeError("ControlNetTrainer._run_step called before setup completed.")
        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"
        current_micro = self._initial_microbatch_size(batch)
        grad_snapshots = self._capture_optimizer_grads(self.optimizer) if train else []

        while True:
            try:
                chunks = self._split_batch(batch, current_micro)
                total_samples = sum(self._batch_size_from_batch(chunk) for chunk in chunks)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                self.base_unet.eval()

                total_loss = 0.0
                with torch.set_grad_enabled(train):
                    for chunk in chunks:
                        target = chunk["target"].to(self.device)
                        source = chunk.get("image")
                        if not torch.is_tensor(source):
                            source = target
                        else:
                            source = source.to(self.device)
                        context_ca = self._extract_context(chunk)
                        noisy_batch = self.noise_process(target, self.device)
                        chunk_bs = target.size(0)

                        with torch.autocast(device_type=self.device.type, enabled=use_amp):
                            residuals = self.model(
                                noisy_batch.noisy,
                                noisy_batch.timesteps,
                                source,
                                encoder_hidden_states=context_ca,
                            )
                            pred = self.base_unet(
                                noisy_batch.noisy,
                                noisy_batch.timesteps,
                                context_ca=context_ca,
                                controlnet_residuals={
                                    "down_residuals": list(residuals["down_residuals"]),
                                    "mid_residual": residuals["mid_residual"],
                                },
                            )
                            pred = unwrap_model_prediction(pred)
                            loss = F.mse_loss(pred, noisy_batch.target)
                        if train:
                            self._backward(loss * (chunk_bs / max(1, total_samples)))
                        total_loss += float(loss.detach().item()) * chunk_bs

                if train:
                    self._step_optimizers(self.optimizer)
                return {"loss": total_loss / max(1, total_samples)}
            except RuntimeError as err:
                if (not train) or (not self.allow_microbatching) or (not self._is_oom_error(err)):
                    raise
                if current_micro <= 1:
                    raise
                current_micro = max(1, current_micro // 2)
                self._prepare_retry_after_oom(grad_snapshots=grad_snapshots)

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        del epoch
        return self._run_step(batch, train=True)

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        del epoch
        return self._run_step(batch, train=False)


__all__ = ["ControlNetTrainer"]

_module = _sys.modules[__name__]
if __name__.startswith("genlib.training."):
    _sys.modules.setdefault(__name__.replace("genlib.training.", "training.", 1), _module)
elif __name__.startswith("src.training."):
    _sys.modules.setdefault(__name__.replace("src.training.", "training.", 1), _module)
    _sys.modules.setdefault(__name__.replace("src.training.", "genlib.training.", 1), _module)
elif __name__.startswith("training."):
    _sys.modules.setdefault(__name__.replace("training.", "src.training.", 1), _module)
    _sys.modules.setdefault(__name__.replace("training.", "genlib.training.", 1), _module)
del _module, _sys
