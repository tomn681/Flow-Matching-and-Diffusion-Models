from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from core.types import unwrap_model_prediction
from scheduling.lr import build_lr_scheduler
from utils.model_utils.diffusion_utils import build_diffusion_model
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback, TensorBoardCallback
from .registry import TRAINER_REGISTRY
import utils


@TRAINER_REGISTRY.register("unet")
class UNetTrainer(BaseTrainer):
    """Supervised UNet trainer for direct input->target objectives.

    Note: current UNet build path uses diffusion-style UNets with timestep input.
    We pass `t=0` as a deterministic sentinel for supervised forward passes.
    """

    checkpoint_prefix = "unet"
    supports_model_override = True

    def __init__(
        self,
        config: dict,
        callbacks: list[Any] | None = None,
        event_bus=None,
        model_override: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks, event_bus=event_bus)
        self._model_override = model_override
        self.loss_name = str(self.training_cfg.get("loss", "mse")).lower()

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
    def from_config(cls, path_or_dict: str | Path | dict) -> "UNetTrainer":
        if isinstance(path_or_dict, (str, Path)):
            cfg = utils.load_json_config(path_or_dict)
        else:
            cfg = path_or_dict
        return cls(config=cfg)

    def _build_model(self) -> torch.nn.Module:
        if self._model_override is not None:
            return self._model_override
        return build_diffusion_model(self.raw_config, self.device, ckpt_path=None, set_eval=False)

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.optimizer is None:
            raise RuntimeError("UNetTrainer._build_lr_scheduler called before optimizer initialization.")
        return build_lr_scheduler(self.optimizer, self._lr_scheduler_config())

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "mse":
            return F.mse_loss(pred, target)
        if self.loss_name in {"l1", "mae"}:
            return F.l1_loss(pred, target)
        raise ValueError(f"Unsupported UNetTrainer loss '{self.loss_name}'. Use one of: mse, l1.")

    def _run_step(self, batch: dict, *, train: bool) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("UNetTrainer._run_step called before model initialization.")
        if self.optimizer is None:
            raise RuntimeError("UNetTrainer._run_step called before optimizer initialization.")
        if self.scaler is None:
            raise RuntimeError("UNetTrainer._run_step called before AMP scaler initialization.")

        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"
        current_micro = self._initial_microbatch_size(batch)
        grad_snapshots = self._capture_optimizer_grads(self.optimizer) if train else []

        while True:
            try:
                chunks = self._split_batch(batch, current_micro)
                total_samples = sum(self._batch_size_from_batch(chunk) for chunk in chunks)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss = 0.0
                for chunk in chunks:
                    inputs = chunk.get("image", chunk["target"]).to(self.device)
                    targets = chunk["target"].to(self.device)
                    timesteps = torch.zeros(inputs.size(0), device=self.device, dtype=torch.long)
                    chunk_bs = inputs.size(0)

                    with torch.autocast(device_type=self.device.type, enabled=use_amp):
                        pred = self.model(inputs, timesteps)
                        pred = unwrap_model_prediction(pred)
                        loss = self._compute_loss(pred, targets)

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
        _ = epoch
        return self._run_step(batch, train=True)

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        _ = epoch
        with torch.no_grad():
            return self._run_step(batch, train=False)
