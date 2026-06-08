from __future__ import annotations
from pathlib import Path
from typing import Any

import torch

import utils
from .events import TrainingEventBus
from .registry import TRAINER_REGISTRY


class TrainerBuilder:
    """Declarative builder for training pipeline construction."""

    def __init__(self) -> None:
        self._config: dict | None = None
        self._model: torch.nn.Module | None = None
        self._noise_process: Any = None
        self._losses: Any = None
        self._callbacks: list[Any] | None = None
        self._event_bus: TrainingEventBus | None = None
        self._ema_decay: float | None = None

    def with_config(self, path_or_dict) -> "TrainerBuilder":
        if isinstance(path_or_dict, (str, Path)):
            self._config = utils.load_json_config(path_or_dict)
        elif isinstance(path_or_dict, dict):
            self._config = dict(path_or_dict)
        else:
            raise TypeError("with_config expects a path or dict.")
        return self

    def with_model(self, model) -> "TrainerBuilder":
        self._model = model
        return self

    def with_noise(self, noise_process) -> "TrainerBuilder":
        self._noise_process = noise_process
        return self

    def with_losses(self, losses) -> "TrainerBuilder":
        self._losses = losses
        return self

    def with_callbacks(self, callbacks) -> "TrainerBuilder":
        self._callbacks = list(callbacks)
        return self

    def with_frozen_vae(self, path) -> "TrainerBuilder":
        if self._config is None:
            raise ValueError("with_frozen_vae requires with_config(...) to be called first.")
        model_cfg = dict(self._config.get("model", {}))
        model_cfg["vae_checkpoint"] = str(path)
        self._config["model"] = model_cfg
        return self

    def with_event_listener(self, event, listener) -> "TrainerBuilder":
        if self._event_bus is None:
            self._event_bus = TrainingEventBus()
        self._event_bus.on(str(event), listener)
        return self

    def with_ema(self, decay=0.9999) -> "TrainerBuilder":
        decay = float(decay)
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must be in (0, 1).")
        self._ema_decay = decay
        return self

    def build(self):
        if self._config is None:
            raise ValueError("TrainerBuilder requires with_config(...) before build().")
        cfg = dict(self._config)
        model_type = str(cfg.get("model", {}).get("model_type", "")).lower()
        if not model_type:
            raise ValueError("Config must include model.model_type.")

        trainer_cls = TRAINER_REGISTRY.get(model_type)
        init_kwargs = {"config": cfg, "callbacks": self._callbacks}
        if bool(getattr(trainer_cls, "supports_event_bus", False)):
            init_kwargs["event_bus"] = self._event_bus
        if self._model is not None:
            if not bool(getattr(trainer_cls, "supports_model_override", False)):
                raise ValueError("with_model is not supported by this trainer class.")
            init_kwargs["model_override"] = self._model
        if self._noise_process is not None:
            if not bool(getattr(trainer_cls, "supports_noise_override", False)):
                raise ValueError("with_noise is only supported for trainers exposing `noise_override` injection.")
            init_kwargs["noise_override"] = self._noise_process
        if self._losses is not None:
            if not bool(getattr(trainer_cls, "supports_losses_override", False)):
                raise ValueError("with_losses is only supported for trainers exposing `losses_override` injection.")
            init_kwargs["losses_override"] = self._losses

        trainer = trainer_cls(**init_kwargs)

        if self._ema_decay is not None and isinstance(trainer.raw_config, dict):
            training_cfg = dict(trainer.raw_config.get("training", {}))
            training_cfg["ema_decay"] = self._ema_decay
            trainer.raw_config["training"] = training_cfg
            trainer.training_cfg = training_cfg

        return trainer


__all__ = ["TrainerBuilder"]
