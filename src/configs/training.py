from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
import logging

from .base import BaseConfig


_TRAINING_ALIASES = {
    "num_epochs": "epochs",
    "train_batch_size": "batch_size",
    "save_model_epochs": "save_every",
}


@dataclass
class MultiResolutionStageConfig(BaseConfig):
    start_epoch: int = 0
    resolution: int = 0


@dataclass
class TrainingConfig(BaseConfig):
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_workers: int = 4
    save_every: int = 1
    output_dir: str = "checkpoints"
    seed: int | None = None
    multi_resolution: list[MultiResolutionStageConfig] | None = None
    input_normalize: str = "centered"

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        if not isinstance(data, dict):
            raise TypeError(f"training must be a dict, got {type(data).__name__}")

        known_fields = {f.name for f in fields(cls)}
        normalized: dict = {}
        extra: dict = {}

        for key, value in data.items():
            if key in _TRAINING_ALIASES:
                canonical = _TRAINING_ALIASES[key]
                warnings.warn(
                    f"Config key '{key}' is deprecated. Use '{canonical}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                normalized[canonical] = value
            elif key in known_fields:
                normalized[key] = value
            else:
                extra[key] = value

        mr_raw = normalized.get("multi_resolution")
        if mr_raw is not None:
            if not isinstance(mr_raw, list):
                raise TypeError("training.multi_resolution must be a list of stages.")
            stages: list[MultiResolutionStageConfig] = []
            for item in mr_raw:
                if not isinstance(item, dict):
                    raise TypeError("Each multi_resolution stage must be a dict.")
                stages.append(
                    MultiResolutionStageConfig(
                        start_epoch=int(item.get("start_epoch", 0)),
                        resolution=int(item.get("resolution", 0)),
                    )
                )
            normalized["multi_resolution"] = stages

        normalized["extra"] = extra
        cfg = cls(**normalized)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if int(self.epochs) <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if int(self.batch_size) <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if int(self.num_workers) < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if int(self.save_every) <= 0:
            raise ValueError(f"save_every must be > 0, got {self.save_every}")
        if float(self.learning_rate) <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if float(self.weight_decay) < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if str(self.input_normalize).lower() not in {"centered", "symmetric", "positive", "zscore"}:
            raise ValueError(
                "input_normalize must be one of {'centered', 'symmetric', 'positive', 'zscore'}, "
                f"got {self.input_normalize!r}"
            )
        if self.multi_resolution is not None:
            if len(self.multi_resolution) == 0:
                raise ValueError("training.multi_resolution must contain at least one stage when provided.")
            ordered = sorted(self.multi_resolution, key=lambda s: int(s.start_epoch))
            if [int(s.start_epoch) for s in ordered] != [int(s.start_epoch) for s in self.multi_resolution]:
                raise ValueError("training.multi_resolution stages must be sorted by start_epoch ascending.")
            if int(self.multi_resolution[0].start_epoch) != 0:
                raise ValueError("training.multi_resolution first stage must start at epoch 0.")
            for stage in self.multi_resolution:
                if int(stage.start_epoch) < 0:
                    raise ValueError("training.multi_resolution.start_epoch must be >= 0.")
                res = int(stage.resolution)
                if res <= 0:
                    raise ValueError("training.multi_resolution.resolution must be > 0.")
                if res & (res - 1):
                    logging.warning(
                        "training.multi_resolution uses non-power-of-2 resolution=%d at epoch=%d; "
                        "this may mismatch attention/downsampling assumptions in some UNet configs.",
                        res,
                        int(stage.start_epoch),
                    )
