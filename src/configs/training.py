from __future__ import annotations

import warnings
from dataclasses import dataclass, fields

from .base import BaseConfig


_TRAINING_ALIASES = {
    "num_epochs": "epochs",
    "train_batch_size": "batch_size",
    "save_model_epochs": "save_every",
}


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
