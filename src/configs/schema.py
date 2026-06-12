from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .base import BaseConfig
from .migration import normalize_aliases
from .model import BaseModelConfig, build_model_config
from .training import TrainingConfig


def _reject_dead_model_keys(model_cfg: dict) -> None:
    unet_cfg = model_cfg.get("unet")
    if isinstance(unet_cfg, dict) and "use_self_attention" in unet_cfg:
        raise ValueError(
            "model.unet.use_self_attention is not supported. "
            "Attention placement is controlled by the actual UNet implementation/configured attention fields."
        )


@dataclass
class FrameworkConfig(BaseConfig):
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    dataset_class: str | None = None
    data_root: str | None = None
    preprocess_kwargs: dict = field(default_factory=dict)


def validate_config(config: dict, config_path: Path | None = None) -> FrameworkConfig:
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config).__name__}")

    normalized = normalize_aliases(config)
    _reject_dead_model_keys(normalized.get("model", {}) if isinstance(normalized.get("model", {}), dict) else {})

    training_cfg = TrainingConfig.from_dict(normalized.get("training", {}))
    model_cfg = build_model_config(normalized.get("model", {}))

    known_top_keys = {
        "__config_path__",
        "training",
        "model",
        "dataset_class",
        "data_root",
        "preprocess_kwargs",
    }
    extra = {k: v for k, v in normalized.items() if k not in known_top_keys}

    framework_cfg = FrameworkConfig(
        config_path=config_path,
        config_version=1,
        training=training_cfg,
        model=model_cfg,
        dataset_class=normalized.get("dataset_class"),
        data_root=normalized.get("data_root"),
        preprocess_kwargs=normalized.get("preprocess_kwargs", {}) or {},
        extra=extra,
    )
    return framework_cfg


def load_and_validate(config_path: str | Path) -> FrameworkConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return validate_config(raw, config_path=path)


load_config = load_and_validate
