"""Backward-compatible wrapper around the registry-based flow-matching trainer."""

from __future__ import annotations

from pathlib import Path

from compat._deprecation import warn_deprecated
from training import TRAINER_REGISTRY
from utils import load_json_config


def train(dataset, json_path: Path | str, val_dataset=None, resume: str | None = None) -> None:
    warn_deprecated(
        api="pipelines.train.flow_matching_lib.train",
        replacement="training.TRAINER_REGISTRY['flow_matching'].from_config(...).fit(...)",
    )
    cfg = load_json_config(json_path)
    trainer = TRAINER_REGISTRY.get("flow_matching").from_config(cfg)
    trainer.fit(dataset, val_dataset=val_dataset, resume=resume)


def debug_visual_only(*args, **kwargs) -> None:
    warn_deprecated(
        api="pipelines.train.flow_matching_lib.debug_visual_only",
        replacement="FlowMatchingTrainer + visualization callback",
    )
    raise NotImplementedError("Use the registry-based trainer with a visualization callback.")
