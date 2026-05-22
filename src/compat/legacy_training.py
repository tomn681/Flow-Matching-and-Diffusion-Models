from __future__ import annotations

from pathlib import Path

from training import TRAINER_REGISTRY
from utils import load_json_config
from ._deprecation import warn_deprecated


def _train_via_registry(dataset, json_path: Path | str, model_type: str, val_dataset=None, resume: str | None = None) -> None:
    cfg = load_json_config(json_path)
    trainer_cls = TRAINER_REGISTRY.get(model_type)
    trainer = trainer_cls.from_config(cfg)
    trainer.fit(dataset, val_dataset=val_dataset, resume=resume)


def train_diffusion(dataset, json_path: Path | str, val_dataset=None, resume: str | None = None) -> None:
    warn_deprecated(
        api="compat.legacy_training.train_diffusion",
        replacement="training.TRAINER_REGISTRY['diffusion'].from_config(...).fit(...)",
    )
    _train_via_registry(
        dataset=dataset,
        json_path=json_path,
        model_type="diffusion",
        val_dataset=val_dataset,
        resume=resume,
    )


def train_flow_matching(dataset, json_path: Path | str, val_dataset=None, resume: str | None = None) -> None:
    warn_deprecated(
        api="compat.legacy_training.train_flow_matching",
        replacement="training.TRAINER_REGISTRY['flow_matching'].from_config(...).fit(...)",
    )
    _train_via_registry(
        dataset=dataset,
        json_path=json_path,
        model_type="flow_matching",
        val_dataset=val_dataset,
        resume=resume,
    )


def train_vae(dataset, json_path: Path | str, val_dataset=None, resume: str | None = None) -> None:
    warn_deprecated(
        api="compat.legacy_training.train_vae",
        replacement="training.TRAINER_REGISTRY['vae'].from_config(...).fit(...)",
    )
    _train_via_registry(
        dataset=dataset,
        json_path=json_path,
        model_type="vae",
        val_dataset=val_dataset,
        resume=resume,
    )


__all__ = ["train_diffusion", "train_flow_matching", "train_vae"]
