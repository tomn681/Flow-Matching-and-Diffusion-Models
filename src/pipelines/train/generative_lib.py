from __future__ import annotations

from pathlib import Path


def train_via_new_api(dataset, json_path: Path | str, val_dataset=None, resume: str | None = None) -> None:
    """Compatibility wrapper that runs generative training via the new trainer API."""
    from training import TRAINER_REGISTRY
    from utils import load_json_config

    cfg = load_json_config(json_path)
    model_type = str(cfg.get("model", {}).get("model_type", "diffusion")).lower()

    trainer_cls = TRAINER_REGISTRY.get(model_type)
    trainer = trainer_cls.from_config(cfg)
    trainer.fit(dataset, val_dataset=val_dataset, resume=resume)
