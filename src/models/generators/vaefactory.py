"""
Compatibility wrapper that builds VAEs from JSON configs.

Canonical model construction lives in `models.factory.ModelFactory.build(...)`.
This module remains only for legacy imports that still expect a VAE-specific
JSON builder surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..factory import ModelFactory


class VAEFactory:
    """
    Compatibility builder that delegates JSON-backed VAE construction to
    `ModelFactory.build(...)`.

    Expected JSON shape:
    {
      "training": { ... },
      "model": { "model_type": "vae", ... }  # architecture params; latent_type selects KL vs VQ
    }
    """

    def __init__(self) -> None:
        pass

    def build_from_json(self, json_path: Path | str):
        cfg = self._load_config(json_path)
        model_cfg: Dict[str, Any] = cfg["model"]
        model_type = str(model_cfg.get("model_type", "vae")).lower()
        if model_type != "vae":
            raise ValueError(f"Expected model_type 'vae', got '{model_type}'.")
        return ModelFactory.build(cfg)

    @staticmethod
    def _load_config(path: Path | str) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r") as fh:
            cfg = json.load(fh)
        if "model" not in cfg:
            raise ValueError("Config must contain a 'model' section.")
        return cfg

def build_from_json(json_path: Path | str):
    """
    Convenience compatibility builder that returns a VAE model from JSON.
    """
    return VAEFactory().build_from_json(json_path)
