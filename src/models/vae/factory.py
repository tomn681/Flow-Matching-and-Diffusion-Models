"""
Model factory that builds VAEs from JSON configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

from .base import AutoencoderKL
from .vq import VQVAE
from nn.blocks.residual import ResBlockND


class VAEFactory:
    """
    Build VAE models from a JSON config.

    Expected JSON shape (existing style):
    {
      "training": { ... },
      "vae": { ... }  # architecture params; latent_type selects KL vs VQ
    }
    """

    def __init__(self) -> None:
        self._model_registry: Dict[str, Callable[..., Any]] = {
            "kl": AutoencoderKL,
            "vq": VQVAE,
        }

    def build_from_json(self, json_path: Path | str):
        cfg = self._load_config(json_path)
        vae_cfg: Dict[str, Any] = cfg["vae"]
        latent_type = vae_cfg.get("latent_type", "kl").lower()
        model_cls = self._model_registry.get(latent_type)
        if model_cls is None:
            raise ValueError(f"Unsupported latent_type '{latent_type}'. Expected one of {list(self._model_registry)}.")

        block_factory = self._make_block_factory(vae_cfg)

        # Map config keys to model ctor; extra keys are ignored by **vae_cfg
        init_kwargs = dict(vae_cfg)
        init_kwargs.setdefault("in_channels", vae_cfg.get("in_channels", 3))
        init_kwargs.setdefault("out_channels", vae_cfg.get("out_channels", vae_cfg.get("in_channels", 3)))
        init_kwargs.setdefault("resolution", vae_cfg.get("resolution", 256))
        init_kwargs["block_factory"] = block_factory

        return model_cls(**init_kwargs)

    @staticmethod
    def _load_config(path: Path | str) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r") as fh:
            cfg = json.load(fh)
        if "vae" not in cfg:
            raise ValueError("Config must contain a 'vae' section.")
        return cfg

    @staticmethod
    def _make_block_factory(vae_cfg: Dict[str, Any]):
        """
        Create a block factory with norm/activation preferences if provided.
        """
        norm_type = vae_cfg.get("norm_type", "gn")
        act = vae_cfg.get("act", "silu")

        def factory(**kwargs):
            return ResBlockND(norm_type=norm_type, act=act, **kwargs)

        return factory


def build_from_json(json_path: Path | str):
    """
    Convenience builder that returns a ready-to-train VAE model from JSON.
    """
    return VAEFactory().build_from_json(json_path)
