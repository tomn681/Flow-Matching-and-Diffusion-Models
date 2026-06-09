from __future__ import annotations

from typing import Any, Dict

from ..factory import ModelFactory

__all__ = ["DiffusionUNetFactory"]


class DiffusionUNetFactory:
    """
    Backward-compatible diffusion/flow UNet builder.

    This class remains for legacy imports while delegating construction to
    the canonical `ModelFactory.build(...)` API.
    """

    def build(self, model_cfg: Dict[str, Any], conditioning: str | None = None, channels: int | None = None):
        config = {"model": {"model_type": "diffusion", "unet": dict(model_cfg or {}), "conditioning": conditioning}}
        if channels is not None:
            config["training"] = {"channels": channels}
        return ModelFactory.build(config, conditioning=conditioning, channels=channels)
