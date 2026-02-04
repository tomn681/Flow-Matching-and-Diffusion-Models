from __future__ import annotations

from typing import Any, Dict, Iterable

from diffusers import UNet2DModel

__all__ = ["DiffusionUNetFactory"]


def _to_tuple(value: Iterable[int] | int, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value,)
    return tuple(value)


class DiffusionUNetFactory:
    """
    Lightweight builder for Diffusers UNet2DModel instances. Accepts dictionaries
    mirroring the `UNet2DModel` constructor arguments, applies sensible defaults,
    and adjusts channel counts based on the conditioning strategy.
    """

    DEFAULT_DOWNS = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    DEFAULT_UPS = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    DEFAULT_BLOCK_CHANNELS = (128, 128, 256, 256, 512, 512)

    def build(self, model_cfg: Dict[str, Any], conditioning: str | None = None, channels: int | None = None) -> UNet2DModel:
        cfg = dict(model_cfg or {})
        sample_size = cfg.get("sample_size") or cfg.get("image_size") or cfg.get("resolution") or 256

        in_channels = cfg.get("in_channels", channels or 1)
        cond_channels = cfg.get("conditioning_channels", channels or in_channels)
        cond_mode = (conditioning or "").lower()
        if cond_mode == "concatenate":
            in_channels = in_channels + cond_channels

        out_channels = cfg.get("out_channels", channels or 1)
        layers_per_block = cfg.get("layers_per_block", 2)
        block_out_channels = _to_tuple(cfg.get("block_out_channels"), self.DEFAULT_BLOCK_CHANNELS)
        down_block_types = _to_tuple(cfg.get("down_block_types"), self.DEFAULT_DOWNS)
        up_block_types = _to_tuple(cfg.get("up_block_types"), self.DEFAULT_UPS)

        return UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
