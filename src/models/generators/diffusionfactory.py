from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

from models.unet import EfficientUNetND

__all__ = ["DiffusionUNetFactory"]


def _to_tuple(value: Iterable[int] | int | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def _infer_channel_mult(block_out_channels: Sequence[int], base_channels: int) -> tuple[int, ...]:
    if not block_out_channels:
        return ()
    base = base_channels or block_out_channels[0]
    return tuple(max(1, int(ch // base)) for ch in block_out_channels)


class DiffusionUNetFactory:
    """
    Builder for EfficientUNetND instances used by diffusion/flow-matching trainers.

    Accepts either native EfficientUNetND keys or diffusers-style keys (e.g.,
    block_out_channels/layers_per_block) and maps them to EfficientUNetND.
    """

    DEFAULT_BLOCK_CHANNELS = (128, 128, 256, 256, 512, 512)

    def build(self, model_cfg: Dict[str, Any], conditioning: str | None = None, channels: int | None = None):
        cfg = dict(model_cfg or {})
        spatial_dims = int(cfg.get("spatial_dims", 2))
        block_out_channels = _to_tuple(cfg.get("block_out_channels"), self.DEFAULT_BLOCK_CHANNELS)
        model_channels = int(cfg.get("model_channels", block_out_channels[0] if block_out_channels else 128))

        in_channels = cfg.get("in_channels", channels or 1)
        cond_channels = cfg.get("conditioning_channels", channels or in_channels)
        cond_mode = (conditioning or "").lower()
        if cond_mode == "concatenate":
            in_channels = in_channels + cond_channels

        out_channels = cfg.get("out_channels", channels or 1)
        num_res_blocks = int(cfg.get("num_res_blocks", cfg.get("layers_per_block", 2)))
        channel_mult = _to_tuple(cfg.get("channel_mult"), _infer_channel_mult(block_out_channels, model_channels))
        attention_resolutions = _to_tuple(cfg.get("attention_resolutions"), (1,))
        cross_attention_resolutions = cfg.get("cross_attention_resolutions")
        cross_attention_in_middle = bool(cfg.get("cross_attention_in_middle", False))
        if cross_attention_resolutions is None and cond_mode == "attention":
            cross_attention_resolutions = attention_resolutions
            if "cross_attention_in_middle" not in cfg:
                cross_attention_in_middle = True

        return EfficientUNetND(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            cross_attention_resolutions=cross_attention_resolutions,
            cross_attention_dim=int(cfg.get("cross_attention_dim", cond_channels)),
            cross_attention_in_middle=cross_attention_in_middle,
            dropout=float(cfg.get("dropout", 0.0)),
            channel_mult=channel_mult or (1, 2, 3, 4),
            conv_resample=bool(cfg.get("conv_resample", True)),
            dim_head=int(cfg.get("dim_head", 64)),
            num_heads=int(cfg.get("num_heads", 4)),
            use_linear_attn=bool(cfg.get("use_linear_attn", True)),
            use_scale_shift_norm=bool(cfg.get("use_scale_shift_norm", True)),
            pool_factor=int(cfg.get("pool_factor", 1)),
        )
