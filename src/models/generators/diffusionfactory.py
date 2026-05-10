from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

from models.unet import EfficientUNetND, UNetDiffusersND

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
        unet_impl = str(cfg.get("unet_impl", "efficient_nd")).lower()
        if unet_impl in {"diffusers_nd", "diffusers_exact_nd", "exact_nd", "diffusers"}:
            return self._build_diffusers_nd(cfg, conditioning, channels)
        return self._build_efficient_nd(cfg, conditioning, channels)

    def _build_efficient_nd(self, cfg: Dict[str, Any], conditioning: str | None = None, channels: int | None = None):
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
            emb_activation_before_proj=bool(cfg.get("emb_activation_before_proj", False)),
            pool_factor=int(cfg.get("pool_factor", 1)),
        )

    def _build_diffusers_nd(self, cfg: Dict[str, Any], conditioning: str | None = None, channels: int | None = None):
        cond_mode = (conditioning or "").lower()
        spatial_dims = int(cfg.get("spatial_dims", 2))
        in_channels = int(cfg.get("in_channels", channels or 1))
        cond_channels = int(cfg.get("conditioning_channels", channels or in_channels))
        if cond_mode == "concatenate":
            in_channels = in_channels + cond_channels

        out_channels = int(cfg.get("out_channels", channels or 1))
        block_out_channels = _to_tuple(cfg.get("block_out_channels"), (224, 448, 672, 896))
        layers_per_block = int(cfg.get("layers_per_block", 2))
        down_block_types = cfg.get(
            "down_block_types",
            ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        )
        up_block_types = cfg.get(
            "up_block_types",
            ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

        return UNetDiffusersND(
            spatial_dims=spatial_dims,
            sample_size=cfg.get("sample_size"),
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=bool(cfg.get("center_input_sample", False)),
            time_embedding_type=str(cfg.get("time_embedding_type", "positional")),
            freq_shift=int(cfg.get("freq_shift", 0)),
            flip_sin_to_cos=bool(cfg.get("flip_sin_to_cos", True)),
            down_block_types=down_block_types,
            mid_block_type=cfg.get("mid_block_type", "UNetMidBlock2D"),
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=int(cfg.get("downsample_padding", 1)),
            dropout=float(cfg.get("dropout", 0.0)),
            attention_head_dim=int(cfg.get("attention_head_dim", 8)),
            norm_num_groups=int(cfg.get("norm_num_groups", 32)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            resnet_time_scale_shift=str(cfg.get("resnet_time_scale_shift", "default")),
            add_attention=bool(cfg.get("add_attention", True)),
        )
