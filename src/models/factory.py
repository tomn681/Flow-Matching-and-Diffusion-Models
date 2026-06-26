from __future__ import annotations

from collections.abc import Callable
from typing import Any
import warnings

from nn.blocks.residual import ResBlockND

from . import unet as _unet  # noqa: F401 - ensure model classes self-register
from . import vae as _vae  # noqa: F401 - ensure model classes self-register
from . import dit as _dit  # noqa: F401 - ensure model classes self-register
from . import controlnet as _controlnet  # noqa: F401 - ensure model classes self-register
from .registry import MODEL_REGISTRY

ModelBuildStrategy = Callable[[dict, str | None, int | None], Any]


def _build_strategy_vae(model_cfg: dict, conditioning: str | None, channels: int | None) -> Any:
    del conditioning, channels
    return ModelFactory._build_vae(model_cfg)


def _build_strategy_unet(model_cfg: dict, conditioning: str | None, channels: int | None) -> Any:
    return ModelFactory._build_unet(model_cfg, conditioning=conditioning, channels=channels)


def _build_strategy_dit(model_cfg: dict, conditioning: str | None, channels: int | None) -> Any:
    del conditioning
    return ModelFactory._build_dit(model_cfg, channels=channels)


def _build_strategy_controlnet(model_cfg: dict, conditioning: str | None, channels: int | None) -> Any:
    return ModelFactory._build_controlnet(model_cfg, conditioning=conditioning, channels=channels)


MODEL_BUILD_STRATEGY: dict[str, ModelBuildStrategy] = {
    "vae": _build_strategy_vae,
    "unet": _build_strategy_unet,
    "video_unet": _build_strategy_unet,
    "diffusion": _build_strategy_unet,
    "flow_matching": _build_strategy_unet,
    "latent_diffusion": _build_strategy_unet,
    "latent_flow_matching": _build_strategy_unet,
    "latent_rectified_flow": _build_strategy_unet,
    "x0_denoising": _build_strategy_unet,
    "consistency": _build_strategy_unet,
    "edm": _build_strategy_unet,
    "rectified_flow": _build_strategy_unet,
    "reflow": _build_strategy_unet,
    "dit": _build_strategy_dit,
    "patch_transformer": _build_strategy_dit,
    "controlnet": _build_strategy_controlnet,
}


class ModelFactory:
    """Unified model factory backed by MODEL_REGISTRY."""

    @staticmethod
    def _normalize_attention_resolutions(
        values,
        *,
        channel_mult: tuple[int, ...],
        sample_size: int | None = None,
        label: str = "attention_resolutions",
    ) -> tuple[int, ...]:
        raw = tuple(int(v) for v in (values or ()))
        if not raw:
            return ()
        max_downsample = max(1, 2 ** max(0, len(channel_mult) - 1))
        normalized: list[int] = []
        interpreted_absolute = False
        for value in raw:
            if value <= 0:
                raise ValueError(f"{label} entries must be > 0, got {value}.")
            if value <= max_downsample:
                normalized.append(value)
                continue
            if sample_size is not None and sample_size > 0 and sample_size % value == 0:
                factor = sample_size // value
                if 1 <= factor <= max_downsample:
                    normalized.append(int(factor))
                    interpreted_absolute = True
                    continue
            warnings.warn(
                f"{label}={raw} uses values outside the supported downsample-factor range "
                f"[1, {max_downsample}] for this UNet depth. Value {value} will be ignored.",
                stacklevel=3,
            )
        if interpreted_absolute:
            warnings.warn(
                f"{label} is defined in downsample-factor units, not absolute spatial resolutions. "
                f"Converted {raw} -> {tuple(normalized)} using sample_size={sample_size}.",
                stacklevel=3,
            )
        deduped = tuple(dict.fromkeys(normalized))
        return deduped

    @staticmethod
    def build(
        config: dict[str, Any],
        *,
        conditioning: str | None = None,
        channels: int | None = None,
    ) -> Any:
        model_cfg = dict(config.get("model", {}))
        model_type = str(model_cfg.get("model_type", "vae")).lower()
        strategy = MODEL_BUILD_STRATEGY.get(model_type)
        if strategy is None:
            available = ", ".join(sorted(MODEL_BUILD_STRATEGY.keys()))
            raise ValueError(f"Unsupported model_type '{model_type}'. Available strategies: {{{available}}}.")
        return strategy(model_cfg, conditioning, channels)

    @staticmethod
    def _build_vae(model_cfg: dict[str, Any]) -> Any:
        vae_cfg: dict[str, Any] = dict(model_cfg)
        latent_type = str(vae_cfg.get("latent_type", "kl")).lower()
        if latent_type == "kl":
            key = "kl_vae"
        elif latent_type == "monai":
            key = "monai_vae"
        elif latent_type == "vq":
            key = "vq_vae"
        else:
            key = None
        if key is None:
            raise ValueError(f"Unsupported latent_type '{latent_type}'.")

        for name in ("emb_channels", "down_channels"):
            val = vae_cfg.get(name)
            if isinstance(val, str) and val.lower() == "none":
                vae_cfg[name] = None
            if name == "down_channels" and isinstance(val, list):
                vae_cfg[name] = tuple(val)

        norm_type = vae_cfg.get("norm_type", "gn")
        act = vae_cfg.get("act", "silu")

        def block_factory(**kwargs: Any) -> Any:
            return ResBlockND(norm_type=norm_type, act=act, **kwargs)

        init_kwargs = dict(vae_cfg)
        init_kwargs.pop("latent_type", None)
        init_kwargs.pop("model_type", None)
        init_kwargs.pop("ckpt_path", None)
        init_kwargs.pop("norm_type", None)
        init_kwargs.pop("act", None)
        init_kwargs.setdefault("in_channels", vae_cfg.get("in_channels", 3))
        init_kwargs.setdefault("out_channels", vae_cfg.get("out_channels", vae_cfg.get("in_channels", 3)))
        init_kwargs.setdefault("resolution", vae_cfg.get("resolution", 256))
        init_kwargs["block_factory"] = block_factory
        return MODEL_REGISTRY.build(key, **init_kwargs)

    @staticmethod
    def _build_unet(
        model_cfg: dict[str, Any],
        *,
        conditioning: str | None = None,
        channels: int | None = None,
    ) -> Any:
        unet_cfg = dict(model_cfg.get("unet", {}))
        unet_impl = str(unet_cfg.get("unet_impl", "efficient_nd")).lower()
        if unet_impl in {"video_nd", "video_unet"} or str(model_cfg.get("model_type", "")).lower() == "video_unet":
            key = "video_unet"
        elif unet_impl in {"condition_nd", "unet2dcondition_nd", "condition_unet"}:
            key = "condition_unet"
        elif unet_impl in {"hf_diffusers", "hf", "huggingface_diffusers"}:
            key = "hf_diffusers_unet"
        elif unet_impl in {"diffusers_nd", "diffusers_exact_nd", "exact_nd", "diffusers"}:
            key = "diffusers_unet"
        else:
            key = "efficient_unet"
        cond_mode = (conditioning or model_cfg.get("conditioning") or "").lower()
        if not cond_mode and str(model_cfg.get("model_type", "")).lower() == "latent_diffusion":
            cond_mode = "attention"

        if key == "efficient_unet":
            return ModelFactory._build_efficient_unet(unet_cfg, cond_mode=cond_mode, channels=channels)
        if key == "video_unet":
            return ModelFactory._build_video_unet(unet_cfg, cond_mode=cond_mode, channels=channels)

        return ModelFactory._build_diffusers_family_unet(
            key=key,
            unet_cfg=unet_cfg,
            cond_mode=cond_mode,
            channels=channels,
        )

    @staticmethod
    def _build_efficient_unet(
        unet_cfg: dict[str, Any],
        *,
        cond_mode: str,
        channels: int | None,
    ) -> Any:
        block_out = tuple(unet_cfg.get("block_out_channels", (128, 128, 256, 256, 512, 512)))
        model_channels = int(unet_cfg.get("model_channels", block_out[0] if block_out else 128))
        sample_size = unet_cfg.get("sample_size")
        in_channels = int(unet_cfg.get("in_channels", channels or 1))
        cond_channels = int(unet_cfg.get("conditioning_channels", channels or in_channels))
        if cond_mode == "concatenate":
            in_channels += cond_channels
        out_channels = int(unet_cfg.get("out_channels", channels or 1))
        num_res_blocks = int(unet_cfg.get("num_res_blocks", unet_cfg.get("layers_per_block", 2)))
        channel_mult = tuple(unet_cfg.get("channel_mult", tuple(max(1, int(ch // model_channels)) for ch in block_out)))
        attention_resolutions = ModelFactory._normalize_attention_resolutions(
            unet_cfg.get("attention_resolutions", (1,)),
            channel_mult=channel_mult or (1, 2, 3, 4),
            sample_size=int(sample_size) if sample_size is not None else None,
            label="model.unet.attention_resolutions",
        )
        cross_attention_resolutions = unet_cfg.get("cross_attention_resolutions")
        if cross_attention_resolutions is not None:
            cross_attention_resolutions = ModelFactory._normalize_attention_resolutions(
                cross_attention_resolutions,
                channel_mult=channel_mult or (1, 2, 3, 4),
                sample_size=int(sample_size) if sample_size is not None else None,
                label="model.unet.cross_attention_resolutions",
            )
        cross_attention_in_middle = bool(unet_cfg.get("cross_attention_in_middle", False))
        if cross_attention_resolutions is None and cond_mode == "attention":
            cross_attention_resolutions = attention_resolutions
            if "cross_attention_in_middle" not in unet_cfg:
                cross_attention_in_middle = True
        return MODEL_REGISTRY.build(
            "efficient_unet",
            spatial_dims=int(unet_cfg.get("spatial_dims", 2)),
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            cross_attention_resolutions=cross_attention_resolutions,
            cross_attention_dim=int(unet_cfg.get("cross_attention_dim", cond_channels)),
            cross_attention_in_middle=cross_attention_in_middle,
            dropout=float(unet_cfg.get("dropout", 0.0)),
            channel_mult=channel_mult or (1, 2, 3, 4),
            conv_resample=bool(unet_cfg.get("conv_resample", True)),
            dim_head=int(unet_cfg.get("dim_head", 64)),
            num_heads=int(unet_cfg.get("num_heads", 4)),
            use_linear_attn=bool(unet_cfg.get("use_linear_attn", False)),
            use_scale_shift_norm=bool(unet_cfg.get("use_scale_shift_norm", True)),
            emb_activation_before_proj=bool(unet_cfg.get("emb_activation_before_proj", False)),
            pool_factor=int(unet_cfg.get("pool_factor", 1)),
        )

    @staticmethod
    def _build_diffusers_family_unet(
        key: str,
        unet_cfg: dict[str, Any],
        *,
        cond_mode: str,
        channels: int | None,
    ) -> Any:
        in_channels = int(unet_cfg.get("in_channels", channels or 1))
        cond_channels = int(unet_cfg.get("conditioning_channels", channels or in_channels))
        if cond_mode == "concatenate" and not bool(unet_cfg.get("in_channels_already_conditioned", False)):
            in_channels += cond_channels
        out_channels = int(unet_cfg.get("out_channels", channels or 1))
        block_out_channels = tuple(unet_cfg.get("block_out_channels", (224, 448, 672, 896)))
        layers_per_block = int(unet_cfg.get("layers_per_block", 2))
        if cond_mode == "attention":
            default_down = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
            default_up = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
            default_mid = "UNetMidBlock2DCrossAttn"
        else:
            default_down = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
            default_up = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
            default_mid = "UNetMidBlock2D"
        return MODEL_REGISTRY.build(
            key,
            spatial_dims=int(unet_cfg.get("spatial_dims", 2)),
            sample_size=unet_cfg.get("sample_size"),
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=bool(unet_cfg.get("center_input_sample", False)),
            time_embedding_type=str(unet_cfg.get("time_embedding_type", "positional")),
            freq_shift=int(unet_cfg.get("freq_shift", 0)),
            flip_sin_to_cos=bool(unet_cfg.get("flip_sin_to_cos", True)),
            down_block_types=unet_cfg.get("down_block_types", default_down),
            mid_block_type=unet_cfg.get("mid_block_type", default_mid),
            up_block_types=unet_cfg.get("up_block_types", default_up),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=int(unet_cfg.get("downsample_padding", 1)),
            dropout=float(unet_cfg.get("dropout", 0.0)),
            attention_head_dim=int(unet_cfg.get("attention_head_dim", 8)),
            norm_num_groups=int(unet_cfg.get("norm_num_groups", 32)),
            norm_eps=float(unet_cfg.get("norm_eps", 1e-5)),
            resnet_time_scale_shift=str(unet_cfg.get("resnet_time_scale_shift", "default")),
            add_attention=bool(unet_cfg.get("add_attention", True)),
            cross_attention_dim=int(unet_cfg.get("cross_attention_dim", cond_channels)) if cond_mode == "attention" else None,
            class_embed_type=unet_cfg.get("class_embed_type"),
            num_class_embeds=unet_cfg.get("num_class_embeds"),
            time_cond_proj_dim=unet_cfg.get("time_cond_proj_dim"),
            addition_embed_type=unet_cfg.get("addition_embed_type"),
            addition_time_embed_dim=unet_cfg.get("addition_time_embed_dim"),
            mid_block_only_cross_attention=bool(unet_cfg.get("mid_block_only_cross_attention", False)),
            transformer_layers_per_block=int(unet_cfg.get("transformer_layers_per_block", 1)),
        )

    @staticmethod
    def _build_video_unet(
        unet_cfg: dict[str, Any],
        *,
        cond_mode: str,
        channels: int | None,
    ) -> Any:
        block_out = tuple(unet_cfg.get("block_out_channels", (128, 128, 256, 256, 512, 512)))
        model_channels = int(unet_cfg.get("model_channels", block_out[0] if block_out else 128))
        sample_size = unet_cfg.get("sample_size")
        in_channels = int(unet_cfg.get("in_channels", channels or 1))
        cond_channels = int(unet_cfg.get("conditioning_channels", channels or in_channels))
        if cond_mode == "concatenate":
            in_channels += cond_channels
        out_channels = int(unet_cfg.get("out_channels", channels or 1))
        num_res_blocks = int(unet_cfg.get("num_res_blocks", unet_cfg.get("layers_per_block", 2)))
        channel_mult = tuple(unet_cfg.get("channel_mult", tuple(max(1, int(ch // model_channels)) for ch in block_out)))
        attention_resolutions = ModelFactory._normalize_attention_resolutions(
            unet_cfg.get("attention_resolutions", (1,)),
            channel_mult=channel_mult or (1, 2, 3, 4),
            sample_size=int(sample_size) if sample_size is not None else None,
            label="model.unet.attention_resolutions",
        )
        cross_attention_resolutions = unet_cfg.get("cross_attention_resolutions")
        if cross_attention_resolutions is not None:
            cross_attention_resolutions = ModelFactory._normalize_attention_resolutions(
                cross_attention_resolutions,
                channel_mult=channel_mult or (1, 2, 3, 4),
                sample_size=int(sample_size) if sample_size is not None else None,
                label="model.unet.cross_attention_resolutions",
            )
        cross_attention_in_middle = bool(unet_cfg.get("cross_attention_in_middle", False))
        if cross_attention_resolutions is None and cond_mode == "attention":
            cross_attention_resolutions = attention_resolutions
            if "cross_attention_in_middle" not in unet_cfg:
                cross_attention_in_middle = True
        return MODEL_REGISTRY.build(
            "video_unet",
            spatial_dims=int(unet_cfg.get("spatial_dims", 3)),
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            cross_attention_resolutions=cross_attention_resolutions,
            cross_attention_dim=int(unet_cfg.get("cross_attention_dim", cond_channels)),
            cross_attention_in_middle=cross_attention_in_middle,
            dropout=float(unet_cfg.get("dropout", 0.0)),
            channel_mult=channel_mult or (1, 2, 3, 4),
            conv_resample=bool(unet_cfg.get("conv_resample", True)),
            dim_head=int(unet_cfg.get("dim_head", 64)),
            num_heads=int(unet_cfg.get("num_heads", 4)),
            use_linear_attn=bool(unet_cfg.get("use_linear_attn", False)),
            use_scale_shift_norm=bool(unet_cfg.get("use_scale_shift_norm", True)),
            emb_activation_before_proj=bool(unet_cfg.get("emb_activation_before_proj", False)),
            pool_factor=int(unet_cfg.get("pool_factor", 1)),
            temporal_num_heads=unet_cfg.get("temporal_num_heads"),
            temporal_dropout=float(unet_cfg.get("temporal_dropout", 0.0)),
            temporal_after_resblocks=bool(unet_cfg.get("temporal_after_resblocks", True)),
            temporal_after_attn=bool(unet_cfg.get("temporal_after_attn", True)),
        )

    @staticmethod
    def _build_dit(model_cfg: dict[str, Any], *, channels: int | None) -> Any:
        dit_cfg = dict(model_cfg.get("dit", {}))
        preset = dit_cfg.pop("preset", None)
        if preset is not None:
            preset_key = str(preset).strip().lower()
            presets = {
                "s/2": {"patch_size": 2, "hidden_size": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0},
                "b/2": {"patch_size": 2, "hidden_size": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0},
                "l/2": {"patch_size": 2, "hidden_size": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0},
                "xl/2": {"patch_size": 2, "hidden_size": 1152, "depth": 28, "num_heads": 16, "mlp_ratio": 4.0},
                "s": {"patch_size": 2, "hidden_size": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0},
                "b": {"patch_size": 2, "hidden_size": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0},
                "l": {"patch_size": 2, "hidden_size": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0},
                "xl": {"patch_size": 2, "hidden_size": 1152, "depth": 28, "num_heads": 16, "mlp_ratio": 4.0},
                "dit-s/2": {"patch_size": 2, "hidden_size": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0},
                "dit-b/2": {"patch_size": 2, "hidden_size": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0},
                "dit-l/2": {"patch_size": 2, "hidden_size": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0},
                "dit-xl/2": {"patch_size": 2, "hidden_size": 1152, "depth": 28, "num_heads": 16, "mlp_ratio": 4.0},
            }
            if preset_key not in presets:
                available = ", ".join(sorted(presets.keys()))
                raise ValueError(f"Unknown DiT preset '{preset}'. Available: {available}")
            expanded = dict(presets[preset_key])
            expanded.update(dit_cfg)
            dit_cfg = expanded
        if "in_channels" not in dit_cfg:
            dit_cfg["in_channels"] = int(model_cfg.get("in_channels", channels or 4))
        if "out_channels" not in dit_cfg and "out_channels" in model_cfg:
            dit_cfg["out_channels"] = int(model_cfg["out_channels"])
        if "spatial_dims" not in dit_cfg:
            dit_cfg["spatial_dims"] = int(model_cfg.get("spatial_dims", 2))
        dit_cfg.setdefault("use_adaLN", True)
        dit_cfg.setdefault("zero_init_final_layer", True)
        return MODEL_REGISTRY.build("dit", **dit_cfg)

    @staticmethod
    def _build_controlnet(
        model_cfg: dict[str, Any],
        *,
        conditioning: str | None = None,
        channels: int | None = None,
    ) -> Any:
        del conditioning
        controlnet_cfg = dict(model_cfg.get("controlnet", {}))
        if not controlnet_cfg:
            controlnet_cfg = dict(model_cfg.get("unet", {}))
        if not controlnet_cfg:
            controlnet_cfg = dict(model_cfg)
        controlnet_cfg.pop("model_type", None)
        controlnet_cfg.pop("base_unet_checkpoint", None)
        if "in_channels" not in controlnet_cfg:
            controlnet_cfg["in_channels"] = int(model_cfg.get("in_channels", channels or 1))
        if "conditioning_channels" not in controlnet_cfg:
            controlnet_cfg["conditioning_channels"] = int(
                model_cfg.get("conditioning_channels", channels or controlnet_cfg["in_channels"])
            )
        if "spatial_dims" not in controlnet_cfg:
            controlnet_cfg["spatial_dims"] = int(model_cfg.get("spatial_dims", controlnet_cfg.get("spatial_dims", 2)))
        return MODEL_REGISTRY.build("controlnet", **controlnet_cfg)
