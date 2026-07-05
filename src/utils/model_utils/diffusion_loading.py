"""
Helpers for diffusion/flow-like model construction and legacy checkpoint loading.
"""

from __future__ import annotations

import logging

import torch

import utils
from models.factory import ModelFactory
from models.adapters.weight_mappers import map_hf_unet_to_ours
from pipelines.utils import resolve_conditioning_mode


def resolve_diffusion_backbone_config(model_cfg: dict) -> tuple[str, dict]:
    backbone_type = ModelFactory._resolve_diffusion_backbone_type(model_cfg)
    key = "dit" if backbone_type == "dit" else "unet"
    return backbone_type, dict(model_cfg.get(key, {}))


def _load_legacy_unet_state(model: torch.nn.Module, state: dict[str, torch.Tensor], strict_shapes: bool = True) -> None:
    model_state = model.state_dict()
    try:
        converted = map_hf_unet_to_ours(state, target_state_dict=model_state)
    except Exception as exc:
        if strict_shapes:
            raise RuntimeError(
                "Validated HF->diffusers_unet remap failed. "
                "This checkpoint is not safely compatible with the local diffusers_unet backend. "
                "Use model.unet.unet_impl='hf_diffusers' for legacy HF UNet2DModel checkpoints."
            ) from exc
        logging.warning(
            "Validated HF->diffusers_unet remap failed under legacy_strict_shapes=False; "
            "falling back to best-effort partial remap. This path is not parity-safe."
        )
        converted = map_hf_unet_to_ours(state)

    strict = bool(strict_shapes)
    model.load_state_dict(converted, strict=strict)


def build_diffusion_model(
    cfg: dict,
    device: torch.device,
    ckpt_path=None,
    set_eval: bool = True,
    *,
    use_ema: bool = False,
):
    training_cfg = cfg["training"]
    root_model_cfg = cfg["model"]
    _, model_cfg = resolve_diffusion_backbone_config(root_model_cfg)
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or root_model_cfg.get("conditioning")
    )
    channels = int(training_cfg.get("channels", model_cfg.get("out_channels", 1)))
    model = ModelFactory.build(cfg, conditioning=conditioning_mode, channels=channels).to(device)
    if ckpt_path is not None:
        ckpt_path = str(ckpt_path)
        payload = None
        if ckpt_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as safe_load_file
            except Exception as exc:
                raise RuntimeError(
                    "Loading .safetensors checkpoints requires `safetensors` package."
                ) from exc
            state = safe_load_file(ckpt_path, device=str(device))
        else:
            payload = utils.safe_torch_load(ckpt_path, map_location=device)
            state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        load_legacy = bool(model_cfg.get("load_legacy", False))
        if load_legacy:
            _load_legacy_unet_state(model, state, strict_shapes=bool(model_cfg.get("legacy_strict_shapes", True)))
        else:
            try:
                model.load_state_dict(state)
            except RuntimeError:
                _load_legacy_unet_state(model, state, strict_shapes=bool(model_cfg.get("legacy_strict_shapes", True)))
        if use_ema:
            from training.ema import apply_ema_state_to_model

            payload_ema = payload.get("ema") if isinstance(payload, dict) else None
            if not apply_ema_state_to_model(model, payload_ema):
                raise ValueError("Requested EMA weights for diffusion runtime, but checkpoint does not contain EMA state.")
    if set_eval:
        model.eval()
    return model


def warn_attention_conditioning_shape(conditioning_batch: torch.Tensor | None, model_cfg: dict) -> bool:
    if conditioning_batch is None or conditioning_batch.dim() < 2:
        return False
    backbone_type, backbone_cfg = resolve_diffusion_backbone_config(model_cfg) if isinstance(model_cfg, dict) else ("unet", {})
    if backbone_type != "unet":
        return False
    expected = backbone_cfg.get("cross_attention_dim")
    if expected is None:
        return False
    expected = int(expected)
    actual = int(conditioning_batch.shape[1])
    if actual != expected:
        logging.warning(
            "Attention conditioning has %d channels, but model unet.cross_attention_dim is %d. "
            "This often means the evaluation split is pointing at pixel conditioning instead of the expected latent conditioning.",
            actual,
            expected,
        )
        return True
    return False
