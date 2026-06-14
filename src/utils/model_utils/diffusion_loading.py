"""
Helpers for diffusion/flow-like model construction and legacy checkpoint loading.
"""

from __future__ import annotations

import logging

import torch

import utils
from models.factory import ModelFactory
from pipelines.utils import resolve_conditioning_mode


def _remap_legacy_unet_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace(".query.", ".to_q.")
        new_key = new_key.replace(".key.", ".to_k.")
        new_key = new_key.replace(".value.", ".to_v.")
        new_key = new_key.replace(".proj_attn.", ".to_out.0.")
        new_key = new_key.replace(".conv1.weight", ".conv1.conv.weight")
        new_key = new_key.replace(".conv1.bias", ".conv1.conv.bias")
        new_key = new_key.replace(".conv2.weight", ".conv2.conv.weight")
        new_key = new_key.replace(".conv2.bias", ".conv2.conv.bias")
        new_key = new_key.replace(".time_emb_proj.weight", ".emb_layers.weight")
        new_key = new_key.replace(".time_emb_proj.bias", ".emb_layers.bias")
        new_key = new_key.replace(".conv_shortcut.weight", ".skip_connection.conv.weight")
        new_key = new_key.replace(".conv_shortcut.bias", ".skip_connection.conv.bias")
        new_key = new_key.replace(".downsamplers.0.conv.weight", ".downsamplers.0.op.conv.weight")
        new_key = new_key.replace(".downsamplers.0.conv.bias", ".downsamplers.0.op.conv.bias")
        new_key = new_key.replace(".upsamplers.0.conv.weight", ".upsamplers.0.conv.conv.weight")
        new_key = new_key.replace(".upsamplers.0.conv.bias", ".upsamplers.0.conv.conv.bias")
        remapped[new_key] = value
    return remapped


def _load_legacy_unet_state(model: torch.nn.Module, state: dict[str, torch.Tensor], strict_shapes: bool = True) -> None:
    state = _remap_legacy_unet_keys(state)
    model_state = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    shape_mismatch: list[str] = []
    missing: list[str] = []
    unexpected: list[str] = []

    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch.append(f"{key}: ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}")
            continue
        converted[key] = value

    for key in model_state.keys():
        if key not in converted:
            missing.append(key)

    if strict_shapes and shape_mismatch:
        msg = "Legacy load failed due to shape mismatches:\n" + "\n".join(shape_mismatch[:20])
        if len(shape_mismatch) > 20:
            msg += f"\n... and {len(shape_mismatch) - 20} more"
        raise RuntimeError(msg)

    model.load_state_dict(converted, strict=False)

    if strict_shapes and (missing or unexpected):
        details = []
        if missing:
            details.append(f"missing={len(missing)}")
        if unexpected:
            details.append(f"unexpected={len(unexpected)}")
        raise RuntimeError(
            "Legacy load key mismatch after conversion (" + ", ".join(details) + "). "
            "Architecture/config likely differs from the source checkpoint."
        )


def build_diffusion_model(
    cfg: dict,
    device: torch.device,
    ckpt_path=None,
    set_eval: bool = True,
    *,
    use_ema: bool = False,
):
    training_cfg = cfg["training"]
    model_cfg = cfg["model"].get("unet", {})
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or cfg["model"].get("conditioning")
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
    unet_cfg = model_cfg.get("unet", {}) if isinstance(model_cfg, dict) else {}
    expected = unet_cfg.get("cross_attention_dim")
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
