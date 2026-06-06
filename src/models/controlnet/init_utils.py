from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from models.factory import ModelFactory
from utils.sampling_utils import load_run_config, resolve_checkpoint


_INIT_PREFIXES = (
    # EfficientUNetND / framework-style naming
    "input_blocks.",
    "middle_block.",
    "time_embed.",
    # Diffusers-style naming
    "conv_in.",
    "time_embedding.",
    "time_proj.",
    "down_blocks.",
    "mid_block.",
)


def initialize_controlnet_from_unet(
    controlnet: nn.Module,
    base_unet: nn.Module,
) -> None:
    """Copy encoder and mid-block weights from a base UNet into a ControlNet."""
    base_sd = base_unet.state_dict()
    ctrl_sd = controlnet.state_dict()
    copied = 0
    skipped = 0
    for key, tensor in base_sd.items():
        if not any(key.startswith(prefix) for prefix in _INIT_PREFIXES):
            continue
        if key in ctrl_sd and ctrl_sd[key].shape == tensor.shape:
            ctrl_sd[key] = tensor.detach().clone()
            copied += 1
        else:
            logging.debug("initialize_controlnet_from_unet: skipping key %s", key)
            skipped += 1
    controlnet.load_state_dict(ctrl_sd, strict=False)
    logging.info(
        "ControlNet initialised from base UNet: %d keys copied, %d skipped.",
        copied,
        skipped,
    )


def load_frozen_base_unet(
    base_ckpt_dir: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load a pretrained UNet checkpoint and freeze it for ControlNet training."""
    ckpt_dir = Path(base_ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    model = ModelFactory.build(cfg).to(device)

    trainer_type = str(cfg.get("training", {}).get("trainer") or cfg.get("model", {}).get("model_type") or "diffusion")
    ckpt_path = resolve_checkpoint(ckpt_dir, trainer_type)
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    logging.info("Loaded base UNet from %s", ckpt_path)

    model.requires_grad_(False)
    model.eval()
    return model


__all__ = ["initialize_controlnet_from_unet", "load_frozen_base_unet"]
