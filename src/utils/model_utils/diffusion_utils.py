"""
Helpers for diffusion/flow-matching model construction and inference.
"""

from __future__ import annotations

import torch

from models.generators import DiffusionUNetFactory
from pipelines.utils import build_scheduler, resolve_conditioning_mode, sample_with_scheduler
from utils.utils import select_visual_indices


def _remap_legacy_unet_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Remap common legacy/diffusers UNet attention key names to this repo names.
    This keeps shape compatibility while allowing name differences.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        # Diffusers attention projection names -> this repo attention names
        new_key = new_key.replace(".query.", ".to_q.")
        new_key = new_key.replace(".key.", ".to_k.")
        new_key = new_key.replace(".value.", ".to_v.")
        new_key = new_key.replace(".proj_attn.", ".to_out.0.")
        # Diffusers UNet ResNet block names -> this repo ND wrapper names
        new_key = new_key.replace(".conv1.weight", ".conv1.conv.weight")
        new_key = new_key.replace(".conv1.bias", ".conv1.conv.bias")
        new_key = new_key.replace(".conv2.weight", ".conv2.conv.weight")
        new_key = new_key.replace(".conv2.bias", ".conv2.conv.bias")
        new_key = new_key.replace(".time_emb_proj.weight", ".emb_layers.weight")
        new_key = new_key.replace(".time_emb_proj.bias", ".emb_layers.bias")
        new_key = new_key.replace(".conv_shortcut.weight", ".skip_connection.conv.weight")
        new_key = new_key.replace(".conv_shortcut.bias", ".skip_connection.conv.bias")
        # Down/Up sampler conv wrappers
        new_key = new_key.replace(".downsamplers.0.conv.weight", ".downsamplers.0.op.conv.weight")
        new_key = new_key.replace(".downsamplers.0.conv.bias", ".downsamplers.0.op.conv.bias")
        new_key = new_key.replace(".upsamplers.0.conv.weight", ".upsamplers.0.conv.conv.weight")
        new_key = new_key.replace(".upsamplers.0.conv.bias", ".upsamplers.0.conv.conv.bias")
        remapped[new_key] = value
    return remapped


def _load_legacy_unet_state(model: torch.nn.Module, state: dict[str, torch.Tensor], strict_shapes: bool = True) -> None:
    """
    Load legacy state into model, allowing key remapping but enforcing tensor shape compatibility.
    """
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

    # Load only exact-matching tensors; allow non-critical missing/unexpected keys.
    model.load_state_dict(converted, strict=False)

    # Promote key-set mismatch as an actionable error for strict legacy mode.
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


def build_diffusion_model(cfg: dict, device: torch.device, ckpt_path=None, set_eval: bool = True):
    """
    build_diffusion_model Function

    Builds a diffusion/flow model and optionally loads a checkpoint state.

    Inputs:
        - cfg: (dict) Full config dict.
        - device: (torch.device) Target device.
        - ckpt_path: (Path | None) Optional checkpoint path.
        - set_eval: (Boolean) If True, set model to eval() after loading.

    Outputs:
        - model: (torch.nn.Module) Constructed model.
    """
    training_cfg = cfg["training"]
    model_cfg = cfg["model"].get("unet", {})
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or cfg["model"].get("conditioning")
    )
    channels = int(training_cfg.get("channels", model_cfg.get("out_channels", 1)))
    factory = DiffusionUNetFactory()
    model = factory.build(model_cfg, conditioning_mode, channels).to(device)
    if ckpt_path is not None:
        payload = torch.load(ckpt_path, map_location=device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        load_legacy = bool(model_cfg.get("load_legacy", False))
        if load_legacy:
            _load_legacy_unet_state(model, state, strict_shapes=bool(model_cfg.get("legacy_strict_shapes", True)))
        else:
            try:
                model.load_state_dict(state)
            except RuntimeError:
                # Fallback for external diffusers-style checkpoints with equivalent shapes but different key names.
                _load_legacy_unet_state(model, state, strict_shapes=bool(model_cfg.get("legacy_strict_shapes", True)))
    if set_eval:
        model.eval()
    return model


def encode_diffusion_batch(scheduler, targets: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """
    encode_diffusion_batch Function

    Applies forward noise to targets using the scheduler.

    Inputs:
        - scheduler: (object) Diffusers scheduler.
        - targets: (Tensor) Clean targets.
        - timesteps: (Tensor) Timesteps per batch element.

    Outputs:
        - noisy: (Tensor) Noisy targets.
    """
    noise = torch.randn_like(targets)
    return scheduler.add_noise(targets, noise, timesteps)


def decode_diffusion_batch(
    model,
    training_cfg: dict,
    model_cfg: dict,
    device: torch.device,
    batch_shape: tuple[int, ...],
    conditioning_batch: torch.Tensor | None = None,
    timing: dict | None = None,
) -> torch.Tensor:
    """
    decode_diffusion_batch Function

    Generates samples with the diffusion/flow model using its scheduler.

    Inputs:
        - model: (torch.nn.Module) Diffusion model.
        - training_cfg: (dict) Training config.
        - model_cfg: (dict) Model config.
        - device: (torch.device) Target device.
        - batch_shape: (Tuple) Output shape.
        - conditioning_batch: (Tensor | None) Optional conditioning batch.

    Outputs:
        - samples: (Tensor) Generated samples.
    """
    scheduler_cfg = model_cfg.get("scheduler", {})
    scheduler, num_inference = build_scheduler(scheduler_cfg, training_cfg)
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or model_cfg.get("conditioning")
    )
    latent_norm = training_cfg.get("latent_norm")
    return sample_with_scheduler(
        model,
        scheduler,
        num_inference,
        batch_shape,
        device,
        conditioning_mode=conditioning_mode,
        conditioning_batch=conditioning_batch,
        latent_norm=latent_norm,
        timing=timing,
    )


def warn_attention_conditioning_shape(conditioning_batch: torch.Tensor | None, model_cfg: dict) -> bool:
    """
    Warn when attention conditioning channels do not match the configured context width.
    """
    if conditioning_batch is None or conditioning_batch.dim() < 2:
        return False
    unet_cfg = model_cfg.get("unet", {}) if isinstance(model_cfg, dict) else {}
    expected = unet_cfg.get("cross_attention_dim")
    if expected is None:
        return False
    expected = int(expected)
    actual = int(conditioning_batch.shape[1])
    if actual != expected:
        import logging

        logging.warning(
            "Attention conditioning has %d channels, but model unet.cross_attention_dim is %d. "
            "This often means the evaluation split is pointing at pixel conditioning instead of the expected latent conditioning.",
            actual,
            expected,
        )
        return True
    return False


def prepare_diffusion_visual_batch(dataset, count: int, device: torch.device, seed: int | None = None):
    """
    prepare_diffusion_visual_batch Function

    Collects a fixed batch of targets and optional conditioning images.

    Inputs:
        - dataset: (Dataset) Dataset instance.
        - count: (Int) Number of samples to collect.
        - device: (torch.device) Target device.

    Outputs:
        - targets: (Tensor) Target batch.
        - conditioning: (Tensor | None) Conditioning batch if available.
    """
    indices = select_visual_indices(dataset, count, seed=seed)
    targets = []
    conditioning = []
    for idx in indices:
        sample = dataset[idx]
        targets.append(sample["target"])
        conditioning.append(sample.get("image"))
    target_batch = torch.stack(targets, dim=0).to(device)
    if conditioning and all(c is not None for c in conditioning):
        cond_batch = torch.stack(conditioning, dim=0).to(device)
    else:
        cond_batch = None
    return target_batch, cond_batch


def run_self_tests() -> None:
    """
    Lightweight tests for diffusion utility helpers.
    """
    class DummyScheduler:
        def add_noise(self, targets, noise, timesteps):
            return targets + noise * 0.0

    class DummyDataset:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"target": torch.zeros(1, 2, 2), "image": torch.zeros(1, 2, 2)}

    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch unavailable
        raise RuntimeError("torch is required for diffusion utility self-tests.") from exc

    scheduler = DummyScheduler()
    targets = torch.zeros(2, 1, 2, 2)
    timesteps = torch.zeros(2, dtype=torch.long)
    noisy = encode_diffusion_batch(scheduler, targets, timesteps)
    assert noisy.shape == targets.shape

    dataset = DummyDataset()
    target_batch, cond_batch = prepare_diffusion_visual_batch(dataset, 2, torch.device("cpu"))
    assert target_batch.shape[0] == 2
    assert cond_batch is not None
