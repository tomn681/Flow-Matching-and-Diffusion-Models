"""
Helpers for diffusion/flow-matching model construction and inference.
"""

from __future__ import annotations

import torch

from models.generators import DiffusionUNetFactory
from pipelines.utils import build_scheduler, resolve_conditioning_mode, sample_with_scheduler


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
        model.load_state_dict(state)
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
    return sample_with_scheduler(
        model,
        scheduler,
        num_inference,
        batch_shape,
        device,
        conditioning_mode=conditioning_mode,
        conditioning_batch=conditioning_batch,
    )


def prepare_diffusion_visual_batch(dataset, count: int, device: torch.device):
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
    targets = []
    conditioning = []
    for idx in range(min(len(dataset), count)):
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
