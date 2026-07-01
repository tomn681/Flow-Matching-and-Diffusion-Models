"""
Shared runtime helper layer for diffusion-like sampler entrypoints.
"""

from __future__ import annotations

import torch

from models.adapters import build_text_encoder
from core.noise_contracts import effective_noise_family_for_config, noise_family_for_model_type
from pipelines import InferencePipeline
from pipelines.utils import build_scheduler, resolve_conditioning_mode


def stack_optional_tensor_list(samples: list[dict], key: str, device: torch.device) -> torch.Tensor | None:
    tensors = [s.get(key) for s in samples]
    if all(t is not None for t in tensors):
        return torch.stack(tensors, dim=0).to(device)
    return None


def build_conditioning_batch(
    *,
    conditioning_mode: str | None,
    samples: list[dict],
    targets: torch.Tensor,
    device: torch.device,
    text_embeddings: torch.Tensor | None = None,
) -> torch.Tensor | dict[str, torch.Tensor] | None:
    mode = str(conditioning_mode or "none").lower()
    if mode in {"none", "false", "off"}:
        return None
    if mode == "text":
        return text_embeddings
    if mode in {"concatenate", "attention", "latent_attention"}:
        if mode in {"attention", "latent_attention"} and text_embeddings is not None:
            return text_embeddings
        return stack_optional_tensor_list(samples, "image", device)
    if mode == "inpainting":
        mask = stack_optional_tensor_list(samples, "mask", device)
        if mask is None:
            return None
        return {"mask": mask, "original": targets.to(device)}
    if mode == "chain":
        concat_cond = stack_optional_tensor_list(samples, "concat_cond", device)
        attn_cond = stack_optional_tensor_list(samples, "attn_cond", device)
        if concat_cond is None:
            concat_cond = stack_optional_tensor_list(samples, "image", device)
        text_cond = text_embeddings
        if attn_cond is None and text_cond is None:
            attn_cond = stack_optional_tensor_list(samples, "image", device)
        if concat_cond is None and attn_cond is None and text_cond is None:
            return None
        return {"concatenate": concat_cond, "attention": attn_cond, "text": text_cond}
    return stack_optional_tensor_list(samples, "image", device)


class TextConditioningRuntime:
    def __init__(self, sampling_cfg: dict | None, device: torch.device) -> None:
        self.device = device
        self._enabled = bool(isinstance(sampling_cfg, dict) and sampling_cfg.get("text_encoder"))
        self._encoder = None
        self._encoder_cfg = sampling_cfg.get("text_encoder", {}) if isinstance(sampling_cfg, dict) else {}

    def build_batch(self, samples: list[dict]) -> torch.Tensor | None:
        if not self._enabled:
            return None
        texts: list[str] = []
        for sample in samples:
            raw = sample.get("text")
            if raw is None:
                raw = sample.get("prompt")
            if raw is None:
                return None
            texts.append(str(raw))
        if not texts:
            return None
        if self._encoder is None:
            kind = self._encoder_cfg.get("kind", "clip")
            model_name = self._encoder_cfg.get("model_name")
            self._encoder = build_text_encoder(kind=kind, model_name=model_name).to(self.device)
        with torch.no_grad():
            return self._encoder(texts).to(self.device)


def resolve_conditioning_save_tensor(sample: dict, conditioning_mode: str | None) -> torch.Tensor | None:
    mode = str(conditioning_mode or "none").lower()
    if mode in {"concatenate", "attention", "latent_attention", "none", "false", "off"}:
        value = sample.get("image")
        return value if torch.is_tensor(value) else None
    if mode == "text":
        return None
    if mode == "inpainting":
        value = sample.get("mask")
        return value if torch.is_tensor(value) else None
    if mode == "chain":
        value = sample.get("concat_cond")
        if torch.is_tensor(value):
            return value
        value = sample.get("attn_cond")
        if torch.is_tensor(value):
            return value
        value = sample.get("image")
        return value if torch.is_tensor(value) else None
    value = sample.get("image")
    return value if torch.is_tensor(value) else None


def build_inference_pipeline(
    *,
    model,
    training_cfg: dict,
    model_cfg: dict,
    device: torch.device,
):
    scheduler_cfg = dict(model_cfg.get("scheduler", {}))
    scheduler, num_inference = build_scheduler(
        scheduler_cfg,
        training_cfg,
        noise_family=effective_noise_family_for_config(
            str(model_cfg.get("model_type", "")),
            training_cfg=training_cfg,
            model_cfg=model_cfg,
        ),
    )
    conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))
    return InferencePipeline(
        unet=model,
        scheduler=scheduler,
        device=device,
        conditioning_mode=conditioning_mode,
        latent_norm=training_cfg.get("latent_norm"),
    ), int(num_inference)


def tensor_stats(name: str, tensor: torch.Tensor | None) -> dict:
    if tensor is None:
        return {"name": name, "present": False}
    t = torch.as_tensor(tensor).detach().float().cpu()
    return {
        "name": name,
        "present": True,
        "shape": list(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()) if t.numel() > 1 else 0.0,
    }
