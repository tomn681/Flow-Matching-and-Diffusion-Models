"""Inference mediator coordinating UNet, scheduler, text encoder, and ControlNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys as _sys
from typing import Any

import torch
import torch.nn as nn

import utils
from models.autoencoder.base import BaseAutoencoder
from models.autoencoder.utils import decode_from_latent
from models.factory import ModelFactory
from core.noise_contracts import noise_family_for_model_type
from scheduling import build_scheduler, build_text_conditioning_adapter, resolve_conditioning_mode, sample_with_scheduler
from training.ema import apply_ema_state_to_model


@dataclass(frozen=True)
class InferenceInputs:
    """Inputs consumed by :class:`InferencePipeline.generate`."""

    sample_shape: tuple[int, ...]
    num_inference_steps: int
    conditioning_batch: torch.Tensor | dict[str, torch.Tensor] | None = None
    unconditional_conditioning_batch: torch.Tensor | dict[str, torch.Tensor] | None = None
    prompts: list[str] | None = None
    controlnet_cond: torch.Tensor | None = None
    init_image: torch.Tensor | None = None
    strength: float = 1.0
    guidance_scale: float = 1.0
    cfg_rescale: float = 0.0


@dataclass(frozen=True)
class TextToImageInputs:
    """Inputs consumed by :class:`TextToImagePipeline.generate`."""

    prompts: list[str]
    height: int
    width: int
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    cfg_rescale: float = 0.0
    seed: int | None = None


class _ControlNetGuidedUNet(nn.Module):
    def __init__(
        self,
        *,
        unet: nn.Module,
        controlnet: nn.Module,
        controlnet_cond: torch.Tensor,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.controlnet_cond = controlnet_cond

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context_ca: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residuals = self.controlnet(
            x,
            t,
            self.controlnet_cond,
            encoder_hidden_states=context_ca,
        )
        return self.unet(
            x,
            t,
            context_ca=context_ca,
            controlnet_residuals=residuals,
        )


class InferencePipeline:
    """Mediator for diffusion-like generation.

    Coordinates:
    - scheduler sampling loop
    - optional text encoding to cross-attention context
    - optional ControlNet residual generation and UNet injection
    - optional VAE latent-to-image decode
    """

    def __init__(
        self,
        *,
        unet: nn.Module,
        scheduler: Any,
        device: torch.device,
        vae: BaseAutoencoder | None = None,
        text_encoder: nn.Module | None = None,
        controlnet: nn.Module | None = None,
        conditioning_mode: str | None = None,
        latent_norm: str | None = None,
    ) -> None:
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.vae = vae
        self.text_encoder = text_encoder
        self.controlnet = controlnet
        self.conditioning_mode = resolve_conditioning_mode(conditioning_mode)
        self.latent_norm = latent_norm

    def encode_prompts(self, prompts: list[str] | None) -> torch.Tensor | None:
        if prompts is None:
            return None
        if self.text_encoder is None:
            raise ValueError("prompts were provided but text_encoder is not configured.")
        with torch.no_grad():
            encoded = self.text_encoder(prompts)
        return encoded.to(self.device)

    def generate(self, inputs: InferenceInputs) -> torch.Tensor:
        model = self.unet

        text_context = self.encode_prompts(inputs.prompts)
        conditioning_batch = inputs.conditioning_batch
        if text_context is not None and self.conditioning_mode in {"attention", "latent_attention", "text"}:
            conditioning_batch = text_context
        elif text_context is not None and self.conditioning_mode == "chain":
            if conditioning_batch is None:
                conditioning_batch = {"text": text_context}
            elif isinstance(conditioning_batch, dict):
                conditioning_batch = dict(conditioning_batch)
                conditioning_batch.setdefault("text", text_context)
            else:
                conditioning_batch = {"concatenate": conditioning_batch, "text": text_context}

        if inputs.controlnet_cond is not None:
            if self.controlnet is None:
                raise ValueError("controlnet_cond was provided but controlnet is not configured.")
            controlnet_cond = inputs.controlnet_cond.to(self.device)
            model = _ControlNetGuidedUNet(
                unet=self.unet,
                controlnet=self.controlnet,
                controlnet_cond=controlnet_cond,
            )

        return sample_with_scheduler(
            model=model,
            scheduler=self.scheduler,
            num_inference_steps=int(inputs.num_inference_steps),
            sample_shape=tuple(inputs.sample_shape),
            device=self.device,
            conditioning_mode=self.conditioning_mode,
            conditioning_batch=conditioning_batch,
            unconditional_conditioning_batch=inputs.unconditional_conditioning_batch,
            latent_norm=self.latent_norm,
            init_image=inputs.init_image,
            strength=float(inputs.strength),
            guidance_scale=float(inputs.guidance_scale),
            cfg_rescale=float(inputs.cfg_rescale),
        )

    def generate_images(self, inputs: InferenceInputs, *, recon_type: str = "l1") -> torch.Tensor:
        latents = self.generate(inputs)
        if self.vae is None:
            return latents
        return decode_from_latent(self.vae, latents, recon_type=recon_type)


class TextToImagePipeline:
    """Checkpoint-driven text-to-image facade for latent diffusion-style runs."""

    def __init__(
        self,
        *,
        model: nn.Module,
        vae: BaseAutoencoder,
        text_adapter: Any,
        scheduler: Any,
        device: torch.device,
        latent_channels: int,
        compression_factor: int,
        conditioning_mode: str = "text",
        latent_norm: str | None = None,
    ) -> None:
        mode = resolve_conditioning_mode(conditioning_mode) or "text"
        if mode == "text":
            mode = "attention"
        self.model = model
        self.vae = vae
        self.text_adapter = text_adapter
        self.scheduler = scheduler
        self.device = device
        self.latent_channels = int(latent_channels)
        self.compression_factor = int(compression_factor)
        self.conditioning_mode = mode
        self.latent_norm = latent_norm

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        use_ema: bool = False,
    ) -> "TextToImagePipeline":
        from utils.model_utils.diffusion_utils import build_diffusion_model
        from utils.sampling_utils import load_run_config, resolve_checkpoint

        ckpt_dir = Path(ckpt_dir)
        cfg = load_run_config(ckpt_dir)
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_device = utils.resolve_device(device, default_device)

        model_type = str(cfg.get("model", {}).get("model_type", "latent_diffusion"))
        model_ckpt = resolve_checkpoint(ckpt_dir, model_type)
        model = build_diffusion_model(cfg, resolved_device, ckpt_path=model_ckpt, use_ema=use_ema)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        model_cfg = dict(cfg.get("model", {}))
        vae_cfg = dict(model_cfg.get("vae", {}))
        if not vae_cfg:
            raise ValueError("Text-to-image pipeline requires config.model.vae.")
        vae_cfg["model_type"] = "vae"
        vae = ModelFactory.build({"model": vae_cfg}).to(resolved_device)

        vae_ckpt_path = model_cfg.get("vae_checkpoint")
        if not vae_ckpt_path:
            raise ValueError("Text-to-image pipeline requires config.model.vae_checkpoint.")
        vae_payload = utils.safe_torch_load(vae_ckpt_path, map_location=resolved_device)
        vae_state = vae_payload["model"] if isinstance(vae_payload, dict) and "model" in vae_payload else vae_payload
        vae.load_state_dict(vae_state)
        if use_ema and not apply_ema_state_to_model(vae, vae_payload.get("ema") if isinstance(vae_payload, dict) else None):
            raise ValueError("Requested EMA weights for VAE runtime, but checkpoint does not contain EMA state.")
        vae.eval()
        for param in vae.parameters():
            param.requires_grad_(False)

        dummy_res = int(cfg.get("training", {}).get("img_size") or vae_cfg.get("resolution", 256))
        dummy_channels = int(vae_cfg.get("in_channels", 1))
        with torch.no_grad():
            dummy_in = torch.zeros(1, dummy_channels, dummy_res, dummy_res, device=resolved_device)
            dummy_latent = vae.encode(vae.image_to_model_range(dummy_in), normalize=False)
            if not isinstance(dummy_latent, torch.Tensor):
                dummy_latent = dummy_latent.mode()
        compression_factor = dummy_res // int(dummy_latent.shape[-1])
        latent_channels = int(dummy_latent.shape[1])

        text_cfg = cfg.get("training", {}).get("text_encoder", {}) if isinstance(cfg.get("training", {}).get("text_encoder"), dict) else {}
        text_adapter = build_text_conditioning_adapter(
            kind=str(text_cfg.get("kind", "clip")),
            model_name=text_cfg.get("model_name"),
            device=resolved_device,
        )
        scheduler, _num_inference = build_scheduler(
            cfg.get("model", {}).get("scheduler", {}),
            cfg.get("training", {}),
            noise_family=noise_family_for_model_type(str(cfg.get("model", {}).get("model_type", ""))),
        )

        return cls(
            model=model,
            vae=vae,
            text_adapter=text_adapter,
            scheduler=scheduler,
            device=resolved_device,
            latent_channels=latent_channels,
            compression_factor=compression_factor,
            conditioning_mode=str(cfg.get("training", {}).get("conditioning") or cfg.get("model", {}).get("conditioning") or "text"),
            latent_norm=cfg.get("training", {}).get("latent_norm"),
        )

    def generate(self, inputs: TextToImageInputs, *, recon_type: str = "l1") -> torch.Tensor:
        if not inputs.prompts:
            raise ValueError("Text-to-image generation requires at least one prompt.")
        comp = int(self.compression_factor)
        if comp <= 0:
            raise ValueError("compression_factor must be > 0.")
        if inputs.height % comp != 0 or inputs.width % comp != 0:
            raise ValueError(
                f"height={inputs.height} and width={inputs.width} must both be divisible by the VAE compression factor {comp}."
            )
        latent_height = inputs.height // comp
        latent_width = inputs.width // comp
        batch_size = len(inputs.prompts)
        sample_shape = (batch_size, self.latent_channels, latent_height, latent_width)

        if inputs.seed is not None:
            torch.manual_seed(int(inputs.seed))

        dummy = torch.zeros(batch_size, 1, device=self.device)
        _, conditioning_batch = self.text_adapter(dummy, inputs.prompts, self.latent_norm)
        unconditional_batch = None
        if float(inputs.guidance_scale) > 1.0:
            _, unconditional_batch = self.text_adapter(dummy, [""] * batch_size, self.latent_norm)

        latents = sample_with_scheduler(
            model=self.model,
            scheduler=self.scheduler,
            num_inference_steps=int(inputs.num_inference_steps),
            sample_shape=sample_shape,
            device=self.device,
            conditioning_mode=self.conditioning_mode,
            conditioning_batch=conditioning_batch,
            unconditional_conditioning_batch=unconditional_batch,
            latent_norm=self.latent_norm,
            guidance_scale=float(inputs.guidance_scale),
            cfg_rescale=float(inputs.cfg_rescale),
        )
        return decode_from_latent(self.vae, latents, recon_type=recon_type)


__all__ = ["InferenceInputs", "InferencePipeline", "TextToImageInputs", "TextToImagePipeline"]

_module = _sys.modules[__name__]
if __name__.startswith("genlib.pipelines."):
    _sys.modules.setdefault(__name__.replace("genlib.pipelines.", "pipelines.", 1), _module)
elif __name__.startswith("src.pipelines."):
    _sys.modules.setdefault(__name__.replace("src.pipelines.", "pipelines.", 1), _module)
    _sys.modules.setdefault(__name__.replace("src.pipelines.", "genlib.pipelines.", 1), _module)
elif __name__.startswith("pipelines."):
    _sys.modules.setdefault(__name__.replace("pipelines.", "src.pipelines.", 1), _module)
    _sys.modules.setdefault(__name__.replace("pipelines.", "genlib.pipelines.", 1), _module)
del _module, _sys
