"""Inference mediator coordinating UNet, scheduler, text encoder, and ControlNet."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.autoencoder.base import BaseAutoencoder
from models.autoencoder.utils import decode_from_latent
from scheduling import resolve_conditioning_mode, sample_with_scheduler


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

    def forward(self, x: torch.Tensor, t: torch.Tensor, context_ca: torch.Tensor | None = None):
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
        scheduler,
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
        if text_context is not None and self.conditioning_mode in {"attention", "latent_attention"}:
            conditioning_batch = text_context

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
        )

    def generate_images(self, inputs: InferenceInputs, *, recon_type: str = "l1") -> torch.Tensor:
        latents = self.generate(inputs)
        if self.vae is None:
            return latents
        return decode_from_latent(self.vae, latents, recon_type=recon_type)


__all__ = ["InferenceInputs", "InferencePipeline"]
