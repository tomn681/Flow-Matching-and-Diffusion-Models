from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from core.types import ModelOutput, unwrap_model_prediction


def sample_log_normal_sigmas(
    batch_size: int,
    device: torch.device,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    sigmas = torch.randn(batch_size, device=device, dtype=dtype) * float(p_std) + float(p_mean)
    sigmas = sigmas.exp()
    return sigmas.clamp(min=float(sigma_min), max=float(sigma_max))


def karras_sigmas(
    num_steps: int,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    steps = max(1, int(num_steps))
    ramp = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
    inv_rho = 1.0 / float(rho)
    min_inv = float(sigma_min) ** inv_rho
    max_inv = float(sigma_max) ** inv_rho
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** float(rho)
    return torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)


def sigma_to_timestep(
    sigma: torch.Tensor,
    *,
    sigma_min: float,
    sigma_max: float,
    num_train_timesteps: int,
) -> torch.Tensor:
    sigma = sigma.clamp(min=float(sigma_min), max=float(sigma_max))
    if abs(float(sigma_max) - float(sigma_min)) < 1e-12:
        return torch.zeros_like(sigma)
    normalized = (sigma.log() - math.log(float(sigma_min))) / (math.log(float(sigma_max)) - math.log(float(sigma_min)))
    return normalized * float(max(1, int(num_train_timesteps) - 1))


def edm_scalings(
    sigma: torch.Tensor,
    *,
    sigma_data: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = sigma.to(dtype=torch.float32)
    sigma_data = float(sigma_data)
    sigma2 = sigma.square()
    data2 = sigma_data * sigma_data
    denom = sigma2 + data2
    c_skip = data2 / denom
    c_out = sigma * sigma_data / denom.sqrt()
    c_in = denom.rsqrt()
    c_noise = 0.25 * sigma.clamp_min(1e-12).log()
    return c_skip, c_out, c_in, c_noise


def edm_loss_weights(sigmas: torch.Tensor, *, sigma_data: float = 0.5) -> torch.Tensor:
    sigmas = sigmas.to(dtype=torch.float32)
    sigma2 = sigmas.square()
    data2 = float(sigma_data) ** 2
    return (sigma2 + data2) / (sigma2 * data2).clamp_min(1e-12)


def _sigma_view(sigmas: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return sigmas.view(sigmas.size(0), *([1] * (reference.ndim - 1)))


def _forward_model(model, inputs: torch.Tensor, noise_levels: torch.Tensor, context_ca=None) -> torch.Tensor:
    outputs = model(inputs, noise_levels, context_ca=context_ca) if context_ca is not None else model(inputs, noise_levels)
    if isinstance(outputs, tuple):
        return outputs[0]
    if isinstance(outputs, ModelOutput):
        return outputs.reconstruction
    return unwrap_model_prediction(outputs)


def edm_denoise_prediction(
    model,
    noisy: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    sigma_data: float = 0.5,
    conditioning_adapter=None,
    conditioning_batch: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    latent_norm: str | None = None,
    context_override=None,
    conditioned_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas = sigmas.to(device=noisy.device, dtype=torch.float32)
    c_skip, c_out, c_in, c_noise = edm_scalings(sigmas, sigma_data=sigma_data)
    scaled_noisy = noisy * _sigma_view(c_in.to(dtype=noisy.dtype), noisy)
    if conditioned_input is not None:
        model_input = conditioned_input
        context = context_override
    elif conditioning_adapter is not None:
        model_input, context = conditioning_adapter(scaled_noisy, conditioning_batch, latent_norm)
    else:
        model_input, context = scaled_noisy, None
    pred = _forward_model(model, model_input, c_noise.to(device=noisy.device, dtype=noisy.dtype), context_ca=context)
    denoised = _sigma_view(c_skip.to(dtype=noisy.dtype), noisy) * noisy + _sigma_view(c_out.to(dtype=noisy.dtype), noisy) * pred
    return denoised, pred


def edm_sample_heun(
    *,
    model,
    sample_shape: tuple[int, ...],
    device: torch.device,
    num_inference_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    sigma_data: float = 0.5,
    conditioning_adapter=None,
    conditioning_batch: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    unconditional_conditioning_batch: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    latent_norm: str | None = None,
    guidance_scale: float = 1.0,
    cfg_rescale: float = 0.0,
) -> torch.Tensor:
    from .sampling_loop import _apply_cfg_rescale

    sigmas = karras_sigmas(
        num_inference_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )
    current = torch.randn(sample_shape, device=device) * float(sigmas[0].item())
    cond = conditioning_batch
    uncond = unconditional_conditioning_batch

    for idx in range(sigmas.numel() - 1):
        sigma = sigmas[idx].expand(sample_shape[0])
        sigma_next = sigmas[idx + 1].expand(sample_shape[0])
        if guidance_scale != 1.0 and cond is not None and conditioning_adapter is not None:
            denoised_cond, _ = edm_denoise_prediction(
                model,
                current,
                sigma,
                sigma_data=sigma_data,
                conditioning_adapter=conditioning_adapter,
                conditioning_batch=cond,
                latent_norm=latent_norm,
            )
            if uncond is None:
                null_input, null_context = conditioning_adapter.null_conditioning(current * 0.0 + current, cond, latent_norm)
                denoised_uncond, _ = edm_denoise_prediction(
                    model,
                    current,
                    sigma,
                    sigma_data=sigma_data,
                    conditioning_adapter=None,
                    conditioned_input=null_input,
                    context_override=null_context,
                )
            else:
                denoised_uncond, _ = edm_denoise_prediction(
                    model,
                    current,
                    sigma,
                    sigma_data=sigma_data,
                    conditioning_adapter=conditioning_adapter,
                    conditioning_batch=uncond,
                    latent_norm=latent_norm,
                )
            denoised = denoised_uncond + float(guidance_scale) * (denoised_cond - denoised_uncond)
            denoised = _apply_cfg_rescale(
                denoised,
                pred_cond=denoised_cond,
                guidance_scale=float(guidance_scale),
                cfg_rescale=float(cfg_rescale),
            )
        else:
            denoised, _ = edm_denoise_prediction(
                model,
                current,
                sigma,
                sigma_data=sigma_data,
                conditioning_adapter=conditioning_adapter,
                conditioning_batch=cond,
                latent_norm=latent_norm,
            )

        sigma_view = _sigma_view(sigma.to(dtype=current.dtype), current)
        d_cur = (current - denoised) / sigma_view.clamp_min(1e-12)
        dt = _sigma_view((sigma_next - sigma).to(dtype=current.dtype), current)
        x_euler = current + dt * d_cur
        if float(sigma_next[0].item()) == 0.0:
            current = x_euler
            continue
        if guidance_scale != 1.0 and cond is not None and conditioning_adapter is not None:
            denoised_next_cond, _ = edm_denoise_prediction(
                model,
                x_euler,
                sigma_next,
                sigma_data=sigma_data,
                conditioning_adapter=conditioning_adapter,
                conditioning_batch=cond,
                latent_norm=latent_norm,
            )
            if uncond is None:
                null_input, null_context = conditioning_adapter.null_conditioning(x_euler * 0.0 + x_euler, cond, latent_norm)
                denoised_next_uncond, _ = edm_denoise_prediction(
                    model,
                    x_euler,
                    sigma_next,
                    sigma_data=sigma_data,
                    conditioning_adapter=None,
                    conditioned_input=null_input,
                    context_override=null_context,
                )
            else:
                denoised_next_uncond, _ = edm_denoise_prediction(
                    model,
                    x_euler,
                    sigma_next,
                    sigma_data=sigma_data,
                    conditioning_adapter=conditioning_adapter,
                    conditioning_batch=uncond,
                    latent_norm=latent_norm,
                )
            denoised_next = denoised_next_uncond + float(guidance_scale) * (denoised_next_cond - denoised_next_uncond)
            denoised_next = _apply_cfg_rescale(
                denoised_next,
                pred_cond=denoised_next_cond,
                guidance_scale=float(guidance_scale),
                cfg_rescale=float(cfg_rescale),
            )
        else:
            denoised_next, _ = edm_denoise_prediction(
                model,
                x_euler,
                sigma_next,
                sigma_data=sigma_data,
                conditioning_adapter=conditioning_adapter,
                conditioning_batch=cond,
                latent_norm=latent_norm,
            )
        d_next = (x_euler - denoised_next) / _sigma_view(sigma_next.to(dtype=current.dtype), x_euler).clamp_min(1e-12)
        current = current + dt * 0.5 * (d_cur + d_next)
    return current


def consistency_sample(
    *,
    model,
    sample_shape: tuple[int, ...],
    device: torch.device,
    num_inference_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    sigma_data: float = 0.5,
    conditioning_adapter=None,
    conditioning_batch: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    latent_norm: str | None = None,
) -> torch.Tensor:
    sigmas = karras_sigmas(
        num_inference_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
    )
    current = torch.randn(sample_shape, device=device) * float(sigmas[0].item())
    for sigma in sigmas[:-1]:
        sigma_batch = sigma.expand(sample_shape[0])
        current, _ = edm_denoise_prediction(
            model,
            current,
            sigma_batch,
            sigma_data=sigma_data,
            conditioning_adapter=conditioning_adapter,
            conditioning_batch=conditioning_batch,
            latent_norm=latent_norm,
        )
    return current


__all__ = [
    "sample_log_normal_sigmas",
    "karras_sigmas",
    "sigma_to_timestep",
    "edm_scalings",
    "edm_loss_weights",
    "edm_denoise_prediction",
    "edm_sample_heun",
    "consistency_sample",
]
