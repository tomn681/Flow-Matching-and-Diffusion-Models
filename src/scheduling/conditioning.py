from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from models.autoencoder.base import BaseAutoencoder
from models.autoencoder.utils import encode_to_latent

from core.registry import Registry
from .conditioning_chain import ChainAdapterSpec, ConditioningChain
from .sampling_loop import _prepare_attention_context, normalize_latent_conditioning


ConditioningInput = torch.Tensor | Mapping[str, object] | Sequence[str] | str | None
ConditioningAdapter = Callable[[torch.Tensor, ConditioningInput, str | None], tuple[torch.Tensor, torch.Tensor | None]]

CONDITIONING_ADAPTER_REGISTRY = Registry[ConditioningAdapter]("conditioning_adapters")


class LatentAttentionAdapter:
    """Factory for per-trainer latent-attention adapter instances."""

    @staticmethod
    def create(vae_model: BaseAutoencoder) -> ConditioningAdapter:
        def _adapt(
            model_input: torch.Tensor,
            cond: torch.Tensor | None,
            latent_norm: str | None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if cond is None:
                return model_input, None
            encoded = encode_to_latent(vae_model, cond)
            normalized = normalize_latent_conditioning(encoded, latent_norm)
            context = _prepare_attention_context(normalized)
            return model_input, context

        return _adapt


@CONDITIONING_ADAPTER_REGISTRY.register("none")
def _adapt_none(model_input: torch.Tensor, cond: torch.Tensor | None, latent_norm: str | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    return model_input, None


@CONDITIONING_ADAPTER_REGISTRY.register("concatenate")
def _adapt_concatenate(
    model_input: torch.Tensor, cond: ConditioningInput, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cond is None:
        return model_input, None
    if not torch.is_tensor(cond):
        raise TypeError("concatenate conditioning must be a tensor.")
    return torch.cat([model_input, cond], dim=1), None


@CONDITIONING_ADAPTER_REGISTRY.register("attention")
def _adapt_attention(
    model_input: torch.Tensor, cond: ConditioningInput, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cond is None:
        return model_input, None
    if not torch.is_tensor(cond):
        raise TypeError("attention conditioning must be a tensor.")
    normalized = normalize_latent_conditioning(cond, latent_norm)
    context = _prepare_attention_context(normalized)
    return model_input, context


@CONDITIONING_ADAPTER_REGISTRY.register("latent_attention")
def _adapt_latent_attention(
    model_input: torch.Tensor, cond: ConditioningInput, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Stateless default registration. Latent trainers should inject a per-instance
    # adapter created via LatentAttentionAdapter.create(vae_model).
    return _adapt_attention(model_input, cond, latent_norm)


@CONDITIONING_ADAPTER_REGISTRY.register("text")
def _adapt_text(
    model_input: torch.Tensor, cond: ConditioningInput, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del latent_norm
    if cond is None:
        return model_input, None
    if torch.is_tensor(cond):
        return model_input, cond.to(model_input.device)
    raise TypeError(
        "text conditioning requires pre-encoded embeddings when resolved from the registry. "
        "Use TextConditioningAdapter/build_text_conditioning_adapter(...) for raw prompt strings."
    )


_DEFAULT_CHAIN = ConditioningChain(
    [
        ChainAdapterSpec(key="concatenate", adapter=_adapt_concatenate),
        ChainAdapterSpec(key="attention", adapter=_adapt_attention),
        ChainAdapterSpec(key="text", adapter=_adapt_text),
    ]
)


@CONDITIONING_ADAPTER_REGISTRY.register("chain")
def _adapt_chain(
    model_input: torch.Tensor, cond: ConditioningInput, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _DEFAULT_CHAIN(model_input, cond, latent_norm)


@CONDITIONING_ADAPTER_REGISTRY.register("inpainting")
def _adapt_inpainting(
    model_input: torch.Tensor, cond: ConditioningInput, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del latent_norm
    if cond is None:
        return model_input, None
    if not isinstance(cond, Mapping):
        raise TypeError("inpainting conditioning must be a mapping with 'mask' and 'original' tensors.")

    mask = cond.get("mask")
    original = cond.get("original")
    if not torch.is_tensor(mask) or not torch.is_tensor(original):
        raise TypeError("inpainting conditioning requires tensor values for 'mask' and 'original'.")
    if mask.dim() != 4 or original.dim() != 4:
        raise ValueError("inpainting 'mask' and 'original' must be rank-4 tensors.")
    if mask.size(1) != 1:
        raise ValueError("inpainting 'mask' must have shape (B, 1, H, W).")
    if mask.shape[0] != model_input.shape[0] or original.shape[0] != model_input.shape[0]:
        raise ValueError("inpainting tensors must match model_input batch size.")
    if mask.shape[2:] != model_input.shape[2:] or original.shape[2:] != model_input.shape[2:]:
        raise ValueError("inpainting tensors must match model_input spatial dimensions.")
    if original.shape[1] != model_input.shape[1]:
        raise ValueError("inpainting 'original' channels must match model_input channels.")

    masked_original = original * (1.0 - mask)
    return torch.cat([model_input, mask, masked_original], dim=1), None


@CONDITIONING_ADAPTER_REGISTRY.register("super_resolution")
def _adapt_super_resolution(
    model_input: torch.Tensor, cond: ConditioningInput, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del latent_norm
    if cond is None:
        return model_input, None
    if not torch.is_tensor(cond):
        raise TypeError("super_resolution conditioning must be a tensor.")
    if cond.dim() != model_input.dim():
        raise ValueError("super_resolution conditioning tensor rank must match model_input rank.")
    if cond.shape[0] != model_input.shape[0]:
        raise ValueError("super_resolution conditioning batch size must match model_input batch size.")
    if cond.shape[2:] != model_input.shape[2:]:
        interp_mode = "linear" if model_input.dim() == 3 else "bilinear" if model_input.dim() == 4 else "trilinear" if model_input.dim() == 5 else "nearest"
        align = False if interp_mode in {"linear", "bilinear", "trilinear"} else None
        cond = F.interpolate(cond, size=model_input.shape[2:], mode=interp_mode, align_corners=align)
    return torch.cat([model_input, cond], dim=1), None


@CONDITIONING_ADAPTER_REGISTRY.register("depth")
def _adapt_depth(
    model_input: torch.Tensor, cond: ConditioningInput, latent_norm: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del latent_norm
    if cond is None:
        return model_input, None
    if not torch.is_tensor(cond):
        raise TypeError("depth conditioning must be a tensor.")
    if cond.dim() == model_input.dim() - 1:
        cond = cond.unsqueeze(1)
    if cond.dim() != model_input.dim():
        raise ValueError("depth conditioning tensor rank must match model_input rank (or be rank-1 without channel).")
    if cond.shape[0] != model_input.shape[0]:
        raise ValueError("depth conditioning batch size must match model_input batch size.")
    if cond.shape[2:] != model_input.shape[2:]:
        interp_mode = "linear" if model_input.dim() == 3 else "bilinear" if model_input.dim() == 4 else "trilinear" if model_input.dim() == 5 else "nearest"
        align = False if interp_mode in {"linear", "bilinear", "trilinear"} else None
        cond = F.interpolate(cond, size=model_input.shape[2:], mode=interp_mode, align_corners=align)
    return torch.cat([model_input, cond], dim=1), None


def resolve_conditioning_adapter(mode: str | None) -> ConditioningAdapter:
    key = str(mode or "none").strip().lower() or "none"
    return CONDITIONING_ADAPTER_REGISTRY.get(key)
