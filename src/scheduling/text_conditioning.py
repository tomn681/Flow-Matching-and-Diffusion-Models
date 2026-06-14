from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from models.adapters import build_text_encoder


TextConditioningInput = torch.Tensor | str | Sequence[str] | None


class TextConditioningAdapter:
    """Encode text prompts into cross-attention embeddings.

    The adapter accepts raw prompt strings/sequences and returns token embeddings
    shaped as `(B, seq_len, dim)`. If pre-encoded embeddings are already passed
    in as a tensor, they are forwarded directly.
    """

    def __init__(self, text_encoder: nn.Module) -> None:
        self.encoder = text_encoder

    def __call__(
        self,
        model_input: torch.Tensor,
        cond: TextConditioningInput,
        latent_norm: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del latent_norm
        if cond is None:
            return model_input, None
        if torch.is_tensor(cond):
            return model_input, cond.to(model_input.device)

        texts = self._normalize_texts(cond)
        with torch.no_grad():
            embeddings = self.encoder(texts)
        if not torch.is_tensor(embeddings):
            raise TypeError("text encoder must return a tensor of embeddings.")
        return model_input, embeddings.to(model_input.device)

    def null_conditioning(
        self,
        model_input: torch.Tensor,
        cond: TextConditioningInput,
        latent_norm: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del latent_norm
        if cond is None:
            return model_input, None
        if torch.is_tensor(cond):
            return model_input, torch.zeros_like(cond, device=model_input.device)
        texts = self._normalize_texts(cond)
        return model_input, self(model_input, [""] * len(texts))[1]

    @staticmethod
    def _normalize_texts(cond: TextConditioningInput) -> list[str]:
        if isinstance(cond, str):
            return [cond]
        if isinstance(cond, Sequence) and not isinstance(cond, (bytes, bytearray)):
            texts = [str(item) for item in cond]
            if not texts:
                raise ValueError("text conditioning sequence must not be empty.")
            return texts
        raise TypeError("text conditioning must be a string, a sequence of strings, or a tensor of embeddings.")


def build_text_conditioning_adapter(
    *,
    kind: str = "clip",
    model_name: str | None = None,
    device: torch.device | str | None = None,
) -> TextConditioningAdapter:
    encoder = build_text_encoder(kind=kind, model_name=model_name)
    if device is not None:
        encoder = encoder.to(device)
    return TextConditioningAdapter(encoder)


__all__ = ["TextConditioningAdapter", "TextConditioningInput", "build_text_conditioning_adapter"]
