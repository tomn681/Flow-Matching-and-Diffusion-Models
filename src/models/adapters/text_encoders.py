from __future__ import annotations

import torch
import torch.nn as nn
from core.registry import Registry


TEXT_ENCODER_REGISTRY = Registry[type[nn.Module]]("text_encoders")
DEFAULT_TEXT_ENCODER_MODEL_NAMES: dict[str, str] = {
    "clip": "openai/clip-vit-large-patch14",
    "qwen": "Qwen/Qwen2-0.5B",
}


class _BaseTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = None
        self.model: nn.Module

    def _freeze_model(self) -> None:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _tokenize(self, texts: list[str], *, max_length: int) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        return {k: v.to(self.model_device) for k, v in tokens.items()}


@TEXT_ENCODER_REGISTRY.register("clip")
class CLIPTextEncoder(_BaseTextEncoder):
    """Frozen CLIP text-encoder wrapper for cross-attention conditioning.

    Example:
        encoder = CLIPTextEncoder("openai/clip-vit-large-patch14")
        embeddings = encoder(["a ct scan", "a healthy sample"])
        # embeddings: (batch, sequence, hidden_dim)
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14") -> None:
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self._freeze_model()

    def forward(self, texts: list[str]) -> torch.Tensor:
        tokens = self._tokenize(texts, max_length=77)
        output = self.model(**tokens)
        return output.last_hidden_state


@TEXT_ENCODER_REGISTRY.register("qwen")
class QWENTextEncoder(_BaseTextEncoder):
    """Frozen QWEN text-encoder wrapper for cross-attention conditioning.

    Example:
        encoder = QWENTextEncoder("Qwen/Qwen2-0.5B")
        embeddings = encoder(["left lung", "right lung"])
        # embeddings: (batch, sequence, hidden_dim)
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B") -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self._freeze_model()

    def forward(self, texts: list[str]) -> torch.Tensor:
        tokens = self._tokenize(texts, max_length=77)
        output = self.model(**tokens)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("QWEN model output does not expose 'last_hidden_state'.")
        return hidden


def build_text_encoder(kind: str, model_name: str | None = None) -> nn.Module:
    encoder_kind = str(kind).strip().lower()
    try:
        encoder_cls = TEXT_ENCODER_REGISTRY.get(encoder_kind)
    except KeyError as exc:
        available = ", ".join(TEXT_ENCODER_REGISTRY.list())
        raise ValueError(
            f"Unsupported text encoder kind '{kind}'. Expected one of: {available}."
        ) from exc
    resolved_name = model_name or DEFAULT_TEXT_ENCODER_MODEL_NAMES.get(encoder_kind)
    if resolved_name is None:
        raise ValueError(
            f"No default model name configured for text encoder kind '{kind}'. "
            "Pass model_name explicitly."
        )
    return encoder_cls(resolved_name)


__all__ = [
    "TEXT_ENCODER_REGISTRY",
    "DEFAULT_TEXT_ENCODER_MODEL_NAMES",
    "CLIPTextEncoder",
    "QWENTextEncoder",
    "build_text_encoder",
]
