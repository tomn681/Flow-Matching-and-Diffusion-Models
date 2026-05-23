from __future__ import annotations

import torch
import torch.nn as nn


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
    if encoder_kind == "clip":
        return CLIPTextEncoder(model_name or "openai/clip-vit-large-patch14")
    if encoder_kind == "qwen":
        return QWENTextEncoder(model_name or "Qwen/Qwen2-0.5B")
    raise ValueError(f"Unsupported text encoder kind '{kind}'. Expected one of: clip, qwen.")


__all__ = ["CLIPTextEncoder", "QWENTextEncoder", "build_text_encoder"]
