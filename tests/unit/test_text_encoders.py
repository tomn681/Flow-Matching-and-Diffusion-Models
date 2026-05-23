from __future__ import annotations

import sys
import types

import pytest
import torch

from models.adapters.text_encoders import (
    CLIPTextEncoder,
    QWENTextEncoder,
    TEXT_ENCODER_REGISTRY,
    build_text_encoder,
)


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_name: str):
        _ = model_name
        return cls()

    def __call__(self, texts, *, return_tensors, padding, max_length, truncation):
        _ = return_tensors, padding, truncation
        b = len(texts)
        ids = torch.arange(b * max_length, dtype=torch.long).reshape(b, max_length) % 100
        return {"input_ids": ids}


class _FakeModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.embed = torch.nn.Embedding(100, hidden_size)

    @classmethod
    def from_pretrained(cls, model_name: str):
        _ = model_name
        return cls()

    def forward(self, input_ids=None, **kwargs):
        _ = kwargs
        x = self.embed(input_ids)
        return types.SimpleNamespace(last_hidden_state=x)


def _install_fake_transformers(monkeypatch) -> None:
    fake = types.SimpleNamespace(
        CLIPTokenizer=_FakeTokenizer,
        CLIPTextModel=_FakeModel,
        AutoTokenizer=_FakeTokenizer,
        AutoModel=_FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)


def test_clip_text_encoder_forward_shape_and_frozen(monkeypatch) -> None:
    _install_fake_transformers(monkeypatch)
    enc = CLIPTextEncoder("fake/clip")
    out = enc(["a cat", "a dog"])
    assert out.shape == (2, 77, 32)
    assert torch.isfinite(out).all()
    assert all(not p.requires_grad for p in enc.model.parameters())


def test_qwen_text_encoder_forward_shape_and_frozen(monkeypatch) -> None:
    _install_fake_transformers(monkeypatch)
    enc = QWENTextEncoder("fake/qwen")
    out = enc(["hello", "world", "text"])
    assert out.shape == (3, 77, 32)
    assert torch.isfinite(out).all()
    assert all(not p.requires_grad for p in enc.model.parameters())


def test_build_text_encoder_factory(monkeypatch) -> None:
    _install_fake_transformers(monkeypatch)
    clip = build_text_encoder("clip", "fake/clip")
    qwen = build_text_encoder("qwen", "fake/qwen")
    assert isinstance(clip, CLIPTextEncoder)
    assert isinstance(qwen, QWENTextEncoder)


def test_text_encoder_registry_contains_expected_entries() -> None:
    assert TEXT_ENCODER_REGISTRY.list() == ["clip", "qwen"]


def test_build_text_encoder_factory_invalid_kind_raises(monkeypatch) -> None:
    _install_fake_transformers(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported text encoder kind"):
        build_text_encoder("t5")
