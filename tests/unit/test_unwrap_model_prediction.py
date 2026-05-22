from __future__ import annotations

import torch

from core.types import unwrap_model_prediction


def test_unwrap_raw_tensor() -> None:
    t = torch.randn(2, 3, 8, 8)
    assert unwrap_model_prediction(t) is t


def test_unwrap_tuple() -> None:
    t = torch.randn(2, 3, 8, 8)
    noise = torch.randn(2, 3, 8, 8)
    result = unwrap_model_prediction((t, noise))
    assert result is t


def test_unwrap_list() -> None:
    t = torch.randn(2, 3, 8, 8)
    result = unwrap_model_prediction([t, None])
    assert result is t


def test_unwrap_object_with_sample_attr() -> None:
    t = torch.randn(2, 3, 8, 8)

    class FakeOutput:
        sample = t

    result = unwrap_model_prediction(FakeOutput())
    assert result is t


def test_unwrap_object_without_sample_attr() -> None:
    t = torch.randn(2, 3, 8, 8)

    class FakeOutput:
        pass

    fake = FakeOutput()
    fake.data = t
    result = unwrap_model_prediction(fake)
    assert result is fake

