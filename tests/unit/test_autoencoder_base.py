from __future__ import annotations

import torch

from models.autoencoder.base import BaseAutoencoder
from models.autoencoder.utils import (
    apply_input_normalize,
    resolve_input_normalize,
    resolve_model_input_range_from_normalize,
    sync_autoencoder_input_range,
)


class _DummyAutoencoder(BaseAutoencoder):
    def encode(self, x: torch.Tensor, normalize: bool = False):
        return x

    def decode(self, z: torch.Tensor, denorm: bool = False):
        return z


def test_image_to_model_range() -> None:
    model = _DummyAutoencoder()
    x = torch.tensor([0.0, 0.5, 1.0])
    result = model.image_to_model_range(x)
    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(result, expected)


def test_model_to_image_range() -> None:
    model = _DummyAutoencoder()
    x = torch.tensor([-1.0, 0.0, 1.0])
    result = model.model_to_image_range(x)
    expected = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(result, expected)


def test_model_to_image_range_clamps() -> None:
    model = _DummyAutoencoder()
    x = torch.tensor([-2.0, 3.0])
    result = model.model_to_image_range(x)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_raw_output_to_image_l1() -> None:
    model = _DummyAutoencoder()
    x = torch.tensor([0.0])
    result = model.raw_output_to_image(x, recon_type="l1")
    expected = model.model_to_image_range(x)
    assert torch.equal(result, expected)


def test_raw_output_to_image_bce() -> None:
    model = _DummyAutoencoder()
    x = torch.tensor([0.0])
    result = model.raw_output_to_image(x, recon_type="bce")
    expected = torch.sigmoid(x)
    assert torch.allclose(result, expected)


def test_raw_output_to_image_focal() -> None:
    model = _DummyAutoencoder()
    x = torch.tensor([0.0])
    result = model.raw_output_to_image(x, recon_type="focal")
    expected = torch.sigmoid(x)
    assert torch.allclose(result, expected)


def test_zero_to_one_input_range_is_identity() -> None:
    model = _DummyAutoencoder()
    model.input_range = "zero_to_one"
    x = torch.tensor([0.0, 0.5, 1.0])
    result = model.image_to_model_range(x)
    assert torch.allclose(result, x)


def test_zero_to_one_model_to_image_range_clamps_to_unit_interval() -> None:
    model = _DummyAutoencoder()
    model.input_range = "zero_to_one"
    x = torch.tensor([-2.0, 0.5, 3.0])
    result = model.model_to_image_range(x)
    expected = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(result, expected)


def test_apply_input_normalize_centered() -> None:
    x = torch.tensor([0.0, 0.5, 1.0])
    result = apply_input_normalize(x, "centered")
    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(result, expected)


def test_apply_input_normalize_symmetric_alias_matches_centered() -> None:
    x = torch.tensor([0.0, 0.5, 1.0])
    centered = apply_input_normalize(x, "centered")
    symmetric = apply_input_normalize(x, "symmetric")
    assert torch.allclose(centered, symmetric)


def test_apply_input_normalize_positive() -> None:
    x = torch.tensor([0.0, 0.5, 1.0])
    result = apply_input_normalize(x, "positive")
    assert torch.allclose(result, x)


def test_apply_input_normalize_zscore_per_sample() -> None:
    x = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[10.0, 10.0], [10.0, 10.0]]],
        ]
    )
    result = apply_input_normalize(x, "zscore")
    assert torch.allclose(result[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(result[0].std(), torch.tensor(1.0), atol=1e-6)
    assert torch.isfinite(result[1]).all()


def test_apply_input_normalize_rejects_unknown_mode() -> None:
    x = torch.tensor([0.0])
    try:
        apply_input_normalize(x, "nope")
    except ValueError as exc:
        assert "input_normalize" in str(exc)
    else:
        raise AssertionError("Expected unknown input_normalize mode to raise ValueError.")


def test_resolve_input_normalize_maps_legacy_model_flag() -> None:
    model = _DummyAutoencoder()
    model.input_range = "zero_to_one"
    assert resolve_input_normalize(model, None) == "positive"


def test_resolve_input_normalize_maps_symmetric_alias() -> None:
    model = _DummyAutoencoder()
    assert resolve_input_normalize(model, "symmetric") == "centered"


def test_resolve_model_input_range_from_positive_normalize() -> None:
    assert resolve_model_input_range_from_normalize("positive") == "zero_to_one"


def test_sync_autoencoder_input_range_derives_zero_to_one_from_positive() -> None:
    model = _DummyAutoencoder()
    applied = sync_autoencoder_input_range(model, {"training": {"input_normalize": "positive"}, "model": {}})
    assert applied == "zero_to_one"
    assert model.input_range == "zero_to_one"


def test_sync_autoencoder_input_range_preserves_explicit_model_setting() -> None:
    model = _DummyAutoencoder()
    applied = sync_autoencoder_input_range(
        model,
        {"training": {"input_normalize": "positive"}, "model": {"input_range": "minus_one_to_one"}},
    )
    assert applied == "minus_one_to_one"
    assert model.input_range == "minus_one_to_one"
