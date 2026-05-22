from __future__ import annotations

import torch

from models.autoencoder.base import BaseAutoencoder


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

