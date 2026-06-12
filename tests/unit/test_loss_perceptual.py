from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from losses.perceptual import PerceptualLossComponent
from nn.losses.perceptual import PerceptualLoss, _to_2d_batch


class _FakePerceptualLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


def test_perceptual_device_tracking(monkeypatch) -> None:
    monkeypatch.setattr("losses.perceptual.PerceptualLoss", _FakePerceptualLoss)
    comp = PerceptualLossComponent(weight=0.5)
    assert comp._device == torch.device("cpu")
    comp.to(torch.device("cpu"))
    assert comp._device == torch.device("cpu")


def test_perceptual_compute_uses_context_keys(monkeypatch) -> None:
    monkeypatch.setattr("losses.perceptual.PerceptualLoss", _FakePerceptualLoss)
    comp = PerceptualLossComponent(weight=1.0)
    pred = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)
    ctx = {
        "reconstruction_image": pred,
        "target": target,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = comp.compute(context=ctx)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_perceptual_result_on_context_device(monkeypatch) -> None:
    monkeypatch.setattr("losses.perceptual.PerceptualLoss", _FakePerceptualLoss)
    comp = PerceptualLossComponent(weight=1.0)
    pred = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)
    ctx = {
        "reconstruction_image": pred,
        "target": target,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    value = comp.compute(context=ctx)
    assert value.device == torch.device("cpu")


def test_perceptual_component_forwards_backbone_and_lpips_kwargs(monkeypatch) -> None:
    captured = {}

    class _CapturePerceptual(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

        def forward(self, pred, target):
            return torch.mean(torch.abs(pred - target))

    monkeypatch.setattr("losses.perceptual.PerceptualLoss", _CapturePerceptual)
    _ = PerceptualLossComponent(
        weight=1.0,
        resize=False,
        backbone="resnet50",
        use_lpips=True,
        lpips_net="alex",
        data_range="minus_one_to_one",
    )
    assert captured["resize"] is False
    assert captured["backbone"] == "resnet50"
    assert captured["use_lpips"] is True
    assert captured["lpips_net"] == "alex"
    assert captured["data_range"] == "minus_one_to_one"


def test_to_2d_batch_supports_rank5_inputs() -> None:
    x = torch.randn(2, 1, 3, 4, 5)
    y, original_shape = _to_2d_batch(x)
    assert original_shape == (2, 1, 3, 4, 5)
    assert y.shape == (6, 1, 4, 5)


def test_perceptual_loss_raises_when_lpips_requested_but_missing(monkeypatch) -> None:
    monkeypatch.setattr("nn.losses.perceptual._HAS_LPIPS", False)
    with pytest.raises(RuntimeError, match="pip install lpips"):
        PerceptualLoss(use_lpips=True)


def test_prepare_inputs_applies_imagenet_normalization_for_vgg_path(monkeypatch) -> None:
    monkeypatch.setattr("nn.losses.perceptual._HAS_TORCHVISION", False)
    loss = PerceptualLoss(use_lpips=False)
    recon = torch.tensor([[[[1.0]]]])
    target = torch.tensor([[[[0.0]]]])
    recon_2d, target_2d = loss._prepare_inputs(recon, target)
    expected_recon = torch.tensor([[[[(1.0 - 0.485) / 0.229]], [[(1.0 - 0.456) / 0.224]], [[(1.0 - 0.406) / 0.225]]]])
    expected_target = torch.tensor([[[[(0.0 - 0.485) / 0.229]], [[(0.0 - 0.456) / 0.224]], [[(0.0 - 0.406) / 0.225]]]])
    assert torch.allclose(recon_2d, expected_recon, atol=1e-6)
    assert torch.allclose(target_2d, expected_target, atol=1e-6)


def test_prepare_inputs_converts_minus_one_to_one_before_imagenet_normalization(monkeypatch) -> None:
    monkeypatch.setattr("nn.losses.perceptual._HAS_TORCHVISION", False)
    loss = PerceptualLoss(use_lpips=False, data_range="minus_one_to_one")
    recon = torch.tensor([[[[-1.0]]]])
    target = torch.tensor([[[[1.0]]]])
    recon_2d, target_2d = loss._prepare_inputs(recon, target)
    expected_recon = torch.tensor([[[[(0.0 - 0.485) / 0.229]], [[(0.0 - 0.456) / 0.224]], [[(0.0 - 0.406) / 0.225]]]])
    expected_target = torch.tensor([[[[(1.0 - 0.485) / 0.229]], [[(1.0 - 0.456) / 0.224]], [[(1.0 - 0.406) / 0.225]]]])
    assert torch.allclose(recon_2d, expected_recon, atol=1e-6)
    assert torch.allclose(target_2d, expected_target, atol=1e-6)


def test_perceptual_loss_rejects_mismatched_declared_range(monkeypatch) -> None:
    monkeypatch.setattr("nn.losses.perceptual._HAS_TORCHVISION", False)
    loss = PerceptualLoss(use_lpips=False, data_range="zero_to_one")
    recon = torch.full((1, 1, 2, 2), -1.5)
    target = torch.zeros_like(recon)
    with pytest.raises(ValueError, match="expected inputs in \\[0, 1\\]"):
        loss._assert_declared_range(recon, target)
