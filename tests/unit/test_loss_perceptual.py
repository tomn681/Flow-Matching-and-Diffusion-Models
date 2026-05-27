from __future__ import annotations

import torch
import torch.nn as nn

from losses.perceptual import PerceptualLossComponent
from nn.losses.perceptual import _to_2d_batch


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
    )
    assert captured["resize"] is False
    assert captured["backbone"] == "resnet50"
    assert captured["use_lpips"] is True
    assert captured["lpips_net"] == "alex"


def test_to_2d_batch_supports_rank5_inputs() -> None:
    x = torch.randn(2, 1, 3, 4, 5)
    y, original_shape = _to_2d_batch(x)
    assert original_shape == (2, 1, 3, 4, 5)
    assert y.shape == (6, 1, 4, 5)
