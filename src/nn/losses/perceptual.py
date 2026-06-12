"""Perceptual loss functions for image-space supervision."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models

    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover - torchvision may be missing in some envs
    _HAS_TORCHVISION = False

try:
    import lpips as _lpips

    _HAS_LPIPS = True
except Exception:  # pragma: no cover - lpips is optional
    _HAS_LPIPS = False


def _to_2d_batch(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Project ND tensors to BCHW while preserving batch semantics."""
    if x.dim() == 4:
        return x, tuple(x.shape)
    if x.dim() == 3:
        return x.unsqueeze(2), tuple(x.shape)
    if x.dim() == 5:
        b, c, d, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        return y, (b, c, d, h, w)
    raise ValueError(f"PerceptualLoss expects rank 3/4/5 tensors, got rank {x.dim()}.")


def _normalize_declared_data_range(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "0,1": "zero_to_one",
        "zero_to_one": "zero_to_one",
        "positive": "zero_to_one",
        "-1,1": "minus_one_to_one",
        "minus_one_to_one": "minus_one_to_one",
        "centered": "minus_one_to_one",
        "symmetric": "minus_one_to_one",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported perceptual data_range '{mode}'. Expected 'zero_to_one' or 'minus_one_to_one'."
        )
    return resolved


class _FeatureExtractor(nn.Module):
    """Backbone-agnostic feature extractor returning selected feature maps."""

    def __init__(self, backbone: str, layers: tuple[int, ...]) -> None:
        super().__init__()
        if not _HAS_TORCHVISION:
            raise RuntimeError("torchvision is required for feature-based perceptual loss.")
        self.backbone = backbone
        self.layers = tuple(int(v) for v in layers)

        if backbone == "vgg16":
            net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features  # type: ignore[attr-defined]
            self.sequence = nn.ModuleList(list(net))
            self._mode = "sequential"
        elif backbone == "vgg19":
            net = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_FEATURES).features  # type: ignore[attr-defined]
            self.sequence = nn.ModuleList(list(net))
            self._mode = "sequential"
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  # type: ignore[attr-defined]
            self.stages = nn.ModuleList(
                [
                    nn.Sequential(net.conv1, net.bn1, net.relu),
                    nn.Sequential(net.maxpool, net.layer1),
                    net.layer2,
                    net.layer3,
                    net.layer4,
                ]
            )
            self._mode = "resnet"
        else:
            raise ValueError(f"Unsupported perceptual backbone '{backbone}'.")

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs: list[torch.Tensor] = []
        if self._mode == "sequential":
            for idx, layer in enumerate(self.sequence):
                x = layer(x)
                if idx in self.layers:
                    outs.append(x)
        else:
            for idx, stage in enumerate(self.stages):
                x = stage(x)
                if idx in self.layers:
                    outs.append(x)
        return outs


class PerceptualLoss(nn.Module):
    """
    Perceptual loss with configurable torchvision backbone or optional LPIPS backend.

    If `use_lpips=True`, the `lpips` package must be installed or construction
    fails loudly.

    If torchvision is unavailable for feature-based perceptual loss, the module
    falls back to a disabled zero-valued loss for compatibility with minimal
    environments.
    """

    def __init__(
        self,
        resize: bool = False,
        layers: tuple[int, ...] = (3, 8, 15, 22),
        layer_weights: Iterable[float] = (1.0, 1.0, 1.0, 1.0),
        *,
        backbone: str = "vgg16",
        use_lpips: bool = False,
        lpips_net: str = "vgg",
        data_range: str = "zero_to_one",
    ) -> None:
        super().__init__()
        self.resize = bool(resize)
        self.layer_weights = list(layer_weights)
        self.backbone = str(backbone).strip().lower()
        self.use_lpips = bool(use_lpips)
        self.lpips_net = str(lpips_net).strip().lower()
        self.data_range = _normalize_declared_data_range(data_range)
        self.enabled = True
        self._lpips_model: nn.Module | None = None
        self._extractor: _FeatureExtractor | None = None
        self._layers = tuple(int(v) for v in layers)
        self._range_checked = False

        if self.use_lpips:
            if not _HAS_LPIPS:
                raise RuntimeError(
                    "PerceptualLoss(use_lpips=True) requires the optional `lpips` package. "
                    "Install it with `pip install lpips`."
                )
            self._lpips_model = _lpips.LPIPS(net=self.lpips_net).eval()
            for p in self._lpips_model.parameters():
                p.requires_grad = False
            return

        if not _HAS_TORCHVISION:
            self.enabled = False
            self.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
            return

        if self.backbone == "resnet50" and self._layers == (3, 8, 15, 22):
            # Keep backward compatibility for default constructor while giving
            # meaningful defaults for ResNet stage indexing.
            self._layers = (1, 2, 3, 4)

        self._extractor = _FeatureExtractor(self.backbone, self._layers)

    @staticmethod
    def _to_three_channel(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] == 3:
            return x
        if x.shape[1] > 3:
            return x[:, :3, :, :]
        reps = (3 + x.shape[1] - 1) // x.shape[1]
        expanded = x.repeat(1, reps, 1, 1)
        return expanded[:, :3, :, :]

    def _prepare_inputs(self, recon: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon_2d, _ = _to_2d_batch(recon)
        target_2d, _ = _to_2d_batch(target)
        recon_2d = self._to_three_channel(recon_2d)
        target_2d = self._to_three_channel(target_2d)

        if self.resize:
            recon_2d = F.interpolate(recon_2d, size=(224, 224), mode="bilinear", align_corners=False)
            target_2d = F.interpolate(target_2d, size=(224, 224), mode="bilinear", align_corners=False)

        if not self.use_lpips:
            if self.data_range == "minus_one_to_one":
                recon_2d = (recon_2d + 1.0) * 0.5
                target_2d = (target_2d + 1.0) * 0.5
            mean = torch.tensor([0.485, 0.456, 0.406], device=recon_2d.device, dtype=recon_2d.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=recon_2d.device, dtype=recon_2d.dtype).view(1, 3, 1, 1)
            recon_2d = (recon_2d - mean) / std
            target_2d = (target_2d - mean) / std
        return recon_2d, target_2d

    def _assert_declared_range(self, recon: torch.Tensor, target: torch.Tensor) -> None:
        if self._range_checked:
            return
        tol = 1e-3
        low = min(float(recon.min().item()), float(target.min().item()))
        high = max(float(recon.max().item()), float(target.max().item()))
        if self.data_range == "zero_to_one":
            if low < -tol or high > 1.0 + tol:
                raise ValueError(
                    f"PerceptualLoss expected inputs in [0, 1], observed range [{low:.4f}, {high:.4f}]."
                )
        else:
            if low < -1.0 - tol or high > 1.0 + tol:
                raise ValueError(
                    f"PerceptualLoss expected inputs in [-1, 1], observed range [{low:.4f}, {high:.4f}]."
                )
        self._range_checked = True

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=recon.device, dtype=recon.dtype)

        self._assert_declared_range(recon, target)
        recon_2d, target_2d = self._prepare_inputs(recon, target)

        if self.use_lpips and self._lpips_model is not None:
            # LPIPS expects inputs normalized to [-1, 1].
            if self.data_range == "zero_to_one":
                recon_lp = recon_2d * 2.0 - 1.0
                target_lp = target_2d * 2.0 - 1.0
            else:
                recon_lp = recon_2d
                target_lp = target_2d
            value = self._lpips_model(recon_lp, target_lp)
            return value.mean().to(device=recon.device, dtype=recon.dtype)

        assert self._extractor is not None
        recon_feats = self._extractor(recon_2d)
        target_feats = self._extractor(target_2d)
        if len(recon_feats) != len(target_feats):
            raise RuntimeError("Perceptual feature extraction produced mismatched feature counts.")

        loss = torch.tensor(0.0, device=recon_2d.device, dtype=recon_2d.dtype)
        for idx, (fr, ft) in enumerate(zip(recon_feats, target_feats)):
            weight = self.layer_weights[idx] if idx < len(self.layer_weights) else 1.0
            loss = loss + float(weight) * F.l1_loss(fr, ft)
        return loss.to(device=recon.device, dtype=recon.dtype)
