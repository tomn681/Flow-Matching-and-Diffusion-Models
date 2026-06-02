from __future__ import annotations

import torch

from models import MODEL_REGISTRY, ModelFactory
from models.dit import DiTND
from models.unet.base import BaseUNetND


def test_dit_registry_key_present() -> None:
    assert MODEL_REGISTRY.get("dit") is DiTND


def test_dit_forward_shapes_1d_2d_3d() -> None:
    cases = [
        (1, (2, 4, 32)),
        (2, (2, 4, 32, 32)),
        (3, (1, 4, 16, 16, 16)),
    ]
    for spatial_dims, shape in cases:
        model = DiTND(
            spatial_dims=spatial_dims,
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        )
        x = torch.randn(*shape)
        t = torch.randint(0, 1000, (shape[0],), dtype=torch.long)
        out = model(x, t)
        assert out.shape == x.shape


def test_model_factory_builds_dit() -> None:
    cfg = {
        "model": {
            "model_type": "dit",
            "dit": {
                "spatial_dims": 2,
                "in_channels": 4,
                "out_channels": 4,
                "patch_size": 2,
                "hidden_size": 64,
                "depth": 2,
                "num_heads": 4,
            },
        }
    }
    model = ModelFactory.build(cfg)
    assert isinstance(model, DiTND)


def test_dit_extends_base_unet_contract() -> None:
    model = DiTND(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
    )
    assert isinstance(model, BaseUNetND)


def test_dit_accepts_integer_class_conditioning_via_context() -> None:
    model = DiTND(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=10,
    )
    x = torch.randn(2, 4, 16, 16)
    t = torch.randint(0, 1000, (2,), dtype=torch.long)
    y = torch.tensor([1, 3], dtype=torch.long)
    out = model(x, t, context=y)
    assert out.shape == x.shape
