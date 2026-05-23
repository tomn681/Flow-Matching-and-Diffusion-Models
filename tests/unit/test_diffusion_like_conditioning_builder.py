from __future__ import annotations

import torch

from pipelines.samplers.diffusion_like import _build_conditioning_batch


def test_build_conditioning_batch_chain_uses_explicit_keys() -> None:
    samples = [
        {
            "target": torch.zeros(1, 4, 4),
            "concat_cond": torch.ones(1, 4, 4),
            "attn_cond": torch.full((2, 4, 4), 3.0),
        },
        {
            "target": torch.zeros(1, 4, 4),
            "concat_cond": torch.full((1, 4, 4), 2.0),
            "attn_cond": torch.full((2, 4, 4), 4.0),
        },
    ]
    targets = torch.stack([s["target"] for s in samples], dim=0)
    cond = _build_conditioning_batch(
        conditioning_mode="chain",
        samples=samples,
        targets=targets,
        device=torch.device("cpu"),
    )
    assert isinstance(cond, dict)
    assert cond["concatenate"].shape == (2, 1, 4, 4)
    assert cond["attention"].shape == (2, 2, 4, 4)


def test_build_conditioning_batch_chain_falls_back_to_image() -> None:
    samples = [
        {"target": torch.zeros(1, 4, 4), "image": torch.ones(1, 4, 4)},
        {"target": torch.zeros(1, 4, 4), "image": torch.ones(1, 4, 4) * 2},
    ]
    targets = torch.stack([s["target"] for s in samples], dim=0)
    cond = _build_conditioning_batch(
        conditioning_mode="chain",
        samples=samples,
        targets=targets,
        device=torch.device("cpu"),
    )
    assert isinstance(cond, dict)
    assert torch.allclose(cond["concatenate"], cond["attention"])


def test_build_conditioning_batch_inpainting_uses_mask_and_target_as_original() -> None:
    samples = [
        {"target": torch.randn(1, 4, 4), "mask": torch.zeros(1, 4, 4)},
        {"target": torch.randn(1, 4, 4), "mask": torch.ones(1, 4, 4)},
    ]
    targets = torch.stack([s["target"] for s in samples], dim=0)
    cond = _build_conditioning_batch(
        conditioning_mode="inpainting",
        samples=samples,
        targets=targets,
        device=torch.device("cpu"),
    )
    assert isinstance(cond, dict)
    assert cond["mask"].shape == (2, 1, 4, 4)
    assert torch.allclose(cond["original"], targets)
