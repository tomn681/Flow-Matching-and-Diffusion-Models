from __future__ import annotations

import torch

from models.controlnet import ControlNetND
from models.unet.condition import UNet2DConditionND


def _build_unet() -> UNet2DConditionND:
    return UNet2DConditionND(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64, 64, 64),
        cross_attention_dim=16,
        attention_head_dim=8,
    )


def _build_controlnet() -> ControlNetND:
    return ControlNetND(
        in_channels=4,
        conditioning_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64, 64, 64),
        cross_attention_dim=16,
        attention_head_dim=8,
    )


def _collect_unet_down_shapes(unet: UNet2DConditionND, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> list[tuple[int, ...]]:
    x = unet._prepare_input(x, context=None, context_ca=ctx)
    t = unet._normalize_timesteps(t, x)
    emb = unet._build_time_embedding(t, x)
    sample = unet.conv_in(x)
    down_block_res_samples = (sample,)
    for downsample_block in unet.down_blocks:
        sample, res_samples = downsample_block(sample, emb, context=ctx)
        down_block_res_samples += res_samples
    return [tuple(v.shape) for v in down_block_res_samples]


def test_controlnet_zero_init_outputs_zero_residuals() -> None:
    unet = _build_unet()
    controlnet = _build_controlnet()
    x = torch.randn(2, 4, 16, 16)
    cond = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 1000, (2,))
    ctx = torch.randn(2, 12, 16)
    emb = unet._build_time_embedding(unet._normalize_timesteps(t, x), x)
    out = controlnet(x, emb, cond, encoder_hidden_states=ctx)
    assert isinstance(out["down_residuals"], list)
    assert torch.allclose(out["mid_residual"], torch.zeros_like(out["mid_residual"]))
    for residual in out["down_residuals"]:
        assert torch.allclose(residual, torch.zeros_like(residual))


def test_controlnet_residual_shapes_match_unet_skip_contract() -> None:
    unet = _build_unet()
    controlnet = _build_controlnet()
    x = torch.randn(2, 4, 16, 16)
    cond = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 1000, (2,))
    ctx = torch.randn(2, 12, 16)
    emb = unet._build_time_embedding(unet._normalize_timesteps(t, x), x)
    residuals = controlnet(x, emb, cond, encoder_hidden_states=ctx)
    expected_shapes = _collect_unet_down_shapes(unet, x, t, ctx)
    assert len(residuals["down_residuals"]) == len(expected_shapes)
    for tensor, shape in zip(residuals["down_residuals"], expected_shapes):
        assert tuple(tensor.shape) == shape


def test_unet_controlnet_residuals_change_output_and_none_keeps_behavior() -> None:
    unet = _build_unet()
    x = torch.randn(2, 4, 16, 16)
    t = torch.randint(0, 1000, (2,))
    ctx = torch.randn(2, 12, 16)
    out_base_a = unet(x, t, encoder_hidden_states=ctx, controlnet_residuals=None)
    out_base_b = unet(x, t, encoder_hidden_states=ctx)
    assert torch.allclose(out_base_a, out_base_b)

    expected_shapes = _collect_unet_down_shapes(unet, x, t, ctx)
    synthetic = {
        "down_residuals": [torch.randn(shape) * 0.5 for shape in expected_shapes],
        "mid_residual": torch.randn(2, 64, 2, 2) * 0.5,
    }
    out_control = unet(x, t, encoder_hidden_states=ctx, controlnet_residuals=synthetic)
    assert not torch.allclose(out_base_a, out_control)
