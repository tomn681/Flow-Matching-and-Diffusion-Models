from __future__ import annotations

import torch

from models.controlnet import ControlNetND, initialize_controlnet_from_unet
from models.unet.efficient import EfficientUNetND
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


def _build_efficient_unet() -> EfficientUNetND:
    return EfficientUNetND(
        spatial_dims=2,
        in_channels=4,
        model_channels=32,
        out_channels=4,
        num_res_blocks=1,
        attention_resolutions=(1,),
        channel_mult=(1, 2),
        cross_attention_resolutions=(),
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
    out = controlnet(x, t, cond, encoder_hidden_states=ctx)
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
    residuals = controlnet(x, t, cond, encoder_hidden_states=ctx)
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


def test_controlnet_precomputed_timestep_embedding_matches_internal() -> None:
    controlnet = _build_controlnet()
    x = torch.randn(2, 4, 16, 16)
    cond = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 1000, (2,))
    ctx = torch.randn(2, 12, 16)

    out_internal = controlnet(x, t, cond, encoder_hidden_states=ctx)
    emb = controlnet._build_time_embedding(controlnet._normalize_timesteps(t, x), x)
    out_precomputed = controlnet(x, t, cond, encoder_hidden_states=ctx, timesteps_emb=emb)

    assert len(out_internal["down_residuals"]) == len(out_precomputed["down_residuals"])
    for lhs, rhs in zip(out_internal["down_residuals"], out_precomputed["down_residuals"]):
        assert torch.allclose(lhs, rhs)
    assert torch.allclose(out_internal["mid_residual"], out_precomputed["mid_residual"])


def test_initialize_controlnet_from_unet_copies_encoder_prefix() -> None:
    unet = _build_unet()
    controlnet = _build_controlnet()
    initialize_controlnet_from_unet(controlnet, unet)
    assert torch.allclose(controlnet.conv_in.weight, unet.conv_in.weight)
    assert torch.allclose(
        controlnet.time_embedding.linear_1.weight,
        unet.time_embedding.linear_1.weight,
    )


def test_initialize_controlnet_from_unet_leaves_zero_convs_untouched() -> None:
    unet = _build_unet()
    controlnet = _build_controlnet()
    before = controlnet.zero_convs[0].conv.weight.detach().clone()
    initialize_controlnet_from_unet(controlnet, unet)
    after = controlnet.zero_convs[0].conv.weight.detach()
    assert torch.allclose(before, torch.zeros_like(before))
    assert torch.allclose(after, before)


def _collect_efficient_skip_shapes(model: EfficientUNetND, x: torch.Tensor, t: torch.Tensor) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    x = model._prepare_input(x, context=None, context_ca=None)
    t = model._normalize_timesteps(t, x)
    emb = model._build_time_embedding(t, x)
    x = model.pool(x)
    hs: list[tuple[int, ...]] = []
    h = x
    for block in model.input_blocks:
        h = block(h, emb, None)
        hs.append(tuple(h.shape))
    h = model.middle_block(h, emb, None)
    return hs, tuple(h.shape)


def test_efficient_unet_controlnet_residuals_change_output() -> None:
    model = _build_efficient_unet()
    with torch.no_grad():
        model.out[2].conv.weight.fill_(0.1)
        if model.out[2].conv.bias is not None:
            model.out[2].conv.bias.zero_()
    x = torch.randn(2, 4, 16, 16)
    t = torch.randint(0, 1000, (2,))
    base = model(x, t)
    skip_shapes, mid_shape = _collect_efficient_skip_shapes(model, x, t)
    residuals = {
        "down_residuals": [torch.randn(shape) * 0.25 for shape in skip_shapes],
        "mid_residual": torch.randn(mid_shape) * 0.25,
    }
    controlled = model(x, t, controlnet_residuals=residuals)
    assert not torch.allclose(base, controlled)
