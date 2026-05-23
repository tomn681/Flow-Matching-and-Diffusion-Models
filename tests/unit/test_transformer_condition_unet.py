from __future__ import annotations

import torch

from models.factory import ModelFactory
from nn.blocks.legacy_unet import BLOCK_REGISTRY
from nn.blocks.transformer import Transformer2DModelND


def test_transformer2dmodel_nd_shapes_1d_2d_3d() -> None:
    cases = [
        (1, (2, 32, 64)),
        (2, (2, 32, 16, 16)),
        (3, (1, 32, 4, 8, 8)),
    ]
    for spatial_dims, shape in cases:
        x = torch.randn(shape)
        model = Transformer2DModelND(
            spatial_dims=spatial_dims,
            in_channels=32,
            num_attention_heads=4,
            attention_head_dim=8,
            num_layers=1,
            cross_attention_dim=64,
        )
        ctx = torch.randn(shape[0], 10, 64)
        enc_mask = torch.zeros(shape[0], ctx.shape[1], dtype=torch.bool)
        y = model(x, encoder_hidden_states=ctx, encoder_attention_mask=enc_mask)
        assert y.shape == x.shape


def test_cross_attn_blocks_registered() -> None:
    keys = set(BLOCK_REGISTRY.list())
    assert "CrossAttnDownBlock2D" in keys
    assert "CrossAttnUpBlock2D" in keys
    assert "UNetMidBlock2DCrossAttn" in keys


def test_condition_unet_forward_with_masks_and_embeddings() -> None:
    cfg = {
        "model": {
            "model_type": "latent_diffusion",
            "conditioning": "attention",
            "unet": {
                "unet_impl": "condition_nd",
                "spatial_dims": 2,
                "in_channels": 4,
                "out_channels": 4,
                "block_out_channels": [32, 64, 64, 64],
                "layers_per_block": 1,
                "attention_head_dim": 8,
                "cross_attention_dim": 48,
                "class_embed_type": "projection",
                "num_class_embeds": 8,
                "time_cond_proj_dim": 16,
            },
        }
    }
    model = ModelFactory.build(cfg, conditioning="attention", channels=4)
    x = torch.randn(2, 4, 16, 16)
    t = torch.randint(0, 1000, (2,))
    enc = torch.randn(2, 12, 48)
    class_labels = torch.randint(0, 8, (2,))
    timestep_cond = torch.randn(2, 16)
    enc_mask = torch.zeros(2, enc.shape[1], dtype=torch.bool)

    y = model(
        x,
        t,
        encoder_hidden_states=enc,
        class_labels=class_labels,
        timestep_cond=timestep_cond,
        encoder_attention_mask=enc_mask,
    )
    assert y.shape == x.shape


def test_model_factory_routes_latent_diffusion_to_condition_unet() -> None:
    cfg = {
        "model": {
            "model_type": "latent_diffusion",
            "conditioning": "attention",
            "unet": {
                "unet_impl": "condition_nd",
                "spatial_dims": 2,
                "in_channels": 4,
                "out_channels": 4,
                "block_out_channels": [32, 64, 64, 64],
                "layers_per_block": 1,
                "attention_head_dim": 8,
                "cross_attention_dim": 32,
            },
        }
    }
    model = ModelFactory.build(cfg, channels=4)
    assert model.__class__.__name__ == "UNet2DConditionND"
