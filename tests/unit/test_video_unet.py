from __future__ import annotations

import pytest
import torch

from models import MODEL_REGISTRY, ModelFactory
from models.unet import VideoUNetND
from nn.modules import TemporalAttentionND


def test_video_unet_registry_key_present() -> None:
    assert MODEL_REGISTRY.get("video_unet") is VideoUNetND


def test_video_unet_forward_shape() -> None:
    model = VideoUNetND(
        spatial_dims=3,
        in_channels=2,
        model_channels=32,
        out_channels=2,
        num_res_blocks=1,
        attention_resolutions=(1,),
        channel_mult=(1, 2),
        num_heads=4,
        dim_head=8,
        use_linear_attn=False,
    )
    x = torch.randn(2, 2, 4, 16, 16)
    t = torch.randint(0, 1000, (2,), dtype=torch.long)
    y = model(x, t)
    assert y.shape == x.shape


def test_video_unet_injects_temporal_attention_layers() -> None:
    model = VideoUNetND(
        spatial_dims=3,
        in_channels=1,
        model_channels=16,
        out_channels=1,
        num_res_blocks=1,
        attention_resolutions=(1,),
        channel_mult=(1,),
        num_heads=4,
        dim_head=4,
        use_linear_attn=False,
    )
    temporal_layers = [module for module in model.modules() if isinstance(module, TemporalAttentionND)]
    assert temporal_layers


def test_video_unet_requires_spatial_dims_three() -> None:
    with pytest.raises(ValueError, match="spatial_dims=3"):
        VideoUNetND(
            spatial_dims=2,
            in_channels=1,
            model_channels=16,
            out_channels=1,
            num_res_blocks=1,
            attention_resolutions=(),
            channel_mult=(1,),
        )


def test_model_factory_builds_video_unet() -> None:
    cfg = {
        "model": {
            "model_type": "video_unet",
            "unet": {
                "unet_impl": "video_unet",
                "spatial_dims": 3,
                "in_channels": 2,
                "out_channels": 2,
                "model_channels": 32,
                "channel_mult": [1, 2],
                "num_res_blocks": 1,
                "attention_resolutions": [1],
                "num_heads": 4,
                "dim_head": 8,
                "use_linear_attn": False,
            },
        }
    }
    model = ModelFactory.build(cfg, channels=2)
    assert isinstance(model, VideoUNetND)
