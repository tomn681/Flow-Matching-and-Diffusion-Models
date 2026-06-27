from __future__ import annotations

import pytest
import torch

from models.adapters.weight_mappers import map_hf_unet_to_ours
from models.unet.diffusers import UNetDiffusersND
from models.unet.hf_diffusers import HFDiffusersUNet2DAdapter
from models.unet.utils import TimestepEmbedding as LocalTimestepEmbedding
from models.unet.utils import build_timestep_features
from nn.blocks.attention import DiffusersAttentionND
from nn.blocks.residual import ResBlockND


diffusers = pytest.importorskip("diffusers")
from diffusers.models.attention_processor import Attention as HFAttention
from diffusers.models.embeddings import TimestepEmbedding as HFTimestepEmbedding
from diffusers.models.embeddings import get_timestep_embedding as HFGetTimestepEmbedding
from diffusers.models.resnet import ResnetBlock2D as HFResnetBlock2D


def test_local_timestep_features_match_hf_default_positional() -> None:
    t = torch.tensor([0, 1, 17, 999], dtype=torch.long)
    ours = build_timestep_features(t, 32, max_period=10000, flip_sin_to_cos=True, freq_shift=0)
    hf = HFGetTimestepEmbedding(t, 32, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1, max_period=10000)
    assert torch.allclose(ours, hf, atol=0.0, rtol=0.0)


def test_local_timestep_mlp_matches_hf() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    hf = HFTimestepEmbedding(32, 128)
    ours = LocalTimestepEmbedding(32, 128)
    ours.load_state_dict(hf.state_dict(), strict=True)
    y_hf = hf(x)
    y_ours = ours(x)
    assert torch.allclose(y_ours, y_hf, atol=1e-6, rtol=1e-6)


def test_resblock_matches_hf_default_branch() -> None:
    torch.manual_seed(0)
    hf = HFResnetBlock2D(
        in_channels=32,
        out_channels=32,
        temb_channels=128,
        eps=1e-5,
        groups=32,
        groups_out=32,
        dropout=0.0,
        time_embedding_norm="default",
        non_linearity="swish",
        output_scale_factor=1.0,
        pre_norm=True,
    )
    ours = ResBlockND(
        channels=32,
        emb_channels=128,
        out_channels=32,
        dropout=0.0,
        spatial_dims=2,
        norm_type="gn",
        act="silu",
        norm_groups=32,
        norm_eps=1e-5,
        zero_init_last_conv=False,
        groups_out=32,
        pre_norm=True,
        time_embedding_norm="default",
        output_scale_factor=1.0,
    )
    mapped = map_hf_unet_to_ours(hf.state_dict(), target_state_dict=ours.state_dict())
    ours.load_state_dict(mapped, strict=True)
    x = torch.randn(2, 32, 16, 16)
    emb = torch.randn(2, 128)
    y_hf = hf(x, emb)
    y_ours = ours(x, emb)
    assert torch.allclose(y_ours, y_hf, atol=1e-5, rtol=1e-5)


def test_attention_matches_hf_self_attention() -> None:
    torch.manual_seed(0)
    hf = HFAttention(
        query_dim=32,
        heads=4,
        dim_head=8,
        norm_num_groups=32,
        residual_connection=True,
        rescale_output_factor=1.0,
        bias=True,
        upcast_softmax=True,
        _from_deprecated_attn_block=True,
    )
    ours = DiffusersAttentionND(
        channels=32,
        heads=4,
        norm_num_groups=32,
        residual_connection=True,
        rescale_output_factor=1.0,
        upcast_softmax=True,
        bias=True,
        use_efficient_attn=False,
    )
    ours.group_norm.load_state_dict(hf.group_norm.state_dict(), strict=True)
    ours.to_q.load_state_dict(hf.to_q.state_dict(), strict=True)
    ours.to_k.load_state_dict(hf.to_k.state_dict(), strict=True)
    ours.to_v.load_state_dict(hf.to_v.state_dict(), strict=True)
    ours.to_out[0].load_state_dict(hf.to_out[0].state_dict(), strict=True)
    x = torch.randn(2, 32, 8, 8)
    y_hf = hf(x, temb=None)
    y_ours = ours(x)
    assert torch.allclose(y_ours, y_hf, atol=1e-5, rtol=1e-5)


def test_unet_diffusers_nd_matches_hf_unet2dmodel_forward() -> None:
    torch.manual_seed(0)
    cfg = dict(
        spatial_dims=2,
        sample_size=16,
        in_channels=1,
        out_channels=1,
        center_input_sample=False,
        time_embedding_type="positional",
        freq_shift=0,
        flip_sin_to_cos=True,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        mid_block_type="UNetMidBlock2D",
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        downsample_padding=1,
        downsample_type="conv",
        upsample_type="conv",
        dropout=0.0,
        act_fn="silu",
        attention_head_dim=8,
        norm_num_groups=32,
        attn_norm_num_groups=None,
        norm_eps=1e-5,
        resnet_time_scale_shift="default",
        add_attention=True,
    )
    hf = HFDiffusersUNet2DAdapter(**cfg)
    ours = UNetDiffusersND(**cfg)
    mapped = map_hf_unet_to_ours(hf.state_dict(), target_state_dict=ours.state_dict())
    ours.load_state_dict(mapped, strict=True)
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([10, 999], dtype=torch.long)
    y_hf = hf(x, t)
    y_ours = ours(x, t)
    assert torch.allclose(y_ours, y_hf, atol=2e-5, rtol=2e-5)
