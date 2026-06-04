from __future__ import annotations

import torch
from diffusers import AutoencoderKL as HFAutoencoderKL

from models.adapters.weight_mappers import (
    load_hf_vae_weights,
    map_hf_vae_key_to_ours,
    map_hf_vae_to_ours,
)
from models.vae.kl import AutoencoderKL


def _build_hf_vae() -> HFAutoencoderKL:
    return HFAutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        latent_channels=4,
        sample_size=256,
    )


def _build_our_vae() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        resolution=256,
        base_ch=128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        z_channels=4,
        embed_dim=4,
    )


def test_map_hf_vae_key_to_ours_rewrites_expected_patterns() -> None:
    assert (
        map_hf_vae_key_to_ours("encoder.down_blocks.1.resnets.0.conv1.weight")
        == "encoder.downs.1.blocks.0.conv1.conv.weight"
    )
    assert (
        map_hf_vae_key_to_ours("decoder.mid_block.attentions.0.to_out.0.bias")
        == "decoder.mid_attn.proj_out.conv.bias"
    )


def test_map_hf_vae_to_ours_covers_all_keys_and_shapes() -> None:
    hf = _build_hf_vae()
    ours = _build_our_vae()
    mapped = map_hf_vae_to_ours(hf.state_dict(), target_state_dict=ours.state_dict())
    for key, target_tensor in ours.state_dict().items():
        assert key in mapped
        assert tuple(mapped[key].shape) == tuple(target_tensor.shape)


def test_map_hf_vae_to_ours_raises_on_shape_mismatch() -> None:
    hf = _build_hf_vae()
    ours = _build_our_vae()
    state = dict(hf.state_dict())
    state["encoder.conv_in.weight"] = torch.randn(1, 1, 3, 3)
    try:
        map_hf_vae_to_ours(state, target_state_dict=ours.state_dict())
        assert False, "Expected shape mismatch error"
    except ValueError as exc:
        assert "Shape mismatch" in str(exc)


def test_load_hf_vae_weights_smoke_and_roundtrip() -> None:
    hf = _build_hf_vae()
    ours = _build_our_vae()
    load_hf_vae_weights(ours, "dummy/model", hf_state_dict=hf.state_dict())
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = ours(x).reconstruction
    assert torch.isfinite(y).all()

    state = ours.state_dict()
    clone = _build_our_vae()
    clone.load_state_dict(state)
    for k, v in clone.state_dict().items():
        assert torch.allclose(v, state[k])
