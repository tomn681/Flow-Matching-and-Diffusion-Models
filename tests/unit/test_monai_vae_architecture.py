from __future__ import annotations

import torch
import torch.nn as nn

from models import ModelFactory
from nn.modules.vae import MonaiResBlock, MonaiSpatialAttention


def _build_monai_vae():
    return ModelFactory.build(
        {
            "model": {
                "model_type": "vae",
                "latent_type": "monai",
                "in_channels": 1,
                "out_channels": 1,
                "resolution": 32,
                "channels": [32, 64, 64],
                "num_res_blocks": 2,
                "attention_levels": [False, False, True],
                "latent_channels": 4,
                "norm_num_groups": 32,
                "spatial_dims": 2,
            }
        }
    )


def test_monai_vae_matches_reference_topology() -> None:
    model = _build_monai_vae()

    assert sum(parameter.numel() for parameter in model.parameters()) == 1_151_457
    assert model.encoder.conv_out.out_channels == 4
    assert all(len(stage.blocks) == 2 for stage in (*model.encoder.downs, *model.decoder.ups))
    assert all(
        isinstance(block, MonaiResBlock)
        for stage in (*model.encoder.downs, *model.decoder.ups)
        for block in stage.blocks
    )
    assert all(
        isinstance(block, MonaiResBlock)
        for block in (
            model.encoder.mid_block1,
            model.encoder.mid_block2,
            model.decoder.mid_block1,
            model.decoder.mid_block2,
        )
    )
    assert isinstance(model.encoder.downs[-1].attns[0], nn.Identity)
    assert isinstance(model.encoder.downs[-1].attns[1], MonaiSpatialAttention)
    assert isinstance(model.decoder.ups[0].attns[0], nn.Identity)
    assert isinstance(model.decoder.ups[0].attns[1], MonaiSpatialAttention)
    assert isinstance(model.encoder.mid_attn, MonaiSpatialAttention)
    assert isinstance(model.decoder.mid_attn, MonaiSpatialAttention)
    assert all(stage.down.use_asymmetric_padding for stage in model.encoder.downs[:-1])


def test_monai_vae_uses_shared_encoder_features_and_separate_qkv() -> None:
    model = _build_monai_vae()
    attention = model.encoder.mid_attn.attn
    assert {name for name, _ in attention.named_children()} == {"to_q", "to_k", "to_v", "attention", "out_proj"}

    projection_inputs: list[torch.Tensor] = []
    hooks = [
        projection.register_forward_pre_hook(lambda _module, args: projection_inputs.append(args[0]))
        for projection in (model.quant_conv_mu, model.quant_conv_log_sigma)
    ]
    posterior = model.encode(torch.randn(1, 1, 32, 32))
    for hook in hooks:
        hook.remove()
    assert projection_inputs[0] is projection_inputs[1]
    assert posterior.mu.shape[1] == posterior.logvar.shape[1] == 4


def test_monai_vae_forward_preserves_image_shape() -> None:
    output = _build_monai_vae()(torch.randn(1, 1, 32, 32), sample_posterior=False)
    assert output.reconstruction.shape == (1, 1, 32, 32)
    assert output.posterior.mu.shape == (1, 4, 8, 8)
