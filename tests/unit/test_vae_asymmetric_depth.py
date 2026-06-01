from __future__ import annotations

from models.vae.kl import AutoencoderKL
from models.vae.vq import VQVAE


def test_kl_vae_supports_asymmetric_decoder_depth() -> None:
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        resolution=16,
        base_ch=32,
        ch_mult=(1, 2),
        num_res_blocks={"encoder": 1, "decoder": 3},
        attn_resolutions=(),
        z_channels=4,
        embed_dim=4,
        use_attention=False,
        spatial_dims=2,
    )
    encoder_blocks = [len(stage.blocks) for stage in model.encoder.downs]
    decoder_blocks = [len(stage.blocks) for stage in model.decoder.ups]
    assert all(n == 1 for n in encoder_blocks)
    assert all(n == 4 for n in decoder_blocks)


def test_vq_vae_supports_asymmetric_decoder_depth() -> None:
    model = VQVAE(
        in_channels=1,
        out_channels=1,
        resolution=16,
        base_ch=32,
        ch_mult=(1, 2),
        num_res_blocks={"encoder": 1, "decoder": 3},
        attn_resolutions=(),
        z_channels=4,
        embed_dim=4,
        use_attention=False,
        spatial_dims=2,
        codebook_size=16,
        quantizer_type="classic",
    )
    encoder_blocks = [len(stage.blocks) for stage in model.encoder.downs]
    decoder_blocks = [len(stage.blocks) for stage in model.decoder.ups]
    assert all(n == 1 for n in encoder_blocks)
    assert all(n == 4 for n in decoder_blocks)
