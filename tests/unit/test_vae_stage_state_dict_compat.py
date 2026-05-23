from models.vae.kl import AutoencoderKL


def test_vae_stage_state_dict_uses_down_up_keys() -> None:
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        resolution=16,
        base_ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(),
        use_attention=False,
        z_channels=4,
        embed_dim=4,
        spatial_dims=2,
    )
    keys = set(model.state_dict().keys())

    assert any(k.startswith("encoder.downs.0.down.") for k in keys)
    assert any(k.startswith("decoder.ups.0.up.") for k in keys)
