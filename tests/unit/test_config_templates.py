from __future__ import annotations

from configs import from_template, validate_config


def test_template_names_build_valid_configs() -> None:
    names = [
        "sd15_vae",
        "sd15_latent_ddpm",
        "fmboost_latent_fm",
        "pixel_ddpm_1d",
        "vqgan_magvit",
    ]
    for name in names:
        cfg = from_template(name)
        validated = validate_config(cfg)
        assert validated.training.epochs > 0
        assert validated.training.batch_size > 0


def test_template_overrides_are_deep_merged() -> None:
    cfg = from_template(
        "sd15_latent_ddpm",
        training={"epochs": 3},
        model={"unet": {"cross_attention_dim": 512}},
    )
    assert cfg["training"]["epochs"] == 3
    assert cfg["model"]["unet"]["cross_attention_dim"] == 512
    assert cfg["model"]["model_type"] == "latent_diffusion"


def test_unknown_template_raises_clear_error() -> None:
    try:
        from_template("does_not_exist")
        raise AssertionError("Expected KeyError for unknown template.")
    except KeyError as exc:
        assert "Unknown template" in str(exc)

