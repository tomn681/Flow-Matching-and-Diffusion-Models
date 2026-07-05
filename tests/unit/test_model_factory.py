from __future__ import annotations

import pytest
import torch

from models.factory import MODEL_BUILD_STRATEGY, ModelFactory
from models.dit import DiTND
from models.generators.diffusionfactory import DiffusionUNetFactory
from models.generators.vaefactory import VAEFactory
from models.registry import MODEL_REGISTRY


def test_model_registry_has_expected_entries() -> None:
    keys = set(MODEL_REGISTRY.list())
    assert {
        "kl_vae",
        "monai_vae",
        "vq_vae",
        "efficient_unet",
        "diffusers_unet",
        "hf_diffusers_unet",
        "condition_unet",
        "video_unet",
        "controlnet",
    }.issubset(keys)


def test_model_factory_routes_to_vae(monkeypatch) -> None:
    sentinel = object()

    def _fake_build_vae(model_cfg: dict):
        assert model_cfg["model_type"] == "vae"
        return sentinel

    monkeypatch.setattr(ModelFactory, "_build_vae", staticmethod(_fake_build_vae))
    out = ModelFactory.build({"model": {"model_type": "vae"}})
    assert out is sentinel


def test_model_factory_builds_monai_vae_from_latent_type() -> None:
    model = ModelFactory.build(
        {
            "model": {
                "model_type": "vae",
                "latent_type": "monai",
                "in_channels": 1,
                "out_channels": 1,
                "resolution": 32,
                "channels": [32, 64, 64],
                "attention_levels": [False, False, True],
                "latent_channels": 4,
                "norm_num_groups": 32,
                "spatial_dims": 2,
                "zero_init_attn_out": True,
            }
        }
    )
    assert model.__class__.__name__ == "MonaiStyleVAE"
    disc = model.make_discriminator()
    assert any(hasattr(module, "weight_orig") for module in disc.modules())


def test_model_factory_routes_to_unet(monkeypatch) -> None:
    sentinel = object()

    def _fake_build_unet(model_cfg: dict, *, conditioning: str | None = None, channels: int | None = None):
        assert model_cfg["model_type"] == "diffusion"
        assert conditioning == "attention"
        assert channels == 2
        return sentinel

    monkeypatch.setattr(ModelFactory, "_build_unet", staticmethod(_fake_build_unet))
    out = ModelFactory.build({"model": {"model_type": "diffusion"}}, conditioning="attention", channels=2)
    assert out is sentinel


def test_model_factory_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unsupported model_type"):
        ModelFactory.build({"model": {"model_type": "unknown"}})


def test_model_factory_build_is_canonical_and_no_model_from_config_alias_exists() -> None:
    assert callable(ModelFactory.build)
    assert not hasattr(ModelFactory, "from_config")


def test_model_factory_defaults_efficient_unet_to_exact_attention(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_build(key: str, **kwargs):
        captured["key"] = key
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("models.factory.MODEL_REGISTRY.build", _fake_build)
    ModelFactory._build_efficient_unet({"attention_resolutions": [1]}, cond_mode="", channels=1)
    assert captured["key"] == "efficient_unet"
    assert captured["use_linear_attn"] is False


def test_model_factory_builds_hf_diffusers_legacy_unet() -> None:
    model = ModelFactory.build(
        {
            "model": {
                "model_type": "flow_matching",
                "unet": {
                    "unet_impl": "hf_diffusers",
                    "spatial_dims": 2,
                    "sample_size": 32,
                    "in_channels": 1,
                    "out_channels": 1,
                    "in_channels_already_conditioned": False,
                    "center_input_sample": False,
                    "time_embedding_type": "positional",
                    "freq_shift": 0,
                    "flip_sin_to_cos": True,
                    "down_block_types": [
                        "DownBlock2D",
                        "DownBlock2D",
                        "AttnDownBlock2D",
                    ],
                    "up_block_types": [
                        "AttnUpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D",
                    ],
                    "block_out_channels": [32, 64, 64],
                    "layers_per_block": 1,
                    "downsample_padding": 1,
                    "dropout": 0.0,
                    "attention_head_dim": 8,
                    "norm_num_groups": 32,
                    "norm_eps": 1e-5,
                    "resnet_time_scale_shift": "default",
                    "add_attention": True,
                },
            }
        },
        conditioning="concatenate",
        channels=1,
    )
    x = torch.zeros(2, 2, 32, 32)
    t = torch.zeros(2, dtype=torch.long)
    y = model(x, t)
    assert y.shape == (2, 1, 32, 32)


def test_model_factory_builds_flow_matching_dit_backbone() -> None:
    model = ModelFactory.build(
        {
            "model": {
                "model_type": "flow_matching",
                "backbone_type": "dit",
                "conditioning": "none",
                "dit": {
                    "spatial_dims": 2,
                    "in_channels": 1,
                    "out_channels": 1,
                    "patch_size": 2,
                    "hidden_size": 64,
                    "depth": 2,
                    "num_heads": 4,
                },
            }
        },
        conditioning="none",
        channels=1,
    )
    assert isinstance(model, DiTND)
    x = torch.zeros(2, 1, 32, 32)
    t = torch.zeros(2, dtype=torch.long)
    y = model(x, t)
    assert y.shape == (2, 1, 32, 32)


def test_model_build_strategy_contains_phase_i_types() -> None:
    keys = set(MODEL_BUILD_STRATEGY.keys())
    assert {
        "vae",
        "controlnet",
        "diffusion",
        "video_unet",
        "flow_matching",
        "latent_diffusion",
        "latent_flow_matching",
        "latent_rectified_flow",
        "x0_denoising",
        "consistency",
        "edm",
        "rectified_flow",
        "reflow",
    }.issubset(keys)


def test_legacy_vae_factory_delegates_to_unified_model_factory(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"model": {"model_type": "vae"}}')
    sentinel = object()

    def _fake_build(cfg: dict, *, conditioning=None, channels=None):
        assert cfg["model"]["model_type"] == "vae"
        assert conditioning is None
        assert channels is None
        return sentinel

    monkeypatch.setattr("models.generators.vaefactory.ModelFactory.build", _fake_build)
    out = VAEFactory().build_from_json(cfg_path)
    assert out is sentinel


def test_legacy_diffusion_factory_delegates_to_unified_model_factory(monkeypatch) -> None:
    sentinel = object()

    def _fake_build(cfg: dict, *, conditioning=None, channels=None):
        assert cfg["model"]["model_type"] == "diffusion"
        assert cfg["model"]["unet"]["in_channels"] == 1
        assert conditioning == "concatenate"
        assert channels == 1
        return sentinel

    monkeypatch.setattr("models.generators.diffusionfactory.ModelFactory.build", _fake_build)
    out = DiffusionUNetFactory().build({"in_channels": 1}, conditioning="concatenate", channels=1)
    assert out is sentinel
