from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKL as HFAutoencoderKL

from losses import LOSS_REGISTRY
from models import merge_models
from models.adapters.weight_mappers import map_hf_vae_to_ours
from models.vae.kl import AutoencoderKL
from nn.losses.adversarial import PatchDiscriminator


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


def test_map_hf_vae_to_ours_merges_mid_attention_qkv_for_mha() -> None:
    hf = _build_hf_vae()
    ours = _build_our_vae()
    mapped = map_hf_vae_to_ours(hf.state_dict(), target_state_dict=ours.state_dict())
    assert "encoder.mid_attn.attn.in_proj_weight" in mapped
    assert "decoder.mid_attn.attn.in_proj_weight" in mapped
    assert "encoder.mid_attn.q.conv.weight" not in mapped
    assert tuple(mapped["encoder.mid_attn.attn.in_proj_weight"].shape) == tuple(
        ours.state_dict()["encoder.mid_attn.attn.in_proj_weight"].shape
    )


def test_merge_models_is_exported_from_top_level_src() -> None:
    import src

    assert callable(src.merge_models)
    assert src.merge_models.__name__ == "merge_models"


def test_new_gan_loss_components_are_registered() -> None:
    keys = set(LOSS_REGISTRY.list())
    assert "gan_generator_wgan" in keys
    assert "gan_discriminator_wgangp" in keys
    assert "gan_discriminator_r1" in keys


def test_wgangp_discriminator_loss_computes_with_penalty() -> None:
    component = LOSS_REGISTRY.build("gan_discriminator_wgangp", gp_weight=2.0)
    disc = PatchDiscriminator(in_channels=1, base_channels=8, spatial_dims=2)
    real = torch.randn(2, 1, 16, 16)
    fake = torch.randn(2, 1, 16, 16)
    value = component.compute(
        context={
            "real_pred": disc(real),
            "fake_pred": disc(fake),
            "discriminator": disc,
            "real": real,
            "fake": fake,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
    )
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_r1_discriminator_loss_computes() -> None:
    component = LOSS_REGISTRY.build("gan_discriminator_r1", r1_weight=0.25)
    disc = PatchDiscriminator(in_channels=1, base_channels=8, spatial_dims=2)
    real = torch.randn(2, 1, 16, 16)
    value = component.compute(
        context={
            "discriminator": disc,
            "real": real,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
    )
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_no_hardcoded_absolute_paths_in_docs() -> None:
    docs_root = Path(__file__).resolve().parents[2] / "docs"
    forbidden = ("/home/delas/", "/Users/", "C:\\\\Users\\\\")
    for path in docs_root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert not any(pattern in text for pattern in forbidden), f"Hardcoded absolute path found in {path}"


def test_core_api_page_does_not_duplicate_protocols_mkdocstrings() -> None:
    page = (Path(__file__).resolve().parents[2] / "docs" / "api" / "core.md").read_text(encoding="utf-8")
    assert "::: src.core.protocols" not in page


def test_controlnet_guide_documents_registered_trainer() -> None:
    page = (Path(__file__).resolve().parents[2] / "docs" / "guides" / "controlnet.md").read_text(encoding="utf-8")
    assert "`ControlNetTrainer` is now registered" in page
    assert "python3 train.py --config configs/<controlnet_config>.json" in page
    assert "python3 run_model.py" in page


def test_distillation_guide_documents_progressive_mode_honestly() -> None:
    page = (
        Path(__file__).resolve().parents[2] / "docs" / "guides" / "distillation_workflow.md"
    ).read_text(encoding="utf-8")
    assert '`training.distillation_mode: "feature_matching"`' in page
    assert '`training.distillation_mode: "progressive"`' in page
    assert "teacher_steps` / `student_steps` actively affect the training target" in page
    assert "progressive distillation is not implemented" not in page.lower()
