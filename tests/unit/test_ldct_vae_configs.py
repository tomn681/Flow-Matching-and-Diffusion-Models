from __future__ import annotations

import json
from pathlib import Path


def test_all_ldct_vae_configs_set_input_normalize() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "LDCT" / "vae"
    missing: list[str] = []
    invalid: list[str] = []
    for path in sorted(root.glob("*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        mode = cfg.get("training", {}).get("input_normalize")
        if mode is None:
            missing.append(path.name)
            continue
        if str(mode).lower() not in {"positive", "centered", "symmetric", "zscore"}:
            invalid.append(f"{path.name}: {mode!r}")
    assert not missing, f"LDCT VAE configs missing training.input_normalize: {missing}"
    assert not invalid, f"LDCT VAE configs with invalid training.input_normalize: {invalid}"


def test_all_ldct_vae_configs_enable_zero_init_attn_out() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "LDCT" / "vae"
    missing: list[str] = []
    disabled: list[str] = []
    for path in sorted(root.glob("*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        value = cfg.get("model", {}).get("zero_init_attn_out")
        if value is None:
            missing.append(path.name)
            continue
        if value is not True:
            disabled.append(f"{path.name}: {value!r}")
    assert not missing, f"LDCT VAE configs missing model.zero_init_attn_out: {missing}"
    assert not disabled, f"LDCT VAE configs with model.zero_init_attn_out != true: {disabled}"


def test_all_direct_vae_configs_enable_zero_init_attn_out() -> None:
    root = Path(__file__).resolve().parents[2] / "configs"
    missing: list[str] = []
    disabled: list[str] = []
    for path in sorted(root.rglob("*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        model_cfg = cfg.get("model", {})
        model_type = str(model_cfg.get("model_type", "")).lower()
        if model_type not in {"vae", "kl_vae", "vq_vae"}:
            continue
        value = model_cfg.get("zero_init_attn_out")
        rel = str(path.relative_to(root))
        if value is None:
            missing.append(rel)
            continue
        if value is not True:
            disabled.append(f"{rel}: {value!r}")
    assert not missing, f"Direct VAE configs missing model.zero_init_attn_out: {missing}"
    assert not disabled, f"Direct VAE configs with model.zero_init_attn_out != true: {disabled}"


def test_latent_dropout_experiment_config_exists_and_sets_dropout() -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "LDCT" / "vae" / "vae_exp_e_latent_dropout_ssim_lpips_gan.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["model"]["latent_dropout"] == 0.1
    assert cfg["training"]["ssim_weight"] == 0.3
    assert cfg["training"]["perceptual_use_lpips"] is True


def test_sharpness_experiment_configs_set_loss_start_epochs() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "LDCT" / "vae"
    names = [
        "vae_exp_a_ssim_gan.json",
        "vae_exp_b_lpips_gan.json",
        "vae_exp_c_ssim_lpips_gan.json",
        "vae_exp_d_grad_ssim_gan.json",
        "vae_exp_e_latent_dropout_ssim_lpips_gan.json",
    ]
    for name in names:
        cfg = json.loads((root / name).read_text(encoding="utf-8"))
        assert cfg["training"]["ssim_start"] == 20
    exp_b = json.loads((root / "vae_exp_b_lpips_gan.json").read_text(encoding="utf-8"))
    assert exp_b["training"]["perceptual_start"] == 20


def test_sharpness_experiment_configs_enable_ema_and_spectral_norm() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "LDCT" / "vae"
    names = [
        "vae_exp_a_ssim_gan.json",
        "vae_exp_b_lpips_gan.json",
        "vae_exp_c_ssim_lpips_gan.json",
        "vae_exp_d_grad_ssim_gan.json",
        "vae_exp_e_latent_dropout_ssim_lpips_gan.json",
    ]
    for name in names:
        cfg = json.loads((root / name).read_text(encoding="utf-8"))
        assert cfg["training"]["ema_decay"] == 0.9999
        assert cfg["training"]["spectral_norm"] is True


def test_reconstruction_only_and_production_ldct_vae_configs_exist() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "LDCT" / "vae"

    r0 = json.loads((root / "vae_exp_r0_recon_only.json").read_text(encoding="utf-8"))
    assert r0["training"]["gan_weight"] == 0.0
    assert r0["training"]["perceptual_weight"] == 0.0
    assert r0["training"]["ssim_weight"] == 0.0
    assert r0["training"]["ema_decay"] == 0.9999

    l6 = json.loads((root / "vae_exp_l6_prod.json").read_text(encoding="utf-8"))
    assert l6["training"]["perceptual_use_lpips"] is True
    assert l6["training"]["perceptual_start"] == 20
    assert l6["training"]["ssim_weight"] == 0.3
    assert l6["training"]["ssim_start"] == 20
    assert l6["training"]["gan_weight"] == 0.1
    assert l6["training"]["spectral_norm"] is True
    assert l6["training"]["ema_decay"] == 0.9999


def test_monai_ldct_vae_configs_exist_and_use_monai_latent_type() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "LDCT" / "vae"
    names = [
        "monai_m1_4x_pure_recon.json",
        "monai_m2a_kl1e7.json",
        "monai_m2c_kl1e5.json",
        "monai_m3_8x_reference.json",
        "monai_m4_l1_ssim.json",
        "monai_m5_l1_gradient.json",
        "monai_m6_l1_ssim_gradient.json",
        "monai_m7_l1_vgg.json",
        "monai_m8_l1_ssim_grad_vgg.json",
        "monai_m9_l1_ffl.json",
        "monai_m10_best_plus_gan.json",
        "monai_m11_production.json",
    ]
    for name in names:
        cfg = json.loads((root / name).read_text(encoding="utf-8"))
        assert cfg["model"]["model_type"] == "vae"
        assert cfg["model"]["latent_type"] == "monai"
        assert cfg["model"]["attention_impl"] == "qkv"
        assert cfg["model"]["zero_init_attn_out"] is True
        assert cfg["training"]["input_normalize"] == "positive"
        assert cfg["training"]["ema_decay"] == 0.9999
        assert cfg["training"]["max_grad_norm"] == 1.0
        assert cfg["dataset"]["class"] == "datasets.ldct:LDCTDataset"

    m9 = json.loads((root / "monai_m9_l1_ffl.json").read_text(encoding="utf-8"))
    assert m9["training"]["focal_frequency_weight"] == 1.0
    assert m9["training"]["focal_frequency_alpha"] == 1.0
