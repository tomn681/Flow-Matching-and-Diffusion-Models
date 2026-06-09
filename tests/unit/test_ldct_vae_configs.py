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
