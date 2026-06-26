from __future__ import annotations

import json
from pathlib import Path

from configs import from_template, validate_config
from models.factory import ModelFactory


def _config_paths() -> list[Path]:
    return sorted(Path("configs").rglob("*.json"))


def test_all_shipped_configs_validate_and_build_models() -> None:
    bad_keys: list[str] = []
    for path in _config_paths():
        raw = json.loads(path.read_text())
        text = path.read_text()
        for key in ("mixed_precision", "use_self_attention", "model_best.pt", '"attention_impl": "compvis"', '"attention_impl": "legacy_qkv"'):
            if key in text:
                bad_keys.append(f"{path}: {key}")
        validate_config(raw, config_path=path)
        ModelFactory.build(raw)
    assert not bad_keys, (
        "Shipped configs still contain forbidden dead keys, stale VAE checkpoint names, "
        "or deprecated attention_impl aliases:\n" + "\n".join(bad_keys)
    )


def test_all_templates_validate_and_build_models() -> None:
    for name in ("sd15_vae", "sd15_latent_ddpm", "fmboost_latent_fm", "pixel_ddpm_1d", "vqgan_magvit"):
        cfg = from_template(name)
        validate_config(cfg)
        ModelFactory.build(cfg)


def test_legacy_ldct_pixel_configs_preserve_old_training_contract() -> None:
    root = Path("configs")
    targets = {
        "LDCT/pixel/concat/pixel_fm_concat_legacy_ldct.json": {
            "epochs": 500,
            "weight_decay": 0.01,
        },
        "LDCT/pixel/concat/pixel_ddpm_concat_legacy_ldct.json": {
            "epochs": 500,
            "weight_decay": 0.01,
        },
        "LDCT/pixel/concat/pixel_rf_concat_legacy_ldct.json": {
            "epochs": 500,
            "weight_decay": 0.01,
        },
    }
    for rel_path, expected in targets.items():
        cfg = json.loads((root / rel_path).read_text())
        training = cfg["training"]
        assert training["epochs"] == expected["epochs"]
        assert training["weight_decay"] == expected["weight_decay"]
        assert training["lr_warmup_steps"] == 500
        assert training["lr_scheduler"]["name"] == "warmup_cosine"

    ddpm_cfg = json.loads((root / "LDCT/pixel/concat/pixel_ddpm_concat_legacy_ldct.json").read_text())
    assert "beta_schedule" not in ddpm_cfg["model"]["scheduler"]


def test_legacy_ldct_reflow_configs_use_old_optimizer_schedule_contract() -> None:
    root = Path("configs/experiments")
    for name in (
        "reflow_round1_concat_legacy_ldct.json",
        "reflow_round2_concat_legacy_ldct.json",
        "reflow_round3_concat_legacy_ldct.json",
    ):
        cfg = json.loads((root / name).read_text())
        training = cfg["training"]
        assert training["weight_decay"] == 0.01
        assert training["lr_warmup_steps"] == 500
        assert training["lr_scheduler"]["name"] == "warmup_cosine"
