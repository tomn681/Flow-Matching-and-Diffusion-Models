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
        for key in ("mixed_precision", "use_self_attention", "model_best.pt"):
            if key in text:
                bad_keys.append(f"{path}: {key}")
        validate_config(raw, config_path=path)
        ModelFactory.build(raw)
    assert not bad_keys, "Shipped configs still contain forbidden dead keys or stale VAE checkpoint names:\n" + "\n".join(bad_keys)


def test_all_templates_validate_and_build_models() -> None:
    for name in ("sd15_vae", "sd15_latent_ddpm", "fmboost_latent_fm", "pixel_ddpm_1d", "vqgan_magvit"):
        cfg = from_template(name)
        validate_config(cfg)
        ModelFactory.build(cfg)
