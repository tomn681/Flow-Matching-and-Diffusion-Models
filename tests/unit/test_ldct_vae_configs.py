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

