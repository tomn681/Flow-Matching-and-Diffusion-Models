from __future__ import annotations

import json
import re
from pathlib import Path

from configs.v2 import resolve_config


def load_json_config(path: Path | str, overrides: dict | list[str] | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return resolve_config(path, overrides=overrides)


def save_json_config(path: Path | str, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(cfg, fh, indent=2)


def allocate_run_dir(base: Path | str) -> Path:
    base = Path(base)
    parent = base.parent
    stem = base.name
    parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(stem)}_run(\d+)$")
    existing = []
    for entry in parent.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                existing.append(int(m.group(1)))
    next_id = (max(existing) + 1) if existing else 1
    return parent / f"{stem}_run{next_id}"
