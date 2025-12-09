from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    candidates = list(output_dir.glob("vae_last.pt")) + list(output_dir.glob("vae_best.pt"))
    if not candidates:
        candidates = list(output_dir.glob("*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
