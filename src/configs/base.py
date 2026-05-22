from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BaseConfig:
    """Common config metadata shared across typed config blocks."""

    config_path: Optional[Path] = None
    config_version: int = 1
    extra: dict = field(default_factory=dict)
