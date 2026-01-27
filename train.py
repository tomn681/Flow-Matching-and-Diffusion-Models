"""
Library-friendly training entrypoint.

Usage:
    python train.py --config path/to/config.json

The config must declare the model (currently VAEs) and the data root inside
its "training" section. Dispatches to the appropriate pipeline based on the
config contents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

# Ensure local `src` package is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pipelines.train.vae_lib import train as train_vae
from pipelines.train.flow_matching_lib import train as train_flow_matching
from pipelines.train.diffusion_lib import train as train_diffusion
from utils import build_train_val_datasets, load_json_config

TRAINERS: list[tuple[str, Callable]] = [
    ("vae", train_vae),
    ("flow_matching", train_flow_matching),
    ("diffusion", train_diffusion),
]


def dispatch_train(cfg_path: Path, resume: str | None) -> None:
    cfg = load_json_config(cfg_path)
    for key, trainer in TRAINERS:
        if key in cfg:
            train_ds, val_ds = build_train_val_datasets(cfg)
            trainer(train_ds, cfg_path, val_dataset=val_ds, resume=resume)
            return
    available = ", ".join(k for k, _ in TRAINERS)
    raise ValueError(f"Unsupported config: expected one of {{{available}}} sections.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models from JSON configs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from (optional).")
    args = parser.parse_args()
    dispatch_train(args.config, args.resume)


if __name__ == "__main__":
    main()
