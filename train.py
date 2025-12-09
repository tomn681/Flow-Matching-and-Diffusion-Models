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
import json
from pathlib import Path

from pipelines.train.vae_lib import train as train_vae
from utils import build_train_val_datasets


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as fh:
        return json.load(fh)


def dispatch_train(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    if "vae" in cfg:
        train_ds, val_ds = build_train_val_datasets(cfg)
        train_vae(train_ds, cfg_path, val_dataset=val_ds)
    else:
        raise ValueError("Unsupported config: could not infer model type (expecting 'vae' section).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models from JSON configs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    args = parser.parse_args()
    dispatch_train(args.config)


if __name__ == "__main__":
    main()
