"""
Compatibility trainer entrypoint.

Usage:
    python -m src.train --trainer vae --config configs/autoencoder_kl.json --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import warnings

from compat._deprecation import warn_deprecated
from training import TRAINER_REGISTRY
from utils import build_train_val_datasets, load_json_config


def _merge_overrides(cfg: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(cfg)
    for section, values in overrides.items():
        if not values:
            continue
        target = merged.setdefault(section, {})
        if not isinstance(target, dict):
            target = {}
            merged[section] = target
        target.update(values)
    return merged


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper that builds datasets from config overrides and "
            "dispatches training through TRAINER_REGISTRY."
        )
    )
    parser.add_argument("--trainer", type=str, required=True, help="Trainer registry key (for example: 'vae').")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root directory override.")
    parser.add_argument("--device", type=str, default=None, help="Override training device (e.g., 'cuda', 'cuda:1').")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--img-size", type=int, default=None, help="Override image size/resolution.")
    parser.add_argument("--in-channels", type=int, default=None, help="Override VAE input channels.")
    parser.add_argument("--out-channels", type=int, default=None, help="Override VAE output channels.")
    parser.add_argument("--perceptual-device", type=str, default=None, help="Optional device for perceptual loss (e.g., cuda:1).")
    parser.add_argument("--gan-device", type=str, default=None, help="Optional device for discriminator (e.g., cuda:1).")
    return parser


def main() -> None:
    """Compatibility entrypoint that dispatches through the registry with config overrides."""
    parser = _build_parser()
    args = parser.parse_args()
    warn_deprecated(
        api="python -m src.train",
        replacement="python train.py --config ...",
        stacklevel=2,
    )

    overrides = {
        "training": {"data_root": str(args.data_root)},
        "model": {},
    }
    if args.device is not None:
        overrides["training"]["device"] = args.device
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if args.img_size is not None:
        overrides["training"]["img_size"] = args.img_size
        overrides["model"]["resolution"] = args.img_size
    if args.in_channels is not None:
        overrides["model"]["in_channels"] = args.in_channels
    if args.out_channels is not None:
        overrides["model"]["out_channels"] = args.out_channels
    if args.perceptual_device is not None:
        overrides["training"]["perceptual_device"] = args.perceptual_device
    if args.gan_device is not None:
        overrides["training"]["disc_device"] = args.gan_device

    cfg = load_json_config(args.config)
    cfg = _merge_overrides(cfg, overrides)
    cfg.setdefault("model", {})
    configured_model_type = str(cfg["model"].get("model_type", "")).strip().lower()
    requested_trainer = str(args.trainer).strip().lower()
    if configured_model_type and configured_model_type != requested_trainer:
        warnings.warn(
            f"`src.train --trainer {requested_trainer}` overrides config model_type "
            f"`{configured_model_type}` for compatibility.",
            DeprecationWarning,
            stacklevel=2,
        )
    cfg["model"]["model_type"] = requested_trainer

    train_ds, val_ds = build_train_val_datasets(cfg)
    trainer_cls = TRAINER_REGISTRY.get(requested_trainer)
    trainer = trainer_cls.from_config(cfg)
    trainer.fit(train_ds, val_dataset=val_ds)


if __name__ == "__main__":
    main()
