"""
Generic trainer entrypoint.

Usage:
    python -m src.train --trainer vae --config configs/vae.json --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def main() -> None:
    """Dispatch to a trainer under src.pipelines.train with optional overrides."""
    parser = argparse.ArgumentParser(description="Dispatch training to a specific model trainer.")
    parser.add_argument("--trainer", type=str, required=True, help="Trainer name under src.pipelines.train (e.g., 'vae').")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--img-size", type=int, default=None, help="Override image size/resolution.")
    parser.add_argument("--in-channels", type=int, default=None, help="Override VAE input channels.")
    parser.add_argument("--out-channels", type=int, default=None, help="Override VAE output channels.")
    parser.add_argument("--perceptual-device", type=str, default=None, help="Optional device for perceptual loss (e.g., cuda:1).")
    parser.add_argument("--gan-device", type=str, default=None, help="Optional device for discriminator (e.g., cuda:1).")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="Optional micro batch size for gradient accumulation.")
    args = parser.parse_args()

    module = importlib.import_module(f"pipelines.train.{args.trainer}")
    if not hasattr(module, "train"):
        raise AttributeError(f"Trainer '{args.trainer}' does not expose a train(config_path, data_root) function.")

    overrides = {
        "training": {},
        "vae": {},
    }
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if args.img_size is not None:
        overrides["training"]["img_size"] = args.img_size
        overrides["vae"]["resolution"] = args.img_size
    if args.in_channels is not None:
        overrides["vae"]["in_channels"] = args.in_channels
    if args.out_channels is not None:
        overrides["vae"]["out_channels"] = args.out_channels
    if args.perceptual_device is not None:
        overrides["training"]["perceptual_device"] = args.perceptual_device
    if args.gan_device is not None:
        overrides["training"]["disc_device"] = args.gan_device
    if args.micro_batch_size is not None:
        overrides["training"]["micro_batch_size"] = args.micro_batch_size

    module.train(args.config, args.data_root, overrides=overrides)


if __name__ == "__main__":
    main()
