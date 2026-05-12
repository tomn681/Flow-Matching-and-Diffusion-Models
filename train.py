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
from pipelines.train.vae_lib import debug_visual_only as vae_debug_visual_only
from pipelines.train.flow_matching_lib import train as train_flow_matching
from pipelines.train.flow_matching_lib import debug_visual_only as flow_debug_visual_only
from pipelines.train.diffusion_lib import train as train_diffusion
from pipelines.train.diffusion_lib import debug_visual_only as diffusion_debug_visual_only
from utils import build_train_val_datasets, load_json_config

TRAINERS: dict[str, Callable] = {
    "vae": train_vae,
    "flow_matching": train_flow_matching,
    "diffusion": train_diffusion,
}


def dispatch_train(cfg_path: Path, resume: str | None) -> None:
    cfg = load_json_config(cfg_path)
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("model_type", "")).lower()
    trainer = TRAINERS.get(model_type)
    if trainer is None:
        available = ", ".join(TRAINERS.keys())
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of {{{available}}}.")
    train_ds, val_ds = build_train_val_datasets(cfg)
    trainer(train_ds, cfg_path, val_dataset=val_ds, resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models from JSON configs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from (optional).")
    parser.add_argument("--debug_visual_only", action="store_true", help="Diffusion-only: load checkpoint and save visual generations without training.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for --debug_visual_only.")
    parser.add_argument("--visual_samples", type=int, default=10, help="Number of samples for --debug_visual_only.")
    parser.add_argument("--debug_split", type=str, choices=("train", "test"), default="test", help="Split used by --debug_visual_only.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory override for --debug_visual_only.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override for --debug_visual_only.")
    args = parser.parse_args()
    if args.debug_visual_only:
        cfg = load_json_config(args.config)
        model_type = str(cfg.get("model", {}).get("model_type", "")).lower()
        if not args.ckpt:
            raise ValueError("--ckpt is required when using --debug_visual_only.")
        train_ds, val_ds = build_train_val_datasets(cfg)
        ds = train_ds if args.debug_split == "train" else val_ds
        if model_type == "diffusion":
            diffusion_debug_visual_only(
                ds,
                args.config,
                args.ckpt,
                output_dir=args.output_dir,
                visual_samples=args.visual_samples,
                seed=args.seed,
            )
        elif model_type == "flow_matching":
            flow_debug_visual_only(
                ds,
                args.config,
                args.ckpt,
                output_dir=args.output_dir,
                visual_samples=args.visual_samples,
                seed=args.seed,
            )
        elif model_type == "vae":
            vae_debug_visual_only(
                ds,
                args.config,
                args.ckpt,
                output_dir=args.output_dir,
                visual_samples=args.visual_samples,
                seed=args.seed,
            )
        else:
            raise ValueError(f"--debug_visual_only unsupported model_type '{model_type}'.")
        return
    dispatch_train(args.config, args.resume)


if __name__ == "__main__":
    main()
