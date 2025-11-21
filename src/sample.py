"""
Generic sampler dispatcher.

Usage:
    python -m src.sample --sampler vae --checkpoints-root checkpoints --run-name my_run
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def main() -> None:
    """Dispatch to a sampler under src.pipelines.samplers."""
    parser = argparse.ArgumentParser(description="Dispatch sampling to a specific model sampler.")
    parser.add_argument("--sampler", type=str, required=True, help="Sampler name under src.pipelines.samplers (e.g., 'vae').")
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"), help="Root directory containing run folders.")
    parser.add_argument("--run-name", type=str, default=None, help="Run directory name under checkpoints root (defaults to latest).")
    parser.add_argument("--samples", type=int, default=25, help="Images per grid (must be a perfect square).")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--checkpoint", type=str, choices=("best", "last", "both"), default="best", help="Which checkpoint(s) to sample.")
    args = parser.parse_args()

    module = importlib.import_module(f"pipelines.samplers.{args.sampler}")
    if not hasattr(module, "sample"):
        raise AttributeError(f"Sampler '{args.sampler}' does not expose a sample(...) function.")

    module.sample(
        checkpoints_root=args.checkpoints_root,
        run_name=args.run_name,
        samples=args.samples,
        device=args.device,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
