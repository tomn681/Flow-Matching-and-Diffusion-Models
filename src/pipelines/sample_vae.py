"""
Utility script to sample generations and reconstructions from a trained AutoencoderKL run.

Usage:
    python -m src.pipelines.sample_vae --run-name vae_run1

If --run-name is omitted, the newest run directory under checkpoints/ is selected.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ..models import AutoencoderKL
from ..utils import DefaultDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VAE samples and reconstructions.")
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory containing VAE training runs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the run directory under checkpoints/. If omitted pick latest.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of images per grid (must be a square number, default 25 → 5x5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for noise sampling.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=("best", "last", "both"),
        default="best",
        help="Select which checkpoint to sample: best, last, or both.",
    )
    return parser.parse_args()


def resolve_run_directory(root: Path, run_name: str | None) -> Path:
    if run_name is not None:
        run_dir = root / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_training_config(run_dir: Path) -> dict:
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing train_config.json in {run_dir}")
    with config_path.open("r") as fh:
        return json.load(fh)


def select_checkpoints(run_dir: Path, preference: str) -> list[tuple[str, Path]]:
    best = sorted(run_dir.glob("vae_best_epoch*.pt"))
    ckpts = sorted(run_dir.glob("vae_ckpt_epoch*.pt"))
    generic = sorted(run_dir.glob("*.pt"))

    best_path = best[-1] if best else None
    last_ckpt = ckpts[-1] if ckpts else None
    last_any = generic[-1] if generic else None

    def ensure(path: Path | None, label: str) -> list[tuple[str, Path]]:
        if path is None:
            raise FileNotFoundError(f"No checkpoint ({label}) found in {run_dir}")
        return [(label, path)]

    if preference == "best":
        return ensure(best_path or last_ckpt or last_any, "best")
    if preference == "last":
        return ensure(last_ckpt or last_any, "last")

    # preference == "both"
    selected: list[tuple[str, Path]] = []
    if best_path is not None:
        selected.append(("best", best_path))
    if last_ckpt is not None and (not selected or last_ckpt != selected[-1][1]):
        selected.append(("last", last_ckpt))
    elif last_any is not None and (not selected or last_any != selected[-1][1]):
        selected.append(("last", last_any))
    if not selected:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return selected


def suffix_for(label: str) -> str:
    return f"_{label}"
    ckpts = sorted(run_dir.glob("vae_ckpt_epoch*.pt"))
    if ckpts:
        return ckpts[-1]
    generic = sorted(run_dir.glob("*.pt"))
    if generic:
        return generic[-1]
    raise FileNotFoundError(f"No checkpoint (*.pt) found in {run_dir}")


def build_model(model_cfg: dict, checkpoint: Path, device: torch.device) -> AutoencoderKL:
    model = AutoencoderKL(**model_cfg).to(device)
    payload = torch.load(checkpoint, map_location=device)
    if "model" in payload:
        model.load_state_dict(payload["model"])
    else:
        model.load_state_dict(payload)
    model.eval()
    return model


def gather_validation_batch(cfg_args: dict, samples: int) -> torch.Tensor:
    dataset = DefaultDataset(
        str(cfg_args["data_root"]),
        s_cnt=cfg_args.get("slice_count", 1),
        img_size=cfg_args.get("img_size", 256),
        train=False,
        diff=cfg_args.get("diff", True),
        norm=cfg_args.get("norm", True),
    )
    count = min(len(dataset), samples)
    tensors = [dataset[i]["target"] for i in range(count)]
    if not tensors:
        raise RuntimeError("Validation dataset appears empty.")
    return torch.stack(tensors, dim=0)


def latent_shape(model_cfg: dict) -> Tuple[int, ...]:
    spatial_dims = model_cfg.get("spatial_dims", 2)
    embed_dim = model_cfg["embed_dim"]
    resolution = model_cfg["resolution"]
    ch_mult = tuple(model_cfg["ch_mult"])
    factor = 2 ** (len(ch_mult) - 1)
    base_size = resolution // factor
    if spatial_dims == 2:
        return (embed_dim, base_size, base_size)
    if spatial_dims == 1:
        return (embed_dim, base_size)
    if spatial_dims == 3:
        return (embed_dim, base_size, base_size, base_size)
    raise ValueError(f"Unsupported spatial_dims={spatial_dims}")


def build_grid(tensor: torch.Tensor, grid_size: Tuple[int, int]) -> np.ndarray:
    n, c, *spatial = tensor.shape
    rows, cols = grid_size
    expected = rows * cols
    if n < expected:
        raise ValueError(f"Need at least {expected} images to build the grid, found {n}")

    tensor = tensor[:expected]
    if c == 1:
        tensor = tensor.expand(-1, 3, *spatial)  # repeat to RGB for nicer viewing
        c = 3

    # Normalise from [-1, 1] to [0, 1] and clamp
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0

    h = spatial[-2] if len(spatial) >= 2 else spatial[-1]
    w = spatial[-1]

    tensor = tensor.reshape(rows, cols, c, h, w)
    tensor = tensor.permute(2, 0, 3, 1, 4).contiguous()
    grid = tensor.reshape(c, rows * h, cols * w)

    grid_np = (grid.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    grid_np = np.transpose(grid_np, (1, 2, 0))
    return grid_np


def save_grid_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    logging.info("Saved %s", path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    run_dir = resolve_run_directory(args.checkpoints_root, args.run_name)
    logging.info("Using run directory: %s", run_dir)

    cfg = load_training_config(run_dir)
    cfg_args = cfg["args"]
    model_cfg = cfg["model"]

    # Prepare validation batch
    val_batch = gather_validation_batch(cfg_args, args.samples).to(device)
    val_batch = val_batch * 2.0 - 1.0  # match training normalisation
    side = int(args.samples ** 0.5)
    if side * side != args.samples:
        raise ValueError("Number of samples must be a perfect square (e.g., 25).")

    # Fixed latent samples for reproducibility across checkpoints
    latent_shape_ = latent_shape(model_cfg)
    noise = torch.randn((args.samples, *latent_shape_), device=device)

    checkpoints = select_checkpoints(run_dir, args.checkpoint)

    for label, checkpoint_path in checkpoints:
        logging.info("Loading checkpoint (%s): %s", label, checkpoint_path)
        model = build_model(model_cfg, checkpoint_path, device)

        # Reconstructions
        with torch.no_grad():
            recon, _ = model(val_batch, sample_posterior=False)
        recon_grid = build_grid(recon, (side, side))
        save_grid_image(recon_grid, run_dir / f"vae_recon_grid{suffix_for(label)}.png")

        # Generations
        with torch.no_grad():
            generated = model.decode(noise)
        gen_grid = build_grid(generated, (side, side))
        save_grid_image(gen_grid, run_dir / f"vae_generated_grid{suffix_for(label)}.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
