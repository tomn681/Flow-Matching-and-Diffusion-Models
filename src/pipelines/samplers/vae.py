"""
Sampler for VAE runs: generates reconstruction and random grids from checkpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from PIL import Image

from models import AutoencoderKL
from utils import build_dataset_from_config


def resolve_run_directory(root: Path, run_name: str | None) -> Path:
    """Pick a run directory under the checkpoints root (latest if not provided)."""
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
    """Load the saved training config from a run directory."""
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing train_config.json in {run_dir}")
    with config_path.open("r") as fh:
        return json.load(fh)


def select_checkpoints(run_dir: Path, preference: str) -> list[tuple[str, Path]]:
    """Choose checkpoints (best/last/both)."""
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


def build_model(model_cfg: dict, checkpoint: Path, device: torch.device) -> AutoencoderKL:
    """Instantiate and load the VAE from a checkpoint."""
    model = AutoencoderKL(**model_cfg).to(device)
    payload = torch.load(checkpoint, map_location=device)
    if "model" in payload:
        model.load_state_dict(payload["model"])
    else:
        model.load_state_dict(payload)
    model.eval()
    return model


def gather_validation_batch(training_cfg: dict, vae_cfg: dict, samples: int) -> torch.Tensor:
    """Collect a small batch from the validation dataset for reconstruction."""
    dataset = build_dataset_from_config(training_cfg, vae_cfg, train=False)
    count = min(len(dataset), samples)
    tensors = [dataset[i]["target"] for i in range(count)]
    if not tensors:
        raise RuntimeError("Validation dataset appears empty.")
    return torch.stack(tensors, dim=0)


def latent_shape(model_cfg: dict) -> Tuple[int, ...]:
    """Infer latent spatial shape given the training model config."""
    spatial_dims = model_cfg.get("spatial_dims", 2)
    embed_dim = model_cfg["embed_dim"]
    resolution = model_cfg["resolution"]
    ch_mult: Iterable[int] = model_cfg["ch_mult"]
    factor = 2 ** (len(tuple(ch_mult)) - 1)
    base_size = resolution // factor
    if spatial_dims == 2:
        return (embed_dim, base_size, base_size)
    if spatial_dims == 1:
        return (embed_dim, base_size)
    if spatial_dims == 3:
        return (embed_dim, base_size, base_size, base_size)
    raise ValueError(f"Unsupported spatial_dims={spatial_dims}")


def build_grid(tensor: torch.Tensor, grid_size: Tuple[int, int]) -> np.ndarray:
    """Turn a batch of images in [-1,1] into a single grid uint8 array."""
    n, c, *spatial = tensor.shape
    rows, cols = grid_size
    expected = rows * cols
    if n < expected:
        raise ValueError(f"Need at least {expected} images to build the grid, found {n}")

    tensor = tensor[:expected]
    if c == 1:
        tensor = tensor.expand(-1, 3, *spatial)  # repeat to RGB
        c = 3

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
    """Save a numpy image array as PNG and log the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    logging.info("Saved %s", path)


def sample(
    checkpoints_root: Path | str,
    run_name: str | None = None,
    samples: int = 25,
    device: str | torch.device = "cuda",
    seed: int = 42,
    checkpoint: str = "best",
) -> None:
    """Generate reconstructions and random samples for a VAE run."""
    if isinstance(device, str):
        device = torch.device(device)
    checkpoints_root = Path(checkpoints_root)
    torch.manual_seed(seed)

    run_dir = resolve_run_directory(checkpoints_root, run_name)
    logging.info("Using run directory: %s", run_dir)

    cfg = load_training_config(run_dir)
    training_cfg = cfg["training"]
    model_cfg = cfg["vae"]

    val_batch = gather_validation_batch(training_cfg, model_cfg, samples).to(device)
    val_batch = val_batch * 2.0 - 1.0
    side = int(samples ** 0.5)
    if side * side != samples:
        raise ValueError("Number of samples must be a perfect square (e.g., 25).")

    input_grid = build_grid(val_batch, (side, side))
    save_grid_image(input_grid, run_dir / "vae_input_grid.png")

    latent_shape_ = latent_shape(model_cfg)
    noise = torch.randn((samples, *latent_shape_), device=device)

    checkpoints = select_checkpoints(run_dir, checkpoint)

    for label, checkpoint_path in checkpoints:
        logging.info("Loading checkpoint (%s): %s", label, checkpoint_path)
        model = build_model(model_cfg, checkpoint_path, device)

        with torch.no_grad():
            outputs = model(val_batch, sample_posterior=False)
            recon = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        recon_grid = build_grid(recon, (side, side))
        save_grid_image(recon_grid, run_dir / f"vae_recon_grid_{label}.png")

        with torch.no_grad():
            generated = model.decode(noise)
        gen_grid = build_grid(generated, (side, side))
        save_grid_image(gen_grid, run_dir / f"vae_generated_grid_{label}.png")


__all__ = ["sample"]
