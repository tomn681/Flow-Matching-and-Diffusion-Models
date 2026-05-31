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
import logging
import sys
from pathlib import Path
from typing import Callable

import torch

# Ensure local `src` package is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pipelines.train.flow_matching_lib import train as train_flow_matching
from pipelines.train.flow_matching_lib import debug_visual_only as flow_debug_visual_only
from pipelines.train.diffusion_lib import train as train_diffusion
from pipelines.train.diffusion_lib import debug_visual_only as diffusion_debug_visual_only
from training import TRAINER_REGISTRY
from utils import build_train_val_datasets, load_json_config
from datasets import LatentCacheDataset
from models.autoencoder.utils import encode_to_latent
from models.factory import ModelFactory

TRAINERS: dict[str, Callable] = {
    "vae": lambda dataset, json_path, val_dataset=None, resume=None: _train_via_registry(
        "vae", dataset, json_path, val_dataset=val_dataset, resume=resume
    ),
    "unet": lambda dataset, json_path, val_dataset=None, resume=None: _train_via_registry(
        "unet", dataset, json_path, val_dataset=val_dataset, resume=resume
    ),
    "flow_matching": train_flow_matching,
    "diffusion": train_diffusion,
    "consistency": lambda dataset, json_path, val_dataset=None, resume=None: _train_via_registry(
        "consistency", dataset, json_path, val_dataset=val_dataset, resume=resume
    ),
    "edm": lambda dataset, json_path, val_dataset=None, resume=None: _train_via_registry(
        "edm", dataset, json_path, val_dataset=val_dataset, resume=resume
    ),
    "rectified_flow": lambda dataset, json_path, val_dataset=None, resume=None: _train_via_registry(
        "rectified_flow", dataset, json_path, val_dataset=val_dataset, resume=resume
    ),
    "gan": lambda dataset, json_path, val_dataset=None, resume=None: _train_via_registry(
        "gan", dataset, json_path, val_dataset=val_dataset, resume=resume
    ),
    "latent_diffusion": train_diffusion,
    "latent_flow_matching": train_flow_matching,
    "latent_rectified_flow": lambda dataset, json_path, val_dataset=None, resume=None: _train_via_registry(
        "latent_rectified_flow", dataset, json_path, val_dataset=val_dataset, resume=resume
    ),
    "reflow": lambda dataset, json_path, val_dataset=None, resume=None: _train_via_registry(
        "reflow", dataset, json_path, val_dataset=val_dataset, resume=resume
    ),
}


def _train_via_registry(
    trainer_key: str,
    dataset,
    json_path: Path | str,
    *,
    val_dataset=None,
    resume: str | None = None,
) -> None:
    cfg = load_json_config(json_path)
    trainer = TRAINER_REGISTRY.get(trainer_key).from_config(cfg)
    trainer.fit(dataset, val_dataset=val_dataset, resume=resume)


def dispatch_train(cfg_path: Path, resume: str | None) -> None:
    cfg = load_json_config(cfg_path)
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("model_type", "")).lower()
    trainer = TRAINERS.get(model_type)
    if trainer is None:
        available = ", ".join(TRAINERS.keys())
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected one of {{{available}}}.")
    use_presaved_latents = bool(model_cfg.get("use_presaved_latents", False))
    if model_type in {"latent_diffusion", "latent_flow_matching", "latent_rectified_flow"} and use_presaved_latents:
        latent_cache_dir = model_cfg.get("latent_cache_dir")
        if not latent_cache_dir:
            raise ValueError("Presaved latent training requires model.latent_cache_dir.")
        train_ds = LatentCacheDataset(latent_cache_dir, split="train")
        val_ds = LatentCacheDataset(latent_cache_dir, split="val")
    else:
        train_ds, val_ds = build_train_val_datasets(cfg)
    trainer(train_ds, cfg_path, val_dataset=val_ds, resume=resume)


def _load_frozen_vae_from_cfg(cfg: dict, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    vae_cfg = dict(model_cfg.get("vae", {}))
    if not vae_cfg:
        raise ValueError("--mode encode_latents requires config.model.vae to build the VAE architecture.")
    vae_cfg["model_type"] = "vae"
    vae_cfg.setdefault("latent_type", "kl")

    ckpt_path = model_cfg.get("vae_checkpoint")
    if not ckpt_path:
        raise ValueError("--mode encode_latents requires config.model.vae_checkpoint.")

    vae = ModelFactory.build({"model": vae_cfg}).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    vae.load_state_dict(state)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def encode_latents_from_config(cfg_path: Path) -> None:
    cfg = load_json_config(cfg_path)
    train_ds, val_ds = build_train_val_datasets(cfg)
    cfg_training = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    batch_size = int(cfg_training.get("batch_size", 4))
    num_workers = int(cfg_training.get("num_workers", 0))
    manual_device = cfg_training.get("manual_device")
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(manual_device) if manual_device else default_device

    cache_dir_raw = model_cfg.get("latent_cache_dir")
    if not cache_dir_raw:
        raise ValueError("--mode encode_latents requires config.model.latent_cache_dir.")
    cache_dir = Path(cache_dir_raw)
    train_cache = cache_dir / "train"
    val_cache = cache_dir / "val"
    train_cache.mkdir(parents=True, exist_ok=True)
    val_cache.mkdir(parents=True, exist_ok=True)

    vae = _load_frozen_vae_from_cfg(cfg, device)

    def _run_split(dataset, out_dir: Path, split_name: str) -> int:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        index = 0
        with torch.no_grad():
            for batch in loader:
                target = encode_to_latent(vae, batch["target"].to(device)).cpu()
                image = batch.get("image")
                image_latent = encode_to_latent(vae, image.to(device)).cpu() if image is not None else None
                for b in range(target.size(0)):
                    payload = {"target": target[b]}
                    if image_latent is not None:
                        payload["image"] = image_latent[b]
                    torch.save(payload, out_dir / f"{index:08d}.pt")
                    index += 1
        logging.info("Encoded %d %s samples to %s", index, split_name, out_dir)
        return index

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    train_n = _run_split(train_ds, train_cache, "train")
    val_n = _run_split(val_ds, val_cache, "val")
    logging.info("Latent encoding complete. train=%d val=%d root=%s", train_n, val_n, cache_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train models from JSON configs.")
    parser.add_argument("--mode", type=str, default="train", choices=("train", "encode_latents"), help="Execution mode.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from (optional).")
    parser.add_argument("--debug_visual_only", action="store_true", help="Diffusion-only: load checkpoint and save visual generations without training.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for --debug_visual_only.")
    parser.add_argument("--visual_samples", type=int, default=10, help="Number of samples for --debug_visual_only.")
    parser.add_argument("--debug_split", type=str, choices=("train", "test"), default="test", help="Split used by --debug_visual_only.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory override for --debug_visual_only.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override for --debug_visual_only.")
    args = parser.parse_args() if argv is None else parser.parse_args(argv)

    if args.mode == "encode_latents":
        if args.debug_visual_only:
            raise ValueError("--debug_visual_only cannot be combined with --mode encode_latents.")
        if args.resume is not None:
            raise ValueError("--resume is not used with --mode encode_latents.")
        encode_latents_from_config(args.config)
        return

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
            from pipelines.train.vae_lib import debug_visual_only as vae_debug_visual_only

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
