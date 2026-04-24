"""
Sampling, encoding, decoding, and evaluation for VAE models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

import utils
from utils.dataset_utils import iter_batches, save_output_tensor
from utils.model_utils.vae_utils import build_vae_model, decode_vae_batch, encode_vae_batch, reconstruct_vae_batch
from utils.sampling_utils import (
    build_sampling_dataset,
    load_run_config,
    resolve_checkpoint,
    resolve_output_root,
)


def encode(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
    timestep: int | None = None,
) -> None:
    """
    Encode inputs into latent representations.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "vae")

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    for indices, samples in iter_batches(dataset, batch_size):
        inputs = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            latents = encode_vae_batch(model, inputs)
        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, latents[batch_idx].cpu(), output_root)

    logging.info("VAE encode completed for %d samples.", len(dataset))


def decode(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
) -> None:
    """
    Decode latent representations into images.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "vae")

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    for indices, samples in iter_batches(dataset, batch_size):
        latents = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            recon = decode_vae_batch(model, latents, recon_type=cfg.get("training", {}).get("recon_type", "l1"))
        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, recon[batch_idx].cpu(), output_root)

    logging.info("VAE decode completed for %d samples.", len(dataset))


def sample(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
) -> None:
    """
    Reconstruct inputs using encode+decode.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "vae")

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    for indices, samples in iter_batches(dataset, batch_size):
        inputs = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            recon = reconstruct_vae_batch(model, inputs, recon_type=cfg.get("training", {}).get("recon_type", "l1"))
        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, recon[batch_idx].cpu(), output_root)

    logging.info("VAE sample completed for %d samples.", len(dataset))


def evaluate(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
) -> None:
    """
    Evaluate reconstructions against targets using MSE/PSNR (SSIM if available).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:  # pragma: no cover - optional
        ssim = None

    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "vae")

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for indices, samples in iter_batches(dataset, batch_size):
        inputs = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            recon = reconstruct_vae_batch(model, inputs, recon_type=cfg.get("training", {}).get("recon_type", "l1"))
        targets = inputs

        mse = torch.mean((recon - targets) ** 2, dim=(1, 2, 3))
        total_mse += mse.sum().item()
        total_psnr += torch.sum(10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))).item()
        if ssim is not None:
            for idx in range(recon.size(0)):
                gen_np = recon[idx].detach().cpu().numpy().transpose(1, 2, 0)
                tgt_np = targets[idx].detach().cpu().numpy().transpose(1, 2, 0)
                total_ssim += float(ssim(gen_np, tgt_np, channel_axis=2, data_range=1.0))
        count += recon.size(0)

    if count == 0:
        raise RuntimeError("No samples available for evaluation.")

    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    logging.info("Eval MSE: %.6f | PSNR: %.3f", avg_mse, avg_psnr)
    if ssim is not None:
        logging.info("Eval SSIM: %.4f", total_ssim / count)
