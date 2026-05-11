"""
Sampling, encoding, decoding, and evaluation for VAE models.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

import utils
from pipelines.utils import sync_if_cuda
from utils.dataset_utils import save_output_tensor
from utils.evaluation_utils import compute_ssim_sample
from utils.model_utils.vae_utils import build_vae_model, decode_vae_batch, encode_vae_batch, reconstruct_vae_batch
from utils.sampling_utils import (
    append_eval_metrics,
    append_per_image_eval_metrics,
    build_sampling_dataset,
    load_run_config,
    progress_batches,
    resolve_sample_indices,
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
    num_samples: int | None = None,
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
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    for indices, samples in progress_batches(dataset, batch_size, "VAE encode", indices=selected_indices):
        inputs = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            latents = encode_vae_batch(model, inputs)
        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, latents[batch_idx].cpu(), output_root)

    logging.info("VAE encode completed for %d samples.", len(selected_indices))


def decode(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
    num_samples: int | None = None,
    save_input: bool = False,
    save_conditioning: bool = False,
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
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    for indices, samples in progress_batches(dataset, batch_size, "VAE decode", indices=selected_indices):
        latents = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            recon = decode_vae_batch(model, latents, recon_type=cfg.get("training", {}).get("recon_type", "l1"))
        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, recon[batch_idx].cpu(), output_root)
                if save_input:
                    save_output_tensor(dataset, row, dataset.target_key, samples[batch_idx]["target"], output_root / "input")
                if save_conditioning and dataset.conditioning_key is not None:
                    save_output_tensor(dataset, row, dataset.conditioning_key, samples[batch_idx]["image"], output_root / "conditioning")

    logging.info("VAE decode completed for %d samples.", len(selected_indices))


def sample(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
    num_samples: int | None = None,
    save_input: bool = False,
    save_conditioning: bool = False,
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
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    for indices, samples in progress_batches(dataset, batch_size, "VAE sample", indices=selected_indices):
        inputs = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            recon = reconstruct_vae_batch(model, inputs, recon_type=cfg.get("training", {}).get("recon_type", "l1"))
        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, recon[batch_idx].cpu(), output_root)
                if save_input:
                    save_output_tensor(dataset, row, dataset.target_key, samples[batch_idx]["target"], output_root / "input")
                if save_conditioning and dataset.conditioning_key is not None:
                    save_output_tensor(dataset, row, dataset.conditioning_key, samples[batch_idx]["image"], output_root / "conditioning")

    logging.info("VAE sample completed for %d samples.", len(selected_indices))


def evaluate(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
    num_samples: int | None = None,
    save_input: bool = False,
    save_conditioning: bool = False,
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
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)
    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    ssim_count = 0

    for indices, samples in progress_batches(dataset, batch_size, "VAE evaluate", indices=selected_indices):
        inputs = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            sync_if_cuda(device)
            start = time.perf_counter()
            recon = reconstruct_vae_batch(model, inputs, recon_type=cfg.get("training", {}).get("recon_type", "l1"))
            sync_if_cuda(device)
            model_seconds += time.perf_counter() - start
            model_calls += 1
        targets = inputs

        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, recon[batch_idx].cpu(), output_root)
                if save_input:
                    save_output_tensor(dataset, row, dataset.target_key, samples[batch_idx]["target"], output_root / "input")
                if save_conditioning and dataset.conditioning_key is not None:
                    save_output_tensor(dataset, row, dataset.conditioning_key, samples[batch_idx]["image"], output_root / "conditioning")

        reduce_dims = tuple(range(1, recon.ndim))
        mse = torch.mean((recon - targets) ** 2, dim=reduce_dims)
        total_mse += mse.sum().item()
        total_psnr += torch.sum(psnr_values).item()
        ssim_values = [None] * recon.size(0)
        if ssim is not None:
            for idx in range(recon.size(0)):
                value = compute_ssim_sample(recon[idx], targets[idx], ssim)
                if value is not None:
                    total_ssim += value
                    ssim_count += 1
        count += recon.size(0)

    if count == 0:
        raise RuntimeError("No samples available for evaluation.")

    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    model_sps = count / model_seconds if model_seconds > 0 else 0.0
    model_s_per_sample = model_seconds / count if count else 0.0
    logging.info("Eval MSE: %.6f | PSNR: %.3f", avg_mse, avg_psnr)
    if ssim is not None and ssim_count > 0:
        logging.info("Eval SSIM: %.4f", total_ssim / ssim_count)
