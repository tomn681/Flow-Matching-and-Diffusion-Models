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
from utils.model_utils.vae_utils import build_vae_model, decode_vae_batch, encode_vae_batch, reconstruct_vae_batch
from utils.sampling_utils import (
    append_eval_metrics,
    append_per_image_eval_metrics,
    build_sampling_dataset,
    load_run_config,
    progress_batches,
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

    for indices, samples in progress_batches(dataset, batch_size, "VAE encode"):
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

    for indices, samples in progress_batches(dataset, batch_size, "VAE decode"):
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

    for indices, samples in progress_batches(dataset, batch_size, "VAE sample"):
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

    dataset = build_sampling_dataset(cfg, data_txt, evaluate=True)
    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    model_seconds = 0.0
    model_calls = 0
    per_image_rows = []

    for indices, samples in progress_batches(dataset, batch_size, "VAE evaluate"):
        inputs = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            sync_if_cuda(device)
            start = time.perf_counter()
            recon = reconstruct_vae_batch(model, inputs, recon_type=cfg.get("training", {}).get("recon_type", "l1"))
            sync_if_cuda(device)
            model_seconds += time.perf_counter() - start
            model_calls += 1
        targets = inputs

        mse = torch.mean((recon - targets) ** 2, dim=(1, 2, 3))
        psnr_values = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
        total_mse += mse.sum().item()
        total_psnr += torch.sum(psnr_values).item()
        ssim_values = [None] * recon.size(0)
        if ssim is not None:
            for idx in range(recon.size(0)):
                gen_np = recon[idx].detach().cpu().numpy().transpose(1, 2, 0)
                tgt_np = targets[idx].detach().cpu().numpy().transpose(1, 2, 0)
                ssim_values[idx] = float(ssim(gen_np, tgt_np, channel_axis=2, data_range=1.0))
                total_ssim += ssim_values[idx]
        for batch_idx, sample_idx in enumerate(indices):
            sample = samples[batch_idx]
            per_image_rows.append(
                {
                    "sample_index": sample_idx,
                    "img_id": sample.get("img_id"),
                    "img_path": sample.get("img_path"),
                    "mse": f"{mse[batch_idx].item():.8f}",
                    "psnr": f"{psnr_values[batch_idx].item():.6f}",
                    "ssim": "" if ssim_values[batch_idx] is None else f"{ssim_values[batch_idx]:.6f}",
                }
            )
        count += recon.size(0)

    if count == 0:
        raise RuntimeError("No samples available for evaluation.")

    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    model_sps = count / model_seconds if model_seconds > 0 else 0.0
    model_s_per_sample = model_seconds / count if count else 0.0
    logging.info("Eval MSE: %.6f | PSNR: %.3f", avg_mse, avg_psnr)
    print(f"Eval MSE: {avg_mse:.6f} | PSNR: {avg_psnr:.3f}")
    print(
        f"Model throughput: {model_sps:.3f} samples/s | "
        f"{model_s_per_sample:.6f} s/sample | model time {model_seconds:.3f}s"
    )
    avg_ssim = None
    if ssim is not None:
        avg_ssim = total_ssim / count
        logging.info("Eval SSIM: %.4f", avg_ssim)
        print(f"Eval SSIM: {avg_ssim:.4f}")
    else:
        logging.warning("Eval SSIM unavailable because scikit-image is not installed.")
        print("Eval SSIM: unavailable (install scikit-image)")
    metrics_path = append_eval_metrics(
        ckpt_dir,
        {
            "samples": count,
            "mse": f"{avg_mse:.8f}",
            "psnr": f"{avg_psnr:.6f}",
            "ssim": "" if avg_ssim is None else f"{avg_ssim:.6f}",
            "ssim_enabled": ssim is not None,
            "model_seconds": f"{model_seconds:.6f}",
            "model_samples_per_second": f"{model_sps:.6f}",
            "model_seconds_per_sample": f"{model_s_per_sample:.8f}",
            "model_calls": model_calls,
        },
    )
    logging.info("Wrote eval metrics: %s", metrics_path)
    per_image_metrics_path = append_per_image_eval_metrics(ckpt_dir, per_image_rows)
    logging.info("Wrote per-image eval metrics: %s", per_image_metrics_path)
