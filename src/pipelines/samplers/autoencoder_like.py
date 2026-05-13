"""
Sampling, encoding, decoding, and evaluation for autoencoder-style models (VAE now).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
import json

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
    save_tensor_cache: bool = False,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "vae")

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(
        cfg, data_txt, evaluate=True, save_tensor_cache_override=save_tensor_cache
    )
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)
    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    for indices, samples in progress_batches(dataset, batch_size, "Autoencoder encode", indices=selected_indices):
        inputs = torch.stack([s["target"] for s in samples], dim=0).to(device)
        with torch.no_grad():
            latents = encode_vae_batch(model, inputs)
        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, latents[batch_idx].cpu(), output_root)

    logging.info("Autoencoder encode completed for %d samples.", len(selected_indices))


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
    save_tensor_cache: bool = False,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "vae")

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt, save_tensor_cache_override=save_tensor_cache)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)
    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    for indices, samples in progress_batches(dataset, batch_size, "Autoencoder decode", indices=selected_indices):
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

    logging.info("Autoencoder decode completed for %d samples.", len(selected_indices))


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
    save_tensor_cache: bool = False,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "vae")

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt, save_tensor_cache_override=save_tensor_cache)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)
    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    for indices, samples in progress_batches(dataset, batch_size, "Autoencoder sample", indices=selected_indices):
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

    logging.info("Autoencoder sample completed for %d samples.", len(selected_indices))


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
    save_tensor_cache: bool = False,
) -> None:
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

    dataset = build_sampling_dataset(
        cfg, data_txt, evaluate=True, save_tensor_cache_override=save_tensor_cache
    )
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)
    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    ssim_count = 0
    model_seconds = 0.0
    model_calls = 0
    per_image_rows: list[dict] = []

    batch_iter = progress_batches(dataset, batch_size, "Autoencoder evaluate", indices=selected_indices)
    for indices, samples in batch_iter:
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
        psnr_values = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
        total_mse += mse.sum().item()
        total_psnr += torch.sum(psnr_values).item()
        ssim_values = [None] * recon.size(0)
        if ssim is not None:
            for idx in range(recon.size(0)):
                value = compute_ssim_sample(recon[idx], targets[idx], ssim)
                if value is not None:
                    total_ssim += value
                    ssim_count += 1
                    ssim_values[idx] = value
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
        if hasattr(batch_iter, "set_postfix"):
            running = {
                "mse": f"{(total_mse / max(count, 1)):.6f}",
                "psnr": f"{(total_psnr / max(count, 1)):.3f}",
                "sps": f"{(count / max(model_seconds, 1e-12)):.3f}",
            }
            if ssim_count > 0:
                running["ssim"] = f"{(total_ssim / ssim_count):.4f}"
            batch_iter.set_postfix(running)

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
    if ssim is not None and ssim_count > 0:
        avg_ssim = total_ssim / ssim_count
        logging.info("Eval SSIM: %.4f", avg_ssim)
        print(f"Eval SSIM: {avg_ssim:.4f}")
    elif ssim is None:
        print("Eval SSIM: unavailable (install scikit-image)")

    metrics_root = Path(output_dir) if output_dir else ckpt_dir
    metrics_path = append_eval_metrics(
        metrics_root,
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
    per_image_metrics_path = append_per_image_eval_metrics(metrics_root, per_image_rows)
    logging.info("Wrote per-image eval metrics: %s", per_image_metrics_path)


def debug_compare(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    output_dir: str | None = None,
    device: str | None = None,
    seed: int = 42,
    num_samples: int | None = None,
    save_tensor_cache: bool = False,
) -> None:
    """
    One-sample VAE debug artifact dump (target/reconstruction/conditioning + stats).
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "vae")
    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(
        cfg, data_txt, evaluate=True, save_tensor_cache_override=save_tensor_cache
    )
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    if not selected_indices:
        raise RuntimeError("No samples available for debug_compare.")
    sample_idx = int(selected_indices[0])
    sample = dataset[sample_idx]
    row = dataset.data[sample_idx]

    model = build_vae_model(cfg, device, ckpt_path=ckpt_path)
    target = sample["target"].unsqueeze(0).to(device)
    conditioning = sample.get("image")
    with torch.no_grad():
        sync_if_cuda(device)
        start = time.perf_counter()
        generated = reconstruct_vae_batch(model, target, recon_type=cfg.get("training", {}).get("recon_type", "l1"))
        sync_if_cuda(device)
        model_seconds = time.perf_counter() - start

    debug_root = Path(output_dir) if output_dir else (ckpt_dir / "debug_compare")
    debug_root.mkdir(parents=True, exist_ok=True)

    torch.save(target.detach().cpu(), debug_root / "target.pt")
    torch.save(generated.detach().cpu(), debug_root / "generated.pt")
    if conditioning is not None:
        torch.save(conditioning.detach().cpu(), debug_root / "conditioning.pt")

    save_output_tensor(dataset, row, dataset.target_key, target[0].detach().cpu(), debug_root / "target")
    save_output_tensor(dataset, row, dataset.target_key, generated[0].detach().cpu(), debug_root / "generated")
    if conditioning is not None and dataset.conditioning_key is not None:
        save_output_tensor(dataset, row, dataset.conditioning_key, conditioning.detach().cpu(), debug_root / "conditioning_export")

    t = target.detach().float().cpu()
    g = generated.detach().float().cpu()
    c = conditioning.detach().float().cpu() if conditioning is not None else None
    stats = {
        "model_type": "vae",
        "sample_index": sample_idx,
        "img_id": sample.get("img_id"),
        "img_path": sample.get("img_path"),
        "timing": {"model_seconds": model_seconds, "model_calls": 1},
        "target": {"shape": list(t.shape), "min": float(t.min()), "max": float(t.max()), "mean": float(t.mean()), "std": float(t.std())},
        "generated": {"shape": list(g.shape), "min": float(g.min()), "max": float(g.max()), "mean": float(g.mean()), "std": float(g.std())},
        "conditioning": None
        if c is None
        else {"shape": list(c.shape), "min": float(c.min()), "max": float(c.max()), "mean": float(c.mean()), "std": float(c.std())},
    }
    with (debug_root / "stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)

    logging.info("Autoencoder debug compare completed. Artifacts written to: %s", debug_root)
    print(f"Autoencoder debug compare completed. Artifacts written to: {debug_root}")
