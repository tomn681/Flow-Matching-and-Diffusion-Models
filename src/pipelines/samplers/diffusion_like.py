"""
Shared sampling/encoding/decoding/evaluation for diffusion-like generators.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

import utils
from pipelines.utils import build_scheduler, resolve_conditioning_mode
from utils.dataset_utils import iter_batches, save_output_tensor
from utils.evaluation_utils import compute_ssim_sample
from utils.model_utils.diffusion_utils import build_diffusion_model, decode_diffusion_batch, encode_diffusion_batch
from utils.sampling_utils import (
    build_sampling_dataset,
    load_run_config,
    resolve_checkpoint,
    resolve_output_root,
    resolve_sample_indices,
)


def _run_encode(
    *,
    ckpt_dir: Path | str,
    model_type: str,
    data_txt: str | None = None,
    save: bool = False,
    output_dir: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
    timestep: int | None = None,
    num_samples: int | None = None,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    scheduler, _ = build_scheduler(model_cfg.get("scheduler", {}), training_cfg)

    for indices, samples in iter_batches(dataset, batch_size, indices=selected_indices):
        targets = torch.stack([s["target"] for s in samples], dim=0).to(device)
        if timestep is None:
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (targets.size(0),), device=device).long()
        else:
            timesteps = torch.full((targets.size(0),), int(timestep), device=device, dtype=torch.long)
        noisy = encode_diffusion_batch(scheduler, targets, timesteps)

        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, noisy[batch_idx].cpu(), output_root)

    logging.info("%s encode completed for %d samples.", model_type.replace("_", "-").title(), len(selected_indices))


def _run_decode(
    *,
    ckpt_dir: Path | str,
    model_type: str,
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
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, model_type)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path)
    conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))

    for indices, samples in iter_batches(dataset, batch_size, indices=selected_indices):
        targets = torch.stack([s["target"] for s in samples], dim=0)
        batch_shape = targets.shape
        cond = None
        if conditioning_mode in {"concatenate", "attention"}:
            cond_list = [s.get("image") for s in samples]
            if all(c is not None for c in cond_list):
                cond = torch.stack(cond_list, dim=0).to(device)
        generated = decode_diffusion_batch(model, training_cfg, model_cfg, device, batch_shape, cond).clamp(0.0, 1.0)

        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, generated[batch_idx].cpu(), output_root)
                if save_input:
                    save_output_tensor(dataset, row, dataset.target_key, samples[batch_idx]["target"], output_root / "input")
                if save_conditioning and dataset.conditioning_key is not None:
                    save_output_tensor(dataset, row, dataset.conditioning_key, samples[batch_idx]["image"], output_root / "conditioning")

    logging.info("%s decode completed for %d samples.", model_type.replace("_", "-").title(), len(selected_indices))


def _run_evaluate(
    *,
    ckpt_dir: Path | str,
    model_type: str,
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
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:  # pragma: no cover - optional
        ssim = None

    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, model_type)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)
    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path)
    conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    ssim_count = 0

    for indices, samples in iter_batches(dataset, batch_size, indices=selected_indices):
        targets = torch.stack([s["target"] for s in samples], dim=0).to(device)
        batch_shape = targets.shape
        cond = None
        if conditioning_mode in {"concatenate", "attention"}:
            cond_list = [s.get("image") for s in samples]
            if all(c is not None for c in cond_list):
                cond = torch.stack(cond_list, dim=0).to(device)
        generated = decode_diffusion_batch(model, training_cfg, model_cfg, device, batch_shape, cond).clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)

        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, generated[batch_idx].cpu(), output_root)
                if save_input:
                    save_output_tensor(dataset, row, dataset.target_key, samples[batch_idx]["target"], output_root / "input")
                if save_conditioning and dataset.conditioning_key is not None:
                    save_output_tensor(dataset, row, dataset.conditioning_key, samples[batch_idx]["image"], output_root / "conditioning")

        reduce_dims = tuple(range(1, generated.ndim))
        mse = torch.mean((generated - targets) ** 2, dim=reduce_dims)
        total_mse += mse.sum().item()
        total_psnr += torch.sum(10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))).item()
        if ssim is not None:
            for idx in range(generated.size(0)):
                value = compute_ssim_sample(generated[idx], targets[idx], ssim)
                if value is not None:
                    total_ssim += value
                    ssim_count += 1
        count += generated.size(0)

    if count == 0:
        raise RuntimeError("No samples available for evaluation.")

    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    logging.info("Eval MSE: %.6f | PSNR: %.3f", avg_mse, avg_psnr)
    if ssim is not None and ssim_count > 0:
        logging.info("Eval SSIM: %.4f", total_ssim / ssim_count)
