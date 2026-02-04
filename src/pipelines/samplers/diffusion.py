"""
Sampling, encoding, decoding, and evaluation for diffusion models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

import utils
from utils.dataset_utils import iter_batches, save_output_tensor
from utils.model_utils.diffusion_utils import build_diffusion_model, decode_diffusion_batch, encode_diffusion_batch
from pipelines.utils import build_scheduler, resolve_conditioning_mode
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
    Forward diffusion (noise addition) over the dataset.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    scheduler, _ = build_scheduler(model_cfg.get("scheduler", {}), training_cfg)

    for indices, samples in iter_batches(dataset, batch_size):
        targets = torch.stack([s["target"] for s in samples], dim=0).to(device)
        if timestep is None:
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (targets.size(0),), device=device
            ).long()
        else:
            timesteps = torch.full((targets.size(0),), int(timestep), device=device, dtype=torch.long)
        noisy = encode_diffusion_batch(scheduler, targets, timesteps)

        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, noisy[batch_idx].cpu(), output_root)

    logging.info("Diffusion encode completed for %d samples.", len(dataset))


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
    Decode/generate samples with the diffusion model.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "diffusion")
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path)
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or model_cfg.get("conditioning")
    )

    for indices, samples in iter_batches(dataset, batch_size):
        targets = torch.stack([s["target"] for s in samples], dim=0)
        batch_shape = targets.shape
        cond = None
        if conditioning_mode == "concatenate":
            cond_list = [s.get("image") for s in samples]
            if all(c is not None for c in cond_list):
                cond = torch.stack(cond_list, dim=0).to(device)
        generated = decode_diffusion_batch(model, training_cfg, model_cfg, device, batch_shape, cond)
        generated = generated.clamp(0.0, 1.0)

        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, generated[batch_idx].cpu(), output_root)

    logging.info("Diffusion decode completed for %d samples.", len(dataset))


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
    Alias for decode: generate samples from the diffusion model.
    """
    decode(
        ckpt_dir=ckpt_dir,
        data_txt=data_txt,
        save=save,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )


def evaluate(
    ckpt_dir: Path | str,
    data_txt: str | None = None,
    batch_size: int = 4,
    device: str | None = None,
    seed: int = 42,
) -> None:
    """
    Evaluate diffusion samples against targets using MSE/PSNR (SSIM if available).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:  # pragma: no cover - optional
        ssim = None

    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "diffusion")
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt)
    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path)
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or model_cfg.get("conditioning")
    )

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for indices, samples in iter_batches(dataset, batch_size):
        targets = torch.stack([s["target"] for s in samples], dim=0).to(device)
        batch_shape = targets.shape
        cond = None
        if conditioning_mode == "concatenate":
            cond_list = [s.get("image") for s in samples]
            if all(c is not None for c in cond_list):
                cond = torch.stack(cond_list, dim=0).to(device)
        generated = decode_diffusion_batch(model, training_cfg, model_cfg, device, batch_shape, cond).clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)

        mse = torch.mean((generated - targets) ** 2, dim=(1, 2, 3))
        total_mse += mse.sum().item()
        total_psnr += torch.sum(10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))).item()
        if ssim is not None:
            for idx in range(generated.size(0)):
                gen_np = generated[idx].detach().cpu().numpy().transpose(1, 2, 0)
                tgt_np = targets[idx].detach().cpu().numpy().transpose(1, 2, 0)
                total_ssim += float(ssim(gen_np, tgt_np, channel_axis=2, data_range=1.0))
        count += generated.size(0)

    if count == 0:
        raise RuntimeError("No samples available for evaluation.")

    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    logging.info("Eval MSE: %.6f | PSNR: %.3f", avg_mse, avg_psnr)
    if ssim is not None:
        logging.info("Eval SSIM: %.4f", total_ssim / count)
