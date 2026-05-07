"""
Sampling, encoding, decoding, and evaluation for flow-matching models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

import utils
from utils.dataset_utils import save_output_tensor
from utils.model_utils.diffusion_utils import (
    build_diffusion_model,
    decode_diffusion_batch,
    encode_diffusion_batch,
    warn_attention_conditioning_shape,
)
from pipelines.utils import build_scheduler, resolve_conditioning_mode
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
    Forward flow-matching (noise addition) over the dataset.
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

    for indices, samples in progress_batches(dataset, batch_size, "Flow-matching encode"):
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

    logging.info("Flow-matching encode completed for %d samples.", len(dataset))


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
    Decode/generate samples with the flow-matching model.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "flow_matching")
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

    for indices, samples in progress_batches(dataset, batch_size, "Flow-matching decode"):
        targets = torch.stack([s["target"] for s in samples], dim=0)
        batch_shape = targets.shape
        cond = None
        if conditioning_mode in {"concatenate", "attention"}:
            cond_list = [s.get("image") for s in samples]
            if all(c is not None for c in cond_list):
                cond = torch.stack(cond_list, dim=0).to(device)
        generated = decode_diffusion_batch(model, training_cfg, model_cfg, device, batch_shape, cond)
        generated = generated.clamp(0.0, 1.0)

        if output_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, generated[batch_idx].cpu(), output_root)

    logging.info("Flow-matching decode completed for %d samples.", len(dataset))


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
    Alias for decode: generate samples from the flow-matching model.
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
    Evaluate flow-matching samples against targets using MSE/PSNR (SSIM if available).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:  # pragma: no cover - optional
        ssim = None

    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, "flow_matching")
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt, evaluate=True)
    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path)
    conditioning_mode = resolve_conditioning_mode(
        training_cfg.get("conditioning") or model_cfg.get("conditioning")
    )

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    warned_conditioning_shape = False
    model_timing = {"model_seconds": 0.0, "model_calls": 0}
    per_image_rows = []

    for indices, samples in progress_batches(dataset, batch_size, "Flow-matching evaluate"):
        targets = torch.stack([s["target"] for s in samples], dim=0).to(device)
        batch_shape = targets.shape
        cond = None
        if conditioning_mode in {"concatenate", "attention"}:
            cond_list = [s.get("image") for s in samples]
            if all(c is not None for c in cond_list):
                cond = torch.stack(cond_list, dim=0).to(device)
        if conditioning_mode == "attention" and not warned_conditioning_shape:
            warned_conditioning_shape = warn_attention_conditioning_shape(cond, model_cfg)
        generated = decode_diffusion_batch(
            model,
            training_cfg,
            model_cfg,
            device,
            batch_shape,
            cond,
            timing=model_timing,
        ).clamp(0.0, 1.0)
        targets = targets.clamp(0.0, 1.0)

        mse = torch.mean((generated - targets) ** 2, dim=(1, 2, 3))
        psnr_values = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
        total_mse += mse.sum().item()
        total_psnr += torch.sum(psnr_values).item()
        ssim_values = [None] * generated.size(0)
        if ssim is not None:
            for idx in range(generated.size(0)):
                gen_np = generated[idx].detach().cpu().numpy().transpose(1, 2, 0)
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
        count += generated.size(0)

    if count == 0:
        raise RuntimeError("No samples available for evaluation.")

    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    model_seconds = float(model_timing.get("model_seconds", 0.0))
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
            "model_calls": model_timing.get("model_calls", 0),
        },
    )
    logging.info("Wrote eval metrics: %s", metrics_path)
    per_image_metrics_path = append_per_image_eval_metrics(ckpt_dir, per_image_rows)
    logging.info("Wrote per-image eval metrics: %s", per_image_metrics_path)
