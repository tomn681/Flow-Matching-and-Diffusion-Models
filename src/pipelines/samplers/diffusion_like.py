"""
Shared sampling/encoding/decoding/evaluation for diffusion-like generators.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

import utils
from pipelines.utils import build_scheduler, resolve_conditioning_mode
from utils.dataset_utils import save_output_tensor
from utils.evaluation_utils import compute_ssim_sample
from utils.model_utils.diffusion_utils import build_diffusion_model, decode_diffusion_batch, encode_diffusion_batch
from utils.sampling_utils import (
    append_eval_metrics,
    append_per_image_eval_metrics,
    build_sampling_dataset,
    load_run_config,
    progress_batches,
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
    save_tensor_cache: bool = False,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt, save_tensor_cache_override=save_tensor_cache)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    scheduler, _ = build_scheduler(model_cfg.get("scheduler", {}), training_cfg)

    for indices, samples in progress_batches(dataset, batch_size, f"{model_type} encode", indices=selected_indices):
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
    num_inference_steps: int | None = None,
    start_step: int | None = None,
    last_n_steps: int | None = None,
    scheduler: str | None = None,
    save_tensor_cache: bool = False,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, model_type)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)

    dataset = build_sampling_dataset(cfg, data_txt, save_tensor_cache_override=save_tensor_cache)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path)
    conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))

    for indices, samples in progress_batches(dataset, batch_size, f"{model_type} decode", indices=selected_indices):
        targets = torch.stack([s["target"] for s in samples], dim=0)
        batch_shape = targets.shape
        cond = None
        if conditioning_mode in {"concatenate", "attention"}:
            cond_list = [s.get("image") for s in samples]
            if all(c is not None for c in cond_list):
                cond = torch.stack(cond_list, dim=0).to(device)
        generated = decode_diffusion_batch(
            model,
            training_cfg,
            model_cfg,
            device,
            batch_shape,
            cond,
            reference_batch=targets.to(device),
            init_from_reference=(start_step is not None) or (last_n_steps is not None),
            num_inference_steps=num_inference_steps,
            start_step=start_step,
            last_n_steps=last_n_steps,
            scheduler_override=scheduler,
        ).clamp(0.0, 1.0)

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
    num_inference_steps: int | None = None,
    start_step: int | None = None,
    last_n_steps: int | None = None,
    scheduler: str | None = None,
    save_tensor_cache: bool = False,
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

    dataset = build_sampling_dataset(
        cfg, data_txt, evaluate=True, save_tensor_cache_override=save_tensor_cache
    )
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    output_root = resolve_output_root(ckpt_dir, output_dir, save)
    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path)
    conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    ssim_count = 0
    model_timing = {"model_seconds": 0.0, "model_calls": 0}
    per_image_rows: list[dict] = []

    batch_iter = progress_batches(dataset, batch_size, f"{model_type} evaluate", indices=selected_indices)
    for indices, samples in batch_iter:
        targets = torch.stack([s["target"] for s in samples], dim=0).to(device)
        batch_shape = targets.shape
        cond = None
        if conditioning_mode in {"concatenate", "attention"}:
            cond_list = [s.get("image") for s in samples]
            if all(c is not None for c in cond_list):
                cond = torch.stack(cond_list, dim=0).to(device)
        generated = decode_diffusion_batch(
            model,
            training_cfg,
            model_cfg,
            device,
            batch_shape,
            cond,
            timing=model_timing,
            reference_batch=targets,
            init_from_reference=(start_step is not None) or (last_n_steps is not None),
            num_inference_steps=num_inference_steps,
            start_step=start_step,
            last_n_steps=last_n_steps,
            scheduler_override=scheduler,
        ).clamp(0.0, 1.0)
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
        psnr_values = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
        total_mse += mse.sum().item()
        total_psnr += torch.sum(psnr_values).item()
        ssim_values = [None] * generated.size(0)
        if ssim is not None:
            for idx in range(generated.size(0)):
                value = compute_ssim_sample(generated[idx], targets[idx], ssim)
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
        count += generated.size(0)
        if hasattr(batch_iter, "set_postfix"):
            running = {
                "mse": f"{(total_mse / max(count, 1)):.6f}",
                "psnr": f"{(total_psnr / max(count, 1)):.3f}",
                "sps": f"{(count / max(model_timing.get('model_seconds', 1e-12), 1e-12)):.3f}",
            }
            if ssim_count > 0:
                running["ssim"] = f"{(total_ssim / ssim_count):.4f}"
            batch_iter.set_postfix(running)

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
            "model_calls": model_timing.get("model_calls", 0),
        },
    )
    logging.info("Wrote eval metrics: %s", metrics_path)
    per_image_metrics_path = append_per_image_eval_metrics(metrics_root, per_image_rows)
    logging.info("Wrote per-image eval metrics: %s", per_image_metrics_path)


def _tensor_stats(name: str, tensor: torch.Tensor | None) -> dict:
    if tensor is None:
        return {"name": name, "present": False}
    t = torch.as_tensor(tensor).detach().float().cpu()
    return {
        "name": name,
        "present": True,
        "shape": list(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()) if t.numel() > 1 else 0.0,
    }


def _run_debug_compare(
    *,
    ckpt_dir: Path | str,
    model_type: str,
    data_txt: str | None = None,
    output_dir: str | None = None,
    device: str | None = None,
    seed: int = 42,
    num_samples: int | None = None,
    num_inference_steps: int | None = None,
    start_step: int | None = None,
    last_n_steps: int | None = None,
    scheduler: str | None = None,
    save_tensor_cache: bool = False,
) -> None:
    """
    Debug helper for one-sample diffusion-like inference.
    Dumps tensor stats and raw/clamped outputs to inspect evaluation regressions.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, model_type)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

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

    target = sample["target"].unsqueeze(0).to(device)
    cond = sample.get("image")
    cond_batch = cond.unsqueeze(0).to(device) if cond is not None else None

    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path)
    timing = {"model_seconds": 0.0, "model_calls": 0}
    generated_raw = decode_diffusion_batch(
        model,
        training_cfg,
        model_cfg,
        device,
        target.shape,
        cond_batch,
        timing=timing,
        reference_batch=target,
        init_from_reference=(start_step is not None) or (last_n_steps is not None),
        num_inference_steps=num_inference_steps,
        start_step=start_step,
        last_n_steps=last_n_steps,
        scheduler_override=scheduler,
    )
    generated_clamped = generated_raw.clamp(0.0, 1.0)

    conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))
    generated_raw_no_cond = None
    generated_clamped_no_cond = None
    no_cond_error = None
    # For attention-conditioned UNets with cross-attention blocks, context is mandatory.
    # Keep this probe only for concatenate-mode models; otherwise record why it was skipped.
    if conditioning_mode == "concatenate":
        generated_raw_no_cond = decode_diffusion_batch(
            model,
            training_cfg,
            model_cfg,
            device,
            target.shape,
            conditioning_batch=None,
            num_inference_steps=num_inference_steps,
            start_step=start_step,
            last_n_steps=last_n_steps,
            scheduler_override=scheduler,
        )
        generated_clamped_no_cond = generated_raw_no_cond.clamp(0.0, 1.0)
    elif conditioning_mode == "attention":
        no_cond_error = "Skipped no-cond probe: attention model requires context."

    debug_root = Path(output_dir) if output_dir else (ckpt_dir / "debug_compare")
    debug_root.mkdir(parents=True, exist_ok=True)

    # Save tensors for exact inspection.
    torch.save(target.detach().cpu(), debug_root / "target.pt")
    if cond_batch is not None:
        torch.save(cond_batch.detach().cpu(), debug_root / "conditioning.pt")
    torch.save(generated_raw.detach().cpu(), debug_root / "generated_raw.pt")
    torch.save(generated_clamped.detach().cpu(), debug_root / "generated_clamped.pt")
    if generated_raw_no_cond is not None:
        torch.save(generated_raw_no_cond.detach().cpu(), debug_root / "generated_raw_no_cond.pt")
        torch.save(generated_clamped_no_cond.detach().cpu(), debug_root / "generated_clamped_no_cond.pt")

    # Save image-like outputs through dataset writers.
    save_output_tensor(dataset, row, dataset.target_key, generated_clamped[0].detach().cpu(), debug_root / "generated")
    save_output_tensor(dataset, row, dataset.target_key, target[0].detach().cpu(), debug_root / "target")
    if dataset.conditioning_key is not None and cond is not None:
        save_output_tensor(dataset, row, dataset.conditioning_key, cond.detach().cpu(), debug_root / "conditioning_export")
    if generated_clamped_no_cond is not None:
        save_output_tensor(dataset, row, dataset.target_key, generated_clamped_no_cond[0].detach().cpu(), debug_root / "generated_no_cond")

    stats = {
        "model_type": model_type,
        "sample_index": sample_idx,
        "img_id": sample.get("img_id"),
        "img_path": sample.get("img_path"),
        "conditioning_mode": conditioning_mode,
        "timing": timing,
        "num_inference_steps": num_inference_steps,
        "start_step": start_step,
        "last_n_steps": last_n_steps,
        "scheduler_override": scheduler,
        "target": _tensor_stats("target", target),
        "conditioning": _tensor_stats("conditioning", cond_batch),
        "generated_raw": _tensor_stats("generated_raw", generated_raw),
        "generated_clamped": _tensor_stats("generated_clamped", generated_clamped),
        "generated_raw_no_cond": _tensor_stats("generated_raw_no_cond", generated_raw_no_cond),
        "generated_clamped_no_cond": _tensor_stats("generated_clamped_no_cond", generated_clamped_no_cond),
        "no_cond_note": no_cond_error,
    }
    with (debug_root / "stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)

    logging.info("Debug compare completed. Artifacts written to: %s", debug_root)
    print(f"Debug compare completed. Artifacts written to: {debug_root}")
