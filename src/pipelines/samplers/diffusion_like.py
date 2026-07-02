"""
Shared sampling/encoding/decoding/evaluation for diffusion-like generators.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import torch

import utils
from core.noise_contracts import effective_noise_family_for_config, noise_family_for_model_type
from pipelines import InferenceInputs
from pipelines.samplers.diffusion_runtime import (
    TextConditioningRuntime,
    build_conditioning_batch,
    build_inference_pipeline,
    resolve_conditioning_save_tensor,
    tensor_stats,
)
from pipelines.utils import build_scheduler, resolve_conditioning_mode, resolve_scheduler_override
from utils.dataset_utils import save_output_tensor
from utils.evaluation_utils import compute_ssim_batch, compute_ssim_sample
from utils.model_utils.diffusion_utils import build_diffusion_model, decode_diffusion_batch, encode_diffusion_batch
from utils.sampling_utils import (
    append_eval_metrics,
    append_per_image_eval_metrics,
    build_abs_diff_map,
    build_sampling_dataset,
    build_diff_map,
    colorize_red_map,
    create_experiment_dir,
    load_run_config,
    progress_batches,
    resolve_checkpoint,
    resolve_output_root,
    resolve_sample_indices,
    save_diff_map_grid,
    write_eval_metrics,
)

_TextConditioningRuntime = TextConditioningRuntime
_build_conditioning_batch = build_conditioning_batch
_resolve_conditioning_save_tensor = resolve_conditioning_save_tensor
_build_inference_pipeline = build_inference_pipeline
_tensor_stats = tensor_stats


def _build_residual_source_batch(
    model_type: str,
    training_cfg: dict,
    model_cfg: dict,
    samples: list[dict],
    device: torch.device,
) -> torch.Tensor | None:
    noise_family = effective_noise_family_for_config(model_type, training_cfg=training_cfg, model_cfg=model_cfg)
    if noise_family not in {"residual_flow_matching", "residual_rectified_flow", "residual_reflow"}:
        return None
    images = [sample.get("image") for sample in samples]
    if not images or any(not torch.is_tensor(image) for image in images):
        raise ValueError(f"{noise_family} inference requires dataset image tensors for the source endpoint.")
    return torch.stack(images, dim=0).to(device)


def _count_selected_timesteps(
    *,
    scheduler_cfg: dict,
    training_cfg: dict,
    model_type: str,
    num_inference_steps: int | None,
    start_step: int | None,
    last_n_steps: int | None,
    scheduler_override: str | None,
) -> int:
    scheduler_spec = dict(scheduler_cfg or {})
    override_cfg = resolve_scheduler_override(scheduler_override)
    if override_cfg is not None:
        scheduler_spec["name"] = override_cfg["name"]
        override_params = dict(override_cfg.get("params", {}))
        merged_params = dict(scheduler_spec.get("params", {}))
        merged_params.update(override_params)
        scheduler_spec["params"] = merged_params
    scheduler, inferred_steps = build_scheduler(
        scheduler_spec,
        training_cfg,
        noise_family=effective_noise_family_for_config(model_type, training_cfg=training_cfg),
    )
    effective_steps = int(num_inference_steps or inferred_steps)
    scheduler.set_timesteps(effective_steps)
    timesteps = scheduler.timesteps
    if start_step is not None:
        timesteps = timesteps[timesteps <= int(start_step)]
    if last_n_steps is not None:
        timesteps = timesteps[-int(last_n_steps):]
    return int(timesteps.numel())


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
    use_ema: bool = False,
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

    scheduler, _ = build_scheduler(
        model_cfg.get("scheduler", {}),
        training_cfg,
        noise_family=noise_family_for_model_type(model_type),
    )

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
    save_num_samples: int | None = None,
    save_input: bool = False,
    save_conditioning: bool = False,
    save_diff_map: bool = False,
    diff_amplify: float = 5.0,
    num_inference_steps: int | None = None,
    start_step: int | None = None,
    last_n_steps: int | None = None,
    cfg_rescale: float = 0.0,
    scheduler: str | None = None,
    save_tensor_cache: bool = False,
    use_ema: bool = False,
    strict_model_timing: bool = False,
) -> None:
    _ = cfg_rescale
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, model_type)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    sampling_cfg = cfg.get("sampling", {}) if isinstance(cfg, dict) else {}
    img2img_enabled = bool(sampling_cfg.get("init_from_input", False))
    img2img_strength = float(sampling_cfg.get("strength", 1.0))

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)
    text_runtime = _TextConditioningRuntime(sampling_cfg, device)

    dataset = build_sampling_dataset(cfg, data_txt, save_tensor_cache_override=save_tensor_cache)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    save_indices = set(resolve_sample_indices(dataset, save_num_samples, seed=seed)) if save_num_samples else None
    output_root = resolve_output_root(ckpt_dir, output_dir, save)

    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path, use_ema=use_ema)
    conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))
    inference_pipe, default_inference_steps = _build_inference_pipeline(
        model=model, training_cfg=training_cfg, model_cfg=model_cfg, device=device
    )
    effective_steps = _count_selected_timesteps(
        scheduler_cfg=model_cfg.get("scheduler", {}),
        training_cfg=training_cfg,
        model_type=model_type,
        num_inference_steps=int(num_inference_steps or default_inference_steps),
        start_step=start_step,
        last_n_steps=last_n_steps,
        scheduler_override=scheduler,
    )
    logging.info(
        "%s decode runtime: scheduler=%s requested_steps=%s effective_steps=%d batch_size=%d",
        model_type.replace("_", "-").title(),
        scheduler or model_cfg.get("scheduler", {}).get("name", "default"),
        int(num_inference_steps or default_inference_steps),
        effective_steps,
        batch_size,
    )

    predicted_root = output_root / "predicted" if output_root is not None else None
    diff_root = (output_root / "diff") if (output_root is not None and save_diff_map) else None
    diff_amp_root = (output_root / "diff_amplified") if (output_root is not None and save_diff_map) else None
    diff_tensors_for_grid: list[torch.Tensor] = []
    generation_seconds = 0.0
    decode_wall_start = time.perf_counter()
    generated_count = 0
    batch_iter = progress_batches(dataset, batch_size, f"{model_type} decode", indices=selected_indices)
    for indices, samples in batch_iter:
        targets = torch.stack([s["target"] for s in samples], dim=0)
        batch_shape = targets.shape
        text_embeddings = text_runtime.build_batch(samples)
        residual_source = _build_residual_source_batch(model_type, training_cfg, model_cfg, samples, device)
        cond = _build_conditioning_batch(
            conditioning_mode=conditioning_mode,
            samples=samples,
            targets=targets,
            device=device,
            text_embeddings=text_embeddings,
        )
        if residual_source is not None:
            cond = residual_source
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        batch_start = time.perf_counter()
        if (start_step is not None) or (last_n_steps is not None) or (scheduler is not None) or (residual_source is not None):
            generated = decode_diffusion_batch(
                model,
                training_cfg,
                model_cfg,
                device,
                batch_shape,
                cond,
                reference_batch=targets.to(device),
                init_from_reference=(start_step is not None) or (last_n_steps is not None),
                init_image_batch=targets.to(device) if img2img_enabled else None,
                strength=img2img_strength,
                num_inference_steps=num_inference_steps,
                start_step=start_step,
                last_n_steps=last_n_steps,
                scheduler_override=scheduler,
            ).clamp(0.0, 1.0)
        else:
            generated = inference_pipe.generate(
                InferenceInputs(
                    sample_shape=tuple(batch_shape),
                    num_inference_steps=int(num_inference_steps or default_inference_steps),
                    conditioning_batch=cond,
                    init_image=targets.to(device) if img2img_enabled else None,
                    strength=img2img_strength,
                )
            ).clamp(0.0, 1.0)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        generation_seconds += time.perf_counter() - batch_start
        generated_count += generated.size(0)

        if predicted_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                if save_indices is not None and sample_idx not in save_indices:
                    continue
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, generated[batch_idx].cpu(), predicted_root)
                if save_input:
                    save_output_tensor(dataset, row, dataset.target_key, samples[batch_idx]["target"], output_root / "input")
                if save_conditioning and dataset.conditioning_key is not None:
                    cond_tensor = _resolve_conditioning_save_tensor(samples[batch_idx], conditioning_mode)
                    if cond_tensor is not None:
                        save_output_tensor(dataset, row, dataset.conditioning_key, cond_tensor, output_root / "conditioning")
                if diff_root is not None:
                    diff_tensor = build_abs_diff_map(generated[batch_idx], targets[batch_idx])
                    save_output_tensor(dataset, row, dataset.target_key, colorize_red_map(diff_tensor).cpu(), diff_root)
                if diff_amp_root is not None:
                    diff_amp_tensor = build_diff_map(generated[batch_idx], targets[batch_idx], diff_amplify)
                    diff_amp_rgb = colorize_red_map(diff_amp_tensor)
                    save_output_tensor(dataset, row, dataset.target_key, diff_amp_rgb.cpu(), diff_amp_root)
                    diff_tensors_for_grid.append(diff_amp_rgb.detach().cpu().unsqueeze(0))
        if hasattr(batch_iter, "set_postfix"):
            running_wall = time.perf_counter() - decode_wall_start
            running = {
                "sampler_sps": f"{(generated_count / max(generation_seconds, 1e-12)):.3f}",
                "wall_sps": f"{(generated_count / max(running_wall, 1e-12)):.3f}",
            }
            batch_iter.set_postfix(running)

    wall_seconds = time.perf_counter() - decode_wall_start
    sampler_sps = generated_count / generation_seconds if generation_seconds > 0 else 0.0
    wall_sps = generated_count / wall_seconds if wall_seconds > 0 else 0.0
    print(
        f"Sampler throughput: {sampler_sps:.3f} samples/s | "
        f"{(generation_seconds / max(generated_count, 1)):.6f} s/sample | generation time {generation_seconds:.3f}s"
    )
    print(
        f"Decode wall throughput: {wall_sps:.3f} samples/s | "
        f"{(wall_seconds / max(generated_count, 1)):.6f} s/sample | decode wall time {wall_seconds:.3f}s"
    )
    save_diff_map_grid(diff_tensors_for_grid, output_root if save_diff_map else None)
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
    save_num_samples: int | None = None,
    save_input: bool = False,
    save_conditioning: bool = False,
    save_diff_map: bool = False,
    diff_amplify: float = 5.0,
    num_inference_steps: int | None = None,
    start_step: int | None = None,
    last_n_steps: int | None = None,
    cfg_rescale: float = 0.0,
    scheduler: str | None = None,
    save_tensor_cache: bool = False,
    use_ema: bool = False,
    strict_model_timing: bool = False,
) -> None:
    _ = cfg_rescale
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, model_type)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    sampling_cfg = cfg.get("sampling", {}) if isinstance(cfg, dict) else {}
    img2img_enabled = bool(sampling_cfg.get("init_from_input", False))
    img2img_strength = float(sampling_cfg.get("strength", 1.0))

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = utils.resolve_device(device, default_device)
    text_runtime = _TextConditioningRuntime(sampling_cfg, device)

    dataset = build_sampling_dataset(
        cfg, data_txt, evaluate=True, save_tensor_cache_override=save_tensor_cache
    )
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    save_indices = set(resolve_sample_indices(dataset, save_num_samples, seed=seed)) if save_num_samples else None
    experiment_dir = create_experiment_dir(
        output_dir=output_dir,
        mode="evaluate",
        scheduler=scheduler,
        last_n_steps=last_n_steps,
        start_step=start_step,
        num_inference_steps=num_inference_steps,
        num_samples=num_samples,
        seed=seed,
        batch_size=batch_size,
    )
    output_root = (experiment_dir / "samples") if (save and experiment_dir is not None) else resolve_output_root(ckpt_dir, output_dir, save)
    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path, use_ema=use_ema)
    conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))
    inference_pipe, default_inference_steps = _build_inference_pipeline(
        model=model, training_cfg=training_cfg, model_cfg=model_cfg, device=device
    )
    effective_steps = _count_selected_timesteps(
        scheduler_cfg=model_cfg.get("scheduler", {}),
        training_cfg=training_cfg,
        model_type=model_type,
        num_inference_steps=int(num_inference_steps or default_inference_steps),
        start_step=start_step,
        last_n_steps=last_n_steps,
        scheduler_override=scheduler,
    )
    logging.info(
        "%s evaluate runtime: scheduler=%s requested_steps=%s effective_steps=%d batch_size=%d strict_model_timing=%s",
        model_type.replace("_", "-").title(),
        scheduler or model_cfg.get("scheduler", {}).get("name", "default"),
        int(num_inference_steps or default_inference_steps),
        effective_steps,
        batch_size,
        strict_model_timing,
    )

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    ssim_count = 0
    timing_stats = {"forward_seconds": 0.0, "generation_seconds": 0.0, "model_calls": 0}
    per_image_rows: list[dict] = []
    eval_wall_start = time.perf_counter()

    predicted_root = output_root / "predicted" if output_root is not None else None
    diff_root = (output_root / "diff") if (output_root is not None and save_diff_map) else None
    diff_amp_root = (output_root / "diff_amplified") if (output_root is not None and save_diff_map) else None
    diff_tensors_for_grid: list[torch.Tensor] = []
    batch_iter = progress_batches(dataset, batch_size, f"{model_type} evaluate", indices=selected_indices)
    for indices, samples in batch_iter:
        targets = torch.stack([s["target"] for s in samples], dim=0).to(device)
        batch_shape = targets.shape
        text_embeddings = text_runtime.build_batch(samples)
        residual_source = _build_residual_source_batch(model_type, training_cfg, model_cfg, samples, device)
        cond = _build_conditioning_batch(
            conditioning_mode=conditioning_mode,
            samples=samples,
            targets=targets,
            device=device,
            text_embeddings=text_embeddings,
        )
        if residual_source is not None:
            cond = residual_source
        step_timing = {"model_seconds": 0.0, "model_calls": 0} if strict_model_timing else None
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        batch_start = time.perf_counter()
        if (start_step is not None) or (last_n_steps is not None) or (scheduler is not None) or (residual_source is not None):
            generated = decode_diffusion_batch(
                model,
                training_cfg,
                model_cfg,
                device,
                batch_shape,
                cond,
                reference_batch=targets,
                init_from_reference=(start_step is not None) or (last_n_steps is not None),
                init_image_batch=targets if img2img_enabled else None,
                strength=img2img_strength,
                num_inference_steps=num_inference_steps,
                start_step=start_step,
                last_n_steps=last_n_steps,
                scheduler_override=scheduler,
                timing=step_timing,
            ).clamp(0.0, 1.0)
        else:
            generated = inference_pipe.generate(
                InferenceInputs(
                    sample_shape=tuple(batch_shape),
                    num_inference_steps=int(num_inference_steps or default_inference_steps),
                    conditioning_batch=cond,
                    init_image=targets if img2img_enabled else None,
                    strength=img2img_strength,
                ),
                timing=step_timing,
            ).clamp(0.0, 1.0)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        batch_elapsed = time.perf_counter() - batch_start
        timing_stats["generation_seconds"] += batch_elapsed
        if strict_model_timing and step_timing is not None:
            timing_stats["forward_seconds"] += float(step_timing.get("model_seconds", 0.0))
            timing_stats["model_calls"] += int(step_timing.get("model_calls", 0))
        targets = targets.clamp(0.0, 1.0)

        if predicted_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                if save_indices is not None and sample_idx not in save_indices:
                    continue
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, generated[batch_idx].cpu(), predicted_root)
                if save_input:
                    save_output_tensor(dataset, row, dataset.target_key, samples[batch_idx]["target"], output_root / "input")
                if save_conditioning and dataset.conditioning_key is not None:
                    cond_tensor = _resolve_conditioning_save_tensor(samples[batch_idx], conditioning_mode)
                    if cond_tensor is not None:
                        save_output_tensor(dataset, row, dataset.conditioning_key, cond_tensor, output_root / "conditioning")
                if diff_root is not None:
                    diff_tensor = build_abs_diff_map(generated[batch_idx], targets[batch_idx])
                    save_output_tensor(dataset, row, dataset.target_key, colorize_red_map(diff_tensor).cpu(), diff_root)
                if diff_amp_root is not None:
                    diff_amp_tensor = build_diff_map(generated[batch_idx], targets[batch_idx], diff_amplify)
                    diff_amp_rgb = colorize_red_map(diff_amp_tensor)
                    save_output_tensor(dataset, row, dataset.target_key, diff_amp_rgb.cpu(), diff_amp_root)
                    diff_tensors_for_grid.append(diff_amp_rgb.detach().cpu().unsqueeze(0))

        reduce_dims = tuple(range(1, generated.ndim))
        mse = torch.mean((generated - targets) ** 2, dim=reduce_dims)
        psnr_values = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
        total_mse += mse.sum().item()
        total_psnr += torch.sum(psnr_values).item()
        ssim_values = [None] * generated.size(0)
        try:
            ssim_tensor = compute_ssim_batch(generated, targets)
            total_ssim += float(ssim_tensor.sum().item())
            ssim_count += int(ssim_tensor.numel())
            ssim_values = [float(v) for v in ssim_tensor.detach().cpu().tolist()]
        except ValueError:
            for idx in range(generated.size(0)):
                value = compute_ssim_sample(generated[idx], targets[idx], None)
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
            running_wall = time.perf_counter() - eval_wall_start
            running_sampler_sps = count / max(timing_stats.get("generation_seconds", 1e-12), 1e-12)
            running_wall_sps = count / max(running_wall, 1e-12)
            running = {
                "mse": f"{(total_mse / max(count, 1)):.6f}",
                "psnr": f"{(total_psnr / max(count, 1)):.3f}",
                "sampler_sps": f"{running_sampler_sps:.3f}",
                "wall_sps": f"{running_wall_sps:.3f}",
            }
            if strict_model_timing:
                running_model_sps = count / max(timing_stats.get("forward_seconds", 1e-12), 1e-12)
                running["model_sps"] = f"{running_model_sps:.3f}"
            if ssim_count > 0:
                running["ssim"] = f"{(total_ssim / ssim_count):.4f}"
            batch_iter.set_postfix(running)

    if count == 0:
        raise RuntimeError("No samples available for evaluation.")

    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    forward_seconds = float(timing_stats.get("forward_seconds", 0.0))
    generation_seconds = float(timing_stats.get("generation_seconds", 0.0))
    eval_wall_seconds = time.perf_counter() - eval_wall_start
    model_sps = count / forward_seconds if forward_seconds > 0 else 0.0
    model_s_per_sample = forward_seconds / count if count else 0.0
    generation_sps = count / generation_seconds if generation_seconds > 0 else 0.0
    generation_s_per_sample = generation_seconds / count if count else 0.0
    eval_sps = count / eval_wall_seconds if eval_wall_seconds > 0 else 0.0
    eval_s_per_sample = eval_wall_seconds / count if count else 0.0
    logging.info("Eval MSE: %.6f | PSNR: %.3f", avg_mse, avg_psnr)
    print(f"Eval MSE: {avg_mse:.6f} | PSNR: {avg_psnr:.3f}")
    if strict_model_timing:
        print(
            f"Model forward throughput: {model_sps:.3f} samples/s | "
            f"{model_s_per_sample:.6f} s/sample | forward time {forward_seconds:.3f}s"
        )
    else:
        print("Model forward throughput: unavailable (enable --strict_model_timing for synchronized per-step profiling)")
    print(
        f"Sampler throughput: {generation_sps:.3f} samples/s | "
        f"{generation_s_per_sample:.6f} s/sample | generation time {generation_seconds:.3f}s"
    )
    print(
        f"Eval wall throughput: {eval_sps:.3f} samples/s | "
        f"{eval_s_per_sample:.6f} s/sample | eval wall time {eval_wall_seconds:.3f}s"
    )
    avg_ssim = None
    if ssim_count > 0:
        avg_ssim = total_ssim / ssim_count
        logging.info("Eval SSIM: %.4f", avg_ssim)
        print(f"Eval SSIM: {avg_ssim:.4f}")

    row = {
        "samples": count,
        "mse": f"{avg_mse:.8f}",
        "psnr": f"{avg_psnr:.6f}",
        "ssim": "" if avg_ssim is None else f"{avg_ssim:.6f}",
        "ssim_enabled": True,
        "model_timing_enabled": bool(strict_model_timing),
        "model_seconds": "" if not strict_model_timing else f"{forward_seconds:.6f}",
        "model_samples_per_second": "" if not strict_model_timing else f"{model_sps:.6f}",
        "model_seconds_per_sample": "" if not strict_model_timing else f"{model_s_per_sample:.8f}",
        "sampler_seconds": f"{generation_seconds:.6f}",
        "sampler_samples_per_second": f"{generation_sps:.6f}",
        "sampler_seconds_per_sample": f"{generation_s_per_sample:.8f}",
        "eval_wall_seconds": f"{eval_wall_seconds:.6f}",
        "eval_wall_samples_per_second": f"{eval_sps:.6f}",
        "eval_wall_seconds_per_sample": f"{eval_s_per_sample:.8f}",
        "model_calls": timing_stats.get("model_calls", 0) if strict_model_timing else "",
    }
    metrics_root = experiment_dir if experiment_dir is not None else ckpt_dir
    metrics_path = write_eval_metrics(metrics_root, row) if experiment_dir is not None else append_eval_metrics(metrics_root, row)
    logging.info("Wrote eval metrics: %s", metrics_path)
    per_image_metrics_path = append_per_image_eval_metrics(metrics_root, per_image_rows)
    logging.info("Wrote per-image eval metrics: %s", per_image_metrics_path)
    save_diff_map_grid(diff_tensors_for_grid, output_root if save_diff_map else None)
    if experiment_dir is not None:
        run_cfg = {
            "mode": "evaluate",
            "model_type": model_type,
            "ckpt_dir": str(ckpt_dir),
            "data_txt": data_txt,
            "scheduler": scheduler,
            "num_inference_steps": num_inference_steps,
            "start_step": start_step,
            "last_n_steps": last_n_steps,
            "num_samples": num_samples,
            "save_num_samples": save_num_samples,
            "batch_size": batch_size,
            "seed": seed,
            "save": save,
            "save_input": save_input,
            "save_conditioning": save_conditioning,
            "use_ema": use_ema,
        }
        with (experiment_dir / "run_config.json").open("w") as fh:
            json.dump(run_cfg, fh, indent=2)


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
    cfg_rescale: float = 0.0,
    scheduler: str | None = None,
    save_tensor_cache: bool = False,
    use_ema: bool = False,
) -> None:
    _ = cfg_rescale
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

    model = build_diffusion_model(cfg, device, ckpt_path=ckpt_path, use_ema=use_ema)
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
