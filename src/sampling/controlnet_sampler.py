from __future__ import annotations

import json
import logging
from pathlib import Path
import sys as _sys

import torch

import utils
from models.controlnet import load_frozen_base_unet
from models.factory import ModelFactory
from core.noise_contracts import noise_family_for_model_type
from pipelines import InferenceInputs, InferencePipeline
from pipelines.samplers.diffusion_runtime import (
    TextConditioningRuntime,
    resolve_conditioning_save_tensor,
)
from scheduling import build_scheduler
from utils.dataset_utils import save_output_tensor
from utils.evaluation_utils import compute_ssim_sample
from training.ema import apply_ema_state_to_model
from utils.sampling_utils import (
    append_eval_metrics,
    append_per_image_eval_metrics,
    build_sampling_dataset,
    create_experiment_dir,
    load_run_config,
    progress_batches,
    resolve_checkpoint,
    resolve_output_root,
    resolve_sample_indices,
    write_eval_metrics,
)

from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


def _stack_optional_tensor_list(samples: list[dict], key: str, device: torch.device) -> torch.Tensor | None:
    tensors = [s.get(key) for s in samples]
    if all(torch.is_tensor(t) for t in tensors):
        return torch.stack(tensors, dim=0).to(device)
    return None


def _build_context_batch(
    *,
    samples: list[dict],
    device: torch.device,
    text_embeddings: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if text_embeddings is not None:
        return text_embeddings
    for key in ("attn_cond", "context", "attention"):
        batch = _stack_optional_tensor_list(samples, key, device)
        if batch is not None:
            return batch
    return None


def _load_controlnet_model(cfg: dict, ckpt_dir: Path, device: torch.device, *, use_ema: bool = False) -> torch.nn.Module:
    model = ModelFactory.build(cfg).to(device)
    ckpt_path = resolve_checkpoint(ckpt_dir, "controlnet")
    payload = utils.safe_torch_load(ckpt_path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    if use_ema and not apply_ema_state_to_model(model, payload.get("ema") if isinstance(payload, dict) else None):
        raise ValueError("Requested EMA weights for ControlNet runtime, but checkpoint does not contain EMA state.")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    logging.info("Loaded ControlNet from %s", ckpt_path)
    return model


def _build_controlnet_inference_pipeline(
    *,
    cfg: dict,
    ckpt_dir: Path,
    device: torch.device,
    use_ema: bool = False,
) -> tuple[InferencePipeline, int]:
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    base_ckpt = model_cfg.get("base_unet_checkpoint")
    if not base_ckpt:
        raise ValueError("ControlNet runtime requires model.base_unet_checkpoint in train_config.json.")
    base_unet = load_frozen_base_unet(base_ckpt, device)
    controlnet = _load_controlnet_model(cfg, ckpt_dir, device, use_ema=use_ema)
    scheduler, default_steps = build_scheduler(
        model_cfg.get("scheduler", {}),
        training_cfg,
        noise_family=noise_family_for_model_type(str(model_cfg.get("model_type", ""))),
    )
    pipe = InferencePipeline(
        unet=base_unet,
        controlnet=controlnet,
        scheduler=scheduler,
        device=device,
        conditioning_mode="attention",
    )
    return pipe, int(default_steps)


def _run_controlnet_inference(
    *,
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
    num_inference_steps: int | None = None,
    start_step: int | None = None,
    last_n_steps: int | None = None,
    scheduler: str | None = None,
    save_tensor_cache: bool = False,
    evaluate: bool = False,
    use_ema: bool = False,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    cfg = load_run_config(ckpt_dir)
    if start_step is not None or last_n_steps is not None or scheduler is not None:
        raise ValueError(
            "ControlNetSampler does not yet support runtime scheduler overrides, start_step, or last_n_steps."
        )

    utils.set_seed(seed)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_device = utils.resolve_device(device, default_device)
    text_runtime = TextConditioningRuntime(cfg.get("sampling", {}), resolved_device)

    dataset = build_sampling_dataset(
        cfg,
        data_txt,
        evaluate=evaluate,
        save_tensor_cache_override=save_tensor_cache,
    )
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    pipe, default_steps = _build_controlnet_inference_pipeline(
        cfg=cfg,
        ckpt_dir=ckpt_dir,
        device=resolved_device,
        use_ema=use_ema,
    )

    if evaluate:
        try:
            from skimage.metrics import structural_similarity as ssim
        except Exception:  # pragma: no cover - optional
            ssim = None
        experiment_dir = create_experiment_dir(
            output_dir=output_dir,
            mode="evaluate",
            scheduler=None,
            last_n_steps=None,
            start_step=None,
            num_inference_steps=num_inference_steps,
            num_samples=num_samples,
            seed=seed,
            batch_size=batch_size,
        )
        output_root = (experiment_dir / "samples") if (save and experiment_dir is not None) else resolve_output_root(
            ckpt_dir, output_dir, save
        )
    else:
        ssim = None
        experiment_dir = None
        output_root = resolve_output_root(ckpt_dir, output_dir, save)

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    ssim_count = 0
    per_image_rows: list[dict[str, str | int]] = []
    predicted_root = output_root / "predicted" if output_root is not None else None

    mode_label = "controlnet evaluate" if evaluate else "controlnet sample"
    for indices, samples in progress_batches(dataset, batch_size, mode_label, indices=selected_indices):
        targets = torch.stack([s["target"] for s in samples], dim=0).to(resolved_device)
        control_cond = _stack_optional_tensor_list(samples, "image", resolved_device)
        if control_cond is None:
            raise ValueError("ControlNet runtime requires dataset samples with an 'image' tensor for control input.")
        text_embeddings = text_runtime.build_batch(samples)
        context_ca = _build_context_batch(samples=samples, device=resolved_device, text_embeddings=text_embeddings)

        generated = pipe.generate(
            InferenceInputs(
                sample_shape=tuple(targets.shape),
                num_inference_steps=int(num_inference_steps or default_steps),
                conditioning_batch=context_ca,
                controlnet_cond=control_cond,
            )
        ).clamp(0.0, 1.0)

        if predicted_root is not None:
            for batch_idx, sample_idx in enumerate(indices):
                row = dataset.data[sample_idx]
                save_output_tensor(dataset, row, dataset.target_key, generated[batch_idx].cpu(), predicted_root)
                if save_input:
                    save_output_tensor(dataset, row, dataset.target_key, samples[batch_idx]["target"], output_root / "input")
                if save_conditioning and dataset.conditioning_key is not None:
                    cond_tensor = resolve_conditioning_save_tensor(samples[batch_idx], "attention")
                    if cond_tensor is not None:
                        save_output_tensor(dataset, row, dataset.conditioning_key, cond_tensor, output_root / "conditioning")

        if not evaluate:
            continue

        reduce_dims = tuple(range(1, generated.ndim))
        mse = torch.mean((generated - targets.clamp(0.0, 1.0)) ** 2, dim=reduce_dims)
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

    if not evaluate:
        logging.info("ControlNet sampling completed for %d samples.", len(selected_indices))
        return

    if count == 0:
        raise RuntimeError("No samples available for evaluation.")
    avg_mse = total_mse / count
    avg_psnr = total_psnr / count
    avg_ssim = None if ssim_count == 0 else (total_ssim / ssim_count)
    print(f"Eval MSE: {avg_mse:.6f} | PSNR: {avg_psnr:.3f}")
    if avg_ssim is not None:
        print(f"Eval SSIM: {avg_ssim:.4f}")
    row = {
        "samples": count,
        "mse": f"{avg_mse:.8f}",
        "psnr": f"{avg_psnr:.6f}",
        "ssim": "" if avg_ssim is None else f"{avg_ssim:.6f}",
        "ssim_enabled": ssim is not None,
        "model_seconds": "0.000000",
        "model_samples_per_second": "0.000000",
        "model_seconds_per_sample": "0.00000000",
        "model_calls": 0,
    }
    metrics_root = experiment_dir if experiment_dir is not None else ckpt_dir
    metrics_path = write_eval_metrics(metrics_root, row) if experiment_dir is not None else append_eval_metrics(metrics_root, row)
    logging.info("Wrote eval metrics: %s", metrics_path)
    per_image_metrics_path = append_per_image_eval_metrics(metrics_root, per_image_rows)
    logging.info("Wrote per-image eval metrics: %s", per_image_metrics_path)
    if experiment_dir is not None:
        run_cfg = {
            "mode": "evaluate",
            "model_type": "controlnet",
            "ckpt_dir": str(ckpt_dir),
            "data_txt": data_txt,
            "num_inference_steps": num_inference_steps,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "seed": seed,
            "save": save,
            "save_input": save_input,
            "save_conditioning": save_conditioning,
            "use_ema": use_ema,
        }
        with (experiment_dir / "run_config.json").open("w") as fh:
            json.dump(run_cfg, fh, indent=2)


@SAMPLER_REGISTRY.register("controlnet")
class ControlNetSampler(BaseSampler):
    """Runtime sampler for trained ControlNet checkpoints."""

    supported_modes = frozenset({"build_tensor_cache", "sample", "evaluate"})

    def sample(self) -> None:
        _run_controlnet_inference(
            **self._generative_decode_like_kwargs,
        )

    def evaluate(self) -> None:
        _run_controlnet_inference(
            **self._generative_decode_like_kwargs,
            evaluate=True,
        )


_module = _sys.modules[__name__]
if __name__.startswith("genlib.sampling."):
    _sys.modules.setdefault(__name__.replace("genlib.sampling.", "sampling.", 1), _module)
elif __name__.startswith("src.sampling."):
    _sys.modules.setdefault(__name__.replace("src.sampling.", "sampling.", 1), _module)
    _sys.modules.setdefault(__name__.replace("src.sampling.", "genlib.sampling.", 1), _module)
elif __name__.startswith("sampling."):
    _sys.modules.setdefault(__name__.replace("sampling.", "src.sampling.", 1), _module)
    _sys.modules.setdefault(__name__.replace("sampling.", "genlib.sampling.", 1), _module)
del _module, _sys
