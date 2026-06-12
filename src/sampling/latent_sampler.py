from __future__ import annotations

from pathlib import Path

import torch

import utils
from models.autoencoder.utils import decode_from_latent, encode_to_latent
from core.noise_contracts import noise_family_for_model_type
from pipelines.utils import build_scheduler, resolve_conditioning_mode
from utils.dataset_utils import save_output_tensor
from utils.model_utils.diffusion_utils import build_diffusion_model, decode_diffusion_batch
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
)
from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


class LatentSampler(BaseSampler):
    model_type: str

    def _resolve_runtime(self, *, evaluate: bool = False):
        ckpt_dir = Path(self.ckpt_dir)
        cfg = load_run_config(ckpt_dir)
        training_cfg = cfg["training"]
        model_cfg = cfg["model"]
        ckpt_path = resolve_checkpoint(ckpt_dir, self.model_type)

        utils.set_seed(self.seed)
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = utils.resolve_device(self.device, default_device)

        dataset = build_sampling_dataset(cfg, self.data_txt, evaluate=evaluate, save_tensor_cache_override=self.save_tensor_cache)
        selected_indices = resolve_sample_indices(dataset, self.num_samples, seed=self.seed)
        model = build_diffusion_model(
            self._model_cfg_for_build(cfg),
            device,
            ckpt_path=ckpt_path,
            use_ema=self.use_ema,
        )
        vae = self._load_frozen_vae(cfg, device)
        conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))
        sampling_mode = "attention" if conditioning_mode == "latent_attention" else conditioning_mode
        use_presaved = bool(model_cfg.get("use_presaved_latents", False))
        return ckpt_dir, cfg, training_cfg, model_cfg, device, dataset, selected_indices, model, vae, conditioning_mode, sampling_mode, use_presaved

    def _prepare_latent_batches(
        self,
        *,
        samples,
        device: torch.device,
        vae: torch.nn.Module,
        use_presaved: bool,
        conditioning_mode: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        target = torch.stack([s["target"] for s in samples], dim=0).to(device)
        target_latent = target if use_presaved else encode_to_latent(vae, target)
        cond = None
        if conditioning_mode in {"attention", "concatenate", "latent_attention"}:
            cond_list = [s.get("image") for s in samples]
            if all(c is not None for c in cond_list):
                cond_raw = torch.stack(cond_list, dim=0).to(device)
                cond = cond_raw if use_presaved else encode_to_latent(vae, cond_raw)
        return target, cond, target_latent

    def _predict_latent_batch(
        self,
        *,
        model,
        training_cfg: dict,
        model_cfg: dict,
        device: torch.device,
        target_latent: torch.Tensor,
        cond: torch.Tensor | None,
        sampling_mode: str | None,
    ) -> torch.Tensor:
        return decode_diffusion_batch(
            model,
            {**training_cfg, "conditioning": sampling_mode},
            model_cfg,
            device,
            tuple(target_latent.shape),
            cond,
            reference_batch=target_latent,
            init_from_reference=(self.start_step is not None) or (self.last_n_steps is not None),
            num_inference_steps=self.num_inference_steps,
            start_step=self.start_step,
            last_n_steps=self.last_n_steps,
            scheduler_override=self.scheduler,
        )

    def _load_frozen_vae(self, cfg: dict, device: torch.device) -> torch.nn.Module:
        model_cfg = cfg.get("model", {})
        vae_cfg = dict(model_cfg.get("vae", {}))
        if not vae_cfg:
            raise ValueError("Latent sampling requires config.model.vae.")
        vae_cfg["model_type"] = "vae"
        vae_cfg.setdefault("latent_type", "kl")
        ckpt_path = model_cfg.get("vae_checkpoint")
        if not ckpt_path:
            raise ValueError("Latent sampling requires config.model.vae_checkpoint.")
        from utils.model_utils.vae_utils import build_vae_model

        return build_vae_model(
            {"model": vae_cfg},
            device,
            ckpt_path=ckpt_path,
            set_eval=True,
            use_ema=self.use_ema,
        )

    def _model_cfg_for_build(self, cfg: dict) -> dict:
        return cfg

    def encode(self) -> None:
        ckpt_dir = Path(self.ckpt_dir)
        cfg = load_run_config(ckpt_dir)
        training_cfg = cfg["training"]
        model_cfg = cfg["model"]

        utils.set_seed(self.seed)
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = utils.resolve_device(self.device, default_device)

        dataset = build_sampling_dataset(cfg, self.data_txt, save_tensor_cache_override=self.save_tensor_cache)
        selected_indices = resolve_sample_indices(dataset, self.num_samples, seed=self.seed)
        output_root = resolve_output_root(ckpt_dir, self.output_dir, self.save)
        vae = self._load_frozen_vae(cfg, device)
        use_presaved = bool(model_cfg.get("use_presaved_latents", False))
        scheduler, _ = build_scheduler(
            model_cfg.get("scheduler", {}),
            training_cfg,
            noise_family=noise_family_for_model_type(self.model_type),
        )

        for indices, samples in progress_batches(dataset, self.batch_size, f"{self.model_type} encode", indices=selected_indices):
            target = torch.stack([s["target"] for s in samples], dim=0).to(device)
            target_latent = target if use_presaved else encode_to_latent(vae, target)
            if self.timestep is None:
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (target_latent.size(0),), device=device).long()
            else:
                timesteps = torch.full((target_latent.size(0),), int(self.timestep), device=device, dtype=torch.long)
            noise = torch.randn_like(target_latent)
            noisy_latent = scheduler.add_noise(target_latent, noise, timesteps)

            if output_root is not None:
                for batch_idx, sample_idx in enumerate(indices):
                    row = dataset.data[sample_idx]
                    save_output_tensor(dataset, row, dataset.target_key, noisy_latent[batch_idx].cpu(), output_root)

    def decode(self) -> None:
        (
            ckpt_dir,
            _cfg,
            training_cfg,
            model_cfg,
            device,
            dataset,
            selected_indices,
            model,
            vae,
            conditioning_mode,
            sampling_mode,
            use_presaved,
        ) = self._resolve_runtime(evaluate=False)
        output_root = resolve_output_root(ckpt_dir, self.output_dir, self.save)
        predicted_root = output_root / "predicted" if output_root is not None else None

        recon_type = str(training_cfg.get("recon_type", "l1"))

        for indices, samples in progress_batches(dataset, self.batch_size, f"{self.model_type} decode", indices=selected_indices):
            _target, cond, target_latent = self._prepare_latent_batches(
                samples=samples,
                device=device,
                vae=vae,
                use_presaved=use_presaved,
                conditioning_mode=conditioning_mode,
            )
            latent_pred = self._predict_latent_batch(
                model=model,
                training_cfg=training_cfg,
                model_cfg=model_cfg,
                device=device,
                target_latent=target_latent,
                cond=cond,
                sampling_mode=sampling_mode,
            )

            generated = decode_from_latent(vae, latent_pred, recon_type=recon_type).clamp(0.0, 1.0)

            if predicted_root is not None:
                for batch_idx, sample_idx in enumerate(indices):
                    row = dataset.data[sample_idx]
                    save_output_tensor(dataset, row, dataset.target_key, generated[batch_idx].cpu(), predicted_root)
                    if self.save_input:
                        save_output_tensor(dataset, row, dataset.target_key, samples[batch_idx]["target"], output_root / "input")
                    if self.save_conditioning and dataset.conditioning_key is not None and samples[batch_idx].get("image") is not None:
                        save_output_tensor(
                            dataset,
                            row,
                            dataset.conditioning_key,
                            samples[batch_idx]["image"],
                            output_root / "conditioning",
                        )

    def sample(self) -> None:
        self.decode()

    def evaluate(self) -> None:
        (
            ckpt_dir,
            _cfg,
            training_cfg,
            model_cfg,
            device,
            dataset,
            selected_indices,
            model,
            vae,
            conditioning_mode,
            sampling_mode,
            use_presaved,
        ) = self._resolve_runtime(evaluate=True)
        experiment_dir = create_experiment_dir(
            output_dir=self.output_dir,
            mode="evaluate",
            scheduler=self.scheduler,
            last_n_steps=self.last_n_steps,
            start_step=self.start_step,
            num_inference_steps=self.num_inference_steps,
            num_samples=self.num_samples,
            seed=self.seed,
            batch_size=self.batch_size,
        )
        output_root = (experiment_dir / "samples") if (self.save and experiment_dir is not None) else resolve_output_root(ckpt_dir, self.output_dir, self.save)
        predicted_root = output_root / "predicted" if output_root is not None else None

        recon_type = str(training_cfg.get("recon_type", "l1"))

        total_mse = 0.0
        total_psnr = 0.0
        count = 0
        per_image_rows: list[dict] = []
        for indices, samples in progress_batches(dataset, self.batch_size, f"{self.model_type} evaluate", indices=selected_indices):
            target, cond, target_latent = self._prepare_latent_batches(
                samples=samples,
                device=device,
                vae=vae,
                use_presaved=use_presaved,
                conditioning_mode=conditioning_mode,
            )
            target_img = target if use_presaved else target
            latent_pred = self._predict_latent_batch(
                model=model,
                training_cfg=training_cfg,
                model_cfg=model_cfg,
                device=device,
                target_latent=target_latent,
                cond=cond,
                sampling_mode=sampling_mode,
            )
            generated = decode_from_latent(vae, latent_pred, recon_type=recon_type).clamp(0.0, 1.0)
            target_eval = target_img.clamp(0.0, 1.0)
            reduce_dims = tuple(range(1, generated.ndim))
            mse = torch.mean((generated - target_eval) ** 2, dim=reduce_dims)
            psnr_values = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
            total_mse += mse.sum().item()
            total_psnr += torch.sum(psnr_values).item()
            for batch_idx, sample_idx in enumerate(indices):
                sample = samples[batch_idx]
                per_image_rows.append(
                    {
                        "sample_index": sample_idx,
                        "img_id": sample.get("img_id"),
                        "img_path": sample.get("img_path"),
                        "mse": f"{mse[batch_idx].item():.8f}",
                        "psnr": f"{psnr_values[batch_idx].item():.6f}",
                    }
                )
            count += generated.size(0)

            if predicted_root is not None:
                for batch_idx, sample_idx in enumerate(indices):
                    row = dataset.data[sample_idx]
                    save_output_tensor(dataset, row, dataset.target_key, generated[batch_idx].cpu(), predicted_root)

        if count == 0:
            raise RuntimeError("No samples available for latent evaluation.")
        avg_mse = total_mse / count
        avg_psnr = total_psnr / count
        row = {"samples": count, "mse": f"{avg_mse:.8f}", "psnr": f"{avg_psnr:.6f}"}
        metrics_root = experiment_dir if experiment_dir is not None else ckpt_dir
        append_eval_metrics(metrics_root, row)
        append_per_image_eval_metrics(metrics_root, per_image_rows)

    def debug_compare(self) -> None:
        # Reuse evaluate as a practical debug path that also writes metrics/artifacts.
        self.evaluate()


@SAMPLER_REGISTRY.register("latent_diffusion")
class LatentDiffusionSampler(LatentSampler):
    model_type = "latent_diffusion"


@SAMPLER_REGISTRY.register("latent_flow_matching")
class LatentFlowMatchingSampler(LatentSampler):
    model_type = "latent_flow_matching"


@SAMPLER_REGISTRY.register("latent_rectified_flow")
class LatentRectifiedFlowSampler(LatentSampler):
    model_type = "latent_rectified_flow"


__all__ = ["LatentSampler", "LatentDiffusionSampler", "LatentFlowMatchingSampler"]
