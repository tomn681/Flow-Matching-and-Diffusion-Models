from __future__ import annotations

from pathlib import Path

import torch

import utils
from models.factory import ModelFactory
from models.vae.constants import LATENT_SCALE
from pipelines.utils import resolve_conditioning_mode
from utils.dataset_utils import save_output_tensor
from utils.model_utils.diffusion_utils import build_diffusion_model, decode_diffusion_batch
from utils.sampling_utils import (
    build_sampling_dataset,
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

    def _load_frozen_vae(self, cfg: dict, device: torch.device) -> torch.nn.Module:
        model_cfg = cfg.get("model", {})
        vae_cfg = dict(model_cfg.get("vae", {}))
        if not vae_cfg:
            raise ValueError("Latent sampling requires config.model.vae.")
        vae_cfg["model_type"] = "vae"
        vae_cfg.setdefault("latent_type", "kl")
        vae = ModelFactory.build({"model": vae_cfg}).to(device)

        ckpt_path = model_cfg.get("vae_checkpoint")
        if not ckpt_path:
            raise ValueError("Latent sampling requires config.model.vae_checkpoint.")
        payload = torch.load(ckpt_path, map_location=device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        vae.load_state_dict(state)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad_(False)
        return vae

    @staticmethod
    def _encode_vae_tensor(vae: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        inp = vae.image_to_model_range(x) if hasattr(vae, "image_to_model_range") else x
        try:
            encoded = vae.encode(inp, normalize=True)
            if isinstance(encoded, torch.Tensor):
                return encoded
        except TypeError:
            pass
        posterior = vae.encode(inp, normalize=False)
        if isinstance(posterior, torch.Tensor):
            return posterior
        if not hasattr(posterior, "mode"):
            raise TypeError(f"Unsupported VAE encode output type '{type(posterior).__name__}'.")
        return posterior.mode() * LATENT_SCALE

    @staticmethod
    def _decode_vae_tensor(vae: torch.nn.Module, z: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
        raw = vae.decode(z, denorm=True)
        if hasattr(vae, "raw_output_to_image"):
            return vae.raw_output_to_image(raw, recon_type=recon_type)
        return raw

    def _model_cfg_for_build(self, cfg: dict) -> dict:
        mapped = dict(cfg)
        model_cfg = dict(mapped.get("model", {}))
        if self.model_type == "latent_flow_matching":
            model_cfg["model_type"] = "flow_matching"
        mapped["model"] = model_cfg
        return mapped

    def encode(self) -> None:
        raise NotImplementedError("LatentSampler.encode is not implemented for C5.")

    def decode(self) -> None:
        ckpt_dir = Path(self.ckpt_dir)
        cfg = load_run_config(ckpt_dir)
        ckpt_path = resolve_checkpoint(ckpt_dir, self.model_type)
        training_cfg = cfg["training"]
        model_cfg = cfg["model"]

        utils.set_seed(self.seed)
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = utils.resolve_device(self.device, default_device)

        dataset = build_sampling_dataset(cfg, self.data_txt, save_tensor_cache_override=self.save_tensor_cache)
        selected_indices = resolve_sample_indices(dataset, self.num_samples, seed=self.seed)
        output_root = resolve_output_root(ckpt_dir, self.output_dir, self.save)
        predicted_root = output_root / "predicted" if output_root is not None else None

        model = build_diffusion_model(self._model_cfg_for_build(cfg), device, ckpt_path=ckpt_path)
        vae = self._load_frozen_vae(cfg, device)
        conditioning_mode = resolve_conditioning_mode(training_cfg.get("conditioning") or model_cfg.get("conditioning"))
        sampling_mode = "attention" if conditioning_mode == "latent_attention" else conditioning_mode
        use_presaved = bool(model_cfg.get("use_presaved_latents", False))
        recon_type = str(training_cfg.get("recon_type", "l1"))

        for indices, samples in progress_batches(dataset, self.batch_size, f"{self.model_type} decode", indices=selected_indices):
            target = torch.stack([s["target"] for s in samples], dim=0).to(device)
            target_latent = target if use_presaved else self._encode_vae_tensor(vae, target)

            cond = None
            if conditioning_mode in {"attention", "concatenate", "latent_attention"}:
                cond_list = [s.get("image") for s in samples]
                if all(c is not None for c in cond_list):
                    cond_raw = torch.stack(cond_list, dim=0).to(device)
                    cond = cond_raw if use_presaved else self._encode_vae_tensor(vae, cond_raw)

            latent_pred = decode_diffusion_batch(
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

            generated = self._decode_vae_tensor(vae, latent_pred, recon_type=recon_type).clamp(0.0, 1.0)

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
        raise NotImplementedError("LatentSampler.evaluate is not implemented for C5.")

    def debug_compare(self) -> None:
        raise NotImplementedError("LatentSampler.debug_compare is not implemented for C5.")


@SAMPLER_REGISTRY.register("latent_diffusion")
class LatentDiffusionSampler(LatentSampler):
    model_type = "latent_diffusion"


@SAMPLER_REGISTRY.register("latent_flow_matching")
class LatentFlowMatchingSampler(LatentSampler):
    model_type = "latent_flow_matching"


__all__ = ["LatentSampler", "LatentDiffusionSampler", "LatentFlowMatchingSampler"]
