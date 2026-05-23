from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.types import unwrap_model_prediction
from models.factory import ModelFactory
from models.vae.constants import LATENT_SCALE
from scheduling import clear_latent_attention_vae, configure_latent_attention_vae, resolve_conditioning_adapter
from scheduling.builder import build_scheduler
from scheduling.lr import build_lr_scheduler
from noise import NOISE_REGISTRY
from utils.model_utils.diffusion_utils import build_diffusion_model
from .callbacks import CheckpointCallback, MetricsCSVCallback
from .generative_trainer import DiffusionTrainer, FlowMatchingTrainer
from .registry import TRAINER_REGISTRY


class LatentCacheDataset(Dataset):
    """Dataset wrapper that reads precomputed latent tensors from `.pt` files."""

    def __init__(self, cache_dir: str | Path, split: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Latent cache directory not found: {self.cache_dir}")
        if split:
            split_dir = self.cache_dir / str(split)
            if not split_dir.exists():
                raise FileNotFoundError(f"Latent cache split directory not found: {split_dir}")
            self.files = sorted(split_dir.glob("*.pt"))
        else:
            direct = sorted(self.cache_dir.glob("*.pt"))
            if direct:
                self.files = direct
            else:
                self.files = sorted(self.cache_dir.rglob("*.pt"))
        if not self.files:
            raise ValueError(f"No .pt latent files found in: {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        payload = torch.load(self.files[idx], map_location="cpu")
        if isinstance(payload, dict):
            if "target" not in payload:
                raise KeyError(f"Latent file '{self.files[idx]}' must contain a 'target' key.")
            target = payload["target"]
            image = payload.get("image")
            out = {"target": target}
            if image is not None:
                out["image"] = image
            return out
        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"Unsupported latent payload type '{type(payload).__name__}' in '{self.files[idx]}'.")
        return {"target": payload}


class LatentTrainerMixin:
    vae_model: torch.nn.Module | None

    def _load_frozen_vae(self) -> torch.nn.Module:
        vae_cfg = dict(self.model_cfg.get("vae", {}))
        if not vae_cfg:
            raise ValueError("Latent training requires model.vae config to construct the frozen VAE.")
        vae_cfg["model_type"] = "vae"
        vae_cfg.setdefault("latent_type", "kl")

        vae_config = {"model": vae_cfg}
        vae = ModelFactory.build(vae_config).to(self.device)

        ckpt_path = self.model_cfg.get("vae_checkpoint")
        if not ckpt_path:
            raise ValueError("Latent training requires model.vae_checkpoint.")
        payload = torch.load(ckpt_path, map_location=self.device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        vae.load_state_dict(state)
        vae.eval()

        for param in vae.parameters():
            param.requires_grad_(False)
        return vae

    def _encode_batch_to_latent(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        use_presaved = bool(self.model_cfg.get("use_presaved_latents", False))
        if use_presaved:
            target = batch["target"].to(self.device)
            cond = batch.get("image")
            cond = cond.to(self.device) if cond is not None else None
            return target, cond

        if self.vae_model is None:
            raise RuntimeError("Latent trainer called before VAE initialization.")

        def _encode_tensor(x: torch.Tensor) -> torch.Tensor:
            inp = x
            if hasattr(self.vae_model, "image_to_model_range"):
                inp = self.vae_model.image_to_model_range(inp)
            try:
                encoded = self.vae_model.encode(inp, normalize=True)
                if isinstance(encoded, torch.Tensor):
                    return encoded
            except TypeError:
                pass
            posterior = self.vae_model.encode(inp, normalize=False)
            if isinstance(posterior, torch.Tensor):
                return posterior
            if not hasattr(posterior, "mode"):
                raise TypeError(f"Unsupported VAE encode output type '{type(posterior).__name__}'.")
            return posterior.mode() * LATENT_SCALE

        with torch.no_grad():
            target = _encode_tensor(batch["target"].to(self.device))
            cond_raw = batch.get("image")
            mode = str(self.training_cfg.get("conditioning") or self.model_cfg.get("conditioning") or "").lower()
            if cond_raw is None:
                cond = None
            elif mode == "latent_attention":
                cond = cond_raw.to(self.device)
            else:
                cond = _encode_tensor(cond_raw.to(self.device))
        return target, cond


class _LatentGenerativeTrainer(LatentTrainerMixin):
    vae_model: torch.nn.Module | None = None

    def _build_model(self) -> torch.nn.Module:
        cfg = dict(self.raw_config)
        model_cfg = dict(cfg.get("model", {}))
        model_type = str(model_cfg.get("model_type", "")).lower()
        if model_type == "latent_flow_matching":
            model_cfg["model_type"] = "flow_matching"
        cfg["model"] = model_cfg
        return build_diffusion_model(cfg, self.device, ckpt_path=None, set_eval=False)

    def __init__(self, config: dict, callbacks: list[Any] | None = None) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.grad_accum = max(1, int(self.training_cfg.get("gradient_accumulation_steps", 1)))
        self.latent_norm = self.training_cfg.get("latent_norm")
        self.conditioning_dropout = float(self.training_cfg.get("conditioning_dropout", 0.0))
        raw_mode = self.training_cfg.get("conditioning") or self.model_cfg.get("conditioning")
        self.conditioning_adapter = resolve_conditioning_adapter(raw_mode)
        self.noise_process = None

        if callbacks is None:
            self.callbacks = [
                CheckpointCallback(
                    filename_prefix=self.checkpoint_prefix,
                    monitor="val_loss" if self.training_cfg.get("validate", True) else "loss",
                    mode="min",
                    save_every=int(self.training_cfg.get("save_every", 0)),
                ),
                MetricsCSVCallback(),
            ]

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.optimizer is None:
            raise RuntimeError("Latent trainer scheduler called before optimizer initialization.")
        return build_lr_scheduler(self.optimizer, self.training_cfg)

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)
        scheduler_cfg = self.model_cfg.get("scheduler", {})
        train_scheduler, _ = build_scheduler(scheduler_cfg, self.training_cfg)
        self.noise_process = NOISE_REGISTRY.build(self.noise_key, scheduler=train_scheduler)
        if not bool(self.model_cfg.get("use_presaved_latents", False)):
            self.vae_model = self._load_frozen_vae()
            mode = str(self.training_cfg.get("conditioning") or self.model_cfg.get("conditioning") or "").lower()
            if mode == "latent_attention":
                configure_latent_attention_vae(self.vae_model)

    def _run_step(self, batch: dict, *, epoch: int, train: bool) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Latent trainer _run_step called before model initialization.")
        if self.optimizer is None:
            raise RuntimeError("Latent trainer _run_step called before optimizer initialization.")
        if self.scaler is None:
            raise RuntimeError("Latent trainer _run_step called before scaler initialization.")
        if self.noise_process is None:
            raise RuntimeError("Latent trainer _run_step called before noise process initialization.")

        clean, cond = self._encode_batch_to_latent(batch)
        bs = clean.size(0)
        chunk_size = max(1, (bs + self.grad_accum - 1) // self.grad_accum)
        clean_chunks = clean.split(chunk_size)
        cond_chunks = cond.split(chunk_size) if cond is not None else [None] * len(clean_chunks)
        accum_steps = len(clean_chunks)
        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_samples = 0

        for clean_chunk, cond_chunk in zip(clean_chunks, cond_chunks):
            if train and cond_chunk is not None and self.conditioning_dropout > 0.0:
                if torch.rand(1, device=self.device).item() < self.conditioning_dropout:
                    cond_chunk = None
            noisy_batch = self.noise_process(clean_chunk, self.device)
            model_input = noisy_batch.noisy
            model_input, context = self.conditioning_adapter(model_input, cond_chunk, self.latent_norm)

            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                pred = (
                    self.model(model_input, noisy_batch.timesteps, context_ca=context)
                    if context is not None
                    else self.model(model_input, noisy_batch.timesteps)
                )
                pred = unwrap_model_prediction(pred)
                loss = F.mse_loss(pred, noisy_batch.target)

            if train:
                self._backward(loss / accum_steps)

            chunk_bs = clean_chunk.size(0)
            total_loss += float(loss.detach().item()) * chunk_bs
            total_samples += chunk_bs

        if train:
            self._step_optimizers(self.optimizer)

        denom = max(1, total_samples)
        return {"loss": total_loss / denom}

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return self._run_step(batch, epoch=epoch, train=True)

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return self._run_step(batch, epoch=epoch, train=False)

    def fit(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        try:
            return super().fit(train_dataset, val_dataset=val_dataset, resume=resume)
        finally:
            clear_latent_attention_vae()


@TRAINER_REGISTRY.register("latent_diffusion")
class LatentDiffusionTrainer(_LatentGenerativeTrainer, DiffusionTrainer):
    noise_key = "ddpm"
    checkpoint_prefix = "latent_diff"


@TRAINER_REGISTRY.register("latent_flow_matching")
class LatentFlowMatchingTrainer(_LatentGenerativeTrainer, FlowMatchingTrainer):
    noise_key = "flow_matching"
    checkpoint_prefix = "latent_flow"


__all__ = [
    "LatentCacheDataset",
    "LatentDiffusionTrainer",
    "LatentFlowMatchingTrainer",
    "LatentTrainerMixin",
]
