from __future__ import annotations

from typing import Any

import torch

from models.autoencoder.utils import encode_to_latent
from models.factory import ModelFactory
from scheduling import LatentAttentionAdapter, resolve_conditioning_adapter
from scheduling.builder import build_scheduler
from scheduling.lr import build_lr_scheduler
from noise import NOISE_REGISTRY
from utils.model_utils.diffusion_utils import build_diffusion_model
from .callbacks import CheckpointCallback, MetricsCSVCallback
from .generative_trainer import DiffusionTrainer, FlowMatchingTrainer, GenerativeTrainer, RectifiedFlowTrainer
from .registry import TRAINER_REGISTRY


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

        with torch.no_grad():
            target = encode_to_latent(self.vae_model, batch["target"].to(self.device))
            cond_raw = batch.get("image")
            mode = str(self.training_cfg.get("conditioning") or self.model_cfg.get("conditioning") or "").lower()
            if cond_raw is None:
                cond = None
            elif mode == "latent_attention":
                cond = cond_raw.to(self.device)
            else:
                cond = encode_to_latent(self.vae_model, cond_raw.to(self.device))
        return target, cond


class LatentGenerativeTrainer(LatentTrainerMixin, GenerativeTrainer):
    vae_model: torch.nn.Module | None = None
    _model_override: torch.nn.Module | None
    _noise_override: Any

    def _build_model(self) -> torch.nn.Module:
        if self._model_override is not None:
            return self._model_override
        return build_diffusion_model(self.raw_config, self.device, ckpt_path=None, set_eval=False)

    def __init__(
        self,
        config: dict,
        callbacks: list[Any] | None = None,
        event_bus=None,
        model_override: torch.nn.Module | None = None,
        noise_override: Any = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks, event_bus=event_bus)
        self._model_override = model_override
        self._noise_override = noise_override
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
        if self._noise_override is not None:
            self.noise_process = self._noise_override
        else:
            train_scheduler, _ = build_scheduler(scheduler_cfg, self.training_cfg)
            self.noise_process = NOISE_REGISTRY.build(self.noise_key, scheduler=train_scheduler)
        if not bool(self.model_cfg.get("use_presaved_latents", False)):
            self.vae_model = self._load_frozen_vae()
            mode = str(self.training_cfg.get("conditioning") or self.model_cfg.get("conditioning") or "").lower()
            if mode == "latent_attention":
                self.conditioning_adapter = LatentAttentionAdapter.create(self.vae_model)

    def _prepare_model_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._encode_batch_to_latent(batch)

@TRAINER_REGISTRY.register("latent_diffusion")
class LatentDiffusionTrainer(LatentGenerativeTrainer, DiffusionTrainer):
    noise_key = "ddpm"
    checkpoint_prefix = "latent_diff"


@TRAINER_REGISTRY.register("latent_flow_matching")
class LatentFlowMatchingTrainer(LatentGenerativeTrainer, FlowMatchingTrainer):
    noise_key = "flow_matching"
    checkpoint_prefix = "latent_flow"


@TRAINER_REGISTRY.register("latent_rectified_flow")
class LatentRectifiedFlowTrainer(LatentGenerativeTrainer, RectifiedFlowTrainer):
    noise_key = "rectified_flow"
    checkpoint_prefix = "latent_rf"


__all__ = [
    "LatentGenerativeTrainer",
    "LatentDiffusionTrainer",
    "LatentFlowMatchingTrainer",
    "LatentRectifiedFlowTrainer",
    "LatentTrainerMixin",
]
