from __future__ import annotations

import torch
import sys as _sys

import utils
from models.autoencoder.utils import encode_to_latent
from models.factory import ModelFactory
from scheduling import LatentAttentionAdapter
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
        payload = utils.safe_torch_load(ckpt_path, map_location=self.device)
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

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)
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

_module = _sys.modules[__name__]
if __name__.startswith("genlib.training."):
    _sys.modules.setdefault(__name__.replace("genlib.training.", "training.", 1), _module)
elif __name__.startswith("src.training."):
    _sys.modules.setdefault(__name__.replace("src.training.", "training.", 1), _module)
    _sys.modules.setdefault(__name__.replace("src.training.", "genlib.training.", 1), _module)
elif __name__.startswith("training."):
    _sys.modules.setdefault(__name__.replace("training.", "src.training.", 1), _module)
    _sys.modules.setdefault(__name__.replace("training.", "genlib.training.", 1), _module)
del _module, _sys
