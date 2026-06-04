from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from losses.adversarial import GANDiscriminatorLoss, GANGeneratorLoss
from noise import NOISE_REGISTRY
from nn.losses.adversarial import PatchDiscriminator
from scheduling import (
    ChainAdapterSpec,
    ConditioningChain,
    TextConditioningAdapter,
    build_text_conditioning_adapter,
    resolve_conditioning_adapter,
)
from scheduling.lr import build_lr_scheduler
from scheduling.builder import build_scheduler
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback, TensorBoardCallback
from .registry import TRAINER_REGISTRY
from utils.model_utils.diffusion_utils import build_diffusion_model
from core.types import unwrap_model_prediction
from core import Discriminatable
import utils


class GenerativeTrainer(BaseTrainer, abc.ABC):
    """Tier-1 trainer for diffusion and flow-matching UNet models."""

    noise_key: str
    checkpoint_prefix: str

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
        self.conditioning_mode = str(self.training_cfg.get("conditioning") or self.model_cfg.get("conditioning") or "none").strip().lower()
        text_cfg = self.training_cfg.get("text_encoder")
        needs_deferred_text_adapter = self.conditioning_mode == "text" or (
            self.conditioning_mode == "chain" and isinstance(text_cfg, dict) and bool(text_cfg)
        )
        self.conditioning_adapter = (
            resolve_conditioning_adapter("none")
            if needs_deferred_text_adapter
            else self._build_conditioning_adapter(self.conditioning_mode)
        )
        self.noise_process = None
        self.gan_weight = float(self.training_cfg.get("gan_weight", 0.0))
        self.gan_space = str(self.training_cfg.get("gan_space", "auto")).strip().lower()
        self.gan_start_epoch = int(self.training_cfg.get("gan_start", 0))
        gan_start_steps = self.training_cfg.get("gan_start_steps")
        self.gan_start_steps = None if gan_start_steps is None else int(gan_start_steps)
        self.disc_lr = float(self.training_cfg.get("disc_lr", self.training_cfg.get("learning_rate", 1e-4)))
        self.discriminator: torch.nn.Module | None = None
        self.disc_optimizer: torch.optim.Optimizer | None = None
        self.gan_generator_component: GANGeneratorLoss | None = None
        self.gan_discriminator_component: GANDiscriminatorLoss | None = None

    def _build_default_callbacks(self) -> list[Any]:
        return [
            CheckpointCallback(
                filename_prefix=self.checkpoint_prefix,
                monitor="val_loss" if self.training_cfg.get("validate", True) else "loss",
                mode="min",
                save_every=int(self.training_cfg.get("save_every", 0)),
            ),
            MetricsCSVCallback(),
            TensorBoardCallback(),
        ]

    @classmethod
    def from_config(cls, path_or_dict: str | Path | dict) -> "GenerativeTrainer":
        if isinstance(path_or_dict, (str, Path)):
            cfg = utils.load_json_config(path_or_dict)
        else:
            cfg = path_or_dict
        return cls(config=cfg)

    def _build_model(self) -> torch.nn.Module:
        if self._model_override is not None:
            return self._model_override
        return build_diffusion_model(self.raw_config, self.device, ckpt_path=None, set_eval=False)

    def _build_conditioning_adapter(self, mode: str):
        if mode == "text":
            return self._build_text_conditioning_adapter()
        if mode == "chain":
            specs = [
                ChainAdapterSpec("concatenate", resolve_conditioning_adapter("concatenate")),
                ChainAdapterSpec("attention", resolve_conditioning_adapter("attention")),
            ]
            text_cfg = self.training_cfg.get("text_encoder")
            if isinstance(text_cfg, dict) and text_cfg:
                specs.append(ChainAdapterSpec("text", self._build_text_conditioning_adapter()))
            return ConditioningChain(specs)
        return resolve_conditioning_adapter(mode)

    def _build_text_conditioning_adapter(self) -> TextConditioningAdapter:
        text_cfg = self.training_cfg.get("text_encoder")
        if not isinstance(text_cfg, dict) or not text_cfg:
            raise ValueError(
                "Text conditioning requires training.text_encoder configuration with at least a 'kind' field."
            )
        kind = str(text_cfg.get("kind", "clip"))
        model_name = text_cfg.get("model_name")
        return build_text_conditioning_adapter(kind=kind, model_name=model_name, device=self.device)

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.optimizer is None:
            raise RuntimeError("GenerativeTrainer._build_lr_scheduler called before optimizer initialization.")
        return build_lr_scheduler(self.optimizer, self.training_cfg)

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)
        self.conditioning_adapter = self._build_conditioning_adapter(self.conditioning_mode)
        if self._noise_override is not None:
            self.noise_process = self._noise_override
        else:
            scheduler_cfg = self.model_cfg.get("scheduler", {})
            train_scheduler, _ = build_scheduler(scheduler_cfg, self.training_cfg)
            noise_kwargs: dict[str, Any] = {"scheduler": train_scheduler}
            if self.noise_key == "reflow":
                pairs_dir = self.training_cfg.get("reflow_pairs_dir")
                if not pairs_dir:
                    raise ValueError("Reflow training requires training.reflow_pairs_dir.")
                noise_kwargs["pairs_dir"] = str(pairs_dir)
            self.noise_process = NOISE_REGISTRY.build(self.noise_key, **noise_kwargs)
        if self.gan_weight > 0.0:
            if self.gan_space == "auto":
                if self.noise_key == "consistency":
                    self.gan_space = "clean"
                else:
                    raise ValueError(
                        "GenerativeTrainer GAN mode requires explicit training.gan_space for this noise process. "
                        "Use 'prediction' (opt-in, target-space adversarial) or disable gan_weight."
                    )
            if self.gan_space not in {"prediction", "clean"}:
                raise ValueError("training.gan_space must be one of: auto, prediction, clean.")
            self.gan_generator_component = GANGeneratorLoss(
                weight=self.gan_weight,
                start_epoch=self.gan_start_epoch,
                start_step=self.gan_start_steps,
            )
            self.gan_discriminator_component = GANDiscriminatorLoss(
                weight=1.0,
                start_epoch=self.gan_start_epoch,
                start_step=self.gan_start_steps,
            )
            self.discriminator = self._build_discriminator().to(self.device)
            self.disc_optimizer = AdamW(self.discriminator.parameters(), lr=self.disc_lr)

    def _build_discriminator(self) -> torch.nn.Module:
        if self.model is not None and isinstance(self.model, Discriminatable):
            disc = self.model.make_discriminator()
            if disc is not None:
                return disc
        in_channels = int(
            self.model_cfg.get(
                "out_channels",
                self.model_cfg.get("unet", {}).get("out_channels", self.training_cfg.get("channels", 1)),
            )
        )
        spatial_dims = int(self.model_cfg.get("unet", {}).get("spatial_dims", 2))
        return PatchDiscriminator(in_channels=in_channels, spatial_dims=spatial_dims)

    def _extract_text_conditioning(self, batch: dict):
        text = batch.get("text")
        if text is None:
            text = batch.get("prompt")
        return text

    def _prepare_model_batch(self, batch: dict) -> tuple[torch.Tensor, object | None]:
        clean = batch["target"].to(self.device)
        cond = batch.get("image")
        cond = cond.to(self.device) if torch.is_tensor(cond) else cond
        if self.conditioning_mode == "text":
            text_cond = self._extract_text_conditioning(batch)
            return clean, text_cond if text_cond is not None else cond
        if self.conditioning_mode == "chain":
            concat_cond = batch.get("concat_cond")
            attn_cond = batch.get("attn_cond")
            text_cond = self._extract_text_conditioning(batch)
            return clean, {
                "concatenate": concat_cond.to(self.device) if torch.is_tensor(concat_cond) else cond,
                "attention": attn_cond.to(self.device) if torch.is_tensor(attn_cond) else None,
                "text": text_cond,
            }
        return clean, cond

    @staticmethod
    def _split_conditioning_payload(cond, chunk_size: int):
        if cond is None:
            return []
        if torch.is_tensor(cond):
            return list(cond.split(chunk_size))
        if isinstance(cond, dict):
            value_splits = {key: GenerativeTrainer._split_conditioning_payload(value, chunk_size) for key, value in cond.items()}
            chunk_count = max((len(chunks) for chunks in value_splits.values()), default=0)
            return [
                {
                    key: (chunks[idx] if idx < len(chunks) else None)
                    for key, chunks in value_splits.items()
                }
                for idx in range(chunk_count)
            ]
        if isinstance(cond, (list, tuple)):
            return [list(cond[idx : idx + chunk_size]) for idx in range(0, len(cond), chunk_size)]
        raise TypeError(f"Unsupported conditioning payload type for chunking: {type(cond).__name__}")

    def _run_step(self, batch: dict, *, epoch: int, train: bool) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("GenerativeTrainer._run_step called before model initialization.")
        if self.optimizer is None:
            raise RuntimeError("GenerativeTrainer._run_step called before optimizer initialization.")
        if self.scaler is None:
            raise RuntimeError("GenerativeTrainer._run_step called before scaler initialization.")
        if self.noise_process is None:
            raise RuntimeError("GenerativeTrainer._run_step called before noise process initialization.")

        clean, cond = self._prepare_model_batch(batch)

        bs = clean.size(0)
        chunk_size = max(1, (bs + self.grad_accum - 1) // self.grad_accum)
        clean_chunks = clean.split(chunk_size)
        cond_chunks = self._split_conditioning_payload(cond, chunk_size) if cond is not None else [None] * len(clean_chunks)
        if cond is not None and len(cond_chunks) != len(clean_chunks):
            raise ValueError("Conditioning payload chunk count does not match target chunk count.")
        accum_steps = len(clean_chunks)
        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.disc_optimizer is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
                self.discriminator.train()
        elif self.discriminator is not None:
            self.discriminator.eval()

        total_loss = 0.0
        total_d_loss = 0.0
        total_samples = 0

        for clean_chunk, cond_chunk in zip(clean_chunks, cond_chunks):
            noisy_batch = self.noise_process(clean_chunk, self.device)
            model_input = noisy_batch.noisy
            model_input, context = self.conditioning_adapter(model_input, cond_chunk, self.latent_norm)
            if train and self.conditioning_dropout > 0.0:
                drop_mask = torch.rand(clean_chunk.size(0), device=self.device) < self.conditioning_dropout
                if torch.any(drop_mask):
                    if context is not None:
                        context = context.clone()
                        context[drop_mask] = 0.0
                    if model_input.shape[1] > clean_chunk.shape[1]:
                        cond_channels = model_input.shape[1] - clean_chunk.shape[1]
                        model_input = model_input.clone()
                        model_input[drop_mask, -cond_channels:] = 0.0

            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                pred = (
                    self.model(model_input, noisy_batch.timesteps, context_ca=context)
                    if context is not None
                    else self.model(model_input, noisy_batch.timesteps)
                )
                pred = unwrap_model_prediction(pred)
                loss = F.mse_loss(pred, noisy_batch.target)
                if self._disc_is_active(epoch=epoch):
                    fake_for_gan, _real_for_gan = self._resolve_gan_tensors(
                        pred=pred, clean=clean_chunk, target=noisy_batch.target
                    )
                    fake_pred = self.discriminator(fake_for_gan)
                    g_adv = self.gan_generator_component.compute(
                        context={"fake_pred": fake_pred, "device": self.device, "dtype": pred.dtype}
                    )
                    loss = loss + self.gan_generator_component.weight * g_adv

            if train:
                self._backward(loss / accum_steps)

            d_loss = self._discriminator_step(
                pred=pred,
                clean=clean_chunk,
                target=noisy_batch.target,
                epoch=epoch,
                train=train,
                accum_steps=accum_steps,
            )
            chunk_bs = clean_chunk.size(0)
            total_loss += float(loss.detach().item()) * chunk_bs
            total_d_loss += float(d_loss) * chunk_bs
            total_samples += chunk_bs

        if train:
            self._step_optimizers(self.optimizer, self.disc_optimizer)

        denom = max(1, total_samples)
        metrics = {"loss": total_loss / denom}
        if self.gan_weight > 0.0:
            metrics["d_gan"] = total_d_loss / denom
        return metrics

    def _disc_is_active(self, *, epoch: int) -> bool:
        return (
            self.gan_generator_component is not None
            and self.gan_generator_component.is_active(epoch=epoch, global_step=self.global_step)
            and self.discriminator is not None
            and self.gan_discriminator_component is not None
        )

    def _discriminator_step(
        self,
        *,
        pred: torch.Tensor,
        clean: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        train: bool,
        accum_steps: int,
    ) -> float:
        if not self._disc_is_active(epoch=epoch):
            return 0.0
        fake_for_gan, real_for_gan = self._resolve_gan_tensors(pred=pred, clean=clean, target=target)
        fake_pred = self.discriminator(fake_for_gan.detach())
        real_pred = self.discriminator(real_for_gan.detach())
        d_loss = self.gan_discriminator_component.compute(
            context={"real_pred": real_pred, "fake_pred": fake_pred, "device": self.device, "dtype": pred.dtype}
        )
        if train:
            self._backward(d_loss / accum_steps)
        return float(d_loss.detach().item())

    def _resolve_gan_tensors(
        self,
        *,
        pred: torch.Tensor,
        clean: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gan_space == "prediction":
            return pred, target
        return pred, clean

    def _build_state(self, *, epoch: int, metrics: dict[str, float]):
        state = super()._build_state(epoch=epoch, metrics=metrics)
        if self.disc_optimizer is not None:
            state.extra["disc_optimizer"] = self.disc_optimizer.state_dict()
        return state

    def _build_checkpoint_dict(self, state):
        payload = super()._build_checkpoint_dict(state)
        payload["disc_optimizer"] = state.extra.get("disc_optimizer")
        return payload

    def _resume_from_payload(self, payload: dict[str, Any]) -> None:
        if self.disc_optimizer is not None and payload.get("disc_optimizer"):
            self.disc_optimizer.load_state_dict(payload["disc_optimizer"])

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return self._run_step(batch, epoch=epoch, train=True)

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return self._run_step(batch, epoch=epoch, train=False)


@TRAINER_REGISTRY.register("diffusion")
class DiffusionTrainer(GenerativeTrainer):
    noise_key = "ddpm"
    checkpoint_prefix = "diff"


@TRAINER_REGISTRY.register("flow_matching")
class FlowMatchingTrainer(GenerativeTrainer):
    noise_key = "flow_matching"
    checkpoint_prefix = "flow"


@TRAINER_REGISTRY.register("consistency")
class ConsistencyTrainer(GenerativeTrainer):
    noise_key = "consistency"
    checkpoint_prefix = "consistency"


@TRAINER_REGISTRY.register("edm")
class EDMTrainer(GenerativeTrainer):
    noise_key = "edm"
    checkpoint_prefix = "edm"


@TRAINER_REGISTRY.register("rectified_flow")
class RectifiedFlowTrainer(GenerativeTrainer):
    noise_key = "rectified_flow"
    checkpoint_prefix = "rectified_flow"


@TRAINER_REGISTRY.register("reflow")
class ReflowTrainer(GenerativeTrainer):
    noise_key = "reflow"
    checkpoint_prefix = "reflow"
