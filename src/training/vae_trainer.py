from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any

import torch
from torch import autocast
from torch.optim import AdamW
from core.types import TrainingState

from core.types import ModelOutput
from losses import LOSS_REGISTRY, LossAssembler
from losses.adversarial import GANDiscriminatorLoss, GANGeneratorLoss
from losses.perceptual import PerceptualLossComponent
from scheduling.lr import build_lr_scheduler
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback, TensorBoardCallback, VisualizationCallback
from .registry import TRAINER_REGISTRY
from utils.model_utils.vae_utils import build_vae_model
from models.autoencoder.utils import apply_input_normalize, sync_autoencoder_input_range
import utils


@TRAINER_REGISTRY.register("vae")
class VAETrainer(BaseTrainer):
    """Tier-1 VAE trainer using legacy model factory + new callback/loss plumbing."""

    supports_model_override = True
    supports_losses_override = True

    def __init__(
        self,
        config: dict,
        callbacks: list[Any] | None = None,
        event_bus=None,
        model_override: torch.nn.Module | None = None,
        losses_override: LossAssembler | None = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks, event_bus=event_bus)
        self._model_override = model_override
        self._losses_override = losses_override

        training_cfg = self.training_cfg
        self.recon_type = str(training_cfg.get("recon_type", "l1")).lower()
        self.kl_weight = float(training_cfg.get("kl_weight", 0.0))
        self.kl_anneal_steps = int(training_cfg.get("kl_anneal_steps", 0))
        self.codebook_weight = float(training_cfg.get("codebook_weight", 1.0))
        self.allow_microbatching = bool(training_cfg.get("allow_microbatching", True))
        self.perceptual_weight = float(training_cfg.get("perceptual_weight", 0.0))
        self.gan_weight = float(training_cfg.get("gan_weight", 0.0))
        self.gan_start_epoch = int(training_cfg.get("gan_start", 0))
        gan_start_steps = training_cfg.get("gan_start_steps")
        self.gan_start_steps = None if gan_start_steps is None else int(gan_start_steps)
        self.disc_lr = float(training_cfg.get("disc_lr", training_cfg.get("learning_rate", 1e-4)))
        self.input_normalize = str(training_cfg.get("input_normalize", "centered")).lower()

        self.loss_assembler: LossAssembler | None = None
        self.perceptual_component: PerceptualLossComponent | None = None
        self.gan_generator_component: GANGeneratorLoss | None = None
        self.gan_discriminator_component: GANDiscriminatorLoss | None = None
        self.discriminator: torch.nn.Module | None = None
        self.disc_optimizer: torch.optim.Optimizer | None = None
        self.perceptual_device = torch.device("cpu")
        self.disc_device = torch.device("cpu")
        self._metric_keys: list[str] = ["loss"]

    def _build_default_callbacks(self) -> list[Any]:
        return [
            CheckpointCallback(
                filename_prefix="vae",
                monitor="val_loss" if self.training_cfg.get("validate", True) else "loss",
                mode="min",
                save_every=int(self.training_cfg.get("save_every", 0)),
            ),
            MetricsCSVCallback(),
            TensorBoardCallback(),
            VisualizationCallback(every_n_epochs=int(self.training_cfg.get("save_images_every", 1))),
        ]

    @classmethod
    def from_config(cls, path_or_dict: str | Path | dict) -> "VAETrainer":
        if isinstance(path_or_dict, (str, Path)):
            cfg = utils.load_json_config(path_or_dict)
        else:
            cfg = path_or_dict
        return cls(config=cfg)

    def _build_model(self) -> torch.nn.Module:
        if self._model_override is not None:
            return self._model_override
        return build_vae_model(self.raw_config, self.device, ckpt_path=None, set_eval=False)

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.optimizer is None:
            raise RuntimeError("VAETrainer._build_lr_scheduler called before optimizer initialization.")
        return build_lr_scheduler(self.optimizer, self.training_cfg)

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)
        sync_autoencoder_input_range(self.model, self.raw_config, input_normalize=self.input_normalize)
        if "input_normalize" not in self.training_cfg:
            logging.warning(
                "VAE training config does not set training.input_normalize. "
                "Defaulting to 'centered'. Set 'positive' for [0,1] medical-image inputs "
                "or 'symmetric'/'centered' for [-1,1]-style encoder input."
            )

        model_cfg = self.raw_config.get("model", {})
        reg_type = str(self.training_cfg.get("reg_type", "kl")).lower()
        latent_type = str(model_cfg.get("latent_type", "kl")).lower()
        effective_codebook_weight = self.codebook_weight if (latent_type == "vq" or reg_type == "vq") else 0.0

        try:
            self.recon_component = LOSS_REGISTRY.build(self.recon_type, weight=1.0)
        except KeyError as exc:
            available = ", ".join(LOSS_REGISTRY.list())
            raise ValueError(f"Unsupported recon_type '{self.recon_type}'. Available losses: {available}.") from exc
        components = [self.recon_component]
        self.kl_component = None
        self.vq_component = None
        if self.kl_weight > 0 or self.kl_anneal_steps > 0:
            self.kl_component = LOSS_REGISTRY.build("kl", weight=self.kl_weight)
            components.append(self.kl_component)
        if effective_codebook_weight > 0:
            self.vq_component = LOSS_REGISTRY.build("vq", weight=effective_codebook_weight)
            components.append(self.vq_component)

        if self.perceptual_weight > 0:
            self.perceptual_component = LOSS_REGISTRY.build("perceptual", weight=self.perceptual_weight, resize=True)
            self.perceptual_device = utils.resolve_device(self.training_cfg.get("perceptual_device"), self.device)
            self.perceptual_component = self.perceptual_component.to(self.perceptual_device)
            components.append(self.perceptual_component)

        if self.gan_weight > 0:
            self.gan_generator_component = LOSS_REGISTRY.build(
                "gan_generator",
                weight=self.gan_weight,
                start_epoch=self.gan_start_epoch,
                start_step=self.gan_start_steps,
            )
            self.gan_discriminator_component = LOSS_REGISTRY.build(
                "gan_discriminator",
                weight=1.0,
                start_epoch=self.gan_start_epoch,
                start_step=self.gan_start_steps,
            )
            components.append(self.gan_generator_component)

            self.discriminator = self.model.make_discriminator().to(self.device)
            self.disc_device = utils.resolve_device(self.training_cfg.get("disc_device"), self.device)
            self.discriminator = self.discriminator.to(self.disc_device)
            self.disc_optimizer = AdamW(self.discriminator.parameters(), lr=self.disc_lr)

        self.loss_assembler = self._losses_override if self._losses_override is not None else LossAssembler(components)
        assembler_keys = self.loss_assembler.metric_keys()
        self._metric_keys = ["loss"] + assembler_keys + (["d_gan"] if self.gan_weight > 0 else [])

        self.sample_count = int(self.training_cfg.get("visual_samples", 20))
        self.visual_enabled = bool(self.training_cfg.get("save_images", True))
        eval_source = val_dataset if val_dataset is not None else train_dataset
        self.sample_batch = utils.prepare_eval_batch(eval_source, self.sample_count, self.device, seed=self.training_cfg.get("seed"))
        self.latent_shape = utils.latent_shape(model_cfg)

    def _run_step(self, batch: dict, *, epoch: int, train: bool) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("VAETrainer._run_step called before model initialization.")
        if self.optimizer is None:
            raise RuntimeError("VAETrainer._run_step called before optimizer initialization.")
        if self.scaler is None:
            raise RuntimeError("VAETrainer._run_step called before AMP scaler initialization.")
        if self.loss_assembler is None:
            raise RuntimeError("VAETrainer._run_step called before loss assembler initialization.")

        raw_target = batch["target"].to(self.device)
        raw_inputs = batch.get("image", batch["target"]).to(self.device)
        inputs = apply_input_normalize(raw_inputs, self.input_normalize)

        batch_size = int(self.training_cfg.get("batch_size", 4))
        current_micro = batch_size
        if not train:
            current_micro = min(current_micro, inputs.size(0))

        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.disc_optimizer is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)

        totals: dict[str, float] = {k: 0.0 for k in self._metric_keys}
        sample_count = 0

        while True:
            try:
                chunks = inputs.split(current_micro)
                raw_target_chunks = raw_target.split(current_micro)
                accum_steps = len(chunks)

                for chunk, raw_chunk in zip(chunks, raw_target_chunks):
                    with autocast(device_type=self.device.type, enabled=use_amp):
                        output = self.model(chunk, sample_posterior=train)
                        if not isinstance(output, ModelOutput):
                            raise TypeError(f"Expected ModelOutput from model.forward, got {type(output).__name__}.")

                        rec = output.reconstruction
                        rec_img = self.model.raw_output_to_image(rec, recon_type=self.recon_type)

                        if self.kl_component is not None and self.kl_anneal_steps > 0:
                            step_for_anneal = max(1, self.global_step + 1)
                            self.kl_component.weight = self.kl_weight * min(1.0, step_for_anneal / max(1, self.kl_anneal_steps))
                        elif self.kl_component is not None:
                            self.kl_component.weight = self.kl_weight

                        disc_active = self._disc_is_active(epoch=epoch)
                        if disc_active:
                            rec_d = self._ensure_device(rec_img, self.disc_device)
                            fake_pred = self.discriminator(rec_d)
                        else:
                            fake_pred = None

                        total_loss, parts = self.loss_assembler(
                            context={
                                "reconstruction": rec,
                                "reconstruction_image": rec_img,
                                "target": raw_chunk,
                                "posterior": output.posterior,
                                "codebook_loss": output.codebook_loss,
                                "fake_pred": fake_pred,
                                "device": self.device,
                                "dtype": rec.dtype,
                            },
                            epoch=epoch,
                            global_step=self.global_step,
                        )

                    if train:
                        self._backward(total_loss / accum_steps)

                    d_loss = self._discriminator_step(
                        rec_img=rec_img,
                        raw_chunk=raw_chunk,
                        epoch=epoch,
                        train=train,
                        accum_steps=accum_steps,
                        use_amp=use_amp,
                        dtype=rec.dtype,
                    )

                    chunk_bs = chunk.size(0)
                    sample_count += chunk_bs
                    totals["loss"] += float(total_loss.detach().item()) * chunk_bs
                    for name, value in parts.items():
                        totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * chunk_bs
                    if "d_gan" in totals:
                        totals["d_gan"] = totals.get("d_gan", 0.0) + float(d_loss) * chunk_bs

                if train:
                    self._step_optimizers(self.optimizer, self.disc_optimizer)
                break
            except RuntimeError as err:
                if (not train) or ("out of memory" not in str(err).lower()):
                    raise
                if not self.allow_microbatching:
                    raise RuntimeError("Batch too large and microbatching is disabled.") from err
                torch.cuda.empty_cache()
                if current_micro <= 1:
                    raise
                current_micro = max(1, current_micro // 2)

        denom = max(1, sample_count)
        return {k: v / denom for k, v in totals.items()}

    def _build_state(self, *, epoch: int, metrics: dict[str, float]):
        state = super()._build_state(epoch=epoch, metrics=metrics)
        if self.disc_optimizer is not None:
            state.extra["disc_optimizer"] = self.disc_optimizer.state_dict()
        return state

    def _build_checkpoint_dict(self, state: TrainingState) -> dict[str, Any]:
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
        rec_img: torch.Tensor,
        raw_chunk: torch.Tensor,
        epoch: int,
        train: bool,
        accum_steps: int,
        use_amp: bool,
        dtype: torch.dtype,
    ) -> float:
        if not self._disc_is_active(epoch=epoch):
            return 0.0
        with autocast(device_type=self.disc_device.type, enabled=use_amp):
            rec_d = self._ensure_device(rec_img.detach(), self.disc_device)
            raw_d = self._ensure_device(raw_chunk.detach(), self.disc_device)
            real_pred = self.discriminator(raw_d)
            fake_pred = self.discriminator(rec_d)
            d_loss = self.gan_discriminator_component.compute(
                context={
                    "real_pred": real_pred,
                    "fake_pred": fake_pred,
                    "device": self.device,
                    "dtype": dtype,
                }
            )
        if train:
            self._backward(d_loss / accum_steps)
        return float(d_loss.detach().item())

    def render_visuals(self, *, output_root: Path, epoch: int, metrics: dict, state: dict) -> None:
        if not self.visual_enabled:
            return
        if self.model is None:
            raise RuntimeError("VAETrainer.render_visuals called before model initialization.")

        self.model.eval()
        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"
        with torch.no_grad():
            sample_inputs = apply_input_normalize(self.sample_batch, self.input_normalize)
            with autocast(device_type=self.device.type, enabled=use_amp):
                output = self.model(sample_inputs, sample_posterior=False)
            if not isinstance(output, ModelOutput):
                raise TypeError(f"Expected ModelOutput from model.forward, got {type(output).__name__}.")
            rec = output.reconstruction
            rec_vis = self.model.raw_output_to_image(rec, recon_type=self.recon_type)
            input_vis = self.sample_batch.clamp(0.0, 1.0)
            input_grid = utils.make_grid(input_vis, 4, 5)
            rec_grid = utils.make_grid(rec_vis, 4, 5)
            noise = torch.randn((self.sample_count, *self.latent_shape), device=self.device)
            with autocast(device_type=self.device.type, enabled=use_amp):
                gen = self.model.decode(noise)
            gen_vis = self.model.raw_output_to_image(gen, recon_type=self.recon_type)
            gen_grid = utils.make_grid(gen_vis, 4, 5)

        utils.save_image(input_grid, output_root / "input.png")
        utils.save_image(rec_grid, output_root / "recon.png")
        utils.save_image(gen_grid, output_root / "gen.png")
        self.model.train()
