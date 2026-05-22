from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import autocast
from torch.optim import AdamW

from core.types import ModelOutput
from losses import LOSS_REGISTRY, LossAssembler
from losses.adversarial import GANDiscriminatorLoss, GANGeneratorLoss
from losses.perceptual import PerceptualLossComponent
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback, VisualizationCallback
from .registry import TRAINER_REGISTRY
from utils.model_utils.vae_utils import build_vae_model
import utils


@TRAINER_REGISTRY.register("vae")
class VAETrainer(BaseTrainer):
    """Tier-1 VAE trainer using legacy model factory + new callback/loss plumbing."""

    def __init__(self, config: dict, callbacks: list[Any] | None = None) -> None:
        super().__init__(config=config, callbacks=callbacks)

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

        if callbacks is None:
            self.callbacks = [
                CheckpointCallback(filename_prefix="vae", monitor="val_loss" if self.training_cfg.get("validate", True) else "loss", mode="min"),
                MetricsCSVCallback(
                    metric_keys=[
                        "loss",
                        "recon",
                        "kl",
                        "vq",
                        "perceptual",
                        "g_gan",
                        "d_gan",
                        "val_loss",
                        "val_recon",
                        "val_kl",
                        "val_vq",
                        "val_perceptual",
                        "val_g_gan",
                        "val_d_gan",
                    ]
                ),
                VisualizationCallback(every_n_epochs=int(self.training_cfg.get("save_images_every", 1))),
            ]

        self.loss_assembler: LossAssembler | None = None
        self.perceptual_component: PerceptualLossComponent | None = None
        self.gan_generator_component: GANGeneratorLoss | None = None
        self.gan_discriminator_component: GANDiscriminatorLoss | None = None
        self.discriminator: torch.nn.Module | None = None
        self.disc_optimizer: torch.optim.Optimizer | None = None
        self.perceptual_device = torch.device("cpu")
        self.disc_device = torch.device("cpu")

    @classmethod
    def from_config(cls, path_or_dict: str | Path | dict) -> "VAETrainer":
        if isinstance(path_or_dict, (str, Path)):
            cfg = utils.load_json_config(path_or_dict)
        else:
            cfg = path_or_dict
        return cls(config=cfg)

    def _build_model(self) -> torch.nn.Module:
        return build_vae_model(self.raw_config, self.device, ckpt_path=None, set_eval=False)

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)

        model_cfg = self.raw_config.get("model", {})
        reg_type = str(self.training_cfg.get("reg_type", "kl")).lower()
        latent_type = str(model_cfg.get("latent_type", "kl")).lower()
        effective_codebook_weight = self.codebook_weight if (latent_type == "vq" or reg_type == "vq") else 0.0

        try:
            self.recon_component = LOSS_REGISTRY.build(self.recon_type, weight=1.0)
        except KeyError as exc:
            available = ", ".join(LOSS_REGISTRY.list())
            raise ValueError(f"Unsupported recon_type '{self.recon_type}'. Available losses: {available}.") from exc
        self.kl_component = LOSS_REGISTRY.build("kl", weight=self.kl_weight)
        self.vq_component = LOSS_REGISTRY.build("vq", weight=effective_codebook_weight)
        components = [self.recon_component, self.kl_component, self.vq_component]

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

        self.loss_assembler = LossAssembler(components)

        self.sample_count = int(self.training_cfg.get("visual_samples", 20))
        self.visual_enabled = bool(self.training_cfg.get("save_images", True))
        eval_source = val_dataset if val_dataset is not None else train_dataset
        self.sample_batch = utils.prepare_eval_batch(eval_source, self.sample_count, self.device, seed=self.training_cfg.get("seed"))
        self.latent_shape = utils.latent_shape(model_cfg)

        resume_flag = resume if resume is not None else self.training_cfg.get("resume")
        if isinstance(resume_flag, str) and resume_flag.lower() == "none":
            resume_flag = None
        if resume_flag and self.disc_optimizer is not None:
            ckpt_path = Path(resume_flag)
            if ckpt_path.exists():
                payload = torch.load(ckpt_path, map_location=self.device)
                if payload.get("disc_optimizer"):
                    self.disc_optimizer.load_state_dict(payload["disc_optimizer"])

    def _run_step(self, batch: dict, *, epoch: int, train: bool) -> dict[str, float]:
        assert self.model is not None
        assert self.optimizer is not None
        assert self.scaler is not None
        assert self.loss_assembler is not None

        raw_inputs = batch["target"].to(self.device)
        inputs = self.model.image_to_model_range(raw_inputs)

        batch_size = int(self.training_cfg.get("batch_size", 4))
        current_micro = batch_size
        if not train:
            current_micro = min(current_micro, inputs.size(0))

        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.disc_optimizer is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)

        totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "vq": 0.0, "perceptual": 0.0, "g_gan": 0.0, "d_gan": 0.0}
        sample_count = 0

        while True:
            try:
                chunks = inputs.split(current_micro)
                raw_chunks = raw_inputs.split(current_micro)
                accum_steps = len(chunks)

                for chunk, raw_chunk in zip(chunks, raw_chunks):
                    with autocast(device_type=self.device.type, enabled=use_amp):
                        output = self.model(chunk, sample_posterior=train)
                        if not isinstance(output, ModelOutput):
                            raise TypeError(f"Expected ModelOutput from model.forward, got {type(output).__name__}.")

                        rec = output.reconstruction
                        rec_img = self.model.raw_output_to_image(rec, recon_type=self.recon_type)

                        if self.kl_anneal_steps > 0:
                            step_for_anneal = max(1, self.global_step + 1)
                            self.kl_component.weight = self.kl_weight * min(1.0, step_for_anneal / max(1, self.kl_anneal_steps))
                        else:
                            self.kl_component.weight = self.kl_weight

                        disc_active = (
                            self.gan_generator_component is not None
                            and self.gan_generator_component.is_active(epoch=epoch, global_step=self.global_step)
                            and self.discriminator is not None
                        )
                        if disc_active:
                            rec_d = rec_img if rec_img.device == self.disc_device else rec_img.to(self.disc_device)
                            fake_pred = self.discriminator(rec_d)
                        else:
                            fake_pred = None

                        if self.perceptual_component is not None:
                            rec_p = rec_img if rec_img.device == self.perceptual_device else rec_img.to(self.perceptual_device)
                            tgt_p = raw_chunk if raw_chunk.device == self.perceptual_device else raw_chunk.to(self.perceptual_device)
                            perceptual_pred = rec_p
                            perceptual_tgt = tgt_p
                        else:
                            perceptual_pred = rec_img
                            perceptual_tgt = raw_chunk

                        total_loss, parts = self.loss_assembler(
                            rec if self.recon_type in {"bce", "focal", "bce_focal"} else rec_img,
                            raw_chunk,
                            epoch=epoch,
                            global_step=self.global_step,
                            posterior=output.posterior,
                            codebook_loss=output.codebook_loss,
                            fake_pred=fake_pred,
                        )
                        if self.perceptual_component is not None:
                            p_loss = self.perceptual_component.compute(perceptual_pred, perceptual_tgt)
                            p_weighted = p_loss * self.perceptual_component.weight
                            total_loss = total_loss + p_weighted.to(device=self.device, dtype=total_loss.dtype)
                            parts[self.perceptual_component.name] = p_weighted.to(device=self.device, dtype=total_loss.dtype)

                    if train:
                        if self.scaler.is_enabled():
                            self.scaler.scale(total_loss / accum_steps).backward()
                        else:
                            (total_loss / accum_steps).backward()

                    if disc_active:
                        with autocast(device_type=self.disc_device.type, enabled=use_amp):
                            rec_detached = rec_img.detach()
                            raw_detached = raw_chunk.detach()
                            rec_d = rec_detached if rec_detached.device == self.disc_device else rec_detached.to(self.disc_device)
                            raw_d = raw_detached if raw_detached.device == self.disc_device else raw_detached.to(self.disc_device)
                            real_pred = self.discriminator(raw_d)
                            fake_pred_detached = self.discriminator(rec_d)
                            d_loss = self.gan_discriminator_component.compute(
                                rec_d,
                                raw_d,
                                real_pred=real_pred,
                                fake_pred=fake_pred_detached,
                            )
                        if train:
                            if self.scaler.is_enabled():
                                self.scaler.scale(d_loss / accum_steps).backward()
                            else:
                                (d_loss / accum_steps).backward()
                        parts[self.gan_discriminator_component.name] = d_loss.to(device=self.device, dtype=total_loss.dtype)
                    else:
                        d_loss = torch.tensor(0.0, device=self.device, dtype=total_loss.dtype)

                    chunk_bs = chunk.size(0)
                    sample_count += chunk_bs
                    totals["loss"] += float(total_loss.detach().item()) * chunk_bs
                    totals["recon"] += float(parts.get(self.recon_component.name, torch.tensor(0.0, device=self.device)).detach().item()) * chunk_bs
                    totals["kl"] += float(parts.get(self.kl_component.name, torch.tensor(0.0, device=self.device)).detach().item()) * chunk_bs
                    totals["vq"] += float(parts.get(self.vq_component.name, torch.tensor(0.0, device=self.device)).detach().item()) * chunk_bs
                    totals["perceptual"] += float(parts.get("perceptual", torch.tensor(0.0, device=self.device)).detach().item()) * chunk_bs
                    totals["g_gan"] += float(parts.get("g_gan", torch.tensor(0.0, device=self.device)).detach().item()) * chunk_bs
                    totals["d_gan"] += float(parts.get("d_gan", torch.tensor(0.0, device=self.device)).detach().item()) * chunk_bs

                if train:
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        if self.disc_optimizer is not None:
                            self.scaler.step(self.disc_optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        if self.disc_optimizer is not None:
                            self.disc_optimizer.step()
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

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return self._run_step(batch, epoch=epoch, train=True)

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return self._run_step(batch, epoch=epoch, train=False)

    def render_visuals(self, *, output_root: Path, epoch: int, metrics: dict, state: dict) -> None:
        if not self.visual_enabled:
            return
        assert self.model is not None

        self.model.eval()
        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"
        with torch.no_grad():
            sample_inputs = self.model.image_to_model_range(self.sample_batch)
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
