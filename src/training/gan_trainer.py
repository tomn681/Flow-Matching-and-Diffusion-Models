from __future__ import annotations
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from core.protocols import TimestepConditioned
from core.types import ModelOutput
from losses.adversarial import GANDiscriminatorLoss, GANGeneratorLoss
from models.factory import ModelFactory
from models.vae.base import BaseVAE
from nn.losses.adversarial import (
    PatchDiscriminator,
    apply_spectral_norm_,
    discriminator_wgan_loss,
    generator_wgan_loss,
    gradient_penalty,
    r1_regularization,
)
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback, TensorBoardCallback
from .registry import TRAINER_REGISTRY
import utils


@TRAINER_REGISTRY.register("gan")
class GANTrainer(BaseTrainer):
    """Standalone GAN trainer with alternating generator/discriminator updates."""

    supports_model_override = True

    def __init__(
        self,
        config: dict,
        callbacks: list[Any] | None = None,
        event_bus=None,
        model_override: torch.nn.Module | None = None,
        discriminator_override: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks, event_bus=event_bus)
        self._model_override = model_override
        self._discriminator_override = discriminator_override
        self.discriminator: torch.nn.Module | None = None
        self.disc_optimizer: torch.optim.Optimizer | None = None
        self.disc_scaler: torch.amp.GradScaler | None = None
        self.disc_lr = float(self.training_cfg.get("disc_lr", self.training_cfg.get("learning_rate", 1e-4)))
        self.adv_weight = float(self.training_cfg.get("adv_weight", 1.0))
        self.gan_loss_type = str(self.training_cfg.get("gan_loss", "hinge")).lower()
        self.gradient_penalty_weight = float(self.training_cfg.get("gradient_penalty_weight", 0.0))
        self.r1_weight = float(self.training_cfg.get("r1_weight", 0.0))
        self.use_spectral_norm = bool(self.training_cfg.get("spectral_norm", False))
        self.disc_updates_per_gen_step = int(
            self.training_cfg.get(
                "disc_updates_per_gen_step",
                self.training_cfg.get("disc_steps", 1),
            )
        )
        if self.disc_updates_per_gen_step < 1:
            raise ValueError("disc_updates_per_gen_step must be >= 1.")
        self.gan_generator_component = GANGeneratorLoss(weight=self.adv_weight, start_epoch=0, start_step=None)
        self.gan_discriminator_component = GANDiscriminatorLoss(weight=1.0, start_epoch=0, start_step=None)

    def _build_default_callbacks(self) -> list[Any]:
        training_cfg = self.training_cfg
        metric_keys = ["loss", "g_gan", "d_gan"]
        if float(training_cfg.get("gradient_penalty_weight", 0.0)) > 0:
            metric_keys.append("gp")
        if float(training_cfg.get("r1_weight", 0.0)) > 0:
            metric_keys.append("r1")
        return [
            CheckpointCallback(
                filename_prefix="gan",
                monitor="val_loss" if training_cfg.get("validate", True) else "loss",
                mode="min",
                save_every=int(training_cfg.get("save_every", 0)),
            ),
            MetricsCSVCallback(metric_keys=metric_keys),
            TensorBoardCallback(),
        ]

    @classmethod
    def from_config(cls, path_or_dict: str | Path | dict) -> "GANTrainer":
        if isinstance(path_or_dict, (str, Path)):
            cfg = utils.load_json_config(path_or_dict)
        else:
            cfg = path_or_dict
        return cls(config=cfg)

    def _build_model(self) -> torch.nn.Module:
        if self._model_override is not None:
            return self._model_override.to(self.device)
        model_cfg = dict(self.model_cfg)
        generator_cfg = model_cfg.get("generator")
        if not isinstance(generator_cfg, dict):
            raise ValueError(
                "GANTrainer requires model_override or config.model.generator with a valid model config."
            )
        cfg = {"model": dict(generator_cfg)}
        conditioning = self.training_cfg.get("conditioning")
        channels = self.training_cfg.get("channels")
        return ModelFactory.build(cfg, conditioning=conditioning, channels=channels).to(self.device)

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)
        if self._discriminator_override is not None:
            self.discriminator = self._discriminator_override.to(self.device)
        elif isinstance(self._model_module(), BaseVAE):
            self.discriminator = self._model_module().make_discriminator().to(self.device)
        if self.discriminator is None:
            disc_cfg = self.model_cfg.get("discriminator", {}) if isinstance(self.model_cfg, dict) else {}
            in_channels = int(
                disc_cfg.get(
                    "in_channels",
                    self.model_cfg.get("generator", {}).get(
                        "out_channels", self.training_cfg.get("channels", 1)
                    ),
                )
            )
            spatial_dims = int(disc_cfg.get("spatial_dims", 2))
            base_channels = int(disc_cfg.get("base_channels", 64))
            self.discriminator = PatchDiscriminator(
                in_channels=in_channels,
                base_channels=base_channels,
                spatial_dims=spatial_dims,
            ).to(self.device)
        if self.use_spectral_norm:
            self.discriminator = apply_spectral_norm_(self.discriminator).to(self.device)
        self.disc_optimizer = AdamW(self.discriminator.parameters(), lr=self.disc_lr, betas=(0.5, 0.9))
        self.disc_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(self.scaler is not None and self.scaler.is_enabled()),
        )

    def _forward_generator(self, inputs: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        if isinstance(self.model, TimestepConditioned):
            t = torch.zeros(inputs.size(0), device=inputs.device, dtype=torch.long)
            out = self.model(inputs, t)
        else:
            out = self.model(inputs)
        if isinstance(out, ModelOutput):
            return out.reconstruction
        if isinstance(out, tuple):
            return out[0]
        return out

    def _run_step(self, batch: dict, *, train: bool) -> dict[str, float]:
        if self.model is None or self.optimizer is None or self.scaler is None:
            raise RuntimeError("GANTrainer called before initialization.")
        if self.discriminator is None or self.disc_optimizer is None:
            raise RuntimeError("GANTrainer discriminator was not initialized.")

        target = batch["target"].to(self.device)
        source = batch.get("image")
        if source is None:
            source = torch.randn_like(target)
        else:
            source = source.to(self.device)

        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"
        update_generator = (not train) or (self.global_step % self.disc_updates_per_gen_step == 0)
        if train:
            if update_generator:
                self.optimizer.zero_grad(set_to_none=True)
            self.disc_optimizer.zero_grad(set_to_none=True)
            self.discriminator.train()
        else:
            self.discriminator.eval()

        if update_generator:
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                fake = self._forward_generator(source)
                with self.frozen_module(self.discriminator):
                    fake_pred = self.discriminator(fake)
                g_loss = self._generator_loss(fake_pred=fake_pred, dtype=fake.dtype)
            if train:
                self._backward(g_loss)
        else:
            with torch.no_grad():
                fake = self._forward_generator(source)
            g_loss = torch.tensor(0.0, device=self.device, dtype=fake.dtype)

        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            real_pred = self.discriminator(target)
            fake_pred_d = self.discriminator(fake.detach())
            d_adv = self._discriminator_loss(real_pred=real_pred, fake_pred=fake_pred_d, dtype=fake.dtype)

        gp = torch.tensor(0.0, device=self.device, dtype=fake.dtype)
        if train and self.gradient_penalty_weight > 0:
            gp = gradient_penalty(self.discriminator, target.detach(), fake.detach()).to(device=self.device, dtype=fake.dtype)

        r1 = torch.tensor(0.0, device=self.device, dtype=fake.dtype)
        if train and self.r1_weight > 0:
            r1 = r1_regularization(self.discriminator, target.detach()).to(device=self.device, dtype=fake.dtype)

        d_loss = d_adv + self.gradient_penalty_weight * gp + self.r1_weight * r1
        if train:
            self._backward(d_loss, scaler=self.disc_scaler)
            if update_generator:
                self._step_optimizers((self.optimizer, self.scaler), (self.disc_optimizer, self.disc_scaler))
            else:
                self._step_optimizers((self.disc_optimizer, self.disc_scaler))

        total = g_loss + d_loss
        metrics = {
            "loss": float(total.detach().item()),
            "g_gan": float(g_loss.detach().item()),
            "d_gan": float(d_loss.detach().item()),
        }
        if self.gradient_penalty_weight > 0:
            metrics["gp"] = float(gp.detach().item())
        if self.r1_weight > 0:
            metrics["r1"] = float(r1.detach().item())
        return metrics

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        del epoch
        return self._run_step(batch, train=True)

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        del epoch
        return self._run_step(batch, train=False)

    def _build_state(self, *, epoch: int, metrics: dict[str, float]):
        state = super()._build_state(epoch=epoch, metrics=metrics)
        if self.disc_optimizer is not None:
            state.extra["disc_optimizer"] = self.disc_optimizer.state_dict()
        if self.disc_scaler is not None:
            state.extra["disc_scaler"] = self.disc_scaler.state_dict()
        return state

    def _build_checkpoint_dict(self, state) -> dict[str, Any]:
        payload = super()._build_checkpoint_dict(state)
        payload["disc_optimizer"] = state.extra.get("disc_optimizer")
        payload["disc_scaler"] = state.extra.get("disc_scaler")
        return payload

    def _resume_from_payload(self, payload: dict[str, Any]) -> None:
        if self.disc_optimizer is not None and payload.get("disc_optimizer"):
            self.disc_optimizer.load_state_dict(payload["disc_optimizer"])
        if self.disc_scaler is not None and payload.get("disc_scaler"):
            self.disc_scaler.load_state_dict(payload["disc_scaler"])

    def _generator_loss(self, *, fake_pred: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.gan_loss_type == "hinge":
            return self.gan_generator_component.compute(
                context={"fake_pred": fake_pred, "device": self.device, "dtype": dtype}
            )
        if self.gan_loss_type == "wgan":
            return (generator_wgan_loss(fake_pred) * self.adv_weight).to(device=self.device, dtype=dtype)
        raise ValueError(f"Unsupported gan_loss '{self.gan_loss_type}'. Expected 'hinge' or 'wgan'.")

    def _discriminator_loss(self, *, real_pred: torch.Tensor, fake_pred: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.gan_loss_type == "hinge":
            return self.gan_discriminator_component.compute(
                context={"real_pred": real_pred, "fake_pred": fake_pred, "device": self.device, "dtype": dtype}
            )
        if self.gan_loss_type == "wgan":
            return discriminator_wgan_loss(real_pred, fake_pred).to(device=self.device, dtype=dtype)
        raise ValueError(f"Unsupported gan_loss '{self.gan_loss_type}'. Expected 'hinge' or 'wgan'.")
