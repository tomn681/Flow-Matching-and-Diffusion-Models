from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from losses.adversarial import GANDiscriminatorLoss, GANGeneratorLoss
from models.factory import ModelFactory
from nn.losses.adversarial import PatchDiscriminator
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback
from .registry import TRAINER_REGISTRY
import utils


@TRAINER_REGISTRY.register("gan")
class GANTrainer(BaseTrainer):
    """Standalone GAN trainer with alternating generator/discriminator updates."""

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
        self.disc_lr = float(self.training_cfg.get("disc_lr", self.training_cfg.get("learning_rate", 1e-4)))
        self.adv_weight = float(self.training_cfg.get("adv_weight", 1.0))
        self.gan_generator_component = GANGeneratorLoss(weight=self.adv_weight, start_epoch=0, start_step=None)
        self.gan_discriminator_component = GANDiscriminatorLoss(weight=1.0, start_epoch=0, start_step=None)

        if callbacks is None:
            self.callbacks = [
                CheckpointCallback(
                    filename_prefix="gan",
                    monitor="val_loss" if self.training_cfg.get("validate", True) else "loss",
                    mode="min",
                    save_every=int(self.training_cfg.get("save_every", 0)),
                ),
                MetricsCSVCallback(metric_keys=["loss", "g_gan", "d_gan"]),
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
            return self._model_override
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
        elif self.model is not None and hasattr(self.model, "make_discriminator"):
            make_disc = getattr(self.model, "make_discriminator")
            if callable(make_disc):
                self.discriminator = make_disc().to(self.device)
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
        self.disc_optimizer = AdamW(self.discriminator.parameters(), lr=self.disc_lr)

    def _forward_generator(self, inputs: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        try:
            out = self.model(inputs)
        except TypeError:
            t = torch.zeros(inputs.size(0), device=inputs.device, dtype=torch.long)
            out = self.model(inputs, t)
        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "sample"):
            return out.sample
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
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.disc_optimizer.zero_grad(set_to_none=True)
            self.discriminator.train()
        else:
            self.discriminator.eval()

        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            fake = self._forward_generator(source)
            fake_pred = self.discriminator(fake)
            g_gan = self.gan_generator_component.compute(
                context={"fake_pred": fake_pred, "device": self.device, "dtype": fake.dtype}
            )
            g_loss = g_gan

        if train:
            self._backward(g_loss)

        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            real_pred = self.discriminator(target.detach())
            fake_pred_d = self.discriminator(fake.detach())
            d_loss = self.gan_discriminator_component.compute(
                context={"real_pred": real_pred, "fake_pred": fake_pred_d, "device": self.device, "dtype": fake.dtype}
            )
        if train:
            self._backward(d_loss)
            self._step_optimizers(self.optimizer, self.disc_optimizer)

        total = g_loss + d_loss
        return {
            "loss": float(total.detach().item()),
            "g_gan": float(g_loss.detach().item()),
            "d_gan": float(d_loss.detach().item()),
        }

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
        return state

    def _build_checkpoint_dict(self, state) -> dict[str, Any]:
        payload = super()._build_checkpoint_dict(state)
        payload["disc_optimizer"] = state.extra.get("disc_optimizer")
        return payload

    def _resume_from_payload(self, payload: dict[str, Any]) -> None:
        if self.disc_optimizer is not None and payload.get("disc_optimizer"):
            self.disc_optimizer.load_state_dict(payload["disc_optimizer"])
