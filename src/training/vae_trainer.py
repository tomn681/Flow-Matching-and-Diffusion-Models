from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import autocast

from core.types import ModelOutput
from losses import LossAssembler
from losses.reconstruction import BCEFocalLoss, BCELoss, FocalLoss, L1Loss, MSELoss
from losses.regularization import KLLoss, VQLoss
from .base import BaseTrainer
from .callbacks import CheckpointCallback, MetricsCSVCallback, VisualizationCallback
from .registry import TRAINER_REGISTRY
from utils.model_utils.vae_utils import build_vae_model
import utils


_RECON_LOSSES = {
    "l1": L1Loss,
    "mse": MSELoss,
    "bce": BCELoss,
    "focal": FocalLoss,
    "bce_focal": BCEFocalLoss,
}


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

        if callbacks is None:
            self.callbacks = [
                CheckpointCallback(filename_prefix="vae", monitor="val_loss" if self.training_cfg.get("validate", True) else "loss", mode="min"),
                MetricsCSVCallback(metric_keys=["loss", "recon", "kl", "vq", "val_loss", "val_recon", "val_kl", "val_vq"]),
                VisualizationCallback(every_n_epochs=int(self.training_cfg.get("save_images_every", 1))),
            ]

        self.loss_assembler: LossAssembler | None = None

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

        recon_cls = _RECON_LOSSES.get(self.recon_type)
        if recon_cls is None:
            raise ValueError(f"Unsupported recon_type '{self.recon_type}'.")

        self.recon_component = recon_cls(weight=1.0)
        self.kl_component = KLLoss(weight=self.kl_weight)
        self.vq_component = VQLoss(weight=effective_codebook_weight)
        self.loss_assembler = LossAssembler([self.recon_component, self.kl_component, self.vq_component])

        self.sample_count = int(self.training_cfg.get("visual_samples", 20))
        self.visual_enabled = bool(self.training_cfg.get("save_images", True))
        eval_source = val_dataset if val_dataset is not None else train_dataset
        self.sample_batch = utils.prepare_eval_batch(eval_source, self.sample_count, self.device, seed=self.training_cfg.get("seed"))
        self.latent_shape = utils.latent_shape(model_cfg)

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

        totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "vq": 0.0}
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

                        total_loss, parts = self.loss_assembler(
                            rec if self.recon_type in {"bce", "focal", "bce_focal"} else rec_img,
                            raw_chunk,
                            epoch=epoch,
                            global_step=self.global_step,
                            posterior=output.posterior,
                            codebook_loss=output.codebook_loss,
                        )

                    if train:
                        if self.scaler.is_enabled():
                            self.scaler.scale(total_loss / accum_steps).backward()
                        else:
                            (total_loss / accum_steps).backward()

                    chunk_bs = chunk.size(0)
                    sample_count += chunk_bs
                    totals["loss"] += float(total_loss.detach().item()) * chunk_bs
                    totals["recon"] += float(parts.get(self.recon_component.name, torch.tensor(0.0, device=self.device)).detach().item()) * chunk_bs
                    totals["kl"] += float(parts.get(self.kl_component.name, torch.tensor(0.0, device=self.device)).detach().item()) * chunk_bs
                    totals["vq"] += float(parts.get(self.vq_component.name, torch.tensor(0.0, device=self.device)).detach().item()) * chunk_bs

                if train:
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
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
