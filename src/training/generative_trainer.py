from __future__ import annotations

import abc
from pathlib import Path
import sys as _sys
from typing import Any

import torch
from torch.optim import AdamW

from losses import LOSS_REGISTRY, LossAssembler
from losses.adversarial import GANDiscriminatorLoss, GANGeneratorLoss
from core.noise_contracts import warn_if_legacy_family_alias
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
from .callbacks import CheckpointCallback, MetricsCSVCallback, TensorBoardCallback, VisualizationCallback
from .registry import TRAINER_REGISTRY
from utils.model_utils.diffusion_utils import build_diffusion_model
from core.types import unwrap_model_prediction
from core import Discriminatable
import utils


class GenerativeTrainer(BaseTrainer, abc.ABC):
    """Tier-1 trainer for diffusion and flow-matching UNet models."""

    noise_key: str
    checkpoint_prefix: str
    supports_model_override = True
    supports_noise_override = True

    def __init__(
        self,
        config: dict,
        callbacks: list[Any] | None = None,
        event_bus=None,
        model_override: torch.nn.Module | None = None,
        noise_override: Any = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks, event_bus=event_bus)
        warn_if_legacy_family_alias(str(self.model_cfg.get("model_type", "")))
        self._model_override = model_override
        self._noise_override = noise_override
        self.grad_accum = max(1, int(self._training_value("gradient_accumulation_steps", 1)))
        self.latent_norm = self._training_value("latent_norm")
        self.conditioning_mode = str(self._training_value("conditioning", self._model_value("conditioning", "none")) or "none").strip().lower()
        default_dropout = 0.0 if self.conditioning_mode in {"none", "false", "off"} else 0.1
        self.conditioning_dropout = float(self._training_value("conditioning_dropout", default_dropout))
        text_cfg = self._training_value("text_encoder")
        needs_deferred_text_adapter = self.conditioning_mode == "text" or (
            self.conditioning_mode == "chain" and isinstance(text_cfg, dict) and bool(text_cfg)
        )
        self.conditioning_adapter = (
            resolve_conditioning_adapter("none")
            if needs_deferred_text_adapter
            else self._build_conditioning_adapter(self.conditioning_mode)
        )
        self.noise_process = None
        self.gan_weight = float(self._training_value("gan_weight", 0.0))
        self.gan_space = str(self._training_value("gan_space", "auto")).strip().lower()
        self.gan_start_epoch = int(self._training_value("gan_start", 0))
        gan_start_steps = self._training_value("gan_start_steps")
        self.gan_start_steps = None if gan_start_steps is None else int(gan_start_steps)
        self.disc_lr = float(self._training_value("disc_lr", self._training_value("learning_rate", 1e-4)))
        self.discriminator: torch.nn.Module | None = None
        self.disc_optimizer: torch.optim.Optimizer | None = None
        self.disc_scaler: torch.amp.GradScaler | None = None
        self.gan_generator_component: GANGeneratorLoss | None = None
        self.gan_discriminator_component: GANDiscriminatorLoss | None = None
        self.loss_assembler: LossAssembler | None = None
        self._accum_counter = 0

    def _build_default_callbacks(self) -> list[Any]:
        callbacks = [
            CheckpointCallback(
                filename_prefix=self.checkpoint_prefix,
                monitor="val_loss" if self._training_value("validate", True) else "loss",
                mode="min",
                save_every=int(self._training_value("save_every", 0)),
            ),
            MetricsCSVCallback(),
            TensorBoardCallback(),
        ]
        if bool(self._training_value("save_images", False)):
            callbacks.append(
                VisualizationCallback(
                    every_n_epochs=int(self._training_value("save_images_every", 10)),
                )
            )
        return callbacks

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
            text_cfg = self._training_value("text_encoder")
            if isinstance(text_cfg, dict) and text_cfg:
                specs.append(ChainAdapterSpec("text", self._build_text_conditioning_adapter()))
            return ConditioningChain(specs)
        return resolve_conditioning_adapter(mode)

    def _build_text_conditioning_adapter(self) -> TextConditioningAdapter:
        text_cfg = self._training_value("text_encoder")
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
        return build_lr_scheduler(self.optimizer, self._lr_scheduler_config())

    def _init_noise_process(self) -> None:
        if self._noise_override is not None:
            self.noise_process = self._noise_override
            return
        scheduler_cfg = self.model_cfg.get("scheduler", {})
        train_scheduler, _ = build_scheduler(scheduler_cfg, self.training_cfg, noise_family=self.noise_key)
        noise_kwargs: dict[str, Any] = {"scheduler": train_scheduler}
        if self.noise_key in {"flow_matching", "rectified_flow", "reflow"}:
            noise_kwargs.update(
                {
                    "timestep_sampling": str(self._training_value("flow_timestep_sampling", "uniform")),
                    "logit_mean": float(self._training_value("flow_logit_mean", 0.0)),
                    "logit_std": float(self._training_value("flow_logit_std", 1.0)),
                    "shift": self._training_value("flow_shift"),
                }
            )
        if self.noise_key == "reflow":
            pairs_dir = self._training_value("reflow_pairs_dir")
            if not pairs_dir:
                raise ValueError("Reflow training requires training.reflow_pairs_dir.")
            noise_kwargs["pairs_dir"] = str(pairs_dir)
        self.noise_process = NOISE_REGISTRY.build(self.noise_key, **noise_kwargs)

    def _build_loss_assembler(self) -> LossAssembler:
        components = [
            LOSS_REGISTRY.build(
                "denoising_mse",
                weight=1.0,
                min_snr_gamma=float(self._training_value("min_snr_gamma", 0.0) or 0.0),
            )
        ]
        if self.gan_generator_component is not None:
            components.append(self.gan_generator_component)
        return LossAssembler(components)

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        super()._setup(train_dataset, val_dataset=val_dataset, resume=resume)
        self.conditioning_adapter = self._build_conditioning_adapter(self.conditioning_mode)
        self._init_noise_process()
        if self.gan_weight > 0.0:
            if self.gan_space == "auto":
                if self.noise_key == "x0_denoising":
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
            self.disc_optimizer = AdamW(self.discriminator.parameters(), lr=self.disc_lr, betas=(0.5, 0.9))
            self.disc_scaler = torch.amp.GradScaler(
                "cuda",
                enabled=bool(self.scaler is not None and self.scaler.is_enabled()),
            )
        self.loss_assembler = self._build_loss_assembler()

        self.visual_enabled = bool(self._training_value("save_images", False))
        self.visual_batch: torch.Tensor | None = None
        self.visual_cond: torch.Tensor | None = None
        if self.visual_enabled:
            self._prepare_visual_batch(val_dataset if val_dataset is not None else train_dataset)

    def _prepare_visual_batch(self, ds) -> None:
        from utils.indexing_utils import select_visual_indices
        count = int(self._training_value("visual_samples", 20))
        seed = self._training_value("seed")
        indices = select_visual_indices(ds, count, seed=seed)
        targets, conds = [], []
        for i in indices:
            sample = ds[i]
            targets.append(sample["target"])
            cond = sample.get("image")
            if cond is not None:
                conds.append(cond)
        self.visual_batch = torch.stack(targets, dim=0).to(self.device)
        self.visual_cond = torch.stack(conds, dim=0).to(self.device) if len(conds) == len(targets) else None

    def render_visuals(self, *, output_root: Path, epoch: int, metrics: dict, state: dict) -> None:
        if not self.visual_enabled or self.visual_batch is None or self.model is None:
            return
        from scheduling.sampling_loop import sample_with_scheduler

        scheduler = getattr(self.noise_process, "scheduler", None)
        if scheduler is None:
            return

        scheduler_cfg = self.model_cfg.get("scheduler", {})
        num_steps = int(scheduler_cfg.get("num_inference_steps", 50))
        out_channels = int(self.model_cfg.get("unet", {}).get("out_channels", self.model_cfg.get("out_channels", 1)))
        spatial_dims = int(self.model_cfg.get("unet", {}).get("spatial_dims", 2))
        img_size = int(self._training_value("img_size", 256))
        n = self.visual_batch.size(0)
        sample_shape = (n, out_channels, *([img_size] * spatial_dims))

        self.model.eval()
        use_amp = bool(self._training_value("use_amp", False)) and self.device.type == "cuda"
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=use_amp):
            generated = sample_with_scheduler(
                model=self.model,
                scheduler=scheduler,
                num_inference_steps=num_steps,
                sample_shape=sample_shape,
                device=self.device,
                conditioning_mode=self.conditioning_mode if self.conditioning_mode not in {"none", "false", "off"} else None,
                conditioning_batch=self.visual_cond,
                latent_norm=self.latent_norm,
            )
        self.model.train()

        cols = min(n, 5)
        rows = min(n // cols, 4)
        n_grid = rows * cols
        if n_grid == 0:
            return

        target_vis = self.visual_batch[:n_grid].clamp(0.0, 1.0)
        gen_vis = generated[:n_grid].clamp(0.0, 1.0)
        utils.save_image(utils.make_grid(target_vis, rows, cols), output_root / "target.png")
        utils.save_image(utils.make_grid(gen_vis, rows, cols), output_root / "output.png")
        if self.visual_cond is not None:
            cond_vis = self.visual_cond[:n_grid].clamp(0.0, 1.0)
            utils.save_image(utils.make_grid(cond_vis, rows, cols), output_root / "conditioning.png")

    def _build_discriminator(self) -> torch.nn.Module:
        model_module = self._model_module() if self.model is not None else None
        if model_module is not None and isinstance(model_module, Discriminatable):
            disc = model_module.make_discriminator()
            if disc is not None:
                return disc
        in_channels = int(
            self.model_cfg.get(
                "out_channels",
                self.model_cfg.get("unet", {}).get("out_channels", self._training_value("channels", 1)),
            )
        )
        spatial_dims = int(self.model_cfg.get("unet", {}).get("spatial_dims", self._model_value("spatial_dims", 2)))
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

    @staticmethod
    def _index_conditioning_payload(cond, mask: torch.Tensor):
        if cond is None:
            return None
        if torch.is_tensor(cond):
            return cond[mask]
        if isinstance(cond, dict):
            return {
                key: (GenerativeTrainer._index_conditioning_payload(value, mask) if value is not None else None)
                for key, value in cond.items()
            }
        if isinstance(cond, list):
            indices = mask.nonzero(as_tuple=False).flatten().tolist()
            return [cond[idx] for idx in indices]
        if isinstance(cond, tuple):
            indices = mask.nonzero(as_tuple=False).flatten().tolist()
            return tuple(cond[idx] for idx in indices)
        raise TypeError(f"Unsupported conditioning payload type for masking: {type(cond).__name__}")

    @staticmethod
    def _replace_context_rows(
        context: torch.Tensor | None,
        null_context: torch.Tensor | None,
        drop_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if context is None and null_context is None:
            return None
        if context is None:
            full = torch.zeros(
                drop_mask.size(0),
                *null_context.shape[1:],
                device=null_context.device,
                dtype=null_context.dtype,
            )
            full[drop_mask] = null_context
            return full
        out = context.clone()
        if null_context is None:
            out[drop_mask] = 0.0
        else:
            out[drop_mask] = null_context
        return out

    def _apply_conditioning_dropout(
        self,
        *,
        base_input: torch.Tensor,
        conditioned_input: torch.Tensor,
        context: torch.Tensor | None,
        cond_chunk,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.conditioning_dropout <= 0.0:
            return conditioned_input, context
        drop_mask = torch.rand(base_input.size(0), device=self.device) < self.conditioning_dropout
        if not torch.any(drop_mask):
            return conditioned_input, context
        null_cond = self._index_conditioning_payload(cond_chunk, drop_mask)
        null_input, null_context = self.conditioning_adapter.null_conditioning(
            base_input[drop_mask],
            null_cond,
            self.latent_norm,
        )
        out_input = conditioned_input.clone()
        out_input[drop_mask] = null_input
        out_context = self._replace_context_rows(context, null_context, drop_mask)
        return out_input, out_context

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

        use_amp = bool(self._training_value("use_amp", False)) and self.device.type == "cuda"

        if train:
            if self._accum_counter == 0:
                self.optimizer.zero_grad(set_to_none=True)
                if self.disc_optimizer is not None:
                    self.disc_optimizer.zero_grad(set_to_none=True)
            if self.disc_optimizer is not None:
                self.discriminator.train()
        elif self.discriminator is not None:
            self.discriminator.eval()

        total_loss = 0.0
        total_d_loss = 0.0
        total_denoise_loss = 0.0
        total_g_loss = 0.0
        total_samples = 0

        noisy_batch = self.noise_process(clean, self.device)
        base_input = noisy_batch.noisy
        model_input, context = self.conditioning_adapter(base_input, cond, self.latent_norm)
        if train:
            model_input, context = self._apply_conditioning_dropout(
                base_input=base_input,
                conditioned_input=model_input,
                context=context,
                cond_chunk=cond,
            )

        scheduler = getattr(self.noise_process, "scheduler", None)
        prediction_type = str(
            getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon") or "epsilon"
        ).lower()
        snr = self._snr_for_timesteps(scheduler, noisy_batch.timesteps)

        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            pred = (
                self.model(model_input, noisy_batch.timesteps, context_ca=context)
                if context is not None
                else self.model(model_input, noisy_batch.timesteps)
            )
            pred = unwrap_model_prediction(pred)
            assembler_context = {
                "pred": pred,
                "target": noisy_batch.target,
                "snr": snr,
                "prediction_type": prediction_type,
                "device": self.device,
                "dtype": pred.dtype,
            }
            if self._disc_is_active(epoch=epoch):
                fake_for_gan, _real_for_gan = self._resolve_gan_tensors(
                    pred=pred, clean=clean, target=noisy_batch.target
                )
                with self.frozen_module(self.discriminator):
                    fake_pred = self.discriminator(fake_for_gan)
                assembler_context["fake_pred"] = fake_pred
            if self.loss_assembler is None:
                self.loss_assembler = self._build_loss_assembler()
            loss, parts = self.loss_assembler(
                context=assembler_context,
                epoch=epoch,
                global_step=self.global_step,
            )

        if train:
            self._backward(loss / self.grad_accum)

        d_loss = self._discriminator_step(
            pred=pred,
            clean=clean,
            target=noisy_batch.target,
            epoch=epoch,
            train=train,
            accum_steps=self.grad_accum,
        )
        chunk_bs = clean.size(0)
        total_loss += float(loss.detach().item()) * chunk_bs
        total_d_loss += float(d_loss) * chunk_bs
        total_denoise_loss += float(parts.get("denoise_mse", 0.0).detach().item()) * chunk_bs
        if "g_gan" in parts:
            total_g_loss += float(parts["g_gan"].detach().item()) * chunk_bs
        total_samples += chunk_bs

        if train:
            self._accum_counter += 1
            if self._accum_counter >= self.grad_accum:
                if self.disc_optimizer is not None and self.disc_scaler is not None:
                    self._step_optimizers((self.optimizer, self.scaler), (self.disc_optimizer, self.disc_scaler))
                else:
                    self._step_optimizers(self.optimizer, self.disc_optimizer)
                self._accum_counter = 0

        denom = max(1, total_samples)
        metrics = {"loss": total_loss / denom, "denoise_mse": total_denoise_loss / denom}
        if self.gan_weight > 0.0:
            metrics["d_gan"] = total_d_loss / denom
            if total_g_loss > 0.0:
                metrics["g_gan"] = total_g_loss / denom
        return metrics

    @staticmethod
    def _snr_for_timesteps(scheduler, timesteps: torch.Tensor) -> torch.Tensor | None:
        if scheduler is None or not hasattr(scheduler, "alphas_cumprod"):
            return None
        alphas_cumprod = getattr(scheduler, "alphas_cumprod")
        if not torch.is_tensor(alphas_cumprod):
            return None
        if timesteps.dtype.is_floating_point:
            timesteps = timesteps.round().long()
        timesteps = timesteps.clamp(0, alphas_cumprod.numel() - 1)
        alpha = alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[timesteps]
        return alpha / (1.0 - alpha).clamp_min(1e-8)

    def _finalize_train_epoch(self, *, epoch: int) -> None:
        del epoch
        if self._accum_counter <= 0:
            return
        if self.disc_optimizer is not None and self.disc_scaler is not None:
            self._step_optimizers((self.optimizer, self.scaler), (self.disc_optimizer, self.disc_scaler))
        else:
            self._step_optimizers(self.optimizer, self.disc_optimizer)
        self._accum_counter = 0

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
            self._backward(d_loss / accum_steps, scaler=self.disc_scaler)
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
        if self.disc_scaler is not None:
            state.extra["disc_scaler"] = self.disc_scaler.state_dict()
        return state

    def _build_checkpoint_dict(self, state):
        payload = super()._build_checkpoint_dict(state)
        payload["disc_optimizer"] = state.extra.get("disc_optimizer")
        payload["disc_scaler"] = state.extra.get("disc_scaler")
        return payload

    def _resume_from_payload(self, payload: dict[str, Any]) -> None:
        if self.disc_optimizer is not None and payload.get("disc_optimizer"):
            self.disc_optimizer.load_state_dict(payload["disc_optimizer"])
        if self.disc_scaler is not None and payload.get("disc_scaler"):
            self.disc_scaler.load_state_dict(payload["disc_scaler"])

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
    noise_key = "x0_denoising"
    checkpoint_prefix = "consistency"


@TRAINER_REGISTRY.register("x0_denoising")
class X0DenoisingTrainer(GenerativeTrainer):
    noise_key = "x0_denoising"
    checkpoint_prefix = "x0_denoising"


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
