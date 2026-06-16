from __future__ import annotations

import abc
import logging
import os
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import utils
from configs import validate_config, validate_runtime_semantics, write_resolved_config_dump
from configs.training import TrainingConfig
from configs.model import BaseModelConfig
from configs.migration import normalize_aliases
from core.types import TrainingState
from datasets.base import BaseDataset
from .ema import EMAModel
from .events import TrainingEventBus
from .callbacks import MultiResolutionCallback, StepMetricsCallback
from .multi_resolution import _check_multi_resolution_compatibility, build_resolution_schedule


class BaseTrainer(abc.ABC):
    """Base training orchestration with callback hooks and checkpointing."""

    supports_event_bus = True
    supports_model_override = False
    supports_noise_override = False
    supports_losses_override = False

    def _build_default_callbacks(self) -> list[Any]:
        return []

    def __init__(
        self,
        config: dict,
        callbacks: list[Any] | None = None,
        event_bus: TrainingEventBus | None = None,
    ) -> None:
        if not isinstance(config, dict):
            raise TypeError(f"Trainer config must be a dict, got {type(config).__name__}.")
        normalized_config = normalize_aliases(config)
        config_path = normalized_config.get("__config_path__")
        self.validated_config = validate_config(
            normalized_config,
            config_path=Path(config_path) if isinstance(config_path, str) else None,
        )
        self.config = self.validated_config
        self.raw_config = normalized_config
        self.training_cfg = normalized_config.get("training", {})
        self.model_cfg = normalized_config.get("model", {})
        self.training = self.validated_config.training
        self.model_config = self.validated_config.model

        self.callbacks = self._build_default_callbacks() if callbacks is None else list(callbacks)
        self.event_bus = event_bus or TrainingEventBus()
        self._registered_callback_ids: set[int] = set()
        self._register_callback_listeners()

        self.device = torch.device("cpu")
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.ema_model: EMAModel | None = None

        self.output_dir = Path("checkpoints")
        self.best_metric = float("inf")
        self.global_step = 0
        self.start_epoch = 1

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.train_dataset = None
        self.val_dataset = None
        self._resolution_schedule = build_resolution_schedule(self.raw_config if isinstance(self.raw_config, dict) else {})
        self._resolution_stage_idx: int = 0
        self._current_target_resolution: int | None = None
        self._optimizer_stepped_since_scheduler = False
        self._scheduler_step_unit = "epoch"
        self._main_optimizer_step_count = 0
        self._last_step_seconds = 0.0
        self._last_step_batch_size = 0
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        max_grad_norm = self._training_value("max_grad_norm", 1.0)
        self.max_grad_norm = 0.0 if max_grad_norm is None else float(max_grad_norm)
        self._maybe_add_step_metrics_callback()

    @staticmethod
    def _config_value(config_obj, raw_section: dict, key: str, default=None):
        if hasattr(config_obj, key):
            value = getattr(config_obj, key)
            if value is not None:
                return value
        extra = getattr(config_obj, "extra", {})
        if isinstance(extra, dict) and key in extra:
            return extra[key]
        return raw_section.get(key, default)

    def _training_value(self, key: str, default=None):
        return self._config_value(self.training, self.training_cfg, key, default)

    def _model_value(self, key: str, default=None):
        return self._config_value(self.model_config, self.model_cfg, key, default)

    def _lr_scheduler_config(self) -> dict[str, Any]:
        cfg = dict(self.training_cfg)
        cfg.setdefault("epochs", self._training_value("epochs", 1))
        cfg.setdefault("lr_warmup_steps", self._training_value("lr_warmup_steps", 0))
        cfg["steps_per_epoch"] = len(self.train_loader) if self.train_loader is not None else 0
        return cfg

    def _register_callback_listeners(self) -> None:
        for cb in self.callbacks:
            cb_id = id(cb)
            if cb_id in self._registered_callback_ids:
                continue
            on_epoch_start = getattr(cb, "on_epoch_start", None)
            if callable(on_epoch_start):
                self.event_bus.on("epoch_start", on_epoch_start)
            on_epoch_end = getattr(cb, "on_epoch_end", None)
            if callable(on_epoch_end):
                self.event_bus.on("epoch_end", on_epoch_end)
            on_step_end = getattr(cb, "on_step_end", None)
            if callable(on_step_end):
                self.event_bus.on("step_end", on_step_end)
            on_train_end = getattr(cb, "on_train_end", None)
            if callable(on_train_end):
                self.event_bus.on("train_end", on_train_end)
            self._registered_callback_ids.add(cb_id)

    def request_dataloader_rebuild(self, *, target_resolution: int | None) -> None:
        self._rebuild_dataloaders(target_resolution=target_resolution)

    def _maybe_add_step_metrics_callback(self) -> None:
        every_n_steps = int(self._training_value("step_metrics_every", 0) or 0)
        if every_n_steps <= 0:
            return
        if any(isinstance(cb, StepMetricsCallback) for cb in self.callbacks):
            return
        self.callbacks.append(StepMetricsCallback(every_n_steps=every_n_steps))

    @abc.abstractmethod
    def _build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def _validation_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        return self._training_step(batch, epoch=epoch)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self._training_value("learning_rate", 1e-4))
        weight_decay = float(self._training_value("weight_decay", 0.0))
        if self.model is None:
            raise RuntimeError("BaseTrainer._build_optimizer called before model initialization.")
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found for optimizer construction.")
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        return None

    def _model_module(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("BaseTrainer._model_module called before model initialization.")
        return self.model.module if hasattr(self.model, "module") else self.model

    @property
    def is_main_process(self) -> bool:
        return bool(self.rank == 0)

    def _load_model_state(self, state_dict: dict[str, Any]) -> None:
        self._model_module().load_state_dict(state_dict)

    def _maybe_wrap_distributed_model(self) -> None:
        if self.model is None or not self.distributed:
            return
        if hasattr(self.model, "module"):
            return
        ddp_kwargs: dict[str, Any] = {}
        if self.device.type == "cuda":
            ddp_kwargs["device_ids"] = [self.local_rank]
            ddp_kwargs["output_device"] = self.local_rank
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        self.distributed = bool(utils.setup_distributed(self._training_value("distributed_backend")))
        self.rank = int(utils.get_rank()) if self.distributed else 0
        self.world_size = int(utils.get_world_size()) if self.distributed else 1
        self.local_rank = int(self._training_value("local_rank", int(os.environ.get("LOCAL_RANK", "0"))))

        base_seed = int(self._training_value("seed", 0) or 0)
        utils.set_seed(base_seed + self.rank)

        manual_device = self._training_value("manual_device")
        if self.distributed and torch.cuda.is_available():
            default_device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        else:
            default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = utils.resolve_device(manual_device, default_device)

        base_output_dir = Path(self._training_value("output_dir", "checkpoints"))
        if resume is None:
            if self.is_main_process:
                resolved_output_dir = utils.allocate_run_dir(base_output_dir)
            else:
                resolved_output_dir = None
            resolved_output_dir = utils.broadcast_object(str(resolved_output_dir) if resolved_output_dir is not None else None)
            self.output_dir = Path(resolved_output_dir) if resolved_output_dir is not None else base_output_dir
        else:
            self.output_dir = base_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = self.output_dir / "train_config.json"
        if self.is_main_process and not cfg_path.exists():
            utils.save_json_config(cfg_path, self.raw_config)
            write_resolved_config_dump(self.output_dir, self.raw_config)
        utils.barrier()

        self.model = self._build_model()
        if self._resolution_schedule is not None:
            _check_multi_resolution_compatibility(self.model, self._resolution_schedule)
            for stage in self._resolution_schedule.stages:
                logging.info(
                    "Multi-resolution stage: epoch=%d -> resolution=%d",
                    int(stage.start_epoch),
                    int(stage.resolution),
                )
        self._maybe_apply_lora()
        self._maybe_wrap_distributed_model()
        self.optimizer = self._build_optimizer()
        ema_decay = self._training_value("ema_decay")
        ema_track_all = bool(self._training_value("ema_track_all", False))
        self.ema_model = (
            EMAModel(self._model_module(), decay=float(ema_decay), track_all=ema_track_all)
            if ema_decay is not None
            else None
        )

        use_amp = bool(self._training_value("use_amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        batch_size = int(self._training_value("batch_size", 4))
        num_workers = int(self._training_value("num_workers", 4))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self._rebuild_dataloaders(
            target_resolution=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        validate_runtime_semantics(self.raw_config, steps_per_epoch=len(self.train_loader) if self.train_loader is not None else None)

        if self._resolution_schedule is not None:
            # Register callback after model exists and before training loop starts.
            self.callbacks.append(MultiResolutionCallback())
            self._register_callback_listeners()
            initial_resolution = int(self._resolution_schedule.current_resolution(0))
            self._rebuild_dataloaders(
                target_resolution=initial_resolution,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )
        self.lr_scheduler = self._build_lr_scheduler()
        self._scheduler_step_unit = str(getattr(self.lr_scheduler, "_step_unit", "epoch")) if self.lr_scheduler is not None else "epoch"

        resume_flag = resume if resume is not None else self._training_value("resume")
        if isinstance(resume_flag, str) and resume_flag.lower() == "none":
            resume_flag = None
        if resume_flag:
            ckpt_path = Path(resume_flag)
            if ckpt_path.exists():
                payload = utils.safe_torch_load(ckpt_path, map_location=self.device)
                self._load_model_state(payload["model"])
                if self.optimizer is not None and payload.get("optimizer"):
                    self.optimizer.load_state_dict(payload["optimizer"])
                if self.lr_scheduler is not None and payload.get("scheduler"):
                    self.lr_scheduler.load_state_dict(payload["scheduler"])
                if self.scaler is not None and payload.get("scaler"):
                    self.scaler.load_state_dict(payload["scaler"])
                if self.ema_model is not None and payload.get("ema"):
                    self.ema_model.load_state_dict(payload["ema"])
                self._resume_from_payload(payload)
                self.best_metric = payload.get("best_metric", self.best_metric)
                if isinstance(payload.get("global_step"), int):
                    self.global_step = int(payload["global_step"])
                resumed_epoch = self._resolve_resume_epoch(payload, ckpt_path=None)
                extra_payload = payload.get("extra", {}) if isinstance(payload.get("extra", {}), dict) else {}
                if isinstance(extra_payload.get("resolution_stage"), int):
                    self._resolution_stage_idx = int(extra_payload["resolution_stage"])
                elif isinstance(payload.get("resolution_stage"), int):
                    self._resolution_stage_idx = int(payload["resolution_stage"])
                self.start_epoch = resumed_epoch + 1
                for cb in self.callbacks:
                    if hasattr(cb, "best_metric"):
                        try:
                            setattr(cb, "best_metric", self.best_metric)
                        except Exception:
                            pass
                logging.info("Resumed from %s (epoch %d)", ckpt_path, resumed_epoch)

    class _ResolutionDatasetView:
        def __init__(self, dataset: BaseDataset, target_resolution: int | None) -> None:
            self._dataset = dataset
            self._target_resolution = target_resolution

        def __len__(self) -> int:
            return len(self._dataset)

        def __getattr__(self, name: str):
            return getattr(self._dataset, name)

        def __getitem__(self, idx: int):
            if isinstance(self._dataset, BaseDataset):
                sample = self._dataset.__getitem__(idx, target_resolution=self._target_resolution)
            else:
                sample = self._dataset[idx]
            if isinstance(sample, dict) and "__sample_index__" not in sample:
                out = dict(sample)
                out["__sample_index__"] = int(idx)
                return out
            return sample

    def _rebuild_dataloaders(
        self,
        *,
        target_resolution: int | None,
        train_dataset=None,
        val_dataset=None,
    ) -> None:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if val_dataset is None:
            val_dataset = self.val_dataset
        if train_dataset is None:
            raise RuntimeError("Cannot rebuild dataloaders before train dataset is set.")
        batch_size = int(self._training_value("batch_size", 4))
        num_workers = int(self._training_value("num_workers", 4))
        wrapped_train = self._ResolutionDatasetView(train_dataset, target_resolution)
        wrapped_val = self._ResolutionDatasetView(val_dataset, target_resolution) if val_dataset is not None else None
        train_sampler = (
            DistributedSampler(
                wrapped_train,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(wrapped_train) >= batch_size,
            )
            if self.distributed
            else None
        )
        val_sampler = (
            DistributedSampler(
                wrapped_val,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            if self.distributed and wrapped_val is not None
            else None
        )

        self.train_loader = DataLoader(
            wrapped_train,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=len(wrapped_train) >= batch_size,
            persistent_workers=num_workers > 0,
        )
        self.val_loader = (
            DataLoader(
                wrapped_val,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )
            if val_dataset is not None
            else None
        )
        self._current_target_resolution = target_resolution
        if self._resolution_schedule is not None and target_resolution is not None:
            for idx, stage in enumerate(self._resolution_schedule.stages):
                if int(stage.resolution) == int(target_resolution):
                    self._resolution_stage_idx = idx
            logging.info("Rebuilt dataloaders for target_resolution=%s", str(target_resolution))

    def _maybe_apply_lora(self) -> None:
        if self.model is None:
            raise RuntimeError("BaseTrainer._maybe_apply_lora called before model initialization.")
        lora_cfg = self._training_value("lora")
        if not isinstance(lora_cfg, dict):
            return
        if not bool(lora_cfg.get("enabled", False)):
            return

        from training.lora import LoRAWrapper

        rank = int(lora_cfg.get("rank", 4))
        alpha = float(lora_cfg.get("alpha", 1.0))
        target_modules = lora_cfg.get("target_modules")
        if target_modules is not None and not isinstance(target_modules, list):
            raise TypeError("training.lora.target_modules must be a list when provided.")
        LoRAWrapper.wrap(
            self.model,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
        )

    def _build_state(self, *, epoch: int, metrics: dict[str, float]) -> TrainingState:
        if self.model is None:
            raise RuntimeError("BaseTrainer._build_state called before model initialization.")
        if self.optimizer is None:
            raise RuntimeError("BaseTrainer._build_state called before optimizer initialization.")
        state = TrainingState(
            epoch=epoch,
            global_step=self.global_step,
            model_state=self._model_module().state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            metrics=metrics,
            config=self.raw_config,
            extra={
                "best_metric": self.best_metric,
                "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "ema": self.ema_model.state_dict() if self.ema_model is not None else None,
                "resolution_stage": self._resolution_stage_idx,
                "resolved_config": self.raw_config,
            },
        )
        return state

    def _build_checkpoint_dict(self, state: TrainingState) -> dict[str, Any]:
        return {
            "format_version": utils.CHECKPOINT_FORMAT_VERSION,
            "model": state.model_state,
            "optimizer": state.optimizer_state,
            "scheduler": state.extra.get("scheduler"),
            "scaler": state.extra.get("scaler"),
            "ema": state.extra.get("ema"),
            "epoch": state.epoch,
            "best_metric": self.best_metric,
            "global_step": state.global_step,
            "resolved_config": state.extra.get("resolved_config", self.raw_config),
            "metadata": {
                "trainer": self.__class__.__name__,
                "output_dir": str(self.output_dir),
            },
            "resolution_stage": state.extra.get("resolution_stage"),
            "extra": {"resolution_stage": state.extra.get("resolution_stage")},
        }

    @staticmethod
    def _ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        return tensor if tensor.device == device else tensor.to(device)

    def _backward(self, loss: torch.Tensor, *, scaler: torch.amp.GradScaler | Any | None = None) -> None:
        scaler = self.scaler if scaler is None else scaler
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

    @staticmethod
    def _optimizer_has_any_grad(optimizer: torch.optim.Optimizer) -> bool:
        for group in optimizer.param_groups:
            for param in group.get("params", ()):
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    return True
        return False

    def _clip_optimizer_grads(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        scaler: torch.amp.GradScaler | Any | None = None,
    ) -> None:
        if self.max_grad_norm <= 0:
            return
        scaler = self.scaler if scaler is None else scaler
        if scaler is not None and scaler.is_enabled() and hasattr(scaler, "unscale_"):
            scaler.unscale_(optimizer)
        params: list[torch.Tensor] = []
        for group in optimizer.param_groups:
            for param in group.get("params", ()):
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    params.append(param)
        if params:
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

    def _step_optimizers(self, *optimizers: torch.optim.Optimizer | tuple[torch.optim.Optimizer, torch.amp.GradScaler | Any | None] | None) -> None:
        valid_optimizers: list[tuple[torch.optim.Optimizer, torch.amp.GradScaler | Any | None]] = []
        for item in optimizers:
            if item is None:
                continue
            if isinstance(item, tuple):
                opt, scaler = item
            else:
                opt, scaler = item, self.scaler
            if opt is not None and self._optimizer_has_any_grad(opt):
                valid_optimizers.append((opt, scaler))
        if not valid_optimizers:
            return
        stepped = False
        main_optimizer_stepped = False
        used_scalers: list[Any] = []
        for opt, scaler in valid_optimizers:
            self._clip_optimizer_grads(opt, scaler=scaler)
            if scaler is not None and scaler.is_enabled():
                scaler.step(opt)
                if scaler not in used_scalers:
                    used_scalers.append(scaler)
            else:
                opt.step()
            if opt is self.optimizer or self.optimizer is None:
                main_optimizer_stepped = True
            stepped = True
        for scaler in used_scalers:
            scaler.update()
        if stepped:
            if self.lr_scheduler is not None and self._scheduler_step_unit == "step" and main_optimizer_stepped:
                self.lr_scheduler.step()
                self._main_optimizer_step_count += 1
            elif main_optimizer_stepped:
                self._optimizer_stepped_since_scheduler = True
                self._main_optimizer_step_count += 1
        if self.ema_model is not None and self.model is not None:
            self.ema_model.step(self._model_module())

    def _resume_from_payload(self, payload: dict[str, Any]) -> None:
        """Hook for subclasses to restore extra checkpoint state."""
        return None

    def ema_scope(self):
        if self.ema_model is None or self.model is None:
            return nullcontext()
        return self.ema_model.average_parameters(self._model_module())

    @contextmanager
    def frozen_module(self, module: torch.nn.Module | None):
        if module is None:
            yield
            return
        params = list(module.parameters())
        requires_grad = [param.requires_grad for param in params]
        try:
            for param in params:
                param.requires_grad_(False)
            yield
        finally:
            for param, flag in zip(params, requires_grad):
                param.requires_grad_(flag)

    @staticmethod
    def _resolve_resume_epoch(payload: dict[str, Any], *, ckpt_path: Path | None = None) -> int:
        """
        Resolve last completed epoch from checkpoint payload with legacy fallbacks.
        """
        for key in ("epoch", "current_epoch", "last_epoch"):
            value = payload.get(key)
            if isinstance(value, int):
                return max(0, value)

        return 0

    def _train_epoch(self, *, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("BaseTrainer._train_epoch called before training dataloader initialization.")
        if self.model is None:
            raise RuntimeError("BaseTrainer._train_epoch called before model initialization.")

        self.model.train()
        totals: dict[str, float] = {}
        total_weight = 0.0
        measure_step_timing = any(isinstance(cb, StepMetricsCallback) for cb in self.callbacks)
        sampler = getattr(self.train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        loop = tqdm(
            self.train_loader,
            desc=f"Train epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
            disable=not self.is_main_process,
        )
        for step_idx, batch in enumerate(loop, start=1):
            step_start = (
                torch.cuda.Event(enable_timing=True)
                if measure_step_timing and self.device.type == "cuda" and torch.cuda.is_available()
                else None
            )
            step_end = (
                torch.cuda.Event(enable_timing=True)
                if measure_step_timing and self.device.type == "cuda" and torch.cuda.is_available()
                else None
            )
            if step_start is not None:
                step_start.record()
            wall_time_start = time.perf_counter() if measure_step_timing else 0.0
            step_metrics = self._training_step(batch, epoch=epoch)
            batch_weight = float(self._metric_weight(batch, step_metrics))
            total_weight += batch_weight
            for k, v in step_metrics.items():
                if k.startswith("__"):
                    continue
                totals[k] = totals.get(k, 0.0) + float(v) * batch_weight
            if step_end is not None:
                step_end.record()
                torch.cuda.synchronize(self.device)
                self._last_step_seconds = float(step_start.elapsed_time(step_end)) / 1000.0
            elif measure_step_timing:
                self._last_step_seconds = float(time.perf_counter() - wall_time_start)
            else:
                self._last_step_seconds = 0.0
            self._last_step_batch_size = int(batch_weight)
            self.global_step += 1
            self.event_bus.emit(
                "step_end",
                epoch=epoch,
                step=step_idx,
                global_step=self.global_step,
                metrics=step_metrics,
                trainer=self,
            )

            avg_loss = totals.get("loss", 0.0) / max(1.0, total_weight)
            loop.set_postfix(loss=f"{avg_loss:.4f}")

        self._finalize_train_epoch(epoch=epoch)
        return self._reduce_metrics(totals, total_weight)

    def _validate_epoch(self, *, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        if self.model is None:
            raise RuntimeError("BaseTrainer._validate_epoch called before model initialization.")

        self.model.eval()
        totals: dict[str, float] = {}
        total_weight = 0.0
        sampler = getattr(self.val_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        with self.ema_scope(), torch.no_grad():
            loop = tqdm(
                self.val_loader,
                desc=f"Val epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
                disable=not self.is_main_process,
            )
            for batch_idx, batch in enumerate(loop, start=1):
                batch_weight = float(self._metric_weight(batch))
                step_metrics = self._run_deterministic_validation_step(batch, epoch=epoch, batch_idx=batch_idx)
                total_weight += batch_weight
                for k, v in step_metrics.items():
                    if k.startswith("__"):
                        continue
                    totals[k] = totals.get(k, 0.0) + float(v) * batch_weight
                avg_loss = totals.get("loss", 0.0) / max(1.0, total_weight)
                loop.set_postfix(loss=f"{avg_loss:.4f}")

        return self._reduce_metrics(totals, total_weight)

    def _reduce_metrics(self, totals: dict[str, float], total_weight: float) -> dict[str, float]:
        if not self.distributed:
            return {k: v / max(1.0, total_weight) for k, v in totals.items()}
        keys = sorted(totals.keys())
        payload = torch.tensor(
            [float(total_weight), *[float(totals[key]) for key in keys]],
            device=self.device,
            dtype=torch.float64,
        )
        payload = utils.all_reduce_tensor(payload)
        reduced_weight = float(payload[0].item())
        return {
            key: float(payload[idx + 1].item()) / max(1.0, reduced_weight)
            for idx, key in enumerate(keys)
        }

    @staticmethod
    def _batch_size_from_batch(batch: Any) -> int:
        if isinstance(batch, dict):
            for key in ("target", "image", "__sample_index__"):
                value = batch.get(key)
                if torch.is_tensor(value) and value.ndim >= 1:
                    return int(value.size(0))
                if isinstance(value, list):
                    return int(len(value))
        if torch.is_tensor(batch) and batch.ndim >= 1:
            return int(batch.size(0))
        return 1

    def _metric_weight(self, batch: Any, metrics: dict[str, float] | None = None) -> int:
        if isinstance(metrics, dict) and "__num_samples__" in metrics:
            return int(metrics["__num_samples__"])
        return self._batch_size_from_batch(batch)

    def _validation_seed_for_batch(self, batch: Any, *, batch_idx: int) -> int:
        base_seed = int(self._training_value("validation_seed", self._training_value("seed", 0) or 0))
        if not isinstance(batch, dict):
            return base_seed + int(batch_idx)
        indices = batch.get("__sample_index__")
        if torch.is_tensor(indices):
            values = [int(v) for v in indices.flatten().tolist()]
        elif isinstance(indices, list):
            values = [int(v) for v in indices]
        else:
            values = [int(batch_idx)]
        seed = base_seed
        for value in values:
            seed = ((seed * 1000003) ^ (value + 0x9E3779B9)) & 0x7FFFFFFF
        return int(seed)

    def _run_deterministic_validation_step(self, batch: Any, *, epoch: int, batch_idx: int) -> dict[str, float]:
        if not bool(self._training_value("deterministic_validation", True)):
            return self._validation_step(batch, epoch=epoch)
        devices = [self.device] if self.device.type == "cuda" and torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            seed = self._validation_seed_for_batch(batch, batch_idx=batch_idx)
            torch.manual_seed(seed)
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return self._validation_step(batch, epoch=epoch)

    def _finalize_train_epoch(self, *, epoch: int) -> None:
        del epoch
        return None

    def fit(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        self._register_callback_listeners()
        self._setup(train_dataset, val_dataset=val_dataset, resume=resume)
        self.event_bus.emit("train_start", trainer=self)

        epochs = int(self._training_value("epochs", 1))
        try:
            for epoch in range(self.start_epoch, epochs + 1):
                self.event_bus.emit("epoch_start", epoch=epoch, trainer=self)

                train_metrics = self._train_epoch(epoch=epoch)
                val_metrics = self._validate_epoch(epoch=epoch)
                self.event_bus.emit("validation_end", epoch=epoch, metrics=val_metrics, trainer=self)

                metrics = dict(train_metrics)
                if val_metrics:
                    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                current = metrics.get("val_loss", metrics.get("loss", float("inf")))
                self.best_metric = min(self.best_metric, current)

                state = self._build_state(epoch=epoch, metrics=metrics)
                state_dict = self._build_checkpoint_dict(state)
                train_items = [
                    (k, v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not k.startswith("val_")
                ]
                val_items = [
                    (k[len("val_") :], v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and k.startswith("val_")
                ]

                name_width = 0
                if train_items or val_items:
                    name_width = max(len(k) for k, _ in (train_items + val_items))

                def _fmt(items: list[tuple[str, float]]) -> str:
                    if not items:
                        return "-"
                    return " | ".join(f"{k:<{name_width}}={v:.6f}" for k, v in items)

                line1 = f"Epoch {epoch}/{epochs} | train | {_fmt(train_items)}"
                line2 = f"{' ' * len(f'Epoch {epoch}/{epochs} | ')}val   | {_fmt(val_items)}"
                summary = f"{line1}\n{line2}"
                if self.is_main_process:
                    logging.info("\n%s", summary)
                    print(summary, flush=True)

                self.event_bus.emit("epoch_end", epoch=epoch, metrics=metrics, state=state_dict, trainer=self)

                if (
                    self.lr_scheduler is not None
                    and self._scheduler_step_unit != "step"
                    and self._optimizer_stepped_since_scheduler
                ):
                    self.lr_scheduler.step()
                    self._optimizer_stepped_since_scheduler = False
        except KeyboardInterrupt:
            interrupted_epoch = int(locals().get("epoch", max(0, self.start_epoch - 1)))
            if self.model is not None and self.optimizer is not None:
                try:
                    state = self._build_state(epoch=interrupted_epoch, metrics={})
                    state_dict = self._build_checkpoint_dict(state)
                    state_dict.setdefault("extra", {})
                    state_dict["extra"]["interrupted"] = True
                    utils.save_checkpoint(state_dict, Path(self.output_dir) / "interrupt_last.pt")
                    if self.is_main_process:
                        logging.warning("Saved interrupt checkpoint to %s", Path(self.output_dir) / "interrupt_last.pt")
                except Exception as exc:  # pragma: no cover - best-effort interrupt path
                    logging.exception("Failed to save interrupt checkpoint: %s", exc)
            if self.is_main_process:
                logging.warning("Training interrupted. Terminating...")
                print("\nTraining interrupted. Terminating...", flush=True)
            raise
        finally:
            self.event_bus.emit("train_end", trainer=self)
