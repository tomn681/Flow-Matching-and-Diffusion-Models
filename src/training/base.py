from __future__ import annotations

import abc
import logging
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from configs import validate_config
from configs.training import TrainingConfig
from configs.model import BaseModelConfig
from configs.migration import normalize_aliases
from core.types import TrainingState
from datasets.base import BaseDataset
from .ema import EMAModel
from .events import TrainingEventBus
from .callbacks import MultiResolutionCallback
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
            on_train_end = getattr(cb, "on_train_end", None)
            if callable(on_train_end):
                self.event_bus.on("train_end", on_train_end)
            self._registered_callback_ids.add(cb_id)

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

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        utils.set_seed(self._training_value("seed"))

        manual_device = self._training_value("manual_device")
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = utils.resolve_device(manual_device, default_device)

        base_output_dir = Path(self._training_value("output_dir", "checkpoints"))
        self.output_dir = utils.allocate_run_dir(base_output_dir) if resume is None else base_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = self.output_dir / "train_config.json"
        if not cfg_path.exists():
            utils.save_json_config(cfg_path, self.raw_config)

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
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        ema_decay = self._training_value("ema_decay")
        ema_track_all = bool(self._training_value("ema_track_all", False))
        self.ema_model = (
            EMAModel(self.model, decay=float(ema_decay), track_all=ema_track_all)
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

        resume_flag = resume if resume is not None else self._training_value("resume")
        if isinstance(resume_flag, str) and resume_flag.lower() == "none":
            resume_flag = None
        if resume_flag:
            ckpt_path = Path(resume_flag)
            if ckpt_path.exists():
                payload = utils.safe_torch_load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(payload["model"])
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
                resumed_epoch = self._resolve_resume_epoch(payload, ckpt_path=ckpt_path)
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
            return self._dataset.__getitem__(idx, target_resolution=self._target_resolution)

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
        wrapped_train = (
            self._ResolutionDatasetView(train_dataset, target_resolution)
            if isinstance(train_dataset, BaseDataset)
            else train_dataset
        )
        wrapped_val = (
            self._ResolutionDatasetView(val_dataset, target_resolution)
            if isinstance(val_dataset, BaseDataset)
            else val_dataset
        ) if val_dataset is not None else None

        self.train_loader = DataLoader(
            wrapped_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.val_loader = (
            DataLoader(
                wrapped_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
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
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            metrics=metrics,
            config=self.raw_config,
            extra={
                "best_metric": self.best_metric,
                "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "ema": self.ema_model.state_dict() if self.ema_model is not None else None,
                "resolution_stage": self._resolution_stage_idx,
            },
        )
        return state

    def _build_checkpoint_dict(self, state: TrainingState) -> dict[str, Any]:
        return {
            "model": state.model_state,
            "optimizer": state.optimizer_state,
            "scheduler": state.extra.get("scheduler"),
            "scaler": state.extra.get("scaler"),
            "ema": state.extra.get("ema"),
            "epoch": state.epoch,
            "best_metric": self.best_metric,
            "global_step": state.global_step,
            "resolution_stage": state.extra.get("resolution_stage"),
            "extra": {"resolution_stage": state.extra.get("resolution_stage")},
        }

    @staticmethod
    def _ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        return tensor if tensor.device == device else tensor.to(device)

    def _backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    @staticmethod
    def _optimizer_has_any_grad(optimizer: torch.optim.Optimizer) -> bool:
        for group in optimizer.param_groups:
            for param in group.get("params", ()):
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    return True
        return False

    def _step_optimizers(self, *optimizers: torch.optim.Optimizer | None) -> None:
        valid_optimizers = [
            opt for opt in optimizers if opt is not None and self._optimizer_has_any_grad(opt)
        ]
        if not valid_optimizers:
            return
        stepped = False
        if self.scaler is not None and self.scaler.is_enabled():
            for opt in valid_optimizers:
                self.scaler.step(opt)
                setattr(opt, "_opt_called", True)
                stepped = True
            self.scaler.update()
        else:
            for opt in valid_optimizers:
                opt.step()
                setattr(opt, "_opt_called", True)
                stepped = True
        if stepped:
            self._optimizer_stepped_since_scheduler = True
        if self.ema_model is not None and self.model is not None:
            self.ema_model.step(self.model)

    def _resume_from_payload(self, payload: dict[str, Any]) -> None:
        """Hook for subclasses to restore extra checkpoint state."""
        return None

    def ema_scope(self):
        if self.ema_model is None or self.model is None:
            return nullcontext()
        return self.ema_model.average_parameters(self.model)

    @staticmethod
    def _resolve_resume_epoch(payload: dict[str, Any], *, ckpt_path: Path | None = None) -> int:
        """
        Resolve last completed epoch from checkpoint payload with legacy fallbacks.
        """
        for key in ("epoch", "current_epoch", "last_epoch"):
            value = payload.get(key)
            if isinstance(value, int):
                return max(0, value)

        if ckpt_path is not None:
            for part in (ckpt_path.parent.name, ckpt_path.name):
                lower = part.lower()
                match = re.search(r"epoch[_-]?(\d+)", lower)
                if match:
                    return max(0, int(match.group(1)))

        return 0

    def _train_epoch(self, *, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("BaseTrainer._train_epoch called before training dataloader initialization.")
        if self.model is None:
            raise RuntimeError("BaseTrainer._train_epoch called before model initialization.")

        self.model.train()
        totals: dict[str, float] = {}
        num_batches = 0

        loop = tqdm(self.train_loader, desc=f"Train epoch {epoch}", leave=False, dynamic_ncols=True)
        for step_idx, batch in enumerate(loop, start=1):
            step_metrics = self._training_step(batch, epoch=epoch)
            num_batches += 1
            for k, v in step_metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            self.global_step += 1
            self.event_bus.emit(
                "step_end",
                epoch=epoch,
                step=step_idx,
                global_step=self.global_step,
                metrics=step_metrics,
                trainer=self,
            )

            avg_loss = totals.get("loss", 0.0) / max(1, num_batches)
            loop.set_postfix(loss=f"{avg_loss:.4f}")

        return {k: v / max(1, num_batches) for k, v in totals.items()}

    def _validate_epoch(self, *, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        if self.model is None:
            raise RuntimeError("BaseTrainer._validate_epoch called before model initialization.")

        self.model.eval()
        totals: dict[str, float] = {}
        num_batches = 0
        with self.ema_scope(), torch.no_grad():
            loop = tqdm(self.val_loader, desc=f"Val epoch {epoch}", leave=False, dynamic_ncols=True)
            for batch in loop:
                step_metrics = self._validation_step(batch, epoch=epoch)
                num_batches += 1
                for k, v in step_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                avg_loss = totals.get("loss", 0.0) / max(1, num_batches)
                loop.set_postfix(loss=f"{avg_loss:.4f}")

        return {k: v / max(1, num_batches) for k, v in totals.items()}

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
                logging.info("\n%s", summary)
                print(summary, flush=True)

                self.event_bus.emit("epoch_end", epoch=epoch, metrics=metrics, state=state_dict, trainer=self)
                self.event_bus.emit("checkpoint_saved", epoch=epoch, metrics=metrics, state=state_dict, trainer=self)

                if self.lr_scheduler is not None and self._optimizer_stepped_since_scheduler:
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
                    logging.warning("Saved interrupt checkpoint to %s", Path(self.output_dir) / "interrupt_last.pt")
                except Exception as exc:  # pragma: no cover - best-effort interrupt path
                    logging.exception("Failed to save interrupt checkpoint: %s", exc)
            logging.warning("Training interrupted. Terminating...")
            print("\nTraining interrupted. Terminating...", flush=True)
            raise
        finally:
            self.event_bus.emit("train_end", trainer=self)
