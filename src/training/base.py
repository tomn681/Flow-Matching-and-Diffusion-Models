from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from core.types import TrainingState
from .ema import EMAModel
from .events import TrainingEventBus


class BaseTrainer(abc.ABC):
    """Base training orchestration with callback hooks and checkpointing."""

    def __init__(
        self,
        config: dict,
        callbacks: list[Any] | None = None,
        event_bus: TrainingEventBus | None = None,
    ) -> None:
        self.raw_config = config
        self.training_cfg = config.get("training", {}) if isinstance(config, dict) else {}
        self.model_cfg = config.get("model", {}) if isinstance(config, dict) else {}

        self.callbacks = callbacks or []
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
        lr = float(self.training_cfg.get("learning_rate", 1e-4))
        weight_decay = float(self.training_cfg.get("weight_decay", 0.0))
        if self.model is None:
            raise RuntimeError("BaseTrainer._build_optimizer called before model initialization.")
        return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        return None

    def _setup(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        utils.set_seed(self.training_cfg.get("seed"))

        manual_device = self.training_cfg.get("manual_device")
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = utils.resolve_device(manual_device, default_device)

        base_output_dir = Path(self.training_cfg.get("output_dir", "checkpoints"))
        self.output_dir = utils.allocate_run_dir(base_output_dir) if resume is None else base_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = self.output_dir / "train_config.json"
        if not cfg_path.exists():
            utils.save_json_config(cfg_path, self.raw_config)

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        ema_decay = self.training_cfg.get("ema_decay")
        ema_track_all = bool(self.training_cfg.get("ema_track_all", False))
        self.ema_model = (
            EMAModel(self.model, decay=float(ema_decay), track_all=ema_track_all)
            if ema_decay is not None
            else None
        )

        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        batch_size = int(self.training_cfg.get("batch_size", 4))
        num_workers = int(self.training_cfg.get("num_workers", 4))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.val_loader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            if val_dataset is not None
            else None
        )

        resume_flag = resume if resume is not None else self.training_cfg.get("resume")
        if isinstance(resume_flag, str) and resume_flag.lower() == "none":
            resume_flag = None
        if resume_flag:
            ckpt_path = Path(resume_flag)
            if ckpt_path.exists():
                payload = torch.load(ckpt_path, map_location=self.device)
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
                resumed_epoch = self._resolve_resume_epoch(payload, ckpt_path=ckpt_path)
                self.start_epoch = resumed_epoch + 1
                logging.info("Resumed from %s (epoch %d)", ckpt_path, resumed_epoch)

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
        }

    @staticmethod
    def _ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        return tensor if tensor.device == device else tensor.to(device)

    def _backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _step_optimizers(self, *optimizers: torch.optim.Optimizer | None) -> None:
        valid_optimizers = [opt for opt in optimizers if opt is not None]
        if not valid_optimizers:
            return
        if self.scaler is not None and self.scaler.is_enabled():
            for opt in valid_optimizers:
                self.scaler.step(opt)
            self.scaler.update()
        else:
            for opt in valid_optimizers:
                opt.step()
        if self.ema_model is not None and self.model is not None:
            self.ema_model.step(self.model)

    def _resume_from_payload(self, payload: dict[str, Any]) -> None:
        """Hook for subclasses to restore extra checkpoint state."""
        return None

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
                if "epoch" in lower:
                    digits = "".join(ch for ch in lower if ch.isdigit())
                    if digits:
                        return max(0, int(digits))

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
        with torch.no_grad():
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

        epochs = int(self.training_cfg.get("epochs", 1))
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

            metric_items = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items() if isinstance(v, (int, float)))
            summary = f"Epoch {epoch}/{epochs} | {metric_items}" if metric_items else f"Epoch {epoch}/{epochs}"
            logging.info(summary)
            print(summary, flush=True)

            self.event_bus.emit("epoch_end", epoch=epoch, metrics=metrics, state=state_dict, trainer=self)
            self.event_bus.emit("checkpoint_saved", epoch=epoch, metrics=metrics, state=state_dict, trainer=self)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.event_bus.emit("train_end", trainer=self)
