from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from core.types import TrainingState


class BaseTrainer(abc.ABC):
    """Base training orchestration with callback hooks and checkpointing."""

    def __init__(self, config: dict, callbacks: list[Any] | None = None) -> None:
        self.raw_config = config
        self.training_cfg = config.get("training", {}) if isinstance(config, dict) else {}
        self.model_cfg = config.get("model", {}) if isinstance(config, dict) else {}

        self.callbacks = callbacks or []

        self.device = torch.device("cpu")
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler: GradScaler | None = None

        self.output_dir = Path("checkpoints")
        self.best_metric = float("inf")
        self.global_step = 0
        self.start_epoch = 1

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None

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
        assert self.model is not None
        return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

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

        use_amp = bool(self.training_cfg.get("use_amp", False)) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=use_amp)

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
                self.best_metric = payload.get("best_metric", self.best_metric)
                self.start_epoch = payload.get("epoch", 0) + 1
                logging.info("Resumed from %s (epoch %d)", ckpt_path, self.start_epoch - 1)

    def _build_state(self, *, epoch: int, metrics: dict[str, float]) -> TrainingState:
        assert self.model is not None
        assert self.optimizer is not None
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
            },
        )
        return state

    def _build_checkpoint_dict(self, state: TrainingState) -> dict[str, Any]:
        return {
            "model": state.model_state,
            "optimizer": state.optimizer_state,
            "scheduler": state.extra.get("scheduler"),
            "scaler": state.extra.get("scaler"),
            "epoch": state.epoch,
            "best_metric": self.best_metric,
            "global_step": state.global_step,
        }

    def _train_epoch(self, *, epoch: int) -> dict[str, float]:
        assert self.train_loader is not None
        assert self.model is not None

        self.model.train()
        totals: dict[str, float] = {}
        num_batches = 0

        loop = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False, dynamic_ncols=True)
        for batch in loop:
            step_metrics = self._training_step(batch, epoch=epoch)
            num_batches += 1
            for k, v in step_metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            self.global_step += 1

            avg_loss = totals.get("loss", 0.0) / max(1, num_batches)
            loop.set_postfix(loss=f"{avg_loss:.4f}")

        return {k: v / max(1, num_batches) for k, v in totals.items()}

    def _validate_epoch(self, *, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        assert self.model is not None

        self.model.eval()
        totals: dict[str, float] = {}
        num_batches = 0
        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f"Val {epoch}", leave=False, dynamic_ncols=True)
            for batch in loop:
                step_metrics = self._validation_step(batch, epoch=epoch)
                num_batches += 1
                for k, v in step_metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                avg_loss = totals.get("loss", 0.0) / max(1, num_batches)
                loop.set_postfix(loss=f"{avg_loss:.4f}")

        return {k: v / max(1, num_batches) for k, v in totals.items()}

    def fit(self, train_dataset, val_dataset=None, resume: str | None = None) -> None:
        self._setup(train_dataset, val_dataset=val_dataset, resume=resume)

        epochs = int(self.training_cfg.get("epochs", 1))
        for epoch in range(self.start_epoch, epochs + 1):
            for cb in self.callbacks:
                cb.on_epoch_start(epoch=epoch, trainer=self)

            train_metrics = self._train_epoch(epoch=epoch)
            val_metrics = self._validate_epoch(epoch=epoch)

            metrics = dict(train_metrics)
            if val_metrics:
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            current = metrics.get("val_loss", metrics.get("loss", float("inf")))
            self.best_metric = min(self.best_metric, current)

            state = self._build_state(epoch=epoch, metrics=metrics)
            state_dict = self._build_checkpoint_dict(state)

            for cb in self.callbacks:
                cb.on_epoch_end(epoch=epoch, metrics=metrics, state=state_dict, trainer=self)

        for cb in self.callbacks:
            cb.on_train_end(trainer=self)
