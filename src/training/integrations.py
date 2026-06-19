from __future__ import annotations

from pathlib import Path
from typing import Any


class ThroughputCallback:
    """Write lightweight throughput telemetry on step events."""

    def __init__(self, *, every_n_steps: int = 10, filename: str = "throughput.csv") -> None:
        self.every_n_steps = max(1, int(every_n_steps))
        self.filename = filename
        self._header_written = False

    def on_step_end(self, *, epoch: int, step: int, global_step: int, metrics: dict, trainer: Any) -> None:
        del metrics
        if not getattr(trainer, "is_main_process", True):
            return
        if int(global_step) % self.every_n_steps != 0:
            return
        batch_size = int(getattr(trainer, "_last_step_batch_size", 0) or 0)
        step_seconds = float(getattr(trainer, "_last_step_seconds", 0.0) or 0.0)
        imgs_per_sec = 0.0 if step_seconds <= 0.0 else float(batch_size) / float(step_seconds)
        path = Path(trainer.output_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "epoch": int(epoch),
            "step": int(step),
            "global_step": int(global_step),
            "batch_size": batch_size,
            "step_seconds": step_seconds,
            "imgs_per_sec": imgs_per_sec,
        }
        if not self._header_written and not path.exists():
            path.write_text(",".join(row.keys()) + "\n", encoding="utf-8")
        self._header_written = True
        with path.open("a", encoding="utf-8") as handle:
            handle.write(",".join(str(row[k]) for k in row.keys()) + "\n")


class WandBCallback:
    """Lazy W&B adapter on the trainer event bus."""

    def __init__(self, *, project: str, run_name: str | None = None, config: dict | None = None) -> None:
        self.project = str(project)
        self.run_name = None if run_name is None else str(run_name)
        self.config = dict(config or {})
        self._run = None

    def _ensure_run(self):
        if self._run is not None:
            return self._run
        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("W&B integration requested but package 'wandb' is not installed.") from exc
        self._run = wandb.init(project=self.project, name=self.run_name, config=self.config or None)
        return self._run

    def on_epoch_end(self, *, epoch: int, metrics: dict, state: dict, trainer: Any) -> None:
        del state
        if not getattr(trainer, "is_main_process", True):
            return
        run = self._ensure_run()
        payload = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        payload["epoch"] = int(epoch)
        run.log(payload, step=int(getattr(trainer, "global_step", epoch)))

    def on_train_end(self, *, trainer: Any) -> None:
        del trainer
        if self._run is not None:
            self._run.finish()
            self._run = None


class MLflowCallback:
    """Lazy MLflow adapter on the trainer event bus."""

    def __init__(self, *, experiment_name: str, run_name: str | None = None) -> None:
        self.experiment_name = str(experiment_name)
        self.run_name = None if run_name is None else str(run_name)
        self._mlflow = None
        self._active = False

    def _ensure_run(self):
        if self._active:
            return self._mlflow
        try:
            import mlflow  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("MLflow integration requested but package 'mlflow' is not installed.") from exc
        self._mlflow = mlflow
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        self._active = True
        return mlflow

    def on_epoch_end(self, *, epoch: int, metrics: dict, state: dict, trainer: Any) -> None:
        del state
        if not getattr(trainer, "is_main_process", True):
            return
        mlflow = self._ensure_run()
        payload = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(payload, step=int(getattr(trainer, "global_step", epoch)))
        mlflow.log_metric("epoch", int(epoch), step=int(getattr(trainer, "global_step", epoch)))

    def on_train_end(self, *, trainer: Any) -> None:
        del trainer
        if self._active and self._mlflow is not None:
            self._mlflow.end_run()
            self._active = False


__all__ = ["ThroughputCallback", "WandBCallback", "MLflowCallback"]
