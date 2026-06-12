from __future__ import annotations

from pathlib import Path

import torch

from training.callbacks import CheckpointCallback, MetricsCSVCallback, VisualizationCallback


class _DummyTrainer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir


def test_checkpoint_callback_writes_last_and_best(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    callback = CheckpointCallback(filename_prefix="vae", monitor="loss", mode="min")

    state = {"model": {"w": torch.tensor([1.0])}}
    callback.on_epoch_end(epoch=1, metrics={"loss": 1.5}, state=state, trainer=trainer)

    assert (tmp_path / "vae_last.pt").exists()
    assert (tmp_path / "vae_best.pt").exists()

    callback.on_epoch_end(epoch=2, metrics={"loss": 2.0}, state=state, trainer=trainer)
    assert (tmp_path / "vae_last.pt").exists()
    assert callback.best_metric == 1.5


def test_checkpoint_callback_writes_periodic_epoch_checkpoint(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    callback = CheckpointCallback(filename_prefix="vae", monitor="loss", mode="min", save_every=2)

    state = {"model": {"w": torch.tensor([1.0])}}
    callback.on_epoch_end(epoch=1, metrics={"loss": 1.0}, state=state, trainer=trainer)
    assert not (tmp_path / "epochs" / "epoch0001" / "epoch.pt").exists()

    callback.on_epoch_end(epoch=2, metrics={"loss": 0.9}, state=state, trainer=trainer)
    assert (tmp_path / "epochs" / "epoch0002" / "epoch.pt").exists()


def test_checkpoint_callback_periodic_save_without_monitored_metric(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    callback = CheckpointCallback(filename_prefix="vae", monitor="val_loss", mode="min", save_every=1)

    state = {"model": {"w": torch.tensor([1.0])}}
    callback.on_epoch_end(epoch=1, metrics={"loss": 1.0}, state=state, trainer=trainer)

    assert (tmp_path / "vae_last.pt").exists()
    assert (tmp_path / "epochs" / "epoch0001" / "epoch.pt").exists()


def test_metrics_csv_callback_appends_rows(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    callback = MetricsCSVCallback(metric_keys=["loss", "recon"])

    callback.on_epoch_end(epoch=1, metrics={"loss": 1.0, "recon": 0.7}, state={}, trainer=trainer)
    callback.on_epoch_end(epoch=2, metrics={"loss": 0.9, "recon": 0.6}, state={}, trainer=trainer)

    content = (tmp_path / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    assert content[0] == "epoch,loss,recon"
    assert content[1].startswith("1,1.000000,0.700000")
    assert content[2].startswith("2,0.900000,0.600000")


def test_visualization_callback_calls_renderer(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    called = {"count": 0}

    def _render_visuals(*, output_root: Path, epoch: int, metrics: dict, state: dict) -> None:
        called["count"] += 1
        (output_root / "marker.txt").write_text(f"epoch={epoch}", encoding="utf-8")

    trainer.render_visuals = _render_visuals
    callback = VisualizationCallback(every_n_epochs=2)

    callback.on_epoch_end(epoch=1, metrics={}, state={}, trainer=trainer)
    assert called["count"] == 0

    callback.on_epoch_end(epoch=2, metrics={}, state={}, trainer=trainer)
    assert called["count"] == 1
    assert (tmp_path / "epochs" / "epoch0002" / "marker.txt").exists()


def test_visualization_callback_uses_trainer_ema_scope_when_available(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    called = {"entered": False, "rendered": False}

    class _Scope:
        def __enter__(self):
            called["entered"] = True

        def __exit__(self, exc_type, exc, tb):
            return False

    def _ema_scope():
        return _Scope()

    def _render_visuals(*, output_root: Path, epoch: int, metrics: dict, state: dict) -> None:
        _ = output_root, epoch, metrics, state
        called["rendered"] = True

    trainer.ema_scope = _ema_scope
    trainer.render_visuals = _render_visuals
    callback = VisualizationCallback(every_n_epochs=1)
    callback.on_epoch_end(epoch=1, metrics={}, state={}, trainer=trainer)
    assert called["entered"] is True
    assert called["rendered"] is True
