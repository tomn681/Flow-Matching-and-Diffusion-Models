from __future__ import annotations

from pathlib import Path

from utils.sampling_utils import resolve_checkpoint


def test_resolve_checkpoint_for_latent_rectified_flow(tmp_path: Path) -> None:
    ckpt = tmp_path / "latent_rf_best.pt"
    ckpt.write_bytes(b"x")
    out = resolve_checkpoint(tmp_path, "latent_rectified_flow")
    assert out == ckpt


def test_resolve_checkpoint_for_reflow(tmp_path: Path) -> None:
    ckpt = tmp_path / "reflow_last.pt"
    ckpt.write_bytes(b"x")
    out = resolve_checkpoint(tmp_path, "reflow")
    assert out == ckpt
