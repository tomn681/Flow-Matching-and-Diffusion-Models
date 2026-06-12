from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

import genlib.cli as cli
from src import run_model as run_model_impl


def test_genlib_train_forwards_all_args(monkeypatch) -> None:
    captured = {"argv": None}

    def _fake_train_main(argv=None):
        captured["argv"] = argv

    monkeypatch.setattr(cli.train_entry, "main", _fake_train_main)

    cli.main(["train", "--mode", "encode_latents", "--config", "cfg.json"])

    assert captured["argv"] == ["--mode", "encode_latents", "--config", "cfg.json"]


def test_genlib_sample_maps_to_run_model_mode(monkeypatch) -> None:
    captured = {"argv": None}

    def _fake_run_main(argv=None):
        captured["argv"] = argv

    monkeypatch.setattr(cli.run_model_entry, "main", _fake_run_main)

    cli.main(["sample", "--ckpt_dir", "./ckpt"])

    assert captured["argv"] == ["--mode", "sample", "--ckpt_dir", "./ckpt"]


def test_genlib_evaluate_maps_to_run_model_mode(monkeypatch) -> None:
    captured = {"argv": None}

    def _fake_run_main(argv=None):
        captured["argv"] = argv

    monkeypatch.setattr(cli.run_model_entry, "main", _fake_run_main)

    cli.main(["evaluate", "--ckpt_dir", "./ckpt", "--batch_size", "2"])

    assert captured["argv"] == ["--mode", "evaluate", "--ckpt_dir", "./ckpt", "--batch_size", "2"]


def test_genlib_generate_reflow_pairs_dispatch(monkeypatch, tmp_path) -> None:
    captured = {"called": False, "kwargs": None}

    class _FakeModel(torch.nn.Module):
        def forward(self, x, t, context_ca=None):
            _ = t, context_ca
            return x

    class _FakeScheduler:
        pass

    monkeypatch.setattr(cli, "load_json_config", lambda _: {"training": {}, "model": {"scheduler": {}}})
    monkeypatch.setattr(cli, "build_diffusion_model", lambda cfg, device, ckpt_path=None, set_eval=True: _FakeModel())
    monkeypatch.setattr(cli, "build_scheduler", lambda scheduler_cfg, training_cfg, **kwargs: (_FakeScheduler(), 12))

    def _fake_generate(**kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "generate_reflow_pairs", _fake_generate)

    cli.main(
        [
            "generate-reflow-pairs",
            "--config",
            "cfg.json",
            "--ckpt",
            "model.pt",
            "--num-pairs",
            "7",
            "--output-dir",
            str(tmp_path / "pairs"),
            "--sample-shape",
            "1,8,8",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert captured["called"] is True
    assert captured["kwargs"]["num_pairs"] == 7
    assert captured["kwargs"]["sample_shape"] == (1, 8, 8)


def test_genlib_top_level_help_contains_examples() -> None:
    parser = cli._build_parser()
    help_text = parser.format_help()
    assert "Examples:" in help_text
    assert "generate-reflow-pairs" in help_text


def test_run_model_help_mentions_cache_and_reflow_modes() -> None:
    parser = run_model_impl._build_parser()
    help_text = parser.format_help()
    assert "build_tensor_cache" in help_text
    assert "generate_reflow_pairs" in help_text
    assert "Examples:" in help_text


def test_train_help_mentions_encode_latents_mode() -> None:
    root = Path(__file__).resolve().parents[2]
    train_path = root / "train.py"
    spec = importlib.util.spec_from_file_location("repo_train_entry", train_path)
    assert spec is not None and spec.loader is not None
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    parser = train_mod._build_parser()
    help_text = parser.format_help()
    assert "encode_latents" in help_text
    assert "Examples:" in help_text
