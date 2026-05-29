from __future__ import annotations

import genlib.cli as cli


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
