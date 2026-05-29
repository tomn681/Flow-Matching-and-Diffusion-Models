from __future__ import annotations

import torch

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
    monkeypatch.setattr(cli, "build_scheduler", lambda scheduler_cfg, training_cfg: (_FakeScheduler(), 12))

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
    assert captured["kwargs"]["sample_shape"] == (2, 1, 8, 8)
