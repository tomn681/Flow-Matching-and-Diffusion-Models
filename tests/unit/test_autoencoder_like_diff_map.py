from __future__ import annotations

import json
from pathlib import Path

import torch

import pipelines.samplers.autoencoder_like as auto_mod


class _DummyDataset:
    target_key = "SDCT"
    conditioning_key = None

    def __init__(self) -> None:
        self.data = [{"img_id": "a"}, {"img_id": "b"}]

    def __getitem__(self, idx: int) -> dict:
        base = torch.tensor([[[0.2, 0.6], [0.1, 0.9]]], dtype=torch.float32)
        return {"target": base + (0.1 * idx)}


def test_autoencoder_sample_saves_diff_map_and_grid(monkeypatch, tmp_path: Path) -> None:
    saved: list[tuple[str, torch.Tensor]] = []
    grids: list[torch.Tensor] = []

    dataset = _DummyDataset()
    monkeypatch.setattr(auto_mod, "load_run_config", lambda _: {"training": {"recon_type": "l1", "input_normalize": "positive"}})
    monkeypatch.setattr(auto_mod, "resolve_checkpoint", lambda *_args, **_kwargs: tmp_path / "vae.pt")
    monkeypatch.setattr(auto_mod, "build_sampling_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(auto_mod, "resolve_sample_indices", lambda dataset, num_samples, seed=42: [0, 1])
    monkeypatch.setattr(auto_mod, "resolve_output_root", lambda *args, **kwargs: tmp_path / "outputs")
    monkeypatch.setattr(auto_mod, "build_vae_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        auto_mod,
        "progress_batches",
        lambda dataset, batch_size, desc, indices=None: [([0, 1], [dataset[0], dataset[1]])],
    )
    monkeypatch.setattr(
        auto_mod,
        "reconstruct_vae_batch",
        lambda model, inputs, recon_type="l1", input_normalize="positive": inputs * 0.5,
    )
    monkeypatch.setattr(
        auto_mod,
        "save_output_tensor",
        lambda dataset, row, key, tensor, root: saved.append((Path(root).name, tensor.clone())),
    )
    monkeypatch.setattr(auto_mod, "save_diff_map_grid", lambda diff_batches, output_root: grids.append(torch.cat(diff_batches, dim=0)))

    auto_mod.sample(ckpt_dir=tmp_path, save=True, save_diff_map=True, diff_amplify=5.0)

    diff_roots = [name for name, _tensor in saved if name == "diff_map"]
    assert diff_roots
    diff_tensors = [tensor for name, tensor in saved if name == "diff_map"]
    assert all(float(tensor.max()) <= 1.0 for tensor in diff_tensors)
    assert grids and grids[0].shape[0] == 2


def test_autoencoder_evaluate_saves_diff_map_and_run_config(monkeypatch, tmp_path: Path) -> None:
    saved: list[tuple[str, torch.Tensor]] = []
    grids: list[torch.Tensor] = []

    dataset = _DummyDataset()
    monkeypatch.setattr(auto_mod, "load_run_config", lambda _: {"training": {"recon_type": "l1", "input_normalize": "positive"}})
    monkeypatch.setattr(auto_mod, "resolve_checkpoint", lambda *_args, **_kwargs: tmp_path / "vae.pt")
    monkeypatch.setattr(auto_mod, "build_sampling_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(auto_mod, "resolve_sample_indices", lambda dataset, num_samples, seed=42: [0, 1])
    monkeypatch.setattr(auto_mod, "resolve_output_root", lambda *args, **kwargs: tmp_path / "outputs")
    monkeypatch.setattr(auto_mod, "build_vae_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        auto_mod,
        "progress_batches",
        lambda dataset, batch_size, desc, indices=None: [([0, 1], [dataset[0], dataset[1]])],
    )
    monkeypatch.setattr(
        auto_mod,
        "reconstruct_vae_batch",
        lambda model, inputs, recon_type="l1", input_normalize="positive": inputs * 0.5,
    )
    monkeypatch.setattr(
        auto_mod,
        "save_output_tensor",
        lambda dataset, row, key, tensor, root: saved.append((Path(root).name, tensor.clone())),
    )
    monkeypatch.setattr(auto_mod, "save_diff_map_grid", lambda diff_batches, output_root: grids.append(torch.cat(diff_batches, dim=0)))
    monkeypatch.setattr(auto_mod, "compute_ssim_sample", lambda recon, target, ssim_fn: 0.5)
    monkeypatch.setattr(auto_mod, "append_eval_metrics", lambda *args, **kwargs: tmp_path / "eval_metrics.json")
    monkeypatch.setattr(auto_mod, "append_per_image_eval_metrics", lambda *args, **kwargs: tmp_path / "per_image_eval_metrics.csv")
    monkeypatch.setattr(auto_mod, "write_eval_metrics", lambda *args, **kwargs: tmp_path / "eval_metrics.json")
    def _create_experiment_dir(**kwargs):
        out = tmp_path / "exp"
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(auto_mod, "create_experiment_dir", _create_experiment_dir)

    auto_mod.evaluate(ckpt_dir=tmp_path, save=True, save_diff_map=True, diff_amplify=4.0)

    diff_tensors = [tensor for name, tensor in saved if name == "diff_map"]
    assert diff_tensors
    assert all(float(tensor.max()) <= 1.0 for tensor in diff_tensors)
    assert grids
    run_cfg = (tmp_path / "exp" / "run_config.json")
    assert run_cfg.exists()
    payload = json.loads(run_cfg.read_text(encoding="utf-8"))
    assert payload["save_diff_map"] is True
    assert payload["diff_amplify"] == 4.0
