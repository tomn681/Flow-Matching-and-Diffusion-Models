"""
Shared helpers for sampling/encoding/decoding dispatchers.
"""

from __future__ import annotations

import csv
from pathlib import Path

from utils import build_dataset_from_config, load_json_config


def load_run_config(ckpt_dir: Path) -> dict:
    """
    load_run_config Function

    Loads the saved training config from a checkpoint directory.

    Inputs:
        - ckpt_dir: (Path) Checkpoint directory.

    Outputs:
        - cfg: (dict) Parsed config with __config_path__ injected.
    """
    cfg_path = ckpt_dir / "train_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing train_config.json in {ckpt_dir}")
    cfg = load_json_config(cfg_path)
    existing_path = cfg.get("__config_path__")
    if existing_path:
        existing = Path(existing_path)
        if existing.exists():
            return cfg
    cfg["__config_path__"] = str(cfg_path)
    return cfg


def resolve_checkpoint(ckpt_dir: Path, model_type: str) -> Path:
    """
    resolve_checkpoint Function

    Resolves the best checkpoint path for a given model type.

    Inputs:
        - ckpt_dir: (Path) Checkpoint directory.
        - model_type: (String) Model type name.

    Outputs:
        - path: (Path) Selected checkpoint path.
    """
    model_type = str(model_type).lower()
    candidates = []
    if model_type == "vae":
        candidates = ["vae_best.pt", "vae_last.pt"]
    elif model_type == "diffusion":
        candidates = ["diff_best.pt", "diff_last.pt"]
    elif model_type == "flow_matching":
        candidates = ["flow_best.pt", "flow_last.pt"]
    else:
        candidates = ["*.pt"]

    for name in candidates:
        path = ckpt_dir / name
        if path.exists():
            return path
    if candidates == ["*.pt"]:
        pts = sorted(ckpt_dir.glob("*.pt"))
        if pts:
            return pts[-1]
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def _eval_cache_subdir(cache_subdir: str | None) -> str:
    cache_name = str(cache_subdir or "cache")
    return cache_name if cache_name.endswith("_eval") else f"{cache_name}_eval"


def build_sampling_dataset(cfg: dict, data_txt: str | None, evaluate: bool = False) -> object:
    """
    build_sampling_dataset Function

    Builds a dataset for sampling (test split by default, optional split override).

    Inputs:
        - cfg: (dict) Full config dict.
        - data_txt: (String | None) Optional split file override.
        - evaluate: (Boolean) If True, use the dataset test split and an eval cache namespace.

    Outputs:
        - dataset: (object) Dataset instance.
    """
    training_cfg = dict(cfg.get("training", {}))
    if evaluate:
        if data_txt:
            training_cfg["split_file"] = data_txt
        else:
            training_cfg.pop("split_file", None)
        training_cfg["tensor_cache_subdir"] = _eval_cache_subdir(training_cfg.get("tensor_cache_subdir"))
    elif data_txt:
        training_cfg["split_file"] = data_txt
    cfg_path = Path(cfg.get("__config_path__", "")) if cfg.get("__config_path__") else None
    return build_dataset_from_config(training_cfg, cfg.get("model", {}), train=False, cfg_path=cfg_path)


def resolve_output_root(ckpt_dir: Path, output_dir: str | None, save: bool) -> Path | None:
    """
    resolve_output_root Function

    Resolves output directory root for saved tensors.

    Inputs:
        - ckpt_dir: (Path) Checkpoint directory.
        - output_dir: (String | None) Optional output override.
        - save: (Boolean) Whether to save outputs.

    Outputs:
        - output_root: (Path | None) Output directory or None.
    """
    if not save:
        return None
    if output_dir:
        return Path(output_dir)
    return ckpt_dir / "outputs"


def append_eval_metrics(ckpt_dir: Path, metrics: dict) -> Path:
    """
    Append one evaluation result row under the checkpoint directory.
    """
    metrics_path = ckpt_dir / "eval_metrics.csv"
    fields = [
        "samples",
        "mse",
        "psnr",
        "ssim",
        "ssim_enabled",
        "model_seconds",
        "model_samples_per_second",
        "model_seconds_per_sample",
        "model_calls",
    ]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in fields})
    return metrics_path


def append_per_image_eval_metrics(ckpt_dir: Path, rows: list[dict]) -> Path:
    """
    Append per-sample evaluation metric rows under the checkpoint directory.
    """
    metrics_path = ckpt_dir / "eval_per_image_metrics.csv"
    fields = ["sample_index", "img_id", "img_path", "mse", "psnr", "ssim"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return metrics_path


def run_self_tests() -> None:
    """
    Lightweight tests for sampling utility helpers.
    """
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cfg_path = root / "train_config.json"
        cfg = {"training": {"data_root": str(root)}, "model": {"model_type": "vae"}}
        cfg_path.write_text(json.dumps(cfg))
        loaded = load_run_config(root)
        assert "__config_path__" in loaded
        out = resolve_output_root(root, None, True)
        assert out == root / "outputs"
