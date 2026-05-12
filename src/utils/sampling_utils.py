"""
Shared helpers for sampling/encoding/decoding dispatchers.
"""

from __future__ import annotations

import random
import csv
from pathlib import Path

from utils import build_dataset_from_config, load_json_config
from utils.dataset_utils import iter_batches


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


def resolve_sample_indices(dataset, num_samples: int | None, seed: int = 42) -> list[int]:
    """
    Resolve a deterministic random subset of dataset indices.
    """
    total = len(dataset)
    if total == 0:
        return []
    if num_samples is None or int(num_samples) <= 0 or int(num_samples) >= total:
        return list(range(total))
    rng = random.Random(seed)
    return rng.sample(list(range(total)), int(num_samples))


def progress_batches(dataset, batch_size: int, desc: str, indices: list[int] | None = None):
    """
    Yield dataset batches with a tqdm progress bar when available.
    Falls back to plain iteration if tqdm is unavailable.
    """
    selected = list(range(len(dataset))) if indices is None else list(indices)
    total = len(selected)
    bs = max(int(batch_size), 1)
    total_batches = (total + bs - 1) // bs
    iterator = iter_batches(dataset, batch_size, indices=selected)
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(iterator, total=total_batches, desc=desc, leave=False, dynamic_ncols=True)
    except Exception:
        pass
    return iterator


def build_tensor_cache_from_config(
    cfg: dict,
    data_txt: str | None,
    batch_size: int,
    seed: int,
    num_samples: int | None,
    desc: str = "build_tensor_cache",
    evaluate: bool = True,
) -> int:
    """
    Iterate dataset samples to trigger tensor cache reads/writes without model inference.
    """
    dataset = build_sampling_dataset(cfg, data_txt, evaluate=evaluate)
    selected_indices = resolve_sample_indices(dataset, num_samples, seed=seed)
    total = 0
    for _, samples in progress_batches(dataset, batch_size, desc, indices=selected_indices):
        for sample in samples:
            _ = sample["target"]
            _ = sample.get("image")
        total += len(samples)
    return total


def append_eval_metrics(ckpt_dir: Path, row: dict) -> Path:
    """
    Append one evaluation summary row to ckpt_dir/eval_metrics.csv.
    """
    ckpt_dir = Path(ckpt_dir)
    out_path = ckpt_dir / "eval_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): str(v) for k, v in row.items()}
    fieldnames = list(payload.keys())
    exists = out_path.exists()
    with out_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(payload)
    return out_path


def append_per_image_eval_metrics(ckpt_dir: Path, rows: list[dict]) -> Path:
    """
    Write per-sample evaluation rows to ckpt_dir/eval_metrics_per_image.csv.
    """
    ckpt_dir = Path(ckpt_dir)
    out_path = ckpt_dir / "eval_metrics_per_image.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not out_path.exists():
            out_path.write_text("")
        return out_path
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return out_path


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
