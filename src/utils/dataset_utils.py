"""
Utility helpers for dataset configuration, cache handling, and path resolution.
"""

from __future__ import annotations

import inspect
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .utils import load


def _import_symbol(path: str):
    """
    _import_symbol Method

    Imports a Python symbol from a module path string (module:Symbol).

    Inputs:
        - path: (String) Import path in the form "module:Symbol".

    Outputs:
        - symbol: (Any) Imported symbol.
    """
    if ":" not in path:
        raise ValueError(f"Invalid dataset_class '{path}'. Use 'module:Symbol'.")
    module_name, symbol = path.split(":", 1)
    module = import_module(module_name)
    if not hasattr(module, symbol):
        raise ImportError(f"Cannot find '{symbol}' in module '{module_name}'.")
    return getattr(module, symbol)


def consecutive_paths(directory: str, split: int = 3) -> list[list[str]]:
    """
    consecutive_paths Function

    Returns every n-consecutive-path combination from a given directory.

    Inputs:
        - directory: (String) Path to directory.
        - split: (Int) Consecutive group size (use <0 to return all files as one group).

    Outputs:
        - groups: (list<list<String>>) Consecutive path groups.
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        return []
    if directory_path.is_file():
        return [[str(directory_path)]]

    files = sorted(
        [
            str(directory_path / fname)
            for fname in os.listdir(directory_path)
            if (directory_path / fname).is_file()
        ]
    )
    if not files:
        return []

    if split < 0:
        split = max(len(files), 1)
    if split <= 1:
        return [[f] for f in files]

    return [files[i : i + split] for i in range(0, len(files) - split + 1)]


def absolute_path(root_path: Path, entry) -> Path:
    """
    absolute_path Function

    Resolves an entry path relative to a root directory.

    Inputs:
        - root_path: (Path) Base directory.
        - entry: (Any) Path-like value.

    Outputs:
        - path: (Path) Absolute path.
    """
    entry_path = Path(str(entry))
    return entry_path if entry_path.is_absolute() else root_path / entry_path


def maybe_unwrap(paths):
    """
    maybe_unwrap Function

    Unwraps a single-element list/tuple to its element.

    Inputs:
        - paths: (Any) Candidate list/tuple.

    Outputs:
        - value: (Any) Unwrapped value or original input.
    """
    if isinstance(paths, (list, tuple)) and len(paths) == 1:
        return paths[0]
    return paths


def resolve_entry(root_path: Path, entry, window_size: int) -> list:
    """
    resolve_entry Function

    Resolves an entry into a list of path groups based on window_size.

    Inputs:
        - root_path: (Path) Base directory.
        - entry: (Any) Path or relative entry.
        - window_size: (Int) Window/group size.

    Outputs:
        - entries: (list) List of path groups.
    """
    full_path = absolute_path(root_path, entry)
    if full_path.is_dir():
        splits = consecutive_paths(str(full_path), window_size)
        return [paths for paths in splits if paths]
    return [[str(full_path)]]


def split_volume_entry(path: str, window_size: int) -> list:
    """
    split_volume_entry Function

    Splits a single multi-slice volume into windowed entries.

    Inputs:
        - path: (String) Path to a volume file.
        - window_size: (Int) Window size for each split.

    Outputs:
        - entries: (list) List of split dicts or a single path when no split is needed.
    """
    payload = load(path, id=None)
    image = payload.get("Image") if isinstance(payload, dict) else None
    if image is None:
        return [path]

    if isinstance(image, torch.Tensor):
        depth = image.size(0) if image.dim() >= 3 else 1
    else:
        array = np.asarray(image)
        depth = array.shape[0] if array.ndim >= 3 else 1

    if window_size < 0 or depth <= 1:
        return [path]
    if window_size <= 1:
        return [
            {"path": path, "split_index": idx, "split_count": depth, "window": 1}
            for idx in range(depth)
        ]
    if depth < window_size:
        return [path]
    count = depth - window_size + 1
    return [
        {"path": path, "split_index": idx, "split_count": count, "window": window_size}
        for idx in range(count)
    ]


def build_dataset_from_config(training_cfg: dict, model_cfg: dict | None = None, train: bool = True, cfg_path: Path | None = None):
    """
    build_dataset_from_config Function

    Creates a dataset instance based on the training config and dataset.json.

    Inputs:
        - training_cfg: (dict) Training configuration (must include data_root).
        - model_cfg: (dict | None) Optional model config (unused here).
        - train: (Boolean) If True uses train split, else test split.
        - cfg_path: (Path | None) Path to the config file used to locate dataset.json.

    Outputs:
        - dataset: (object) Instantiated dataset.
    """
    dataset_json = _find_dataset_json(cfg_path)
    if dataset_json is None:
        raise ValueError("dataset.json not found in config directory or parents.")
    dataset_cfg = _read_dataset_config(dataset_json)
    dataset_class = dataset_cfg.get("dataset_class")
    if not dataset_class:
        raise ValueError(f"dataset.json missing 'dataset_class': {dataset_json}")
    merged_cfg = dict(training_cfg or {})
    extra_cfg = {k: v for k, v in dataset_cfg.items() if k != "dataset_class"}
    merged_cfg.update(extra_cfg)
    return _build_from_class(dataset_class, merged_cfg, train)


def build_train_val_datasets(cfg: dict) -> Tuple[object, object]:
    """
    build_train_val_datasets Function

    Convenience helper that builds train/val splits from the full config dict.

    Inputs:
        - cfg: (dict) Full configuration containing training and model sections.

    Outputs:
        - train_ds: (object) Training dataset.
        - val_ds: (object) Validation dataset.
    """
    training_cfg = cfg["training"]
    cfg_path_value = cfg.get("__config_path__") if isinstance(cfg, dict) else None
    cfg_path = Path(cfg_path_value) if cfg_path_value else None
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    train_ds = build_dataset_from_config(training_cfg, model_cfg, train=True, cfg_path=cfg_path)
    val_ds = build_dataset_from_config(training_cfg, model_cfg, train=False, cfg_path=cfg_path)
    return train_ds, val_ds


def _find_dataset_json(cfg_path: Path | None) -> Path | None:
    """
    _find_dataset_json Method

    Walks parent directories to locate a dataset.json file.

    Inputs:
        - cfg_path: (Path | None) Config path used as search anchor.

    Outputs:
        - dataset_json: (Path | None) Located dataset.json path or None.
    """
    if cfg_path is None or not str(cfg_path):
        return None
    cursor = cfg_path.parent
    while True:
        candidate = cursor / "dataset.json"
        if candidate.exists():
            return candidate
        if cursor.parent == cursor:
            return None
        cursor = cursor.parent


def _read_dataset_config(dataset_json: Path) -> dict:
    """
    _read_dataset_config Method

    Reads dataset.json into a dictionary.

    Inputs:
        - dataset_json: (Path) Path to dataset.json.

    Outputs:
        - payload: (dict) Parsed JSON payload.
    """
    with dataset_json.open("r") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"dataset.json must contain a JSON object: {dataset_json}")
    return payload


def _read_dataset_class(dataset_json: Path) -> str:
    """
    _read_dataset_class Method

    Extracts dataset_class from dataset.json.

    Inputs:
        - dataset_json: (Path) Path to dataset.json.

    Outputs:
        - dataset_class: (String) Import string for dataset class.
    """
    payload = _read_dataset_config(dataset_json)
    if "dataset_class" not in payload:
        raise ValueError(f"dataset.json missing 'dataset_class': {dataset_json}")
    return str(payload["dataset_class"])


def _build_from_class(dataset_class: str, training_cfg: dict, train: bool):
    """
    _build_from_class Method

    Instantiates a dataset given a dataset_class import string.

    Inputs:
        - dataset_class: (String) Import string for dataset class.
        - training_cfg: (dict) Training configuration.
        - train: (Boolean) Train/test selection.

    Outputs:
        - dataset: (object) Instantiated dataset.
    """
    target = _import_symbol(dataset_class)
    if inspect.isclass(target):
        return _instantiate_dataset(target, training_cfg, train)
    if callable(target):
        return target(training_cfg, train)
    raise TypeError(f"dataset_class '{dataset_class}' is not callable.")


def _instantiate_dataset(cls, training_cfg: dict, train: bool):
    """
    _instantiate_dataset Method

    Instantiates a dataset class using kwargs mapped from training config.

    Inputs:
        - cls: (type) Dataset class.
        - training_cfg: (dict) Training configuration.
        - train: (Boolean) Train/test selection.

    Outputs:
        - dataset: (object) Instantiated dataset.
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    kwargs = _build_dataset_kwargs(training_cfg, train, params.keys())
    return cls(**kwargs)


def _build_dataset_kwargs(training_cfg: dict, train: bool, keys) -> dict:
    """
    _build_dataset_kwargs Method

    Builds constructor kwargs for dataset instantiation based on config keys.

    Inputs:
        - training_cfg: (dict) Training configuration.
        - train: (Boolean) Train/test selection.
        - keys: (Iterable) Constructor parameter names.

    Outputs:
        - kwargs: (dict) Dataset constructor kwargs.
    """
    mapping = {
        "file_path": "data_root",
        "root": "data_root",
        "img_size": "img_size",
        "window_size": "window_size",
        "load_ldct": "load_ldct",
        "norm": "norm",
        "use_tensor_cache": "use_tensor_cache",
        "save_tensor_cache": "save_tensor_cache",
        "cache_subdir": "tensor_cache_subdir",
        "preprocess_kwargs": "preprocess_kwargs",
        "download": "download",
    }
    kwargs = {}
    for param in keys:
        if param == "self":
            continue
        if param == "train":
            kwargs["train"] = train
            continue
        cfg_key = mapping.get(param, param)
        if cfg_key in training_cfg:
            kwargs[param] = training_cfg[cfg_key]
        elif param == "window_size" and "slice_count" in training_cfg:
            kwargs[param] = training_cfg["slice_count"]
    return kwargs


def cache_path_for_entry(
    base_path: Path,
    cache_root: Path,
    entry,
    split_index: int | None = None,
    split_count: int = 1,
) -> Path | None:
    """
    cache_path_for_entry Function

    Builds the cache file path for a dataset entry.

    Inputs:
        - base_path: (Path) Dataset root.
        - cache_root: (Path) Cache root directory.
        - entry: (Any) Dataset entry (path, list, or dict).
        - split_index: (Int | None) Split index for windowed entries.
        - split_count: (Int) Total split count for the entry.

    Outputs:
        - cache_path: (Path | None) Cache path or None if not resolvable.
    """
    if cache_root is None:
        return None
    if isinstance(entry, list):
        if not entry:
            return None
        base = entry[0]
    elif isinstance(entry, dict):
        base = entry.get("path")
        if base is None and isinstance(entry.get("paths"), (list, tuple)) and entry["paths"]:
            base = entry["paths"][0]
    else:
        base = entry

    if base is None:
        return None
    entry_path = Path(str(base))
    if entry_path.is_absolute():
        try:
            rel = entry_path.relative_to(base_path)
        except Exception:
            rel = Path(entry_path.name)
    else:
        rel = entry_path
    stem = Path(rel).stem
    parent = Path(rel).parent
    if split_count > 1 and split_index is not None:
        filename = f"{stem}_split_{split_index}.pt"
    else:
        filename = f"{stem}.pt"
    return cache_root / parent / filename


def save_tensor_cache(tensor, cache_path: Path) -> None:
    """
    save_tensor_cache Function

    Atomically saves a tensor to the cache path.

    Inputs:
        - tensor: (Tensor) Tensor to save.
        - cache_path: (Path) Destination cache path.
    """
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(tensor, tmp_path)
    try:
        with open(tmp_path, "rb+") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass
    os.replace(tmp_path, cache_path)


def run_self_tests() -> None:
    """
    Lightweight unit tests for dataset utility helpers.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # consecutive_paths
        files = []
        for idx in range(3):
            p = data_dir / f"img_{idx}.npy"
            np.save(p, np.zeros((2, 2), dtype=np.float32))
            files.append(str(p))
        assert consecutive_paths(str(data_dir), 1) == [[f] for f in sorted(files)]
        assert len(consecutive_paths(str(data_dir), 2)) == 2
        assert consecutive_paths(str(data_dir), -1) == [sorted(files)]

        # resolve_entry
        resolved = resolve_entry(root, "data", 2)
        assert len(resolved) == 2
        resolved_file = resolve_entry(root, "data/img_0.npy", 2)
        assert resolved_file == [[str(data_dir / "img_0.npy")]]

        # split_volume_entry
        volume_path = data_dir / "volume.npy"
        np.save(volume_path, np.arange(12, dtype=np.float32).reshape(3, 2, 2))
        splits = split_volume_entry(str(volume_path), 1)
        assert len(splits) == 3
        assert isinstance(splits[0], dict) and splits[0]["window"] == 1
        splits_w2 = split_volume_entry(str(volume_path), 2)
        assert len(splits_w2) == 2

        # cache_path_for_entry
        cache_root = root / "cache"
        cache_path = cache_path_for_entry(root, cache_root, "data/img_0.npy", 0, 3)
        assert cache_path == cache_root / "data" / "img_0_split_0.pt"

        # save_tensor_cache
        tensor = torch.zeros((2, 2), dtype=torch.float32)
        save_tensor_cache(tensor, cache_path)
        assert cache_path.exists()

        # build_dataset_from_config (smoke)
        (root / "train.txt").write_text("target\n" + "data/img_0.npy\n")
        dataset_json = root / "dataset.json"
        dataset_json.write_text(json.dumps({"dataset_class": "src.utils.dataset:BaseDataset"}))
        cfg_path = root / "train_config.json"
        cfg_path.write_text("{}")
        dataset = build_dataset_from_config({"data_root": str(root)}, train=True, cfg_path=cfg_path)
        assert len(dataset) == 1
