"""
Utility helpers for dataset configuration, cache handling, and path resolution.
"""

from __future__ import annotations

import inspect
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


def build_dataset_from_config(
    training_cfg: dict,
    model_cfg: dict | None = None,
    train: bool = True,
    cfg_path: Path | None = None,
    dataset_cfg: dict | None = None,
):
    """
    build_dataset_from_config Function

    Creates a dataset instance based on the training config and dataset section.

    Inputs:
        - training_cfg: (dict) Training configuration (must include data_root).
        - model_cfg: (dict | None) Optional model config (unused here).
        - train: (Boolean) If True uses train split, else test split.
        - cfg_path: (Path | None) Config path (kept for compatibility; no lookup side effects).
        - dataset_cfg: (dict | None) Optional dataset config section (preferred source).

    Outputs:
        - dataset: (object) Instantiated dataset.
    """
    del cfg_path  # retained to avoid breaking existing callsites
    dataset_cfg = dict(dataset_cfg or {})
    merged_cfg = dict(training_cfg or {})
    if dataset_cfg:
        extra_cfg = {k: v for k, v in dataset_cfg.items() if k not in {"class", "dataset_class"}}
        merged_cfg.update(extra_cfg)

    dataset_class = dataset_cfg.get("class") or dataset_cfg.get("dataset_class") or merged_cfg.get("dataset_class")
    if not dataset_class:
        dataset_class = _infer_dataset_class(merged_cfg, model_cfg)
    if not dataset_class:
        raise ValueError(
            "Could not resolve dataset class. Set config.dataset.class (preferred) "
            "or use a legacy config with inferable training.dataset/split_file."
        )
    return _build_from_class(str(dataset_class), merged_cfg, train)


def _infer_dataset_class(training_cfg: dict, model_cfg: dict | None = None) -> str | None:
    """
    Best-effort dataset class inference for legacy runs that do not ship dataset.json.
    """
    model_cfg = model_cfg or {}
    dataset_name = str(training_cfg.get("dataset", "")).strip().lower()
    conditioning = str(training_cfg.get("conditioning", model_cfg.get("conditioning", ""))).strip().lower()
    split_file = str(training_cfg.get("split_file", ""))

    if dataset_name == "mnist":
        return "datasets.mnist:MNISTDataset"
    if dataset_name == "ldct":
        if conditioning == "attention" or "encodeddataset" in split_file.lower() or "pixelattention" in split_file.lower():
            return "datasets.ldct:LDCTAttentionDataset"
        return "datasets.ldct:LDCTDataset"

    # Heuristic fallback from split-file path/content naming.
    if "mnist" in split_file.lower():
        return "datasets.mnist:MNISTDataset"
    if "ldct" in split_file.lower():
        if conditioning == "attention" or "encodeddataset" in split_file.lower() or "pixelattention" in split_file.lower():
            return "datasets.ldct:LDCTAttentionDataset"
        return "datasets.ldct:LDCTDataset"
    return None


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
    training_cfg = dict(cfg["training"])
    cfg_path_value = cfg.get("__config_path__") if isinstance(cfg, dict) else None
    cfg_path = Path(cfg_path_value) if cfg_path_value else None
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if "img_size" not in training_cfg:
        resolution = model_cfg.get("resolution") if isinstance(model_cfg, dict) else None
        if resolution is not None:
            training_cfg["img_size"] = resolution
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
    if isinstance(cfg, dict) and "dataset_class" in cfg and "class" not in dataset_cfg and "dataset_class" not in dataset_cfg:
        # Backward compatibility for legacy top-level dataset_class.
        dataset_cfg = dict(dataset_cfg)
        dataset_cfg["dataset_class"] = cfg.get("dataset_class")
    train_ds = build_dataset_from_config(training_cfg, model_cfg, train=True, cfg_path=cfg_path, dataset_cfg=dataset_cfg)
    val_ds = build_dataset_from_config(training_cfg, model_cfg, train=False, cfg_path=cfg_path, dataset_cfg=dataset_cfg)
    return train_ds, val_ds


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
        "split_file": "split_file",
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


def iter_batches(dataset, batch_size: int, indices: list[int] | None = None):
    """
    iter_batches Function

    Yields index lists and sample batches from a dataset.

    Inputs:
        - dataset: (Dataset) Dataset instance.
        - batch_size: (Int) Batch size.

    Outputs:
        - indices: (list<Int>) Sample indices.
        - samples: (list<dict>) Dataset samples.
    """
    selected = list(range(len(dataset))) if indices is None else list(indices)
    total = len(selected)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_indices = selected[start:end]
        samples = [dataset[i] for i in batch_indices]
        yield batch_indices, samples


def save_output_tensor(dataset, row: dict, key: str, tensor, output_root: Path) -> None:
    """
    save_output_tensor Function

    Saves a tensor using the cache path structure under an output root.

    Inputs:
        - dataset: (Dataset) Dataset instance.
        - row: (dict) Dataset row metadata.
        - key: (String) Target/conditioning key.
        - tensor: (Tensor) Tensor to save.
        - output_root: (Path) Base output directory.
    """
    entry = row.get(key)
    split_index, split_count = dataset._cache_info(entry, row, key)
    out_path = cache_path_for_entry(dataset.base_path, output_root, entry, split_index, split_count)
    if out_path is None:
        return
    writer = getattr(dataset, "save_output", None)
    if callable(writer):
        writer(row=row, key=key, tensor=tensor, output_root=output_root)
        return
    save_tensor_cache(tensor, out_path)


def to_2d_image(arr: torch.Tensor) -> np.ndarray | None:
    """
    Convert common tensor layouts to uint8 grayscale image if possible.
    Supports [H,W], [1,H,W], [C,H,W] with C in {1,3}.
    """
    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3 and arr.shape[0] == 1:
        img = arr[0]
    elif arr.ndim == 3 and arr.shape[0] in (3,):
        img = arr.mean(dim=0)
    else:
        return None
    img = img.clamp(0.0, 1.0).numpy()
    return (img * 255.0).round().astype(np.uint8)
