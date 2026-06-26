"""
Utility helpers for dataset configuration, cache handling, and path resolution.
"""

from __future__ import annotations

import inspect
import os
import re
from importlib import import_module
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .dataset_runtime import (
    cache_path_for_entry,
    iter_batches,
    save_output_tensor,
    save_tensor_cache,
    to_2d_image,
)
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
    merged_cfg["tensor_cache_subdir"] = resolve_tensor_cache_subdir(
        merged_cfg,
        dataset_class=str(dataset_class),
        train=train,
    )
    return _build_from_class(str(dataset_class), merged_cfg, train)


def resolve_tensor_cache_subdir(training_cfg: dict, *, dataset_class: str, train: bool) -> str:
    """
    Build a cache namespace from the effective data contract so all modes can
    safely share train/test caches without collisions across resolutions or
    slice/volume layouts.
    """
    base = str(training_cfg.get("tensor_cache_subdir", "cache")).strip() or "cache"
    dataset_tag = _slugify(dataset_class.split(":")[-1].replace("Dataset", ""))
    split_tag = "train" if train else "test"
    spatial_tag = _cache_spatial_tag(training_cfg)
    window_tag = _cache_window_tag(training_cfg)

    return "/".join(
        [
            base,
            dataset_tag,
            split_tag,
            spatial_tag,
            window_tag,
        ]
    )


def _cache_spatial_tag(training_cfg: dict) -> str:
    size = training_cfg.get("img_size", training_cfg.get("volume_size"))
    if size is None:
        return "native"
    if isinstance(size, int):
        dims = 2
        values = [int(size), int(size)]
    else:
        values = [int(v) for v in size]
        dims = len(values)
    return f"{dims}d_" + "x".join(str(v) for v in values)


def _cache_window_tag(training_cfg: dict) -> str:
    window = training_cfg.get("window_size", training_cfg.get("slice_count", 1))
    try:
        return f"ws{int(window)}"
    except Exception:
        return "ws1"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "dataset"


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
    if dataset_name == "medical3d":
        return "datasets.medical3d:Medical3DDataset"
    if dataset_name == "video":
        return "datasets.video:VideoDataset"
    if dataset_name == "ldct":
        if conditioning == "attention" or "encodeddataset" in split_file.lower() or "pixelattention" in split_file.lower():
            return "datasets.ldct:LDCTAttentionDataset"
        return "datasets.ldct:LDCTDataset"

    # Heuristic fallback from split-file path/content naming.
    if "mnist" in split_file.lower():
        return "datasets.mnist:MNISTDataset"
    if "medical3d" in split_file.lower() or "nifti" in split_file.lower() or "volume" in split_file.lower():
        return "datasets.medical3d:Medical3DDataset"
    if "video" in split_file.lower():
        return "datasets.video:VideoDataset"
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
