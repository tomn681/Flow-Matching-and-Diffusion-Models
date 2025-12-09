from __future__ import annotations

import json
import os
import random
import re
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - torch unavailable
    torch = None

try:
    import pydicom
except ImportError:  # pragma: no cover - optional dependency
    pydicom = None

from PIL import Image
import logging

__all__ = ["lot_id", "n_slice_split", "load", "allocate_run_dir"]


def load_json_config(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as fh:
        return json.load(fh)


def save_json_config(path: Path | str, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(cfg, fh, indent=2)


def allocate_run_dir(base: Path | str) -> Path:
    """
    Given a base directory, pick the next available run directory with suffix _runN.
    Example: base checkpoints/mnist -> checkpoints/mnist_run1, _run2, etc.
    """
    base = Path(base)
    parent = base.parent
    stem = base.name
    parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(stem)}_run(\d+)$")
    existing = []
    for entry in parent.iterdir():
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                existing.append(int(m.group(1)))
    next_id = (max(existing) + 1) if existing else 1
    return parent / f"{stem}_run{next_id}"


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(value, default: "torch.device") -> "torch.device":
    if value is None:
        return default
    if isinstance(value, str) and value.lower() == "none":
        return default
    return value if isinstance(value, torch.device) else torch.device(value)


def summarize_model(model: "torch.nn.Module", vae_cfg: dict, training_cfg: dict) -> None:
    show = training_cfg.get("show_model_summary", True)
    if not show:
        return

    def fmt(count: int) -> str:
        if count >= 1e6:
            return f"{count/1e6:.2f}M"
        if count >= 1e3:
            return f"{count/1e3:.2f}K"
        return str(count)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Build an input shape for summary
    spatial_dims = vae_cfg.get("spatial_dims", 2)
    in_ch = vae_cfg.get("in_channels", 3)
    res = vae_cfg.get("resolution", 256)
    if spatial_dims == 3:
        input_size = (1, in_ch, res, res, res)
    elif spatial_dims == 1:
        input_size = (1, in_ch, res)
    else:
        input_size = (1, in_ch, res, res)

    lines = []
    try:
        from torchinfo import summary  # type: ignore

        summary_obj = summary(
            model,
            input_size=input_size,
            col_names=("output_size", "num_params"),
            verbose=0,
            depth=10,
        )
        allowed = ("conv", "linear", "attention", "pool")
        header = f"{'idx':>4}  {'module':<40}  {'params':>10}  {'output':<20}"
        lines.append("Model summary (compact):")
        lines.append(header)
        lines.append("-" * len(header))
        for layer in summary_obj.summary_list:
            cls_name = layer.class_name.lower()
            if not any(k in cls_name for k in allowed):
                continue
            name = layer.layer_name
            params = layer.num_params
            out = str(layer.output_size)
            lines.append(f"{layer.depth:>4}  {name[:40]:<40}  {fmt(params):>10}  {out:<20}")
    except Exception as exc:
        logging.debug("torchinfo summary skipped (%s)", exc)
        header = f"{'idx':>4}  {'module':<40}  {'params':>10}"
        lines.append("Model summary (compact):")
        lines.append(header)
        lines.append("-" * len(header))
        for idx, (name, module) in enumerate(model.named_modules()):
            if name == "":
                continue
            params = sum(p.numel() for p in module.parameters())
            if params == 0:
                continue
            cls_lower = module.__class__.__name__.lower()
            if not any(k in cls_lower for k in ("conv", "linear", "attention", "pool")):
                continue
            lines.append(f"{idx:>4}  {name[:40]:<40}  {fmt(params):>10}")

    lines.append(f"Total params: {fmt(total_params)} ({total_params}) | Trainable: {fmt(trainable_params)} ({trainable_params})")

    for line in lines:
        logging.info(line)
        print(line, flush=True)


def lot_id(df: pd.DataFrame, case_column: str, number_column: str) -> pd.DataFrame:
    """
    Generate unique identifiers for each multi-file image split lot.
    """
    df = df.copy()
    grouped = df.groupby(case_column)

    for case, group in grouped:
        for idx, row in group.iterrows():
            files = row[number_column]
            if not isinstance(files, (list, tuple)) or not files:
                continue
            first_elem = os.path.basename(files[0]).split(".")[0]
            last_elem = os.path.basename(files[-1]).split(".")[0]
            new_name = f"I{case}S{idx}F{first_elem}T{last_elem}C{len(files)}"
            df.at[idx, case_column] = new_name
    return df


def n_slice_split(directory: str, split: int = 3) -> list[list[str]]:
    """
    Return every n-consecutive-path combination from a given directory.
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


def load_image(path: str, id: str | None = None) -> dict:
    """
    Single-file image loader supporting DICOM, numpy, PyTorch, and standard images.
    """
    ext = Path(path).suffix.lower()

    if ext == ".dcm":
        if pydicom is None:
            raise ImportError("pydicom is required to load DICOM files.")
        image = pydicom.dcmread(path)
        metadata = {
            str(element.name): str(element.value)
            for element in image
            if element.name != "Pixel Data"
        }
        return {
            "Image": image.pixel_array,
            "Metadata": metadata,
            "Id": id if id else path,
        }

    if ext in {".npz", ".npy"}:
        return {
            "Image": np.load(path),
            "Metadata": None,
            "Id": id if id else path,
        }

    if ext in {".pt", ".pth"}:
        if torch is None:
            raise ImportError("torch is required to load .pt/.pth tensors.")
        tensor = torch.load(path)
        array = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
        return {
            "Image": array,
            "Metadata": None,
            "Id": id if id else path,
        }

    image = Image.open(path)
    return {
        "Image": np.array(image),
        "Metadata": None,
        "Id": id if id else path,
    }


def load_composite(
    path_list: Sequence[str],
    id: str | None = None,
    dim: int = 3,
    metadata: str | None = "first",
    multi_cpu: bool = False,
) -> dict:
    """
    Multi-file image loader with optional multiprocessing and metadata retention.
    """
    assert dim in (2, 3), "Dimension dim in load() must be an integer between 2 and 3"
    assert metadata in ("first", "last", None), f"Metadata option unavailable: {metadata}"

    if multi_cpu and len(path_list) > 1:
        with Pool(processes=cpu_count()) as pool:
            files = pool.map(load_image, path_list)
    else:
        files = [load_image(path) for path in path_list]

    files = [f for f in files if f is not None]
    files.sort(key=lambda x: x["Id"])

    meta = None
    if metadata:
        meta = files[0 if metadata == "first" else -1]["Metadata"]

    images = [f["Image"] for f in files]
    image_stack = np.stack(images) if dim == 3 else np.hstack(images)

    return {
        "Image": image_stack,
        "Metadata": meta,
        "Id": id,
    }


def load(path: str | Sequence[str], id: str | None, dim: int = 3) -> dict:
    """
    Default image loader wrapping both single and multi-file inputs.
    """
    if isinstance(path, str):
        if os.path.isdir(path):
            files = [os.path.join(path, name) for name in os.listdir(path)]
            return load_composite(files, id, dim)
        return load_image(path, id)
    return load_composite(path, id, dim)
