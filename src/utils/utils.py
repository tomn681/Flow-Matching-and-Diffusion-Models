from __future__ import annotations

import json
import os
import random
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

__all__ = ["lot_id", "n_slice_split", "load"]


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

    rows = []
    for idx, (name, module) in enumerate(model.named_modules()):
        if name == "":
            continue
        params = sum(p.numel() for p in module.parameters())
        if params == 0:
            continue
        rows.append((idx, name, module.__class__.__name__, params))

    header = f"{'idx':>4}  {'module':<40}  {'params':>10}"
    lines = ["Model summary (compact):", header, "-" * len(header)]
    for idx, name, cls_name, params in rows:
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
