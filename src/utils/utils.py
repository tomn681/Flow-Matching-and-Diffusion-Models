from __future__ import annotations

import os
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Sequence

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

__all__ = ["lot_id", "load", "load_image", "load_composite", "select_visual_indices"]


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

def load_image(path: str, id: str | None = None) -> dict:
    """
    Single-file image loader supporting DICOM, numpy, PyTorch, and standard images.
    """
    ext = Path(path).suffix.lower()

    if ext == ".dcm":
        if pydicom is None:
            raise ImportError("pydicom is required to load DICOM files.")
        image = pydicom.dcmread(path)
        metadata = {}
        for element in image:
            if getattr(element, "name", None) == "Pixel Data":
                continue
            try:
                name = str(element.name)
                value = element.value
                if value is None:
                    continue
                metadata[name] = str(value)
            except Exception:
                continue
        return {
            "Image": image.pixel_array,
            "Metadata": metadata if metadata else None,
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


def select_visual_indices(ds, count: int, seed: int | None = None) -> list[int]:
    """
    select_visual_indices Function

    Selects indices for visual batches, preferring one sample per case when available.

    Inputs:
        - ds: (Dataset) Dataset instance.
        - count: (Int) Number of indices to return.
        - seed: (Int | None) Optional random seed for deterministic selection.

    Outputs:
        - indices: (list<Int>) Selected dataset indices.
    """
    total = len(ds)
    if total <= 0:
        return []
    rng = random.Random(seed)
    indices = []
    if hasattr(ds, "data") and isinstance(getattr(ds, "data"), list):
        cases = {}
        for idx, row in enumerate(ds.data):
            case_id = row.get("Case") or row.get("case") or row.get("case_id")
            if case_id is None:
                continue
            cases.setdefault(case_id, []).append(idx)
        if cases:
            case_ids = list(cases.keys())
            rng.shuffle(case_ids)
            for case_id in case_ids[:count]:
                indices.append(rng.choice(cases[case_id]))
    if not indices:
        indices = list(range(total))
        rng.shuffle(indices)
        indices = indices[:count]
    return indices
