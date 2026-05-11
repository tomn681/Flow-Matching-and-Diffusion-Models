from __future__ import annotations

import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch unavailable
    torch = None

try:
    import pydicom
except ImportError:  # pragma: no cover - optional dependency
    pydicom = None

from PIL import Image


def load_image(path: str, id: str | None = None) -> dict:
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
        return {"Image": np.load(path), "Metadata": None, "Id": id if id else path}

    if ext in {".pt", ".pth"}:
        if torch is None:
            raise ImportError("torch is required to load .pt/.pth tensors.")
        tensor = torch.load(path)
        array = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
        return {"Image": array, "Metadata": None, "Id": id if id else path}

    image = Image.open(path)
    return {"Image": np.array(image), "Metadata": None, "Id": id if id else path}


def load_composite(
    path_list: Sequence[str],
    id: str | None = None,
    dim: int = 3,
    metadata: str | None = "first",
    multi_cpu: bool = False,
) -> dict:
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

    return {"Image": image_stack, "Metadata": meta, "Id": id}


def load(path: str | Sequence[str], id: str | None, dim: int = 3) -> dict:
    if isinstance(path, str):
        if os.path.isdir(path):
            files = [os.path.join(path, name) for name in os.listdir(path)]
            return load_composite(files, id, dim)
        return load_image(path, id)
    return load_composite(path, id, dim)
