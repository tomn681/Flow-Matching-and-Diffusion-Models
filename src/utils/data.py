from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .dataset import DefaultDataset
from .mnist import MNISTDataset


def _infer_img_size(training_cfg: dict, vae_cfg: dict) -> int:
    """
    Prefer the training.img_size override, otherwise fall back to the VAE resolution.
    """
    return int(training_cfg.get("img_size", vae_cfg.get("resolution", 256)))


def build_dataset_from_config(training_cfg: dict, vae_cfg: dict, train: bool = True):
    """
    Create a dataset instance based on the training config.
    Supports the legacy LDCT layout (DefaultDataset) and MNIST.
    """
    dataset_name = (training_cfg.get("dataset") or "ldct").lower()
    img_size = _infer_img_size(training_cfg, vae_cfg)

    if dataset_name == "mnist":
        root = Path(training_cfg.get("data_root", "data/mnist"))
        download = bool(training_cfg.get("download", True))
        return MNISTDataset(root=str(root), train=train, img_size=img_size, download=download)

    if dataset_name in ("ldct", "default", "ct"):
        data_root = Path(training_cfg["data_root"])
        common = dict(
            s_cnt=training_cfg.get("slice_count", 1),
            img_size=img_size,
            diff=training_cfg.get("diff", True),
            norm=training_cfg.get("norm", True),
        )
        return DefaultDataset(str(data_root), train=train, **common)

    raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected 'mnist' or 'ldct'.")


def build_train_val_datasets(cfg: dict) -> Tuple[object, object]:
    """
    Convenience helper that builds train/val splits from the full config dict.
    """
    training_cfg = cfg["training"]
    vae_cfg = cfg.get("vae", {})
    train_ds = build_dataset_from_config(training_cfg, vae_cfg, train=True)
    val_ds = build_dataset_from_config(training_cfg, vae_cfg, train=False)
    return train_ds, val_ds
