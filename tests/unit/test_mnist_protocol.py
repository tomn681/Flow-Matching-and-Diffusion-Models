from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from core.protocols import SamplerCompatibleDataset
from datasets.mnist import MNISTDataset


class _FakeMNIST:
    def __init__(self, root: str, train: bool, download: bool) -> None:
        self._size = 5

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int):
        img = Image.fromarray((torch.rand(28, 28).numpy() * 255).astype("uint8"), mode="L")
        label = idx % 10
        return img, label


def test_mnist_dataset_satisfies_sampler_protocol(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("datasets.mnist.datasets.MNIST", _FakeMNIST)
    ds = MNISTDataset(root=str(tmp_path), train=True, img_size=32, download=False)

    assert isinstance(ds, SamplerCompatibleDataset)
    assert ds.target_key == "target"
    assert ds.conditioning_key is None
    assert isinstance(ds.data, list)
    assert len(ds.data) == len(ds)

    sample = ds[0]
    assert sample["target"].shape == (1, 32, 32)
    assert sample["image"].shape == (1, 32, 32)
    assert sample["img_id"].startswith("train_")
