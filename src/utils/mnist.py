from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


class MNISTDataset(Dataset):
    """
    Simple MNIST loader that auto-downloads and resizes digits to a square tensor.
    Returns dicts compatible with the VAE training pipeline (uses the `target` key).
    """

    def __init__(self, root: str, train: bool = True, img_size: int = 32, download: bool = True) -> None:
        self.root = Path(root)
        self.train = train
        self.img_size = img_size

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # scales to [0,1], keeps single channel
            ]
        )

        self.dataset = datasets.MNIST(
            root=str(self.root),
            train=self.train,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        image = self.transform(image)

        return {
            "target": image,
            "image": image,
            "label": int(label),
            "img_id": f"{'train' if self.train else 'test'}_{idx}",
            "img_size": (self.img_size, self.img_size),
        }
