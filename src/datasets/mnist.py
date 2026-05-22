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
        self.base_path = self.root
        self.train = train
        self.img_size = img_size
        self.target_key = "target"
        self.conditioning_key = None

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
                transforms.PILToTensor(),  # returns uint8 in [0, 255]
                transforms.Lambda(lambda x: x.float() / 255.0),  # explicit [0,1] normalization
            ]
        )

        self.dataset = datasets.MNIST(
            root=str(self.root),
            train=self.train,
            download=download,
        )
        split_name = "train" if self.train else "test"
        self.data = [{self.target_key: f"mnist/{split_name}_{idx}.png"} for idx in range(len(self.dataset))]

    def __len__(self) -> int:
        return len(self.dataset)

    def to_image(self, image):
        """Map MNIST pixels into canonical image space [0, 1]."""
        if hasattr(image, "float"):
            image = image.float()
        return image / 255.0

    def from_image(self, image):
        """Invert canonical image space [0, 1] back into the native MNIST [0, 255] scale."""
        if hasattr(image, "clamp"):
            return image.clamp(0.0, 1.0) * 255.0
        return image.clip(0.0, 1.0) * 255.0

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

    def _cache_info(self, entry, row, key: str | None):
        """Mirror BaseDataset cache metadata contract for sampler compatibility."""
        return None, 1
