"""
Abstract handler interface for sampling/encoding/decoding workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ModelHandler(ABC):
    """
    Interface for model-specific sampling workflows.
    """

    def __init__(
        self,
        ckpt_dir: Path,
        data_txt: str | None = None,
        save: bool = False,
        output_dir: str | None = None,
        batch_size: int = 4,
        device: str | None = None,
        seed: int = 42,
        timestep: int | None = None,
        num_samples: int | None = None,
        save_input: bool = False,
        save_conditioning: bool = False,
    ) -> None:
        """
        Constructor Method

        Inputs:
            - ckpt_dir: (Path) Checkpoint directory.
            - data_txt: (String | None) Optional split file override.
            - save: (Boolean) Whether to save outputs.
            - output_dir: (String | None) Output directory override.
            - batch_size: (Int) Batch size.
            - device: (String | None) Torch device.
            - seed: (Int) Random seed.
            - timestep: (Int | None) Optional timestep for encode.

        Outputs:
            - handler: (ModelHandler) Initialized handler.
        """
        self.ckpt_dir = Path(ckpt_dir)
        self.data_txt = data_txt
        self.save = save
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.timestep = timestep
        self.num_samples = num_samples
        self.save_input = save_input
        self.save_conditioning = save_conditioning

    @property
    def sampler(self):
        if not hasattr(self, "_sampler"):
            self._sampler = self.create_sampler()
        return self._sampler

    @abstractmethod
    def create_sampler(self):
        """
        Create concrete sampler instance for this handler.
        """

    def encode(self) -> None:
        self.sampler.encode()

    def decode(self) -> None:
        self.sampler.decode()

    def build_tensor_cache(self) -> None:
        self.sampler.build_tensor_cache()

    def sample(self) -> None:
        self.sampler.sample()

    def evaluate(self) -> None:
        self.sampler.evaluate()
