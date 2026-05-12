"""
Primary abstract sampler interface.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path

from utils.sampling_utils import build_tensor_cache_from_config, load_run_config


class BaseSampler(abc.ABC):
    """
    Shared sampler base that carries common runtime state and cache builder.
    Abstract interfaces build on top of this.
    """

    def __init__(
        self,
        *,
        ckpt_dir: Path | str,
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
        num_inference_steps: int | None = None,
        start_step: int | None = None,
        last_n_steps: int | None = None,
    ) -> None:
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
        self.num_inference_steps = num_inference_steps
        self.start_step = start_step
        self.last_n_steps = last_n_steps

    def build_tensor_cache(self) -> None:
        cfg = load_run_config(self.ckpt_dir)
        if not bool(cfg.get("training", {}).get("save_tensor_cache", False)):
            logging.warning(
                "build_tensor_cache requested but training.save_tensor_cache is false. "
                "No cache files will be written unless you set it to true."
            )
        total = build_tensor_cache_from_config(
            cfg=cfg,
            data_txt=self.data_txt,
            batch_size=self.batch_size,
            seed=self.seed,
            num_samples=self.num_samples,
            desc="build_tensor_cache",
            evaluate=True,
        )
        logging.info("Tensor cache build completed for %d samples.", total)
        print(f"Tensor cache build completed for {total} samples.")


class AbstractSampler(BaseSampler, abc.ABC):
    """
    Interface contract for all sampler implementations.
    """

    @abc.abstractmethod
    def encode(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError
