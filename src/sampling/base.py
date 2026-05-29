from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from utils.sampling_utils import build_tensor_cache_from_config, load_run_config


class BaseSampler:
    """Top-level sampler base used by run_model dispatch."""

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
        scheduler: str | None = None,
        save_tensor_cache: bool = False,
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
        self.scheduler = scheduler
        self.save_tensor_cache = bool(save_tensor_cache)

    @property
    def _common_kwargs(self) -> dict[str, Any]:
        return {
            "ckpt_dir": self.ckpt_dir,
            "data_txt": self.data_txt,
            "save": self.save,
            "output_dir": self.output_dir,
            "batch_size": self.batch_size,
            "device": self.device,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "save_tensor_cache": self.save_tensor_cache,
        }

    @property
    def _decode_like_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self._common_kwargs)
        kwargs["save_input"] = self.save_input
        kwargs["save_conditioning"] = self.save_conditioning
        return kwargs

    @property
    def _generative_decode_like_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self._decode_like_kwargs)
        kwargs["num_inference_steps"] = self.num_inference_steps
        kwargs["start_step"] = self.start_step
        kwargs["last_n_steps"] = self.last_n_steps
        kwargs["scheduler"] = self.scheduler
        return kwargs

    @property
    def _debug_compare_kwargs(self) -> dict[str, Any]:
        return {
            "ckpt_dir": self.ckpt_dir,
            "data_txt": self.data_txt,
            "output_dir": self.output_dir,
            "device": self.device,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "save_tensor_cache": self.save_tensor_cache,
        }

    @property
    def _generative_debug_compare_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self._debug_compare_kwargs)
        kwargs["num_inference_steps"] = self.num_inference_steps
        kwargs["start_step"] = self.start_step
        kwargs["last_n_steps"] = self.last_n_steps
        kwargs["scheduler"] = self.scheduler
        return kwargs

    def build_tensor_cache(self) -> None:
        cfg = load_run_config(self.ckpt_dir)
        if self.save_tensor_cache:
            cfg.setdefault("training", {})["save_tensor_cache"] = True
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

    def encode(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement encode().")

    def decode(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement decode().")

    def sample(self) -> None:
        self.decode()

    def evaluate(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement evaluate().")

    def debug_compare(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement debug_compare().")
