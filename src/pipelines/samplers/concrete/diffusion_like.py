"""
Concrete sampler for diffusion-like models (DDPM / Flow Matching).
"""

from __future__ import annotations

from pipelines.samplers.abstract import AbstractSampler
from pipelines.samplers.diffusion_like import _run_debug_compare, _run_decode, _run_encode, _run_evaluate


class DiffusionLikeSampler(AbstractSampler):
    def __init__(self, *, model_type: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_type = str(model_type)

    def encode(self) -> None:
        _run_encode(
            ckpt_dir=self.ckpt_dir,
            model_type=self.model_type,
            data_txt=self.data_txt,
            save=self.save,
            output_dir=self.output_dir,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            timestep=self.timestep,
            num_samples=self.num_samples,
        )

    def decode(self) -> None:
        _run_decode(
            ckpt_dir=self.ckpt_dir,
            model_type=self.model_type,
            data_txt=self.data_txt,
            save=self.save,
            output_dir=self.output_dir,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            num_samples=self.num_samples,
            save_input=self.save_input,
            save_conditioning=self.save_conditioning,
        )

    def sample(self) -> None:
        self.decode()

    def evaluate(self) -> None:
        _run_evaluate(
            ckpt_dir=self.ckpt_dir,
            model_type=self.model_type,
            data_txt=self.data_txt,
            save=self.save,
            output_dir=self.output_dir,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            num_samples=self.num_samples,
            save_input=self.save_input,
            save_conditioning=self.save_conditioning,
        )

    def debug_compare(self) -> None:
        _run_debug_compare(
            ckpt_dir=self.ckpt_dir,
            model_type=self.model_type,
            data_txt=self.data_txt,
            output_dir=self.output_dir,
            device=self.device,
            seed=self.seed,
            num_samples=self.num_samples,
        )
