"""
Handler for diffusion model sampling workflows.
"""

from __future__ import annotations

from pathlib import Path

from .base import ModelHandler
from pipelines.samplers import diffusion as diffusion_sampler


class DiffusionHandler(ModelHandler):
    """
    Diffusion handler that exposes encode/decode/sample/evaluate.
    """

    def encode(self) -> None:
        diffusion_sampler.encode(
            ckpt_dir=self.ckpt_dir,
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
        diffusion_sampler.decode(
            ckpt_dir=self.ckpt_dir,
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
        diffusion_sampler.sample(
            ckpt_dir=self.ckpt_dir,
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

    def evaluate(self) -> None:
        diffusion_sampler.evaluate(
            ckpt_dir=self.ckpt_dir,
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
