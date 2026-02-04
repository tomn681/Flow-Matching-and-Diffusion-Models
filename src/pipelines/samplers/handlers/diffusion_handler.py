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
        )

    def evaluate(self) -> None:
        diffusion_sampler.evaluate(
            ckpt_dir=self.ckpt_dir,
            data_txt=self.data_txt,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
        )
