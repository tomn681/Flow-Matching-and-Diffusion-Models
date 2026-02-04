"""
Handler for VAE model sampling workflows.
"""

from __future__ import annotations

from .base import ModelHandler
from pipelines.samplers import vae as vae_sampler


class VAEHandler(ModelHandler):
    """
    VAE handler that exposes encode/decode/sample/evaluate.
    """

    def encode(self) -> None:
        vae_sampler.encode(
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
        vae_sampler.decode(
            ckpt_dir=self.ckpt_dir,
            data_txt=self.data_txt,
            save=self.save,
            output_dir=self.output_dir,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
        )

    def sample(self) -> None:
        vae_sampler.sample(
            ckpt_dir=self.ckpt_dir,
            data_txt=self.data_txt,
            save=self.save,
            output_dir=self.output_dir,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
        )

    def evaluate(self) -> None:
        vae_sampler.evaluate(
            ckpt_dir=self.ckpt_dir,
            data_txt=self.data_txt,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
        )
