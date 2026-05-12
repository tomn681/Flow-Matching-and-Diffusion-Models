"""
Concrete sampler for VAE models.
"""

from __future__ import annotations

from .autoencoder import AutoencoderSampler
from pipelines.samplers import autoencoder_like as autoencoder_sampler


class VAESampler(AutoencoderSampler):
    def encode(self) -> None:
        autoencoder_sampler.encode(
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
        autoencoder_sampler.decode(
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
        autoencoder_sampler.sample(
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
        autoencoder_sampler.evaluate(
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

    def debug_compare(self) -> None:
        autoencoder_sampler.debug_compare(
            ckpt_dir=self.ckpt_dir,
            data_txt=self.data_txt,
            output_dir=self.output_dir,
            device=self.device,
            seed=self.seed,
            num_samples=self.num_samples,
        )
