"""
Handler for VAE model sampling workflows.
"""

from __future__ import annotations

from .base import ModelHandler
from pipelines.samplers.concrete import VAESampler


class VAEHandler(ModelHandler):
    """
    VAE handler that exposes encode/decode/sample/evaluate.
    """

    def create_sampler(self):
        return VAESampler(
            ckpt_dir=self.ckpt_dir,
            data_txt=self.data_txt,
            save=self.save,
            output_dir=self.output_dir,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            timestep=self.timestep,
            num_samples=self.num_samples,
            save_input=self.save_input,
            save_conditioning=self.save_conditioning,
            num_inference_steps=self.num_inference_steps,
            start_step=self.start_step,
            last_n_steps=self.last_n_steps,
            scheduler=self.scheduler,
        )
