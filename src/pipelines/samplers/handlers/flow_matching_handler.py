"""
Handler for flow-matching model sampling workflows.
"""

from __future__ import annotations

from .base import ModelHandler
from pipelines.samplers import flow_matching as flow_sampler


class FlowMatchingHandler(ModelHandler):
    """
    Flow-matching handler that exposes encode/decode/sample/evaluate.
    """

    def encode(self) -> None:
        flow_sampler.encode(
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
        flow_sampler.decode(
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
        flow_sampler.sample(
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
        flow_sampler.evaluate(
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
