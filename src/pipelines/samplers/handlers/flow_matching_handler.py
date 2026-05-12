"""
Handler for flow-matching model sampling workflows.
"""

from __future__ import annotations

from .base import ModelHandler
from pipelines.samplers.concrete import DiffusionLikeSampler


class FlowMatchingHandler(ModelHandler):
    """
    Flow-matching handler that exposes encode/decode/sample/evaluate.
    """

    def create_sampler(self):
        return DiffusionLikeSampler(
            ckpt_dir=self.ckpt_dir,
            model_type="flow_matching",
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
        )
