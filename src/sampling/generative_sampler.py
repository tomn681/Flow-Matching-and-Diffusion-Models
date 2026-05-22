from __future__ import annotations

from pipelines.samplers.diffusion_like import _run_debug_compare, _run_decode, _run_encode, _run_evaluate

from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


class GenerativeSampler(BaseSampler):
    model_type: str

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
            save_tensor_cache=self.save_tensor_cache,
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
            num_inference_steps=self.num_inference_steps,
            start_step=self.start_step,
            last_n_steps=self.last_n_steps,
            scheduler=self.scheduler,
            save_tensor_cache=self.save_tensor_cache,
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
            num_inference_steps=self.num_inference_steps,
            start_step=self.start_step,
            last_n_steps=self.last_n_steps,
            scheduler=self.scheduler,
            save_tensor_cache=self.save_tensor_cache,
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
            num_inference_steps=self.num_inference_steps,
            start_step=self.start_step,
            last_n_steps=self.last_n_steps,
            scheduler=self.scheduler,
            save_tensor_cache=self.save_tensor_cache,
        )


@SAMPLER_REGISTRY.register("diffusion")
class DiffusionSampler(GenerativeSampler):
    model_type = "diffusion"


@SAMPLER_REGISTRY.register("flow_matching")
class FlowMatchingSampler(GenerativeSampler):
    model_type = "flow_matching"

