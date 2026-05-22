from __future__ import annotations

from pathlib import Path

from sampling import SAMPLER_REGISTRY
from ._deprecation import warn_deprecated


class ModelHandler:
    """
    Deprecated compatibility wrapper around the new `sampling` package.
    """

    sampler_key: str | None = None

    def __init__(
        self,
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
        warn_deprecated(
            api=f"{self.__class__.__module__}.{self.__class__.__name__}",
            replacement="sampling.* samplers via SAMPLER_REGISTRY or run_model.py",
            stacklevel=3,
        )
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
    def sampler(self):
        if self.sampler_key is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not define `sampler_key`; use a concrete compat handler."
            )
        sampler_cls = SAMPLER_REGISTRY.get(self.sampler_key)
        return sampler_cls(
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
            save_tensor_cache=self.save_tensor_cache,
        )

    def encode(self) -> None:
        self.sampler.encode()

    def decode(self) -> None:
        self.sampler.decode()

    def build_tensor_cache(self) -> None:
        self.sampler.build_tensor_cache()

    def sample(self) -> None:
        self.sampler.sample()

    def evaluate(self) -> None:
        self.sampler.evaluate()

    def debug_compare(self) -> None:
        self.sampler.debug_compare()


class DiffusionHandler(ModelHandler):
    sampler_key = "diffusion"


class FlowMatchingHandler(ModelHandler):
    sampler_key = "flow_matching"


class VAEHandler(ModelHandler):
    sampler_key = "vae"


__all__ = ["ModelHandler", "DiffusionHandler", "FlowMatchingHandler", "VAEHandler"]

