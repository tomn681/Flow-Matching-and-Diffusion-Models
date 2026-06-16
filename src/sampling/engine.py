from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys as _sys
from typing import Any

from core.families import model_family_for_model_type

from .base import BaseSampler
from .registry import SAMPLER_REGISTRY


@dataclass(frozen=True)
class SamplingRequest:
    ckpt_dir: Path | str
    model_type: str
    mode: str
    data_txt: str | None = None
    save: bool = False
    output_dir: str | None = None
    batch_size: int = 4
    device: str | None = None
    seed: int = 42
    timestep: int | None = None
    num_samples: int | None = None
    save_input: bool = False
    save_conditioning: bool = False
    save_diff_map: bool = False
    diff_amplify: float = 5.0
    num_inference_steps: int | None = None
    start_step: int | None = None
    last_n_steps: int | None = None
    cfg_rescale: float = 0.0
    scheduler: str | None = None
    save_tensor_cache: bool = False
    num_pairs: int | None = None
    use_ema: bool = False

    def sampler_kwargs(self) -> dict[str, Any]:
        return {
            "ckpt_dir": self.ckpt_dir,
            "data_txt": self.data_txt,
            "save": self.save,
            "output_dir": self.output_dir,
            "batch_size": self.batch_size,
            "device": self.device,
            "seed": self.seed,
            "timestep": self.timestep,
            "num_samples": self.num_samples,
            "save_input": self.save_input,
            "save_conditioning": self.save_conditioning,
            "save_diff_map": self.save_diff_map,
            "diff_amplify": self.diff_amplify,
            "num_inference_steps": self.num_inference_steps,
            "start_step": self.start_step,
            "last_n_steps": self.last_n_steps,
            "cfg_rescale": self.cfg_rescale,
            "scheduler": self.scheduler,
            "save_tensor_cache": self.save_tensor_cache,
            "num_pairs": self.num_pairs,
            "use_ema": self.use_ema,
        }


class CheckpointResolver:
    @staticmethod
    def model_type_from_config(cfg: dict) -> str:
        return str(cfg.get("model", {}).get("model_type", "vae")).strip().lower()


class SamplingEngine:
    def resolve_sampler_cls(self, model_type: str):
        family = model_family_for_model_type(model_type)
        sampler_key = family.sampler_key if family is not None and family.sampler_key is not None else str(model_type).strip().lower()
        return SAMPLER_REGISTRY.get(sampler_key)

    @staticmethod
    def supports_mode(sampler, mode: str) -> bool:
        mode_key = str(mode).strip().lower()
        supported_modes = getattr(sampler, "supported_modes", None)
        if supported_modes is not None:
            return mode_key in {str(v).strip().lower() for v in supported_modes}
        sampler_type = type(sampler)
        if mode_key == "encode":
            return sampler_type.encode is not BaseSampler.encode
        if mode_key == "decode":
            return sampler_type.decode is not BaseSampler.decode
        if mode_key == "sample":
            return sampler_type.sample is not BaseSampler.sample or sampler_type.decode is not BaseSampler.decode
        if mode_key == "evaluate":
            return sampler_type.evaluate is not BaseSampler.evaluate
        if mode_key in {"build_tensor_cache", "debug_compare", "generate_reflow_pairs"}:
            return hasattr(sampler, mode_key)
        return False

    def build_sampler(self, request: SamplingRequest):
        sampler_cls = self.resolve_sampler_cls(request.model_type)
        return sampler_cls(**request.sampler_kwargs())

    def run(self, request: SamplingRequest) -> None:
        sampler = self.build_sampler(request)
        method = getattr(sampler, request.mode, None)
        if method is None:
            raise ValueError(f"Unknown mode '{request.mode}'.")
        if not self.supports_mode(sampler, request.mode):
            raise ValueError(
                f"Mode '{request.mode}' is not implemented by sampler '{type(sampler).__name__}'."
            )
        method()

    def run_many(self, requests: list[SamplingRequest]) -> None:
        for request in requests:
            self.run(request)


__all__ = ["SamplingRequest", "SamplingEngine", "CheckpointResolver"]

_module = _sys.modules[__name__]
if __name__.startswith("genlib.sampling."):
    _sys.modules.setdefault(__name__.replace("genlib.sampling.", "sampling.", 1), _module)
elif __name__.startswith("src.sampling."):
    _sys.modules.setdefault(__name__.replace("src.sampling.", "sampling.", 1), _module)
    _sys.modules.setdefault(__name__.replace("src.sampling.", "genlib.sampling.", 1), _module)
elif __name__.startswith("sampling."):
    _sys.modules.setdefault(__name__.replace("sampling.", "src.sampling.", 1), _module)
    _sys.modules.setdefault(__name__.replace("sampling.", "genlib.sampling.", 1), _module)
del _module, _sys
