"""
Training and inference pipelines for the diffusion / VAE components.

Training entrypoints live under `src.pipelines.train` and are dispatched via
`python -m src.train --trainer <name> ...`. Inference/visualisation helpers
live under `src.pipelines.samplers` and are dispatched via
`python -m src.sample --sampler <name> ...`.
"""

from . import samplers, train

__all__ = ["samplers", "train"]
