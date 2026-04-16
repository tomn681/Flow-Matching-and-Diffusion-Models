"""
Training and inference pipelines for the diffusion / VAE components.

Training entrypoints live under `src.pipelines.train` and are dispatched via
`python train.py --config <json>`. Inference/visualisation helpers live under
`src.pipelines.samplers` and are dispatched via `python run_model.py --mode ...`.
"""

__all__ = []
