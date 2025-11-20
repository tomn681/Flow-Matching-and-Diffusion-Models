"""
Training entrypoints for individual models.

Use via `python -m src.train --trainer <name> --config <json> --data-root <path>`.
"""

from .vae import train as train_vae

__all__ = ["train_vae"]
