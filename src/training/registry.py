from __future__ import annotations

from core.registry import Registry


TRAINER_REGISTRY = Registry("trainers")

__all__ = ["TRAINER_REGISTRY"]
