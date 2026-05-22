from __future__ import annotations

import torch.nn as nn

from core.registry import Registry


MODEL_REGISTRY = Registry[nn.Module]("models", base_type=nn.Module)


__all__ = ["MODEL_REGISTRY"]

