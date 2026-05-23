from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class EMAModel:
    """Track an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, track_all: bool = False) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self.track_all = bool(track_all)
        self.shadow_params: OrderedDict[str, torch.Tensor] = OrderedDict()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if (not self.track_all) and (not param.requires_grad):
                    continue
                self.shadow_params[name] = param.detach().clone()

    def step(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.shadow_params:
                    continue
                shadow = self.shadow_params[name]
                shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.shadow_params:
                    continue
                param.copy_(self.shadow_params[name])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "track_all": self.track_all,
            "shadow_params": {name: tensor.clone() for name, tensor in self.shadow_params.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.track_all = bool(state.get("track_all", False))
        shadow = state.get("shadow_params", {})
        self.shadow_params = OrderedDict((name, tensor.clone()) for name, tensor in shadow.items())


__all__ = ["EMAModel"]
