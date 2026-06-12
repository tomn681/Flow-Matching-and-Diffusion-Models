from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager

import torch
import torch.nn as nn


class EMAModel:
    """Track an exponential moving average of model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        track_all: bool = False,
        use_warmup: bool = True,
    ) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self.track_all = bool(track_all)
        self.use_warmup = bool(use_warmup)
        self.num_updates = 0
        self.shadow_params: OrderedDict[str, torch.Tensor] = OrderedDict()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if (not self.track_all) and (not param.requires_grad):
                    continue
                self.shadow_params[name] = param.detach().clone()

    def _effective_decay(self) -> float:
        if not self.use_warmup:
            return self.decay
        warmup_decay = (1.0 + float(self.num_updates)) / (10.0 + float(self.num_updates))
        return min(self.decay, warmup_decay)

    def step(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self._effective_decay()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.shadow_params:
                    continue
                shadow = self.shadow_params[name]
                shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.shadow_params:
                    continue
                param.copy_(self.shadow_params[name])

    def backup_from(self, model: nn.Module) -> OrderedDict[str, torch.Tensor]:
        backup: OrderedDict[str, torch.Tensor] = OrderedDict()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.shadow_params:
                    continue
                backup[name] = param.detach().clone()
        return backup

    def restore(self, model: nn.Module, backup: OrderedDict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in backup:
                    continue
                param.copy_(backup[name])

    @contextmanager
    def average_parameters(self, model: nn.Module):
        if not self.shadow_params:
            yield
            return
        backup = self.backup_from(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model, backup)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "track_all": self.track_all,
            "use_warmup": self.use_warmup,
            "num_updates": self.num_updates,
            "shadow_params": {name: tensor.clone() for name, tensor in self.shadow_params.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.track_all = bool(state.get("track_all", False))
        self.use_warmup = bool(state.get("use_warmup", True))
        self.num_updates = int(state.get("num_updates", 0))
        shadow = state.get("shadow_params", {})
        self.shadow_params = OrderedDict((name, tensor.clone()) for name, tensor in shadow.items())


def apply_ema_state_to_model(model: nn.Module, ema_state: dict | None) -> bool:
    if not isinstance(ema_state, dict):
        return False
    shadow = ema_state.get("shadow_params")
    if not isinstance(shadow, dict):
        return False
    with torch.no_grad():
        for name, param in model.named_parameters():
            tensor = shadow.get(name)
            if tensor is None:
                continue
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
    return True


__all__ = ["EMAModel", "apply_ema_state_to_model"]
