from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear`` base projection.

    The wrapped module computes:
        ``base(x) + (x @ A @ B) * (alpha / rank)``
    where only ``A`` and ``B`` are trainable.
    """

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear base, got {type(base).__name__}.")
        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(base.in_features, self.rank))
        self.lora_B = nn.Parameter(torch.empty(self.rank, base.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out


__all__ = ["LoRALinear"]
