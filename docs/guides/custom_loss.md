# Custom Loss Guide

Framework losses use the context-based API assembled by `LossAssembler`.

## 1. Implement a Loss Component

```python
import torch

from losses.base import BaseLossComponent
from losses.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("toy_l1")
class ToyL1Loss(BaseLossComponent):
    name = "toy_l1"

    def compute(self, *, context: dict) -> torch.Tensor:
        pred = context["reconstruction"]
        target = context["target"]
        return torch.mean(torch.abs(pred - target))
```

## 2. Use It from a Trainer

Either:

- make it selectable by config in an existing trainer, or
- compose it explicitly in a trainer override / builder path

## 3. Loss Context Rules

The framework passes a context dictionary instead of positional tensors. That is
intentional:

- trainers can expose heterogeneous model outputs
- loss components stay decoupled from model subclasses
- activation scheduling stays per-component

## 4. Add Tests

- registry presence
- finite scalar output
- zero / missing-input behavior if relevant
- assembler integration if it contributes to a composed objective
