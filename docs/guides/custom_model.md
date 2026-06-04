# Custom Model Guide

This guide shows how to add a new model family to the framework without
patching the training loop directly.

## 1. Implement the Model

Create a normal `nn.Module` under `src/models/...`.

Minimal example:

```python
import torch
import torch.nn as nn

from core.types import ModelOutput
from models.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("toy_autoencoder")
class ToyAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.decoder = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.encoder(x)
        rec = self.decoder(h)
        return ModelOutput(reconstruction=rec)
```

## 2. Add Factory Routing

If the model needs a new `model_type`, update [factory.py](/home/delas/Documents/LDCT/Flow-Matching-and-Diffusion-Models/src/models/factory.py) so `ModelFactory.build(...)` can route configs to it.

If it fits an existing family contract, register it under the existing branch and
reuse the current factory path.

## 3. Export the Public API

Update:

- [src/models/__init__.py](/home/delas/Documents/LDCT/Flow-Matching-and-Diffusion-Models/src/models/__init__.py)
- optionally [src/__init__.py](/home/delas/Documents/LDCT/Flow-Matching-and-Diffusion-Models/src/__init__.py)

## 4. Add a Config

```json
{
  "model": {
    "model_type": "toy_autoencoder",
    "in_channels": 1,
    "hidden_channels": 16
  },
  "training": {
    "epochs": 1,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "output_dir": "checkpoints/toy_autoencoder"
  }
}
```

## 5. Add Tests

At minimum:

- registry presence
- factory construction
- one forward-pass smoke test
- public import coverage if exported

## Rule of Thumb

Do not add a new inheritance layer unless the framework actually needs shared
behavior from it. Prefer protocols and explicit factory routing over speculative
base classes.
