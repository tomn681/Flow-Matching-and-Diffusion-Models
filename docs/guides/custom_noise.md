# Custom Noise Guide

Noise processes define how clean data is corrupted for diffusion-like training.

## 1. Implement the Noise Process

Create a class under `src/noise/` and register it in `NOISE_REGISTRY`.

```python
import torch

from core.types import NoisyBatch
from noise.registry import NOISE_REGISTRY


@NOISE_REGISTRY.register("toy_noise")
class ToyNoise:
    def __init__(self, scheduler=None) -> None:
        self.scheduler = scheduler

    def __call__(self, clean: torch.Tensor, device: torch.device) -> NoisyBatch:
        noise = torch.randn_like(clean, device=device)
        noisy = clean + 0.1 * noise
        timesteps = torch.zeros(clean.shape[0], device=device, dtype=torch.long)
        return NoisyBatch(noisy=noisy, target=noise, timesteps=timesteps)
```

## 2. Select It from a Trainer

Wire the corresponding trainer class to use it via `noise_key`, or create a new
trainer family if the target semantics differ.

## 3. Add Config Coverage

```json
{
  "model": {
    "model_type": "diffusion"
  },
  "training": {
    "epochs": 1,
    "batch_size": 4
  }
}
```

Trainer-side selection happens through the trainer implementation, not directly
from the JSON unless that trainer exposes the key.

## 4. Add Tests

- registry presence
- output shape contract
- scheduler passthrough if required
- trainer smoke test if the noise is trainable in practice
