# Registries API

This page documents the registry backbone and the concrete public registries
exposed by the library.

## Registry Backbone

::: src.core.registry

## Public Registries

These registries are importable from the public API:

- `MODEL_REGISTRY`
- `TRAINER_REGISTRY`
- `SAMPLER_REGISTRY`
- `NOISE_REGISTRY`
- `LOSS_REGISTRY`
- `CONDITIONING_ADAPTER_REGISTRY`
- `LR_SCHEDULER_REGISTRY`
- `TEXT_ENCODER_REGISTRY`

## Typical Usage

```python
from src.models import MODEL_REGISTRY

model_cls = MODEL_REGISTRY.get("kl_vae")
```

```python
from src.training import TRAINER_REGISTRY

trainer_cls = TRAINER_REGISTRY.get("diffusion")
```

```python
from src.sampling import SAMPLER_REGISTRY

sampler_cls = SAMPLER_REGISTRY.get("vae")
```

## Module References

::: src.models.registry

::: src.training.registry

::: src.sampling.registry

::: src.noise.registry

::: src.losses.registry

::: src.scheduling.lr

::: src.scheduling.conditioning

::: src.models.adapters.text_encoders
