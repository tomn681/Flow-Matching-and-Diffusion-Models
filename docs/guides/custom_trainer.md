# Custom Trainer Guide

Trainers are registered orchestration units. They own model construction,
optimizer setup, per-batch logic, and checkpoint state.

## 1. Implement a Trainer

```python
from training.base import BaseTrainer
from training.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("toy_trainer")
class ToyTrainer(BaseTrainer):
    def _build_model(self):
        ...

    def _training_step(self, batch: dict, *, epoch: int) -> dict[str, float]:
        ...
```

## 2. Required Hooks

- `_build_model()`
- `_training_step(...)`

Optional hooks:

- `_validation_step(...)`
- `_build_optimizer()`
- `_build_lr_scheduler()`
- `_build_default_callbacks()`
- `_resume_from_payload(...)`

## 3. Keep the Base Contracts

Do not reimplement:

- checkpoint writing
- event bus handling
- dataloader construction
- AMP scaler creation
- EMA orchestration

Those belong to `BaseTrainer`.

## 4. Register by `model_type`

`train.py` dispatches using `config["model"]["model_type"]`.

If you register:

```python
@TRAINER_REGISTRY.register("toy_trainer")
```

then your config must contain:

```json
{
  "model": {
    "model_type": "toy_trainer"
  }
}
```

## 5. Add Tests

- registry presence
- one-step smoke fit
- checkpoint save/load if custom state exists
- callback/metric behavior if the trainer adds special metrics
