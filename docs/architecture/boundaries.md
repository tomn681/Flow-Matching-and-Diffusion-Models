# Package Boundaries

This page defines the intended ownership lines between the packages that still
look similar from the outside but are not interchangeable.

## Runtime Inference

- `src.sampling`: the canonical runtime sampler layer.
  - Owns sampler classes, `SAMPLER_REGISTRY`, checkpoint-driven runtime wiring,
    and the public runtime surface used by `run_model.py`.
- `src.pipelines.inference`: higher-level facade layer.
  - Owns orchestration helpers such as `InferencePipeline` and
    `TextToImagePipeline`.

## Compatibility Sampling Layer

- `src.pipelines.samplers`: compatibility and delegation layer.
  - Keeps the older handler/abstract/concrete structure available for legacy
    imports and internal delegation.
  - Should not grow new runtime ownership; new sampler features belong in
    `src.sampling` first.
- `src.pipelines.utils`: backward-compatible re-export surface for scheduling
  and sampling helpers.

## Loss Boundaries

- `src.losses`: composable training loss components and registries.
  - Owns `LOSS_REGISTRY`, `LossAssembler`, and trainer-facing objects such as
    `GANGeneratorLoss`, `KLLoss`, and `PerceptualLossComponent`.
- `src.nn.losses`: tensor-level math and small discriminators.
  - Owns raw functions like `generator_hinge_loss(...)`,
    `gradient_penalty(...)`, and `PatchDiscriminator`.

Rule of thumb:
- if a trainer activates, schedules, or registers a loss, it belongs under
  `src.losses`
- if the code is pure tensor math or a small NN primitive, it belongs under
  `src.nn.losses`

## Utility Boundaries

- `src.utils.dataset_utils`: dataset config resolution and dataset construction.
- `src.utils.dataset_runtime`: dataset cache-path generation, output writing,
  and simple batch iteration.
- `src.utils.training_utils`: config IO, checkpoint helpers, and generic
  training-runtime utilities.
- `src.utils.model_utils`: model-family-specific helpers for VAE/diffusion
  build and encode/decode behavior.
- `src.utils.utils`: compatibility shim only.

## Training Boundaries

- `src.training`: canonical trainer implementation layer.
- `src.compat` and `src.pipelines.train.*_lib`: compatibility wrappers only.
  - They may remain callable for a transition period.
  - They should not contain the only copy of a training algorithm.

## What Phase 5 Does Not Do

- no repo-wide package move
- no deletion of root CLIs
- no merge of `src.sampling` and `src.pipelines.samplers`

The goal is clarity of ownership first, then small extractions where that
clarity removes real ambiguity.
