# Architecture Overview

## Design Principles

- Composition-first modules
- Explicit registries for pluggable components
- Config validation before runtime actions
- Backward compatibility through `src.compat`

## Runtime Layers

1. CLI layer (`genlib`, legacy wrappers)
2. Config and registry resolution
3. Model/trainer/sampler construction
4. Training loops or inference pipeline execution

## Key Packages

- `src.configs`: schema and validation
- `src.models`: model factories and families
- `src.training`: trainer abstractions and concrete trainers
- `src.sampling`: inference samplers
- `src.pipelines`: higher-level inference orchestration

## Ownership Rules

- `src.sampling` is the canonical runtime sampler layer.
- `src.pipelines.samplers` and `src.pipelines.utils` are compatibility /
  delegation surfaces, not the primary ownership layer for new runtime work.
- `src.losses` owns composable trainer-facing loss components and registries.
- `src.nn.losses` owns tensor-level loss math and small discriminator modules.
- `src.compat` owns migration shims and deprecation-preserving wrappers.

See [Package Boundaries](boundaries.md) for the policy contract and
[Ownership Matrix](ownership-matrix.md) for the concrete canonical/compatibility
file inventory used by the follow-up cleanup tracks.

## Training Config Keys

- `training.multi_resolution`: optional progressive schedule list:
  - `start_epoch`: stage start (first stage must be `0`)
  - `resolution`: target side length for that stage
