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

## Training Config Keys

- `training.multi_resolution`: optional progressive schedule list:
  - `start_epoch`: stage start (first stage must be `0`)
  - `resolution`: target side length for that stage
