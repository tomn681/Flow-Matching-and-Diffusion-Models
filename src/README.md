# `src` Package Overview

The `src` package is the canonical implementation surface for training, sampling, configuration, and extension.

## Main Areas

- `core/`: registry primitives, protocols, shared types, plugin discovery.
- `configs/`: config schema/models and migration helpers.
- `datasets/`: dataset implementations and dataset-level cache behavior.
- `models/`: model assemblies and factory/build strategies.
- `nn/`: reusable building blocks, ops, and neural modules.
- `noise/`: noise-process implementations (including reflow pair generation utilities).
- `scheduling/`: scheduler builder/registry, conditioning adapters, sampling loop.
- `training/`: trainer implementations, callbacks, builder, EMA, event bus.
- `sampling/`: runtime sampler classes and registry dispatch.
- `pipelines/`: high-level inference/training workflow glue.
- `utils/`: config, dataset, sampling, IO, and evaluation helpers.

## Entry Points

- Training dispatcher: `train.py` (root) and `python -m genlib train ...`
- Runtime sampler dispatcher: `run_model.py` (root) and `python -m genlib sample ...`

## Notes

- Public API exports are curated via `src/__init__.py`.
- Legacy compatibility aliases are preserved for root-level imports used by existing scripts.
- Per-submodule details live in local README/docs pages (for example `src/pipelines/README.md`).

