# `src` Package Overview

The `src` directory contains the reusable Python package. Importing `src` exposes four subpackages:

- `src.nn` – Low-level neural network building blocks that are dimension-agnostic (1D/2D/3D).
- `src.models` – High-level model assemblies constructed from the blocks (AutoencoderKL, EfficientUNetND).
- `src.utils` – Dataset loaders and helper utilities for LDCT experiments.
- `src.pipelines` – Executable training / evaluation scripts that orchestrate datasets and models (dispatched via `python -m src.train`).

Each subfolder ships with its own README describing the available utilities and how they interoperate.
