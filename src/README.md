# `src` Package Overview

The `src` directory contains the reusable Python package. Importing `src` exposes four subpackages:

- `src.nn` – Low-level neural network building blocks that are dimension-agnostic (1D/2D/3D).
- `src.models` – High-level model assemblies constructed from the blocks (AutoencoderKL, VQVAE/MagvitVQVAE, EfficientUNetND).
- `src.utils` – Dataset loaders and helper utilities (LDCT loaders plus MNIST and config-driven builders).
- `src.pipelines` – Executable training / evaluation scripts that orchestrate datasets and models (dispatched via `python -m src.train` / `python -m src.sample`).

Each subfolder ships with its own README describing the available utilities and how they interoperate.
