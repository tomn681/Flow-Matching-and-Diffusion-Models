# `src` Package Overview

The `src` directory contains the reusable Python package. Importing `src` exposes four subpackages:

- `src.nn` – Low-level neural network building blocks that are dimension-agnostic (1D/2D/3D).
- `src.models` – High-level model assemblies constructed from the blocks (AutoencoderKL, VQVAE/MagvitVQVAE, EfficientUNetND, Diffusers-style UNets).
- `src.utils` – Dataset loaders, config IO, checkpoint helpers, and distributed utilities (LDCT loaders plus MNIST and config-driven builders).
- `src.pipelines` – Executable training / evaluation scripts (VAE, flow matching, diffusion) plus shared pipeline utilities for schedulers/samplers. Dispatch via `python train.py --config <json>` or programmatically (`from pipelines.train import train_*`).

Each subfolder ships with its own README describing the available utilities and how they interoperate. Refer to them when extending the codebase (e.g., adding new trainers, schedulers, or dataset backends).
