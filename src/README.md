# `src` Package Overview

The `src` directory contains the reusable Python package. Importing `src` exposes four subpackages:

- `src.nn` – Low-level neural network building blocks that are dimension-agnostic (1D/2D/3D).
- `src.models` – High-level model assemblies constructed from the blocks (AutoencoderKL, VQVAE/MagvitVQVAE, EfficientUNetND, Diffusers-style UNets).
- `src.utils` – Dataset loaders, config IO, checkpoint helpers, and distributed utilities (LDCT loaders plus MNIST and config-driven builders).
- `src.pipelines` – Executable training / evaluation scripts (VAE, flow matching, diffusion) plus shared pipeline utilities for schedulers/samplers. Dispatch via `python train.py --config <json>` or programmatically (`from pipelines.train import train_*`).

Each subfolder ships with its own README describing the available utilities and how they interoperate. Refer to them when extending the codebase (e.g., adding new trainers, schedulers, or dataset backends).

## FileTree (02/04/2026)

```
├── 📁 models
│   ├── 📁 generators
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 diffusionfactory.py
│   │   └── 🐍 vaefactory.py
│   ├── 📁 unet
│   │   ├── 📝 README.md
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 unet.py
│   ├── 📁 vae
│   │   ├── 📝 README.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 base.py
│   │   ├── 🐍 kl.py
│   │   ├── 🐍 magvit.py
│   │   └── 🐍 vq.py
│   ├── 📝 README.md
│   └── 🐍 __init__.py
├── 📁 nn
│   ├── 📁 blocks
│   │   ├── 📝 README.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 attention.py
│   │   ├── 🐍 residual.py
│   │   └── 🐍 timestep.py
│   ├── 📁 losses
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 vae.py
│   ├── 📁 modules
│   │   ├── 📁 vae
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 codebook.py
│   │   │   ├── 🐍 decoder.py
│   │   │   ├── 🐍 discriminators.py
│   │   │   ├── 🐍 encoder.py
│   │   │   └── 🐍 reparameterizer.py
│   │   └── 🐍 __init__.py
│   ├── 📁 ops
│   │   ├── 📝 README.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 convolution.py
│   │   ├── 🐍 normalization.py
│   │   ├── 🐍 pooling.py
│   │   ├── 🐍 time_embedding.py
│   │   └── 🐍 upsampling.py
│   ├── 📝 README.md
│   └── 🐍 __init__.py
├── 📁 datasets
│   ├── 📝 README.md
│   ├── 🐍 __init__.py
│   ├── 🐍 base.py
│   ├── 🐍 ldct.py
│   └── 🐍 mnist.py
├── 📁 pipelines
│   ├── 📁 samplers
│   │   ├── 📁 handlers
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 base.py
│   │   │   ├── 🐍 diffusion_handler.py
│   │   │   ├── 🐍 flow_matching_handler.py
│   │   │   └── 🐍 vae_handler.py
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 diffusion.py
│   │   ├── 🐍 flow_matching.py
│   │   └── 🐍 vae.py
│   ├── 📁 train
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 diffusion_lib.py
│   │   ├── 🐍 flow_matching_lib.py
│   │   └── 🐍 vae_lib.py
│   ├── 📝 README.md
│   ├── 🐍 __init__.py
│   └── 🐍 utils.py
├── 📁 utils
│   ├── 📝 README.md
│   ├── 🐍 __init__.py
│   ├── 🐍 dataset_utils.py
│   ├── 🐍 evaluation_utils.py
│   ├── 📁 model_utils
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 diffusion_utils.py
│   │   └── 🐍 vae_utils.py
│   ├── 🐍 sampling_utils.py
│   ├── 🐍 training_utils.py
│   └── 🐍 utils.py
├── 📝 README.md
├── 🐍 __init__.py
├── 🐍 run_model.py
└── 🐍 train.py
```
