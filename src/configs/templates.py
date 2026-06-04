from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


_TEMPLATES: dict[str, dict[str, Any]] = {
    "sd15_vae": {
        "training": {
            "epochs": 100,
            "batch_size": 12,
            "num_workers": 4,
            "learning_rate": 4.5e-06,
            "weight_decay": 0.0,
            "save_every": 10,
            "output_dir": "checkpoints/sd15_vae",
            "seed": 42,
            "img_size": 256,
            "slice_count": 1,
            "use_amp": False,
        },
        "model": {
            "model_type": "vae",
            "latent_type": "kl",
            "in_channels": 3,
            "out_channels": 3,
            "resolution": 256,
            "base_ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "z_channels": 4,
            "embed_dim": 4,
            "attention_impl": "compvis",
        },
    },
    "sd15_latent_ddpm": {
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "num_workers": 4,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "save_every": 10,
            "output_dir": "checkpoints/sd15_latent_ddpm",
            "seed": 42,
            "scheduler": "ddpm",
            "num_train_timesteps": 1000,
            "num_inference_steps": 1000,
            "conditioning": "attention",
            "latent_norm": "standardize",
        },
        "model": {
            "model_type": "latent_diffusion",
            "vae_checkpoint": "checkpoints/sd15_vae/vae_best.pt",
            "use_presaved_latents": False,
            "latent_cache_dir": "data/latents/sd15",
            "conditioning": "attention",
            "unet": {
                "unet_impl": "condition_nd",
                "sample_size": 32,
                "in_channels": 4,
                "out_channels": 4,
                "layers_per_block": 2,
                "block_out_channels": [320, 640, 1280, 1280],
                "cross_attention_dim": 768,
            },
            "scheduler": {
                "name": "ddpm",
                "num_train_timesteps": 1000,
                "num_inference_steps": 1000,
                "params": {"beta_start": 0.00085, "beta_end": 0.012},
            },
        },
    },
    "fmboost_latent_fm": {
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "num_workers": 4,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "save_every": 10,
            "output_dir": "checkpoints/fmboost_latent_fm",
            "seed": 42,
            "scheduler": "flow_match_euler",
            "num_train_timesteps": 1000,
            "num_inference_steps": 1000,
            "conditioning": "attention",
            "latent_norm": "standardize",
        },
        "model": {
            "model_type": "latent_flow_matching",
            "vae_checkpoint": "checkpoints/fmboost_vae/vae_best.pt",
            "use_presaved_latents": True,
            "latent_cache_dir": "data/latents/fmboost",
            "conditioning": "attention",
            "unet": {
                "unet_impl": "condition_nd",
                "sample_size": 32,
                "in_channels": 4,
                "out_channels": 4,
                "layers_per_block": 2,
                "block_out_channels": [320, 640, 1280, 1280],
                "cross_attention_dim": 768,
            },
            "scheduler": {
                "name": "flow_match_euler",
                "num_train_timesteps": 1000,
                "num_inference_steps": 1000,
                "params": {},
            },
        },
    },
    "pixel_ddpm_1d": {
        "training": {
            "epochs": 100,
            "batch_size": 16,
            "num_workers": 4,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "save_every": 10,
            "output_dir": "checkpoints/pixel_ddpm_1d",
            "seed": 42,
            "scheduler": "ddpm",
            "num_train_timesteps": 1000,
            "num_inference_steps": 1000,
            "conditioning": "none",
        },
        "model": {
            "model_type": "diffusion",
            "unet": {
                "unet_impl": "efficient_nd",
                "spatial_dims": 1,
                "in_channels": 1,
                "out_channels": 1,
                "model_channels": 128,
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 4, 4],
            },
            "scheduler": {
                "name": "ddpm",
                "num_train_timesteps": 1000,
                "num_inference_steps": 1000,
                "params": {"beta_start": 0.0001, "beta_end": 0.02},
            },
        },
    },
    "vqgan_magvit": {
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "num_workers": 4,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "save_every": 10,
            "output_dir": "checkpoints/vqgan_magvit",
            "seed": 42,
            "recon_type": "l1",
            "perceptual_weight": 1.0,
            "gan_weight": 0.5,
        },
        "model": {
            "model_type": "vae",
            "latent_type": "vq",
            "in_channels": 1,
            "out_channels": 1,
            "resolution": 256,
            "base_ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "z_channels": 4,
            "embed_dim": 4,
            "codebook_size": 8192,
        },
    },
}


def from_template(name: str, **overrides: Any) -> dict[str, Any]:
    raw_key = str(name).strip()
    key = raw_key if raw_key in _TEMPLATES else None
    if key is None:
        lowered = raw_key.lower()
        matches = [k for k in _TEMPLATES.keys() if k.lower() == lowered]
        if len(matches) == 1:
            key = matches[0]
    if key is None:
        available = ", ".join(sorted(_TEMPLATES.keys()))
        raise KeyError(f"Unknown template '{name}'. Available: [{available}]")
    config = deepcopy(_TEMPLATES[key])
    if overrides:
        config = _deep_update(config, overrides)
    return config


__all__ = ["from_template"]
