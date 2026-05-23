from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from sampling import LatentDiffusionSampler


class _DummyDataset:
    def __init__(self) -> None:
        self.base_path = Path("/")
        self.target_key = "target"
        self.conditioning_key = "image"
        self.data = [{"target": "a.pt", "image": "b.pt"}]


class _DummyVAE(nn.Module):
    def decode(self, z: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        return z

    def raw_output_to_image(self, x: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
        return x


def test_latent_sampler_decode_smoke_presaved_latents(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "training": {"conditioning": "latent_attention", "recon_type": "l1"},
        "model": {
            "model_type": "latent_diffusion",
            "conditioning": "latent_attention",
            "use_presaved_latents": True,
            "vae": {"latent_type": "kl"},
            "vae_checkpoint": str(tmp_path / "vae.pt"),
        },
    }
    captured = {}

    def _fake_decode_diffusion_batch(
        model, training_cfg, model_cfg, device, batch_shape, conditioning_batch, **kwargs
    ):
        captured["shape"] = batch_shape
        captured["cond_shape"] = None if conditioning_batch is None else tuple(conditioning_batch.shape)
        return torch.zeros(batch_shape, device=device)

    monkeypatch.setattr("sampling.latent_sampler.load_run_config", lambda _p: cfg)
    monkeypatch.setattr("sampling.latent_sampler.resolve_checkpoint", lambda _d, _m: tmp_path / "latent_diff_last.pt")
    monkeypatch.setattr("sampling.latent_sampler.build_sampling_dataset", lambda *_args, **_kwargs: _DummyDataset())
    monkeypatch.setattr(
        "sampling.latent_sampler.progress_batches",
        lambda _dataset, _batch_size, _desc, indices=None: [
            (
                [0],
                [{"target": torch.zeros(1, 8, 8), "image": torch.zeros(1, 8, 8)}],
            )
        ],
    )
    monkeypatch.setattr("sampling.latent_sampler.resolve_sample_indices", lambda _dataset, _n, seed=42: [0])
    monkeypatch.setattr("sampling.latent_sampler.resolve_output_root", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("sampling.latent_sampler.build_diffusion_model", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr("sampling.latent_sampler.decode_diffusion_batch", _fake_decode_diffusion_batch)
    monkeypatch.setattr("sampling.latent_sampler.LatentSampler._load_frozen_vae", lambda *_args, **_kwargs: _DummyVAE())

    sampler = LatentDiffusionSampler(ckpt_dir=tmp_path, save=False, batch_size=1, device="cpu")
    sampler.decode()

    assert captured["shape"] == (1, 1, 8, 8)
    assert captured["cond_shape"] == (1, 1, 8, 8)

