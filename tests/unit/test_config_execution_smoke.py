from __future__ import annotations

import copy
import json
from pathlib import Path
import types

import torch
import torch.nn as nn

from core.types import ModelOutput
from sampling import SAMPLER_REGISTRY
from training import TRAINER_REGISTRY


class _DummyPosterior:
    def __init__(self, z: torch.Tensor) -> None:
        self._z = z

    def kl(self) -> torch.Tensor:
        return torch.zeros(self._z.size(0), device=self._z.device)

    def mode(self) -> torch.Tensor:
        return self._z


class _DummyVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.scaling_factor = 1.0
        self.input_range = "minus_one_to_one"

    def image_to_model_range(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def raw_output_to_image(self, x: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
        return x.clamp(0.0, 1.0)

    def encode(self, x: torch.Tensor, normalize: bool = False):
        _ = normalize
        return _DummyPosterior(x * self.weight)

    def decode(self, z: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        _ = denorm
        return z * self.weight

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> ModelOutput:
        _ = sample_posterior
        rec = x * self.weight
        return ModelOutput(
            reconstruction=rec,
            posterior=_DummyPosterior(rec),
            codebook_loss=torch.tensor(0.0, device=x.device),
        )

    def make_discriminator(self) -> nn.Module:
        return nn.Sequential(nn.Conv2d(1, 1, kernel_size=1))


class _DummyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor, context_ca=None, **kwargs) -> torch.Tensor:
        _ = t, context_ca, kwargs
        return x * self.weight


class _DummyNoiseBatch:
    def __init__(self, clean: torch.Tensor) -> None:
        self.noisy = clean
        self.target = torch.zeros_like(clean)
        self.timesteps = torch.zeros(clean.size(0), device=clean.device, dtype=torch.long)


class _DummyNoiseProcess:
    def __call__(self, clean: torch.Tensor, device: torch.device):
        _ = device
        return _DummyNoiseBatch(clean)


class _TinyDataset:
    target_key = "target"
    conditioning_key = "image"

    def __init__(self) -> None:
        self.data = [{"id": i} for i in range(2)]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict:
        _ = idx
        x = torch.zeros(1, 8, 8)
        return {"target": x, "image": x, "text": "stub"}


class _FakeInferencePipe:
    def generate(self, inputs) -> torch.Tensor:
        return torch.zeros(inputs.sample_shape)


def _config_paths() -> list[Path]:
    return sorted(Path("configs").rglob("*.json"))


def _runtime_cfg(raw: dict, tmp_path: Path, name: str) -> dict:
    cfg = copy.deepcopy(raw)
    training = cfg.setdefault("training", {})
    training["epochs"] = 1
    training["batch_size"] = 2
    training["num_workers"] = 0
    training["manual_device"] = "cpu"
    training["use_amp"] = False
    training["output_dir"] = str(tmp_path / f"{name}_out")
    training["save_images"] = False
    training["save_images_every"] = 1
    training.setdefault("recon_type", "l1")
    training["disc_lr"] = training.get("disc_lr") or training.get("learning_rate", 1e-3)
    training["gan_weight"] = 0.0
    training["perceptual_weight"] = 0.0
    training["perceptual_use_lpips"] = False
    model = cfg.setdefault("model", {})
    if model.get("model_type") == "vae":
        model.setdefault("latent_type", "kl")
    return cfg


def test_all_shipped_configs_and_templates_execute_one_trainer_and_sampler_step(monkeypatch, tmp_path: Path) -> None:
    import pipelines.samplers.autoencoder_like as autoencoder_like
    import pipelines.samplers.diffusion_like as diffusion_like
    import sampling.latent_sampler as latent_sampler_mod
    import training.latent_trainer as latent_trainer_mod

    dataset = _TinyDataset()

    monkeypatch.setattr(latent_trainer_mod.LatentTrainerMixin, "_load_frozen_vae", lambda self: _DummyVAE().to(self.device))

    monkeypatch.setattr(autoencoder_like, "load_run_config", lambda ckpt_dir: json.loads((Path(ckpt_dir) / "train_config.json").read_text()))
    monkeypatch.setattr(autoencoder_like, "resolve_checkpoint", lambda ckpt_dir, model_type: Path(ckpt_dir) / f"{model_type}_last.pt")
    monkeypatch.setattr(autoencoder_like, "build_sampling_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(autoencoder_like, "resolve_sample_indices", lambda dataset, num_samples, seed: [0])
    monkeypatch.setattr(autoencoder_like, "progress_batches", lambda dataset, batch_size, desc, indices=None: [([0], [dataset[0]])])
    monkeypatch.setattr(autoencoder_like, "resolve_output_root", lambda *args, **kwargs: None)
    monkeypatch.setattr(autoencoder_like, "build_vae_model", lambda *args, **kwargs: _DummyVAE())

    monkeypatch.setattr(diffusion_like, "load_run_config", lambda ckpt_dir: json.loads((Path(ckpt_dir) / "train_config.json").read_text()))
    monkeypatch.setattr(diffusion_like, "resolve_checkpoint", lambda ckpt_dir, model_type: Path(ckpt_dir) / f"{model_type}_last.pt")
    monkeypatch.setattr(diffusion_like, "build_sampling_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(diffusion_like, "resolve_sample_indices", lambda dataset, num_samples, seed: [0])
    monkeypatch.setattr(diffusion_like, "progress_batches", lambda dataset, batch_size, desc, indices=None: [([0], [dataset[0]])])
    monkeypatch.setattr(diffusion_like, "resolve_output_root", lambda *args, **kwargs: None)
    monkeypatch.setattr(diffusion_like, "build_diffusion_model", lambda *args, **kwargs: _DummyUNet())
    monkeypatch.setattr(diffusion_like, "_build_inference_pipeline", lambda **kwargs: (_FakeInferencePipe(), 2))

    monkeypatch.setattr(latent_sampler_mod, "progress_batches", lambda dataset, batch_size, desc, indices=None: [([0], [dataset[0]])])
    monkeypatch.setattr(latent_sampler_mod, "resolve_output_root", lambda *args, **kwargs: None)

    raw_configs: list[tuple[str, dict]] = []
    for path in _config_paths():
        raw_configs.append((path.stem, json.loads(path.read_text())))

    from configs import from_template

    for template_name in ("sd15_vae", "sd15_latent_ddpm", "fmboost_latent_fm", "pixel_ddpm_1d", "vqgan_magvit"):
        raw_configs.append((f"template_{template_name}", from_template(template_name)))

    for name, raw in raw_configs:
        cfg = _runtime_cfg(raw, tmp_path, name)
        model_type = str(cfg.get("model", {}).get("model_type", "")).lower()
        trainer_cls = TRAINER_REGISTRY.get(model_type)
        if model_type == "vae":
            trainer = trainer_cls(cfg, model_override=_DummyVAE())
        elif model_type.startswith("latent_"):
            trainer = trainer_cls(cfg, model_override=_DummyUNet(), noise_override=_DummyNoiseProcess())
        else:
            trainer = trainer_cls(cfg, model_override=_DummyUNet(), noise_override=_DummyNoiseProcess())

        trainer._setup(dataset, val_dataset=dataset, resume=None)
        batch = next(iter(trainer.train_loader))
        metrics = trainer._training_step(batch, epoch=1)
        assert "loss" in metrics, name

        ckpt_dir = tmp_path / f"{name}_runtime"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "train_config.json").write_text(json.dumps(cfg), encoding="utf-8")

        sampler_cls = SAMPLER_REGISTRY.get(model_type)
        sampler = sampler_cls(ckpt_dir=ckpt_dir, batch_size=1, num_samples=1, device="cpu")
        if model_type.startswith("latent_"):
            def _fake_latent_resolve_runtime(self, *, evaluate: bool = False):
                cfg = json.loads((self.ckpt_dir / "train_config.json").read_text())
                training_cfg = cfg["training"]
                model_cfg = cfg["model"]
                conditioning_mode = model_cfg.get("conditioning") or training_cfg.get("conditioning")
                return (
                    self.ckpt_dir,
                    cfg,
                    training_cfg,
                    model_cfg,
                    torch.device("cpu"),
                    dataset,
                    [0],
                    _DummyUNet(),
                    _DummyVAE(),
                    conditioning_mode,
                    conditioning_mode,
                    bool(model_cfg.get("use_presaved_latents", False)),
                )

            sampler._resolve_runtime = types.MethodType(_fake_latent_resolve_runtime, sampler)
            sampler._predict_latent_batch = types.MethodType(lambda self, **kwargs: kwargs["target_latent"], sampler)
        sampler.sample()
