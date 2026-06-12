from __future__ import annotations

import inspect
from pathlib import Path

import torch

from models.autoencoder.base import BaseAutoencoder
from pipelines import InferenceInputs, InferencePipeline, TextToImageInputs, TextToImagePipeline


class _FakeScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.arange(4, -1, -1)

    def set_timesteps(self, n: int) -> None:
        self.timesteps = torch.arange(int(n) - 1, -1, -1)

    def step(self, pred: torch.Tensor, t, current: torch.Tensor):
        class _Step:
            def __init__(self, prev_sample: torch.Tensor):
                self.prev_sample = prev_sample

        return _Step(current - 0.1 * pred)


class _CaptureUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_context = None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context_ca: torch.Tensor | None = None,
        controlnet_residuals: dict | None = None,
    ) -> torch.Tensor:
        self.last_context = context_ca
        out = torch.zeros_like(x)
        if controlnet_residuals is not None:
            out = out + controlnet_residuals["mid_residual"].mean() * torch.ones_like(out)
        return out


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        b = len(texts)
        return torch.ones(b, 4, self.dim)


class _FakeControlNet(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        controlnet_cond: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        _ = t, encoder_hidden_states
        b, _, h, w = x.shape
        strength = controlnet_cond.mean().to(x.dtype)
        return {
            "down_residuals": [torch.zeros_like(x)],
            "mid_residual": torch.full((b, 1, h, w), float(strength), dtype=x.dtype, device=x.device),
        }


class _FakeVAE(BaseAutoencoder):
    def __init__(self) -> None:
        super().__init__()
        self.decode_calls = 0

    def encode(self, x: torch.Tensor, normalize: bool = False):
        _ = normalize
        return x

    def decode(self, z: torch.Tensor, denorm: bool = False):
        _ = denorm
        self.decode_calls += 1
        return z

    def raw_output_to_image(self, x: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
        _ = recon_type
        return x + 1.0


class _FakeTextAdapter:
    def __call__(self, model_input: torch.Tensor, cond, latent_norm=None):
        _ = model_input, latent_norm
        batch = len(cond) if isinstance(cond, list) else 1
        return torch.zeros(batch, 1), torch.ones(batch, 4, 8)


def test_inference_pipeline_uses_text_encoder_for_attention_conditioning() -> None:
    unet = _CaptureUNet()
    pipe = InferencePipeline(
        unet=unet,
        scheduler=_FakeScheduler(),
        device=torch.device("cpu"),
        text_encoder=_FakeTextEncoder(),
        conditioning_mode="attention",
    )
    out = pipe.generate(
        InferenceInputs(
            sample_shape=(2, 1, 8, 8),
            num_inference_steps=3,
            prompts=["a", "b"],
        )
    )
    assert out.shape == (2, 1, 8, 8)
    assert unet.last_context is not None
    assert tuple(unet.last_context.shape) == (2, 4, 8)


def test_inference_pipeline_injects_text_into_chain_conditioning() -> None:
    unet = _CaptureUNet()
    pipe = InferencePipeline(
        unet=unet,
        scheduler=_FakeScheduler(),
        device=torch.device("cpu"),
        text_encoder=_FakeTextEncoder(),
        conditioning_mode="chain",
    )
    out = pipe.generate(
        InferenceInputs(
            sample_shape=(2, 1, 8, 8),
            num_inference_steps=3,
            prompts=["a", "b"],
        )
    )
    assert out.shape == (2, 1, 8, 8)
    assert unet.last_context is not None
    assert tuple(unet.last_context.shape) == (2, 4, 8)


def test_inference_pipeline_controlnet_path_changes_output() -> None:
    scheduler = _FakeScheduler()
    unet = _CaptureUNet()
    pipe = InferencePipeline(
        unet=unet,
        scheduler=scheduler,
        device=torch.device("cpu"),
        controlnet=_FakeControlNet(),
        conditioning_mode="none",
    )
    seed = 123
    torch.manual_seed(seed)
    base = pipe.generate(InferenceInputs(sample_shape=(2, 1, 8, 8), num_inference_steps=3))
    torch.manual_seed(seed)
    controlled = pipe.generate(
        InferenceInputs(
            sample_shape=(2, 1, 8, 8),
            num_inference_steps=3,
            controlnet_cond=torch.ones(2, 3, 8, 8),
        )
    )
    assert base.shape == controlled.shape
    assert not torch.allclose(base, controlled)


def test_inference_pipeline_generate_images_decodes_with_vae() -> None:
    pipe = InferencePipeline(
        unet=_CaptureUNet(),
        scheduler=_FakeScheduler(),
        device=torch.device("cpu"),
        vae=_FakeVAE(),
        conditioning_mode="none",
    )
    seed = 77
    torch.manual_seed(seed)
    latents = pipe.generate(InferenceInputs(sample_shape=(1, 1, 4, 4), num_inference_steps=2))
    torch.manual_seed(seed)
    images = pipe.generate_images(InferenceInputs(sample_shape=(1, 1, 4, 4), num_inference_steps=2))
    assert images.shape == latents.shape
    assert torch.allclose(images, latents + 1.0)


def test_text_to_image_pipeline_generate_output_shape() -> None:
    pipe = TextToImagePipeline(
        model=_CaptureUNet(),
        vae=_FakeVAE(),
        text_adapter=_FakeTextAdapter(),
        scheduler=_FakeScheduler(),
        device=torch.device("cpu"),
        latent_channels=4,
        compression_factor=8,
        conditioning_mode="text",
    )
    images = pipe.generate(
        TextToImageInputs(
            prompts=["a", "b"],
            height=64,
            width=64,
            num_inference_steps=3,
            seed=123,
        )
    )
    assert images.shape == (2, 4, 64 // 8, 64 // 8)


def test_text_to_image_pipeline_validates_height_width_divisibility() -> None:
    pipe = TextToImagePipeline(
        model=_CaptureUNet(),
        vae=_FakeVAE(),
        text_adapter=_FakeTextAdapter(),
        scheduler=_FakeScheduler(),
        device=torch.device("cpu"),
        latent_channels=4,
        compression_factor=8,
        conditioning_mode="text",
    )
    try:
        pipe.generate(TextToImageInputs(prompts=["a"], height=100, width=64))
    except ValueError as exc:
        assert "compression factor 8" in str(exc)
    else:
        raise AssertionError("Expected invalid height/width to raise ValueError.")


def test_text_to_image_pipeline_passes_guidance_scale_to_sampling_loop(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_sample_with_scheduler(**kwargs):
        captured.update(kwargs)
        return torch.zeros(kwargs["sample_shape"])

    monkeypatch.setattr("pipelines.inference.sample_with_scheduler", _fake_sample_with_scheduler)
    pipe = TextToImagePipeline(
        model=_CaptureUNet(),
        vae=_FakeVAE(),
        text_adapter=_FakeTextAdapter(),
        scheduler=_FakeScheduler(),
        device=torch.device("cpu"),
        latent_channels=4,
        compression_factor=8,
        conditioning_mode="text",
    )
    pipe.generate(TextToImageInputs(prompts=["a"], height=64, width=64, guidance_scale=7.5))
    assert captured["guidance_scale"] == 7.5


def test_text_to_image_pipeline_seed_reproducibility() -> None:
    pipe = TextToImagePipeline(
        model=_CaptureUNet(),
        vae=_FakeVAE(),
        text_adapter=_FakeTextAdapter(),
        scheduler=_FakeScheduler(),
        device=torch.device("cpu"),
        latent_channels=4,
        compression_factor=8,
        conditioning_mode="text",
    )
    inputs = TextToImageInputs(prompts=["a", "b"], height=64, width=64, num_inference_steps=3, seed=123)
    a = pipe.generate(inputs)
    b = pipe.generate(inputs)
    assert torch.allclose(a, b)


def test_text_to_image_pipeline_calls_vae_decode_once() -> None:
    vae = _FakeVAE()
    pipe = TextToImagePipeline(
        model=_CaptureUNet(),
        vae=vae,
        text_adapter=_FakeTextAdapter(),
        scheduler=_FakeScheduler(),
        device=torch.device("cpu"),
        latent_channels=4,
        compression_factor=8,
        conditioning_mode="text",
    )
    pipe.generate(TextToImageInputs(prompts=["a"], height=64, width=64, num_inference_steps=2, seed=1))
    assert vae.decode_calls == 1


def test_text_to_image_pipeline_source_does_not_reference_latent_scale() -> None:
    source = inspect.getsource(TextToImagePipeline)
    assert "LATENT_SCALE" not in source
    assert "0.18215" not in source


def test_text_to_image_pipeline_from_checkpoint_reads_vae_checkpoint_key(monkeypatch, tmp_path: Path) -> None:
    vae_ckpt = tmp_path / "vae.pt"
    model_ckpt = tmp_path / "latent.pt"
    cfg = {
        "training": {
            "img_size": 64,
            "conditioning": "text",
            "text_encoder": {"kind": "clip", "model_name": "fake/clip"},
        },
        "model": {
            "model_type": "latent_diffusion",
            "vae_checkpoint": str(vae_ckpt),
            "vae": {
                "latent_type": "kl",
                "in_channels": 1,
                "out_channels": 1,
                "resolution": 64,
                "base_ch": 32,
                "ch_mult": [1, 2],
                "num_res_blocks": 1,
                "z_channels": 4,
                "embed_dim": 4,
            },
            "scheduler": {"name": "ddpm"},
        },
    }
    load_calls: list[str] = []

    monkeypatch.setattr("utils.sampling_utils.load_run_config", lambda _ckpt_dir: cfg)
    monkeypatch.setattr("utils.sampling_utils.resolve_checkpoint", lambda _ckpt_dir, _model_type: model_ckpt)
    monkeypatch.setattr(
        "utils.model_utils.diffusion_utils.build_diffusion_model",
        lambda _cfg, _device, ckpt_path=None, **_kwargs: _CaptureUNet(),
    )
    monkeypatch.setattr("pipelines.inference.build_text_conditioning_adapter", lambda **_kwargs: _FakeTextAdapter())
    monkeypatch.setattr("pipelines.inference.build_scheduler", lambda _spec, _training, **kwargs: (_FakeScheduler(), 10))

    def _fake_torch_load(path, *args, **kwargs):
        load_calls.append(str(path))
        return {"model": {}}

    monkeypatch.setattr("pipelines.inference.utils.safe_torch_load", _fake_torch_load)

    class _FactoryVAE(_FakeVAE):
        def load_state_dict(self, state_dict, strict: bool = True):
            return torch.nn.modules.module._IncompatibleKeys([], [])

    monkeypatch.setattr("pipelines.inference.ModelFactory.build", lambda _cfg: _FactoryVAE())
    pipe = TextToImagePipeline.from_checkpoint(tmp_path, device="cpu")
    assert isinstance(pipe, TextToImagePipeline)
    assert load_calls == [str(vae_ckpt)]
