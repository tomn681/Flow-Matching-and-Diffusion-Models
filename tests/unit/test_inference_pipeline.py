from __future__ import annotations

import torch

from models.autoencoder.base import BaseAutoencoder
from pipelines import InferenceInputs, InferencePipeline


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
    def encode(self, x: torch.Tensor, normalize: bool = False):
        _ = normalize
        return x

    def decode(self, z: torch.Tensor, denorm: bool = False):
        _ = denorm
        return z

    def raw_output_to_image(self, x: torch.Tensor, recon_type: str = "l1") -> torch.Tensor:
        _ = recon_type
        return x + 1.0


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
