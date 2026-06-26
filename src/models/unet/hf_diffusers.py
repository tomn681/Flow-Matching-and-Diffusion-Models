"""
Thin adapter around Hugging Face diffusers UNet2DModel.

This exists for strict legacy 2D parity paths where architectural similarity is
not enough and we want the actual diffusers implementation used by the old
LDCT codebase.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Sequence

import torch
import torch.nn as nn

from ..registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("hf_diffusers_unet")
class HFDiffusersUNet2DAdapter(nn.Module):
    """
    Adapter exposing diffusers.UNet2DModel through the framework UNet contract.

    Scope:
    - 2D only
    - no ControlNet residual injection
    - no attention-context conditioning
    - intended for legacy pixel-space DDPM/FM/RF/Reflow runs with concatenate conditioning
    """

    def __init__(
        self,
        *,
        spatial_dims: int = 2,
        sample_size: int | Sequence[int] | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Sequence[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: str = "UNetMidBlock2D",
        up_block_types: Sequence[str] = (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Sequence[int] = (128, 128, 256, 256, 512, 512),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        dropout: float = 0.0,
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: str | None = None,
        num_class_embeds: int | None = None,
        cross_attention_dim: int | None = None,
        time_cond_proj_dim: int | None = None,
        addition_embed_type: str | None = None,
        addition_time_embed_dim: int | None = None,
        mid_block_only_cross_attention: bool = False,
        transformer_layers_per_block: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        if int(spatial_dims) != 2:
            raise ValueError("hf_diffusers_unet only supports spatial_dims=2.")
        unsupported_features: list[str] = []
        if cross_attention_dim is not None:
            unsupported_features.append("cross_attention_dim")
        if time_cond_proj_dim is not None:
            unsupported_features.append("time_cond_proj_dim")
        if addition_embed_type is not None:
            unsupported_features.append("addition_embed_type")
        if addition_time_embed_dim is not None:
            unsupported_features.append("addition_time_embed_dim")
        if mid_block_only_cross_attention:
            unsupported_features.append("mid_block_only_cross_attention")
        if int(transformer_layers_per_block) != 1:
            unsupported_features.append("transformer_layers_per_block")
        if unsupported_features:
            unsupported = ", ".join(sorted(unsupported_features))
            raise TypeError(f"Unsupported hf_diffusers_unet features requested: {unsupported}")
        if kwargs:
            # Keep strict surface control so legacy parity paths fail loudly if
            # someone tries to use advanced features the adapter does not support.
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported hf_diffusers_unet kwargs: {unsupported}")

        try:
            from diffusers import UNet2DModel
        except Exception as exc:  # pragma: no cover - import surface only
            raise ImportError(
                "hf_diffusers_unet requires the `diffusers` package in the active environment."
            ) from exc

        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            center_input_sample=bool(center_input_sample),
            time_embedding_type=str(time_embedding_type),
            freq_shift=int(freq_shift),
            flip_sin_to_cos=bool(flip_sin_to_cos),
            down_block_types=tuple(down_block_types),
            mid_block_type=str(mid_block_type),
            up_block_types=tuple(up_block_types),
            block_out_channels=tuple(int(v) for v in block_out_channels),
            layers_per_block=int(layers_per_block),
            downsample_padding=int(downsample_padding),
            dropout=float(dropout),
            attention_head_dim=int(attention_head_dim),
            norm_num_groups=int(norm_num_groups),
            norm_eps=float(norm_eps),
            resnet_time_scale_shift=str(resnet_time_scale_shift),
            add_attention=bool(add_attention),
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds,
        )
        self.config = self.model.config

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float | int,
        context: torch.Tensor | None = None,
        context_ca: torch.Tensor | None = None,
        controlnet_residuals: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if context is not None:
            x = torch.cat([x, context], dim=1)
        if context_ca is not None:
            raise ValueError("hf_diffusers_unet does not support context_ca cross-attention.")
        if controlnet_residuals is not None:
            raise ValueError("hf_diffusers_unet does not support controlnet_residuals.")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported hf_diffusers_unet forward kwargs: {unsupported}")

        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device, dtype=torch.long)
        if t.ndim == 0:
            t = t[None]
        t = t.expand(x.shape[0]).to(x.device)
        return self.model(x, t, return_dict=False)[0]

    # Delegate checkpoint IO directly to the wrapped diffusers model so key names
    # stay compatible with old checkpoints that were saved from UNet2DModel.
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, nn.Parameter]]:
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
        return self.model.named_buffers(prefix=prefix, recurse=recurse)
