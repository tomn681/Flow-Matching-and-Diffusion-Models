from __future__ import annotations

import torch.nn as nn

from nn.modules import TemporalAttentionND
from nn.blocks.residual import ResBlockND

from ..registry import MODEL_REGISTRY
from .efficient import EfficientUNetND, TimestepEmbedSequential


@MODEL_REGISTRY.register("video_unet")
class VideoUNetND(EfficientUNetND):
    """Efficient UNet extended with temporal attention for video-like feature maps.

    Convolutions operate in 3D over `(T, H, W)`. Temporal attention is inserted
    after spatial mixing layers and attends over the `T` axis for inputs shaped
    as `(B, C, T, H, W)`.
    """

    def __init__(
        self,
        *args,
        temporal_num_heads: int | None = None,
        temporal_dropout: float = 0.0,
        temporal_after_resblocks: bool = True,
        temporal_after_attn: bool = True,
        **kwargs,
    ) -> None:
        spatial_dims = int(kwargs.get("spatial_dims", 2))
        if spatial_dims != 3:
            raise ValueError("VideoUNetND requires spatial_dims=3 for (T, H, W) feature maps.")
        super().__init__(*args, **kwargs)
        self.temporal_num_heads = int(temporal_num_heads or self.num_heads)
        self.temporal_dropout = float(temporal_dropout)
        self.temporal_after_resblocks = bool(temporal_after_resblocks)
        self.temporal_after_attn = bool(temporal_after_attn)

        self.input_blocks = nn.ModuleList([self._inject_temporal(block) for block in self.input_blocks])
        self.middle_block = self._inject_temporal(self.middle_block)
        self.output_blocks = nn.ModuleList([self._inject_temporal(block) for block in self.output_blocks])

    def _should_inject_after(self, layer: nn.Module) -> bool:
        if self.temporal_after_resblocks and isinstance(layer, ResBlockND):
            return True
        if self.temporal_after_attn:
            name = layer.__class__.__name__.lower()
            if "attention" in name and "temporal" not in name:
                return True
        return False

    def _infer_channels(self, layer: nn.Module) -> int | None:
        if isinstance(layer, ResBlockND):
            return int(layer.out_channels)
        for attr in ("dim", "channels", "out_channels"):
            value = getattr(layer, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)
        return None

    def _inject_temporal(self, block: TimestepEmbedSequential) -> TimestepEmbedSequential:
        layers: list[nn.Module] = []
        for layer in block:
            layers.append(layer)
            if not self._should_inject_after(layer):
                continue
            channels = self._infer_channels(layer)
            if channels is None:
                continue
            layers.append(
                TemporalAttentionND(
                    channels=channels,
                    num_heads=self.temporal_num_heads,
                    dropout=self.temporal_dropout,
                )
            )
        return TimestepEmbedSequential(*layers)


__all__ = ["VideoUNetND"]
