from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from nn.blocks.residual import ResBlockND


class VAEBlockFactoryMixin:
    """Shared residual-block construction hooks for VAE encoders and decoders."""

    def _configure_resblock_factory(self, factory: Callable[..., nn.Module], **kwargs: Any) -> None:
        self._resblock_factory = factory
        self._resblock_kwargs = kwargs

    def _build_resblock(self, channels: int, out_channels: int) -> nn.Module:
        return self._resblock_factory(channels=channels, out_channels=out_channels, **self._resblock_kwargs)

    def _build_mid_resblock(self, channels: int) -> nn.Module:
        return ResBlockND(channels=channels, out_channels=channels, **self._resblock_kwargs)

    @staticmethod
    def _use_level_attention(_block_index: int, _num_blocks: int) -> bool:
        return True


class EncoderStage(nn.Module):
    def __init__(self, blocks: list[nn.Module], attns: list[nn.Module], down: nn.Module | None = None):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.attns = nn.ModuleList(attns)
        self.down = down


class DecoderStage(nn.Module):
    def __init__(self, blocks: list[nn.Module], attns: list[nn.Module], up: nn.Module | None = None):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.attns = nn.ModuleList(attns)
        self.up = up

