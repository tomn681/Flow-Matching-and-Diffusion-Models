from __future__ import annotations

import torch.nn as nn


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

