from __future__ import annotations

import torch

from losses.base import BaseLossComponent, LossAssembler


class _StubLoss(BaseLossComponent):
    def __init__(self, name: str, weight: float = 1.0):
        super().__init__(weight=weight)
        self.name = name

    def compute(self, *, context):
        return torch.tensor(0.0)


def test_metric_keys_returns_all_component_names() -> None:
    assembler = LossAssembler([_StubLoss("recon"), _StubLoss("kl"), _StubLoss("vq")])
    assert assembler.metric_keys() == ["recon", "kl", "vq"]


def test_metric_keys_deduplicates() -> None:
    assembler = LossAssembler([_StubLoss("recon"), _StubLoss("recon")])
    assert assembler.metric_keys() == ["recon"]


def test_metric_keys_stable_order() -> None:
    assembler = LossAssembler([_StubLoss("z"), _StubLoss("a"), _StubLoss("m")])
    assert assembler.metric_keys() == ["z", "a", "m"]


def test_metric_keys_empty_assembler() -> None:
    assembler = LossAssembler([])
    assert assembler.metric_keys() == []

