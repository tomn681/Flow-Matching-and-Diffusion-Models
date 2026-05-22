import torch

from losses.base import BaseLossComponent, LossAssembler


class ConstantLoss(BaseLossComponent):
    name = "const"

    def __init__(self, value: float, weight: float = 1.0) -> None:
        super().__init__(weight=weight)
        self.value = value

    def compute(self, *, context: dict) -> torch.Tensor:
        return torch.tensor(self.value, device=context["device"], dtype=context["dtype"])


class DisabledLoss(ConstantLoss):
    name = "off"

    def is_active(self, epoch: int, global_step: int) -> bool:
        return False


def test_loss_assembler_aggregates_weighted_components() -> None:
    pred = torch.zeros(1)
    assembler = LossAssembler([
        ConstantLoss(value=2.0, weight=0.5),
        ConstantLoss(value=3.0, weight=2.0),
    ])

    total, parts = assembler(context={"device": pred.device, "dtype": pred.dtype}, epoch=1, global_step=5)

    assert torch.isclose(total, torch.tensor(7.0))
    assert set(parts.keys()) == {"const"}
    assert torch.isclose(parts["const"], torch.tensor(7.0))


def test_loss_assembler_skips_inactive_components() -> None:
    pred = torch.zeros(1)
    assembler = LossAssembler([DisabledLoss(value=5.0, weight=1.0)])

    total, parts = assembler(context={"device": pred.device, "dtype": pred.dtype}, epoch=0, global_step=0)

    assert torch.isclose(total, torch.tensor(0.0))
    assert parts == {}
