import pytest

from core.registry import Registry
from losses.base import BaseLossComponent
from losses.registry import LOSS_REGISTRY
from noise import BaseNoiseProcess, NOISE_REGISTRY
from sampling.base import BaseSampler
from sampling.registry import SAMPLER_REGISTRY
from training.base import BaseTrainer
from training.registry import TRAINER_REGISTRY


class Base:
    pass


class ImplA(Base):
    def __init__(self, value: int = 0) -> None:
        self.value = value


class ImplB(Base):
    pass


class NotBase:
    pass


def test_register_and_build() -> None:
    registry = Registry[Base]("test", base_type=Base)

    registry.register("a")(ImplA)

    built = registry.build("a", value=7)
    assert isinstance(built, ImplA)
    assert built.value == 7


def test_duplicate_key_rejected() -> None:
    registry = Registry[Base]("test", base_type=Base)
    registry.register("a")(ImplA)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("a")(ImplB)


def test_base_type_enforced() -> None:
    registry = Registry[Base]("test", base_type=Base)

    with pytest.raises(TypeError, match="does not extend"):
        registry.register("bad")(NotBase)


def test_get_list_contains_and_repr() -> None:
    registry = Registry[Base]("test", base_type=Base)
    registry.register("b")(ImplB)
    registry.register("a")(ImplA)

    assert registry.get("a") is ImplA
    assert registry.list() == ["a", "b"]
    assert "a" in registry
    assert "missing" not in registry
    assert "entries=['a', 'b']" in repr(registry)


def test_unknown_key_error_lists_available() -> None:
    registry = Registry[Base]("test", base_type=Base)
    registry.register("a")(ImplA)

    with pytest.raises(KeyError, match=r"Available: \[a\]"):
        registry.get("missing")

    with pytest.raises(KeyError, match=r"Available: \[a\]"):
        registry.build("missing")


class _NotALoss:
    pass


class _NotANoise:
    pass


class _NotASampler:
    pass


class _NotATrainer:
    pass


def test_framework_registries_expose_base_types() -> None:
    assert LOSS_REGISTRY.base_type is BaseLossComponent
    assert NOISE_REGISTRY.base_type is BaseNoiseProcess
    assert SAMPLER_REGISTRY.base_type is BaseSampler
    assert TRAINER_REGISTRY.base_type is BaseTrainer


def test_framework_registries_reject_invalid_registrations() -> None:
    with pytest.raises(TypeError, match="does not extend"):
        LOSS_REGISTRY.register("bad_test_loss")(_NotALoss)

    with pytest.raises(TypeError, match="does not extend"):
        NOISE_REGISTRY.register("bad_test_noise")(_NotANoise)

    with pytest.raises(TypeError, match="does not extend"):
        SAMPLER_REGISTRY.register("bad_test_sampler")(_NotASampler)

    with pytest.raises(TypeError, match="does not extend"):
        TRAINER_REGISTRY.register("bad_test_trainer")(_NotATrainer)
