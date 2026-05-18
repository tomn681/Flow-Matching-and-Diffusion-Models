import pytest

from core.registry import Registry


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
