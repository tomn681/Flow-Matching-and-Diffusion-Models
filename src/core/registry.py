from __future__ import annotations

from typing import Callable, Generic, Optional, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """Type-safe registry with decorator-based registration."""

    def __init__(self, name: str, base_type: Optional[type] = None) -> None:
        self._name = name
        self._base_type = base_type
        self._entries: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if key in self._entries:
                raise ValueError(
                    f"Registry '{self._name}': key '{key}' already registered "
                    f"by {self._entries[key].__name__}. Cannot register {cls.__name__}."
                )
            if self._base_type and not issubclass(cls, self._base_type):
                raise TypeError(
                    f"Registry '{self._name}': {cls.__name__} does not extend "
                    f"{self._base_type.__name__}."
                )
            self._entries[key] = cls
            return cls

        return decorator

    def register_value(self, key: str, value: type[T]) -> type[T]:
        if key in self._entries:
            raise ValueError(
                f"Registry '{self._name}': key '{key}' already registered "
                f"by {self._entries[key].__name__}. Cannot register {value.__name__}."
            )
        if self._base_type and not issubclass(value, self._base_type):
            raise TypeError(
                f"Registry '{self._name}': {value.__name__} does not extend "
                f"{self._base_type.__name__}."
            )
        self._entries[key] = value
        return value

    def build(self, key: str, **kwargs) -> T:
        if key not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise KeyError(
                f"Registry '{self._name}': unknown key '{key}'. Available: [{available}]"
            )
        return self._entries[key](**kwargs)

    def get(self, key: str) -> type[T]:
        if key not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise KeyError(
                f"Registry '{self._name}': unknown key '{key}'. Available: [{available}]"
            )
        return self._entries[key]

    def __getitem__(self, key: str) -> type[T]:
        return self.get(key)

    def list(self) -> list[str]:
        return sorted(self._entries.keys())

    def keys(self) -> list[str]:
        return self.list()

    def values(self) -> list[type[T]]:
        return [self._entries[key] for key in self.list()]

    def items(self) -> list[tuple[str, type[T]]]:
        return [(key, self._entries[key]) for key in self.list()]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __repr__(self) -> str:
        return f"Registry('{self._name}', entries={self.list()})"
