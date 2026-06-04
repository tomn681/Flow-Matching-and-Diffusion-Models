from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class TrainingEventBus:
    """Simple publish-subscribe bus for trainer lifecycle events."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        self._listeners[str(event)].append(listener)

    def emit(self, event: str, **kwargs: Any) -> None:
        for listener in list(self._listeners.get(str(event), ())):
            listener(**kwargs)

    def remove(self, event: str, listener: Callable[..., Any]) -> None:
        key = str(event)
        listeners = self._listeners.get(key)
        if not listeners:
            return
        self._listeners[key] = [fn for fn in listeners if fn is not listener]
        if not self._listeners[key]:
            self._listeners.pop(key, None)


__all__ = ["TrainingEventBus"]
