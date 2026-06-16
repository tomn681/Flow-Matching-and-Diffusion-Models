from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping


def _freeze_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if mapping is None:
        return MappingProxyType({})
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class TrainEvent:
    trainer: Any


@dataclass(frozen=True)
class EpochEvent:
    epoch: int
    trainer: Any
    metrics: Mapping[str, Any] = MappingProxyType({})
    state: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class StepEvent:
    epoch: int
    step: int
    global_step: int
    metrics: Mapping[str, Any]
    trainer: Any


@dataclass(frozen=True)
class EventSubscription:
    event: str
    listener: Callable[..., Any]


class TrainingEventBus:
    """Simple publish-subscribe bus for trainer lifecycle events."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event: str, listener: Callable[..., Any]) -> EventSubscription:
        key = str(event)
        self._listeners[key].append(listener)
        return EventSubscription(event=key, listener=listener)

    def on(self, event: str, listener: Callable[..., Any]) -> EventSubscription:
        return self.subscribe(event, listener)

    def emit(self, event: str, **kwargs: Any) -> None:
        key = str(event)
        payload = self._build_payload(key, kwargs)
        for listener in list(self._listeners.get(key, ())):
            try:
                listener(payload=payload, **kwargs)
            except TypeError as exc:
                if "payload" not in str(exc):
                    raise
                listener(**kwargs)

    def remove(self, event: str, listener: Callable[..., Any]) -> None:
        key = str(event)
        listeners = self._listeners.get(key)
        if not listeners:
            return
        self._listeners[key] = [fn for fn in listeners if fn is not listener]
        if not self._listeners[key]:
            self._listeners.pop(key, None)

    def unsubscribe(self, subscription: EventSubscription) -> None:
        self.remove(subscription.event, subscription.listener)

    @staticmethod
    def _build_payload(event: str, kwargs: dict[str, Any]) -> Any:
        if event == "step_end":
            if not {"epoch", "step", "global_step", "trainer"} <= kwargs.keys():
                return _freeze_mapping(kwargs)
            return StepEvent(
                epoch=int(kwargs["epoch"]),
                step=int(kwargs["step"]),
                global_step=int(kwargs["global_step"]),
                metrics=_freeze_mapping(kwargs.get("metrics")),
                trainer=kwargs["trainer"],
            )
        if event in {"epoch_start", "epoch_end", "validation_end"}:
            if not {"epoch", "trainer"} <= kwargs.keys():
                return _freeze_mapping(kwargs)
            return EpochEvent(
                epoch=int(kwargs["epoch"]),
                trainer=kwargs["trainer"],
                metrics=_freeze_mapping(kwargs.get("metrics")),
                state=_freeze_mapping(kwargs.get("state")),
            )
        if event in {"train_start", "train_end"}:
            if "trainer" not in kwargs:
                return _freeze_mapping(kwargs)
            return TrainEvent(trainer=kwargs["trainer"])
        return _freeze_mapping(kwargs)


__all__ = [
    "EpochEvent",
    "EventSubscription",
    "StepEvent",
    "TrainEvent",
    "TrainingEventBus",
]
