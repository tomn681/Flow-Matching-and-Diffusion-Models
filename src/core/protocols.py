from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class GenerativeModel(Protocol):
    """Any model that produces a ModelOutput from a forward pass."""

    def forward(self, x: torch.Tensor, **kwargs) -> "ModelOutput": ...

    def encode(self, x: torch.Tensor, **kwargs) -> Any: ...

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor: ...


@runtime_checkable
class TimestepConditioned(Protocol):
    """Model whose forward path is explicitly timestep-conditioned."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> Any: ...


@runtime_checkable
class NoiseProcess(Protocol):
    """Defines how clean data is corrupted for training."""

    scheduler: Any

    def __call__(self, clean: torch.Tensor, device: torch.device) -> "NoisyBatch": ...


@runtime_checkable
class LossComponent(Protocol):
    """A single composable loss term."""

    name: str
    weight: float

    def compute(self, *, context: dict[str, Any]) -> torch.Tensor: ...

    def is_active(self, epoch: int, global_step: int) -> bool: ...


@runtime_checkable
class TrainingCallback(Protocol):
    """Hook for cross-cutting training concerns."""

    def on_epoch_start(self, *, epoch: int, trainer: Any) -> None: ...

    def on_epoch_end(
        self, *, epoch: int, metrics: dict, state: dict, trainer: Any
    ) -> None: ...

    def on_train_end(self, *, trainer: Any) -> None: ...


@runtime_checkable
class SamplerCompatibleDataset(Protocol):
    """What samplers and evaluators require from a dataset."""

    target_key: str
    conditioning_key: Optional[str]
    data: list

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict: ...


@runtime_checkable
class Encodable(Protocol):
    """Capability protocol for samplers that support encode mode."""

    def encode(self) -> None: ...


@runtime_checkable
class Decodable(Protocol):
    """Capability protocol for samplers that support decode mode."""

    def decode(self) -> None: ...


@runtime_checkable
class Sampleable(Protocol):
    """Capability protocol for samplers that support sample mode."""

    def sample(self) -> None: ...


@runtime_checkable
class Evaluatable(Protocol):
    """Capability protocol for samplers that support evaluate mode."""

    def evaluate(self) -> None: ...


@runtime_checkable
class Reflowable(Protocol):
    """Capability protocol for samplers that can generate reflow coupling pairs."""

    def generate_reflow_pairs(self) -> None: ...
