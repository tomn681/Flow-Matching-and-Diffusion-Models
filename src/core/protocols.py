from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MultiResolutionStageConfig:
    start_epoch: int
    resolution: int

@runtime_checkable
class GenerativeModel(Protocol):
    """Any model that produces a ModelOutput from a forward pass."""

    def forward(self, x: torch.Tensor, **kwargs) -> "ModelOutput": ...

    def encode(self, x: torch.Tensor, **kwargs) -> Any: ...

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor: ...


@runtime_checkable
class TimestepConditioned(Protocol):
    """Model whose forward path is explicitly timestep-conditioned."""

    requires_timesteps: bool

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> Any: ...


@runtime_checkable
class NoiseProcess(Protocol):
    """Defines how clean data is corrupted for training."""

    scheduler: Any

    def __call__(self, clean: torch.Tensor, device: torch.device) -> "NoisyBatch": ...


@runtime_checkable
class NoisingScheduler(Protocol):
    """Scheduler capability protocol for adding forward noise to clean samples."""

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor: ...


@runtime_checkable
class LossComponent(Protocol):
    """A single composable loss term."""

    name: str
    weight: float

    def compute(self, *, context: dict[str, Any]) -> torch.Tensor: ...

    def is_active(self, epoch: int, global_step: int) -> bool: ...


@runtime_checkable
class TrainerView(Protocol):
    """Read-only trainer surface exposed to callbacks and event listeners."""

    output_dir: Path
    global_step: int
    is_main_process: bool
    rank: int
    world_size: int

    def ema_scope(self) -> Any: ...

    def request_dataloader_rebuild(self, *, target_resolution: int | None) -> None: ...


@runtime_checkable
class TrainingCallback(Protocol):
    """Hook for cross-cutting training concerns."""

    def on_epoch_start(self, *, epoch: int, trainer: TrainerView) -> None: ...

    def on_epoch_end(
        self, *, epoch: int, metrics: Mapping[str, Any], state: Mapping[str, Any], trainer: TrainerView
    ) -> None: ...

    def on_train_end(self, *, trainer: TrainerView) -> None: ...


@runtime_checkable
class SamplerCompatibleDataset(Protocol):
    """What samplers and evaluators require from a dataset."""

    target_key: str
    conditioning_key: Optional[str]

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


@runtime_checkable
class ResolutionSchedule(Protocol):
    """Capability protocol for epoch-indexed multi-resolution schedules."""

    stages: list[MultiResolutionStageConfig]

    def current_resolution(self, epoch: int) -> int: ...


@runtime_checkable
class Discriminatable(Protocol):
    """Capability protocol for models that can provide a discriminator."""

    def make_discriminator(self) -> nn.Module | None: ...
