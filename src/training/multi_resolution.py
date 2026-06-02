from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.protocols import ResolutionSchedule as ResolutionScheduleProtocol
from configs.training import MultiResolutionStageConfig


@dataclass(frozen=True)
class StepwiseResolutionSchedule(ResolutionScheduleProtocol):
    """Stepwise epoch-resolution schedule.

    Example:
        stages = [
            MultiResolutionStageConfig(start_epoch=0, resolution=64),
            MultiResolutionStageConfig(start_epoch=10, resolution=128),
        ]
    """

    stages: list[MultiResolutionStageConfig]

    def current_resolution(self, epoch: int) -> int:
        if int(epoch) < int(self.stages[0].start_epoch):
            raise ValueError(
                f"Epoch {epoch} is before first stage start_epoch={self.stages[0].start_epoch}."
            )
        current = int(self.stages[0].resolution)
        for stage in self.stages:
            if int(epoch) >= int(stage.start_epoch):
                current = int(stage.resolution)
            else:
                break
        return current


def build_resolution_schedule(cfg: dict[str, Any]) -> ResolutionScheduleProtocol | None:
    training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    stages_raw = training_cfg.get("multi_resolution")
    if stages_raw is None:
        return None
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError("training.multi_resolution must be a non-empty list when provided.")

    stages: list[MultiResolutionStageConfig] = []
    for item in stages_raw:
        if isinstance(item, MultiResolutionStageConfig):
            stage = item
        elif isinstance(item, dict):
            stage = MultiResolutionStageConfig(
                start_epoch=int(item.get("start_epoch", 0)),
                resolution=int(item.get("resolution", 0)),
            )
        else:
            raise TypeError("Each multi_resolution stage must be a dict or MultiResolutionStageConfig.")
        stages.append(stage)

    if sorted(int(s.start_epoch) for s in stages) != [int(s.start_epoch) for s in stages]:
        raise ValueError("training.multi_resolution stages must be sorted by start_epoch ascending.")
    if int(stages[0].start_epoch) != 0:
        raise ValueError("training.multi_resolution first stage must start at epoch 0.")

    for stage in stages:
        if int(stage.resolution) <= 0:
            raise ValueError("training.multi_resolution.resolution must be > 0.")
        res = int(stage.resolution)
        if res & (res - 1):
            logging.warning(
                "Non-power-of-2 multi-resolution stage detected: resolution=%d at epoch=%d.",
                res,
                int(stage.start_epoch),
            )

    return StepwiseResolutionSchedule(stages=stages)


def _check_multi_resolution_compatibility(model, schedule: ResolutionScheduleProtocol) -> None:
    """Check model/schedule compatibility and emit warnings for known mismatch risks."""
    from models.unet.base import BaseUNetND
    from models.vae.base import BaseVAE

    if not isinstance(model, (BaseUNetND, BaseVAE)):
        raise NotImplementedError(
            "Multi-resolution currently supports fully convolutional framework models only "
            "(BaseUNetND or BaseVAE)."
        )

    if hasattr(model, "pos_embed") or hasattr(model, "position_embedding"):
        logging.warning(
            "Model exposes fixed positional embeddings (pos_embed/position_embedding); "
            "this is typically incompatible with dynamic resolution schedules."
        )

    if isinstance(model, BaseUNetND):
        attn_res = getattr(model, "attention_resolutions", None)
        if attn_res:
            for stage in schedule.stages:
                res = int(stage.resolution)
                # Heuristic: if no attention resolution divides stage res, warn.
                missed = all((res % int(a) != 0) for a in attn_res if int(a) > 0)
                if missed:
                    logging.warning(
                        "Multi-resolution stage res=%d (epoch=%d) may miss configured attention_resolutions=%s.",
                        res,
                        int(stage.start_epoch),
                        list(attn_res),
                    )


__all__ = ["StepwiseResolutionSchedule", "build_resolution_schedule", "_check_multi_resolution_compatibility"]

