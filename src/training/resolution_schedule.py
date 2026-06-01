from __future__ import annotations


class ResolutionSchedule:
    """Progressive resolution schedule keyed by start epoch."""

    def __init__(self, schedule: dict[int, int]) -> None:
        if not schedule:
            raise ValueError("ResolutionSchedule requires at least one schedule entry.")
        normalized: dict[int, int] = {}
        for epoch, resolution in schedule.items():
            ep = int(epoch)
            res = int(resolution)
            if ep < 1:
                raise ValueError("Resolution schedule epochs must be >= 1.")
            if res <= 0:
                raise ValueError("Resolution schedule values must be > 0.")
            normalized[ep] = res
        self.schedule = dict(sorted(normalized.items(), key=lambda kv: kv[0]))

    def resolution_for_epoch(self, epoch: int) -> int:
        current = next(iter(self.schedule.values()))
        for start_epoch, resolution in self.schedule.items():
            if int(epoch) >= start_epoch:
                current = resolution
            else:
                break
        return int(current)

