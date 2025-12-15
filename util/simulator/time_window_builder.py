"""Helpers for constructing time windows used by the Poisson sampler."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple

from .config import SimulationConfig


@dataclass(frozen=True)
class TimeWindow:
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        return self.end - self.start


def build_primary_window(config: SimulationConfig) -> TimeWindow:
    start = config.start_datetime
    end = start + config.horizon
    return TimeWindow(start=start, end=end)


def hourly_buckets(window: TimeWindow) -> List[TimeWindow]:
    buckets: List[TimeWindow] = []
    cursor = window.start
    while cursor < window.end:
        next_cursor = min(cursor + timedelta(hours=1), window.end)
        buckets.append(TimeWindow(start=cursor, end=next_cursor))
        cursor = next_cursor
    return buckets


def sliding_windows(
    window: TimeWindow,
    span: timedelta,
) -> Iterable[TimeWindow]:
    cursor = window.start
    while cursor < window.end:
        end = min(cursor + span, window.end)
        yield TimeWindow(cursor, end)
        cursor = end


__all__ = ["TimeWindow", "build_primary_window", "hourly_buckets", "sliding_windows"]

