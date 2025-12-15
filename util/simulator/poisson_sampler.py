"""Non-homogeneous Poisson sampling utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Sequence

from .config import SimulationConfig
from .insight_loader import TimeDistributions
from .time_window_builder import TimeWindow, build_primary_window, hourly_buckets


@dataclass
class TimestampSample:
    timestamp: datetime
    bucket: TimeWindow


def generate_timestamps(
    config: SimulationConfig,
    distributions: TimeDistributions,
    target_count: int | None = None,
) -> List[datetime]:
    """Generate timestamps across the configured horizon using a Poisson process."""
    window = build_primary_window(config)
    buckets = hourly_buckets(window)
    weights = _bucket_weights(buckets, distributions, config)
    normalized_weights = _normalize(weights)

    count_goal = target_count or config.incident_count
    poisson_counts = [
        _sample_poisson(count_goal * weight) for weight in normalized_weights
    ]
    allocated = sum(poisson_counts)

    if allocated == 0:
        poisson_counts = _redistribute_evenly(count_goal, len(buckets))
        allocated = count_goal

    poisson_counts = _adjust_counts(poisson_counts, normalized_weights, count_goal)

    timestamps: List[datetime] = []
    for bucket, count in zip(buckets, poisson_counts):
        if count <= 0:
            continue
        timestamps.extend(_sample_within_bucket(bucket, count))

    timestamps.sort()
    if len(timestamps) > count_goal:
        timestamps = timestamps[:count_goal]
    elif len(timestamps) < count_goal:
        deficit = count_goal - len(timestamps)
        timestamps.extend(_backfill(timestamps, buckets, normalized_weights, deficit))
    return sorted(timestamps)


def _bucket_weights(
    buckets: Sequence[TimeWindow],
    distributions: TimeDistributions,
    config: SimulationConfig,
) -> List[float]:
    weights: List[float] = []
    for bucket in buckets:
        dt = bucket.start
        hour_weight = distributions.hourly[dt.hour]
        weekday_weight = distributions.weekday[dt.weekday()]
        month_weight = distributions.monthly[dt.month - 1]
        hour_weight *= config.peak_hour_multipliers.get(dt.hour, 1.0)
        weekday_weight *= config.peak_weekday_multipliers.get(dt.weekday(), 1.0)
        month_weight *= config.peak_month_multipliers.get(dt.month, 1.0)
        combined = hour_weight * weekday_weight * month_weight
        weights.append(combined)
    return weights


def _normalize(values: Sequence[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        length = len(values)
        return [1.0 / length] * length if length else []
    return [value / total for value in values]


def _sample_poisson(lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        # Knuth's algorithm can get slow for large lambda. Use normal approximation.
        return max(0, int(random.gauss(lam, math.sqrt(lam))))
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def _redistribute_evenly(total: int, buckets: int) -> List[int]:
    base = total // buckets if buckets else 0
    remainder = total % buckets if buckets else 0
    result = [base] * buckets
    for idx in range(remainder):
        result[idx % buckets] += 1
    return result


def _adjust_counts(
    counts: List[int],
    weights: Sequence[float],
    target: int,
) -> List[int]:
    total = sum(counts)
    if total == target:
        return counts

    counts = list(counts)
    if total < target:
        deficit = target - total
        for _ in range(deficit):
            idx = _weighted_index(weights)
            counts[idx] += 1
    else:
        surplus = total - target
        non_zero_indices = [idx for idx, value in enumerate(counts) if value > 0]
        if not non_zero_indices:
            return counts
        for _ in range(surplus):
            idx = random.choice(non_zero_indices)
            counts[idx] -= 1
            if counts[idx] == 0:
                non_zero_indices.remove(idx)
    return counts


def _weighted_index(weights: Sequence[float]) -> int:
    total = sum(weights)
    if total <= 0:
        return random.randrange(len(weights))
    threshold = random.random() * total
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= threshold:
            return idx
    return len(weights) - 1


def _sample_within_bucket(bucket: TimeWindow, count: int) -> List[datetime]:
    duration_seconds = bucket.duration.total_seconds()
    return [
        bucket.start + timedelta(seconds=random.random() * duration_seconds)
        for _ in range(count)
    ]


def _backfill(
    existing: Sequence[datetime],
    buckets: Sequence[TimeWindow],
    weights: Sequence[float],
    deficit: int,
) -> List[datetime]:
    additions: List[datetime] = []
    for _ in range(deficit):
        bucket = buckets[_weighted_index(weights)]
        additions.extend(_sample_within_bucket(bucket, 1))
    return additions


__all__ = ["generate_timestamps", "TimestampSample"]

