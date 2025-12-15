"""Applies overlap logic to incident specs for stress testing scenarios."""

from __future__ import annotations

import random
from collections import Counter
from datetime import timedelta
from typing import Iterable, List, Sequence

from .config import SimulationConfig
from .record_factory import IncidentSpec


def apply_overlaps(
    specs: Sequence[IncidentSpec],
    config: SimulationConfig,
) -> List[IncidentSpec]:
    overlap_cfg = config.overlap
    if not specs or overlap_cfg.probability <= 0:
        return list(specs)

    specs = list(specs)
    overlap_target = min(
        int(len(specs) * overlap_cfg.probability),
        len(specs),
    )
    tracker = Counter(_overlap_key(spec) for spec in specs)

    for _ in range(overlap_target):
        base_idx = random.randrange(len(specs))
        target_idx = random.randrange(len(specs))
        if base_idx == target_idx:
            continue
        base_spec = specs[base_idx]
        target_spec = specs[target_idx]
        new_timestamp = _jitter_timestamp(base_spec.timestamp, overlap_cfg)

        share_location = random.random() < overlap_cfg.same_location_probability
        new_location = base_spec.location if share_location else target_spec.location
        key = _overlap_key_from(new_location.id, new_timestamp)
        if tracker[key] >= overlap_cfg.max_simultaneous:
            continue

        tracker[key] += 1
        tracker[_overlap_key(target_spec)] -= 1
        specs[target_idx] = IncidentSpec(
            timestamp=new_timestamp,
            incident_type=target_spec.incident_type,
            zone=target_spec.zone,
            location=new_location,
        )

    specs.sort(key=lambda spec: spec.timestamp)
    return specs


def _jitter_timestamp(base_time, overlap_cfg) -> "datetime":
    delta_minutes = random.uniform(
        overlap_cfg.time_offset_minutes_min,
        overlap_cfg.time_offset_minutes_max,
    )
    if random.random() < 0.5:
        delta_minutes *= -1
    return base_time + timedelta(minutes=delta_minutes)


def _overlap_key(spec: IncidentSpec):
    return _overlap_key_from(spec.location.id, spec.timestamp)


def _overlap_key_from(location_id: str, timestamp) -> tuple[str, int]:
    epoch_seconds = int(timestamp.timestamp())
    return (location_id, epoch_seconds // 60)


__all__ = ["apply_overlaps"]

