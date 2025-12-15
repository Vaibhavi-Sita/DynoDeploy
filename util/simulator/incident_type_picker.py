"""Incident type and zone selection utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Sequence

from .config import SimulationConfig


def _build_weighted_choices(weights: Dict[str, float]) -> Sequence[tuple[str, float]]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return tuple((key, value / total) for key, value in weights.items())


@dataclass
class IncidentAttributePicker:
    config: SimulationConfig

    def __post_init__(self) -> None:
        self.type_weights = _build_weighted_choices(self.config.incident_type_shares)
        self.zone_weights = _build_weighted_choices(self.config.urban_rural_split)
        self.zone_target_counts = self._build_zone_targets()
        self.zone_actual_counts = {zone: 0 for zone in self.config.urban_rural_split}

    def pick_incident_type(self) -> int:
        label = _weighted_choice(self.type_weights)
        return int(label)

    def pick_zone(self) -> str:
        eligible = [
            zone
            for zone, target in self.zone_target_counts.items()
            if self.zone_actual_counts.get(zone, 0) < target
        ]
        if eligible:
            weights = {zone: self.config.urban_rural_split.get(zone, 0.0) for zone in eligible}
            chosen = _weighted_choice_from_mapping(weights)
        else:
            chosen = _weighted_choice(self.zone_weights)
        self.zone_actual_counts[chosen] = self.zone_actual_counts.get(chosen, 0) + 1
        return chosen

    def _build_zone_targets(self) -> Dict[str, int]:
        shares = self.config.urban_rural_split or {"urban": 0.5, "rural": 0.5}
        total = max(int(self.config.incident_count or 0), 1)
        normalized = _normalize_mapping(shares)
        targets = {zone: int(round(weight * total)) for zone, weight in normalized.items()}
        diff = total - sum(targets.values())
        if diff != 0:
            # Adjust the zone with the highest share
            top_zone = max(normalized.items(), key=lambda item: item[1])[0]
            targets[top_zone] = max(targets.get(top_zone, 0) + diff, 0)
        return targets


def _weighted_choice(choices: Sequence[tuple[str, float]]) -> str:
    threshold = random.random()
    cumulative = 0.0
    for label, weight in choices:
        cumulative += weight
        if threshold <= cumulative:
            return label
    return choices[-1][0]


def _weighted_choice_from_mapping(weights: Dict[str, float]) -> str:
    normalized = _normalize_mapping(weights)
    return _weighted_choice(tuple(normalized.items()))


def _normalize_mapping(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        count = len(weights) or 1
        return {zone: 1.0 / count for zone in weights or {"default": 1.0}}
    return {zone: value / total for zone, value in weights.items()}


__all__ = ["IncidentAttributePicker"]

