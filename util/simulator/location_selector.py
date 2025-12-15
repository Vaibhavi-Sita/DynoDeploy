"""Weighted location selection respecting urban/rural splits and hotspots."""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, Iterable, List, Sequence

from .config import SimulationConfig
from .location_repository import LocationRecord


class LocationSelector:
    def __init__(
        self,
        locations: Sequence[LocationRecord],
        config: SimulationConfig,
    ) -> None:
        if not locations:
            raise ValueError("Location list cannot be empty.")
        self.config = config
        self.usage = Counter()
        self.high_demand_munis = [
            value.strip().lower() for value in config.high_demand_municipalities
        ]
        self.locations = list(locations)
        self.base_weights = {
            location.id: self._base_weight(location) for location in self.locations
        }
        self.zone_targets: Dict[str, float] = {
            "urban": config.urban_rural_split.get("urban", 0.5),
            "rural": config.urban_rural_split.get("rural", 0.5),
        }
        self.balance_tolerance = 0.03
        self.pool = self._build_pool(self.locations)
        self.zone_lookup = self._build_zone_lookup(self.pool)
        self.municipality_lookup = self._build_muni_lookup(self.pool)
        self.urban_reservoir = [loc for loc in self.locations if loc.is_urban]
        self.rural_reservoir = [loc for loc in self.locations if not loc.is_urban]
        self.primary_index = 0
        self.zone_counts: Dict[str, float] = {"urban": 0.0, "rural": 0.0}

    def _build_pool(self, locations: Sequence[LocationRecord]) -> List[LocationRecord]:
        pool_size = min(self.config.location_sample_size, len(locations))
        ranked = sorted(
            locations,
            key=lambda loc: self.base_weights.get(loc.id, 1.0),
            reverse=True,
        )
        rural_sorted = [loc for loc in ranked if not loc.is_urban]
        urban_sorted = [loc for loc in ranked if loc.is_urban]
        min_rural_share = max(self.zone_targets.get("rural", 0.2), 0.2)
        min_rural = min(len(rural_sorted), max(int(pool_size * min_rural_share), 10))
        rural_pick = rural_sorted[:min_rural]
        remaining_slots = pool_size - len(rural_pick)
        # maintain order for remaining slots while avoiding duplicates
        taken_ids = {loc.id for loc in rural_pick}
        remainder: List[LocationRecord] = []
        for loc in ranked:
            if loc.id in taken_ids:
                continue
            remainder.append(loc)
            if len(remainder) >= remaining_slots:
                break
        combined = rural_pick + remainder
        random.shuffle(combined)
        return combined

    @staticmethod
    def _build_zone_lookup(pool: Sequence[LocationRecord]) -> Dict[str, List[LocationRecord]]:
        return {
            "urban": [loc for loc in pool if loc.is_urban],
            "rural": [loc for loc in pool if not loc.is_urban],
        }

    @staticmethod
    def _normalize_muni(location: LocationRecord) -> str | None:
        municipality = (getattr(location, "municipality", None) or "").strip().lower()
        if municipality:
            return municipality
        meta_muni = (
            str(location.metadata.get("municipality") or location.metadata.get("city") or "")
            .strip()
            .lower()
        )
        return meta_muni or None

    def _build_muni_lookup(
        self, pool: Sequence[LocationRecord]
    ) -> Dict[str, List[LocationRecord]]:
        lookup: Dict[str, List[LocationRecord]] = {}
        for loc in pool:
            muni = self._normalize_muni(loc)
            if not muni:
                continue
            lookup.setdefault(muni, []).append(loc)
        return lookup

    def select(self, zone: str) -> LocationRecord:
        zone = self._normalize_zone(zone)
        zone = self._balanced_zone(zone)
        use_primary = (
            self.high_demand_munis
            and random.random() < self.config.primary_location_share
        )
        candidates = []
        if use_primary:
            candidates = self._primary_candidates(zone)
        if not candidates:
            candidates = self._available_candidates(zone)
        if not candidates:
            candidates = self._available_candidates(zone, enforce_limit=False)
        if not candidates:
            alternate_zone = "rural" if zone == "urban" else "urban"
            candidates = self._available_candidates(alternate_zone)
        if not candidates:
            # final fallback: ignore limits but respect zone ordering
            candidates = self._available_candidates(zone, enforce_limit=False) or self._available_candidates(
                alternate_zone, enforce_limit=False
            )
        if not candidates:
            candidates = self._reservoir_candidates(zone) or self._reservoir_candidates(
                alternate_zone
            )
        if not candidates:
            raise RuntimeError("No locations available after enforcing constraints.")
        weights = [self._weight(candidate) for candidate in candidates]
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        self.usage[chosen.id] += 1
        actual_zone = "urban" if chosen.is_urban else "rural"
        self.zone_counts[actual_zone] += 1.0
        return chosen

    def _primary_candidates(self, zone: str) -> List[LocationRecord]:
        if not self.high_demand_munis:
            return []
        for _ in range(len(self.high_demand_munis)):
            muni = self.high_demand_munis[self.primary_index % len(self.high_demand_munis)]
            self.primary_index += 1
            pool = self.municipality_lookup.get(muni, [])
            if not pool:
                continue
            filtered = self._filter_pool(pool, zone, enforce_limit=True)
            if filtered:
                return filtered
        return []

    def _available_candidates(self, zone: str, *, enforce_limit: bool = True) -> List[LocationRecord]:
        pool = self.zone_lookup.get(zone)
        if not pool:
            pool = self.pool
        return self._filter_pool(pool, zone, enforce_limit=enforce_limit)

    def _filter_pool(
        self,
        pool: Sequence[LocationRecord],
        zone: str,
        *,
        enforce_limit: bool,
    ) -> List[LocationRecord]:
        results: List[LocationRecord] = []
        for loc in pool:
            if not self._in_zone(loc, zone):
                continue
            if enforce_limit and not self._within_limit(loc):
                continue
            results.append(loc)
        return results

    def _within_limit(self, location: LocationRecord) -> bool:
        limit = self.config.max_incidents_per_location
        if limit is None:
            return True
        return self.usage[location.id] < limit

    def _in_zone(self, location: LocationRecord, zone: str) -> bool:
        if zone == "urban":
            return location.is_urban
        if zone == "rural":
            return not location.is_urban
        return True

    def _normalize_zone(self, zone: str) -> str:
        zone_lower = (zone or "").strip().lower()
        return "urban" if zone_lower not in {"urban", "rural"} else zone_lower

    def _balanced_zone(self, requested_zone: str) -> str:
        total = sum(self.zone_counts.values())
        if total < 50:  # allow initial ramp without forcing rebalancing
            return requested_zone
        target = self.zone_targets.get(requested_zone, 0.5)
        actual = self.zone_counts.get(requested_zone, 0.0) / max(total, 1.0)
        if actual > target + self.balance_tolerance:
            return "rural" if requested_zone == "urban" else "urban"
        return requested_zone

    def _reservoir_candidates(self, zone: str) -> List[LocationRecord]:
        reservoir = self.urban_reservoir if zone == "urban" else self.rural_reservoir
        if not reservoir:
            return []
        sample_size = min(10, len(reservoir))
        return random.sample(reservoir, sample_size)

    def _weight(self, location: LocationRecord) -> float:
        base = self.base_weights.get(location.id, 1.0)
        usage = self.usage[location.id]
        decay = 1 + 0.25 * usage
        return max(base / decay, 0.01)

    def _base_weight(self, location: LocationRecord) -> float:
        weight = 1.0
        if location.is_urban:
            weight *= self.config.urban_hotspot_weight
        else:
            weight *= self.config.rural_demand_weight
        municipality = (getattr(location, "municipality", None) or "").strip().lower()
        if municipality and municipality in self.high_demand_munis:
            weight *= self.config.urban_hotspot_weight
        if "popular" in location.location_type.lower() or location.source_table == "popular_locations":
            weight *= self.config.hotspot_weight_multiplier
        return max(weight, 0.05)

    def register_usage(self, location: LocationRecord) -> None:
        self.usage[location.id] += 1


__all__ = ["LocationSelector"]

