"""Helpers for caching nearest base/popular relationships."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from util.helper.paths import simulator_root

from .location_repository import IncidentSiteSummary, LocationDataset

CACHE_PATH = simulator_root() / "cache" / "nearest_lookup.json"


@dataclass(frozen=True)
class NearestLookup:
    incident_to_base: Dict[str, str]
    incident_to_popular: Dict[str, str]
    popular_to_base: Dict[str, str]


class NearestLookupBuilder:
    """Builds and caches nearest base/popular mappings."""

    def __init__(
        self,
        dataset: LocationDataset,
        matrices,
        cache_path: Path = CACHE_PATH,
    ) -> None:
        self.dataset = dataset
        self.matrices = matrices
        self.cache_path = cache_path

    def load(self) -> NearestLookup:
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return NearestLookup(
                incident_to_base=payload.get("incident_to_base", {}),
                incident_to_popular=payload.get("incident_to_popular", {}),
                popular_to_base=payload.get("popular_to_base", {}),
            )

        lookup = self._build()
        self._save(lookup)
        return lookup

    def _build(self) -> NearestLookup:
        incident_to_base = self._build_incident_to_base()
        popular_to_base = self._build_popular_to_base(incident_to_base)
        incident_to_popular = self._build_incident_to_popular()
        return NearestLookup(
            incident_to_base=incident_to_base,
            incident_to_popular=incident_to_popular,
            popular_to_base=popular_to_base,
        )

    def _build_incident_to_base(self) -> Dict[str, str]:
        if self.matrices.base_incident.empty:
            return {}
        df = self.matrices.base_incident[
            ["origin_base_id", "destination_incident_id", "travel_time_minutes"]
        ].copy()
        df = df.sort_values("travel_time_minutes")
        best = (
            df.groupby("destination_incident_id")
            .first()
            .reset_index()[["destination_incident_id", "origin_base_id"]]
        )
        return dict(best.values)

    def _build_popular_to_base(
        self, incident_to_base: Dict[str, str]
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for pop_id, site in self.dataset.popular_locations.items():
            base_id = incident_to_base.get(pop_id)
            if not base_id:
                base_id = self._closest_base_by_distance(site)
            mapping[pop_id] = base_id
        return mapping

    def _build_incident_to_popular(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not self.dataset.popular_locations:
            return mapping
        popular_sites = list(self.dataset.popular_locations.values())
        for incident_id, site in self.dataset.incident_locations.items():
            closest = min(
                popular_sites,
                key=lambda pop: _haversine(
                    site.longitude, site.latitude, pop.longitude, pop.latitude
                ),
            )
            mapping[incident_id] = closest.id
        return mapping

    def _closest_base_by_distance(self, site: IncidentSiteSummary) -> str:
        return min(
            self.dataset.bases.values(),
            key=lambda base: _haversine(
                base.longitude, base.latitude, site.longitude, site.latitude
            ),
        ).id

    def _save(self, lookup: NearestLookup) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "incident_to_base": lookup.incident_to_base,
            "incident_to_popular": lookup.incident_to_popular,
            "popular_to_base": lookup.popular_to_base,
        }
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3958.8
    return c * r


__all__ = ["NearestLookup", "NearestLookupBuilder"]
