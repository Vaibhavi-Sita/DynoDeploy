"""Selects realistic incident descriptions for each incident type."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .insight_loader import DEFAULT_INCIDENT_SOURCE


class DescriptionPicker:
    """Weighted description chooser keyed by incident type."""

    def __init__(self, mapping: Dict[int, Sequence[Tuple[str, float]]]) -> None:
        if not mapping:
            raise ValueError("Description mapping cannot be empty.")
        self.mapping = mapping

    @classmethod
    def from_source(
        cls,
        source_path: Path | None = None,
        min_count: int = 5,
        top_n: int = 25,
    ) -> "DescriptionPicker":
        """Build picker from historical incident records."""
        path = source_path or DEFAULT_INCIDENT_SOURCE
        descriptions: Dict[int, Counter[str]] = defaultdict(Counter)
        with path.open("r", encoding="utf-8") as handle:
            import json

            records = json.load(handle)

        for record in records:
            try:
                incident_type = int(record.get("incidentType"))
            except (TypeError, ValueError):
                continue
            description = (record.get("description") or "").strip()
            if not description:
                continue
            descriptions[incident_type][description] += 1

        weighted: Dict[int, List[Tuple[str, float]]] = {}
        for incident_type, counter in descriptions.items():
            ranked = counter.most_common(top_n)
            filtered = [(desc, count) for desc, count in ranked if count >= min_count]
            if not filtered:
                filtered = ranked[:10] or list(counter.items())
            total = sum(weight for _, weight in filtered) or 1.0
            weighted[incident_type] = [(desc, weight / total) for desc, weight in filtered]

        if not weighted:
            raise ValueError("No descriptions extracted from source data.")
        return cls(weighted)

    def pick(self, incident_type: int) -> str:
        candidates = self.mapping.get(incident_type)
        if not candidates:
            # Fallback to the most common type if unknown.
            candidates = next(iter(self.mapping.values()))
        labels, weights = zip(*candidates)
        return random.choices(labels, weights=weights, k=1)[0]


__all__ = ["DescriptionPicker"]

