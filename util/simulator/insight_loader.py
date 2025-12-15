"""Utilities for mining historical incident insights and schema templates."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Sequence

from util.helper.paths import project_root, simulator_root

DEFAULT_INCIDENT_SOURCE = (
    project_root()
    / "resources"
    / "original_incident_records"
    / "lancaster_incidents_filtered_labeled.json"
)
INSIGHT_DIR = simulator_root() / "data" / "insights"


@dataclass(frozen=True)
class TimeDistributions:
    """Holds normalized probability distributions for sim inputs."""

    hourly: tuple[float, ...]  # 24 entries
    weekday: tuple[float, ...]  # Monday (0) .. Sunday (6)
    monthly: tuple[float, ...]  # January (1) .. December (12)


@dataclass(frozen=True)
class InsightSummary:
    """Summary stats plus an example incident schema."""

    distributions: TimeDistributions
    schema_template: dict
    total_source_incidents: int


def _normalize(counter: Counter[int], length: int) -> list[float]:
    if length <= 0:
        raise ValueError("length must be positive")
    total = sum(counter.values())
    if not total:
        return [1.0 / length] * length
    return [counter.get(idx, 0) / total for idx in range(length)]


def _ensure_insight_dir() -> None:
    INSIGHT_DIR.mkdir(parents=True, exist_ok=True)


def _load_source_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Incident source not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_schema_template(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("No records available to infer schema template.")
    # Copy the first record so callers can safely modify it.
    template = dict(records[0])
    # Remove fields that will be re-generated explicitly.
    for key in ("_id", "incidentID", "incidentTime", "_created_at", "_updated_at"):
        template.pop(key, None)
    return template


def build_insight_summary(source_path: Path | None = None) -> InsightSummary:
    """Compute probability distributions + schema template from historical data."""
    path = source_path or DEFAULT_INCIDENT_SOURCE
    records = _load_source_records(path)

    hour_counts: Counter[int] = Counter()
    weekday_counts: Counter[int] = Counter()
    month_counts: Counter[int] = Counter()

    for record in records:
        incident_time = record.get("incidentTime")
        if not incident_time:
            continue
        dt = parsedate_to_datetime(incident_time)
        # Normalize to local timezone if offset provided.
        if dt.tzinfo:
            dt = dt.astimezone(dt.tzinfo)
        hour_counts[dt.hour] += 1
        weekday_counts[dt.weekday()] += 1
        month_counts[dt.month] += 1

    distributions = TimeDistributions(
        hourly=tuple(_normalize(hour_counts, 24)),
        weekday=tuple(_normalize(weekday_counts, 7)),
        monthly=tuple(_normalize(month_counts, 12)),
    )

    schema_template = _extract_schema_template(records)
    summary = InsightSummary(
        distributions=distributions,
        schema_template=schema_template,
        total_source_incidents=len(records),
    )
    _write_insight_artifacts(summary)
    return summary


def _write_insight_artifacts(summary: InsightSummary) -> None:
    """Write the distributions to disk."""
    _ensure_insight_dir()
    hourly_path = INSIGHT_DIR / "hourly_distribution.json"
    weekday_path = INSIGHT_DIR / "weekday_distribution.json"
    monthly_path = INSIGHT_DIR / "monthly_distribution.json"
    meta_path = INSIGHT_DIR / "insight_summary.json"

    def dump(path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    dump(hourly_path, {str(idx): prob for idx, prob in enumerate(summary.distributions.hourly)})
    dump(
        weekday_path,
        {str(idx): prob for idx, prob in enumerate(summary.distributions.weekday)},
    )
    dump(
        monthly_path,
        {str(idx + 1): prob for idx, prob in enumerate(summary.distributions.monthly)},
    )
    dump(
        meta_path,
        {
            "total_source_incidents": summary.total_source_incidents,
            "schema_template_keys": sorted(summary.schema_template.keys()),
        },
    )


def load_cached_distributions() -> TimeDistributions | None:
    """Try to read previously written distributions without reprocessing."""
    hourly_path = INSIGHT_DIR / "hourly_distribution.json"
    weekday_path = INSIGHT_DIR / "weekday_distribution.json"
    monthly_path = INSIGHT_DIR / "monthly_distribution.json"
    try:
        with hourly_path.open("r", encoding="utf-8") as h, \
            weekday_path.open("r", encoding="utf-8") as d, \
            monthly_path.open("r", encoding="utf-8") as m:
            hourly_data = json.load(h)
            weekday_data = json.load(d)
            monthly_data = json.load(m)
            hourly = tuple(float(hourly_data[str(idx)]) for idx in range(24))
            weekday = tuple(float(weekday_data[str(idx)]) for idx in range(7))
            monthly = tuple(float(monthly_data[str(idx + 1)]) for idx in range(12))
            return TimeDistributions(hourly=hourly, weekday=weekday, monthly=monthly)
    except FileNotFoundError:
        return None
    return None


def get_insight_summary(force_recompute: bool = False) -> InsightSummary:
    """Load cached data when possible, otherwise rebuild from source."""
    cached_dists = None if force_recompute else load_cached_distributions()
    if cached_dists:
        schema_template = _extract_schema_template(_load_source_records(DEFAULT_INCIDENT_SOURCE))
        return InsightSummary(
            distributions=cached_dists,
            schema_template=schema_template,
            total_source_incidents=-1,
        )
    return build_insight_summary()


__all__ = [
    "TimeDistributions",
    "InsightSummary",
    "build_insight_summary",
    "get_insight_summary",
    "load_cached_distributions",
]

