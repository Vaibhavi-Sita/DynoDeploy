"""Utilities for normalizing historical incident records for simulation input."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class IncidentEvent:
    """An incident record used by the simulator."""

    id: str
    occurred_at: datetime
    incident_type: str
    priority: int | None
    location_id: str | None
    location_name: str | None
    location_type: str | None
    longitude: float | None
    latitude: float | None
    hospital_id: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "IncidentEvent":
        """Build an `IncidentEvent` from a raw JSON dictionary."""
        occurred_at = _coerce_datetime(
            payload.get("_created_at", {}).get("$date")
            if isinstance(payload.get("_created_at"), dict)
            else payload.get("_created_at"),
            payload.get("incidentTime"),
        )
        incident_id = str(
            payload.get("incidentID")
            or payload.get("_id")
            or payload.get("id")
            or payload.get("uuid")
        )
        priority = _maybe_int(
            payload.get("priority")
            or payload.get("incident_priority")
            or payload.get("incidentType")
        )
        location_id = (
            payload.get("location_id")
            or payload.get("incident_location_id")
            or payload.get("municipalID")
            or payload.get("incidentID")
        )
        location_name = (
            payload.get("location_name")
            or payload.get("municipality")
            or payload.get("street")
            or payload.get("name")
        )
        location_type = payload.get("location_type") or payload.get("type")
        longitude, latitude = _extract_coordinates(payload)

        hospital_id = (
            payload.get("hospital_id")
            or payload.get("transported_to_hospital_id")
            or payload.get("preferred_hospital_id")
        )

        return cls(
            id=incident_id,
            occurred_at=occurred_at,
            incident_type=str(payload.get("incidentType") or payload.get("callType")),
            priority=priority,
            location_id=str(location_id) if location_id else None,
            location_name=str(location_name) if location_name else None,
            location_type=str(location_type) if location_type else None,
            longitude=longitude,
            latitude=latitude,
            hospital_id=str(hospital_id) if hospital_id else None,
            metadata=dict(payload),
        )


def load_incident_history(
    sources: Sequence[str | Path] | str | Path,
    *,
    limit: int | None = None,
) -> List[IncidentEvent]:
    """
    Load incident events from one or more JSON files.

    Each source may contain either:
        * A list of incident dictionaries
        * An envelope with an `"incidents"` key (as produced by the generator)
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]
    events: List[IncidentEvent] = []
    for source in sources:
        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_incidents = payload.get("incidents") if isinstance(payload, dict) else payload
        if not isinstance(raw_incidents, Iterable):
            raise ValueError(f"Source {path} did not contain incidents.")
        for record in raw_incidents:
            event = IncidentEvent.from_payload(record)
            events.append(event)
            if limit is not None and len(events) >= limit:
                return events[:limit]
    return events


def _coerce_datetime(primary: Any, fallback: Any) -> datetime:
    candidates = [primary, fallback, datetime.now(tz=timezone.utc)]
    for candidate in candidates:
        if not candidate:
            continue
        if isinstance(candidate, datetime):
            return candidate if candidate.tzinfo else candidate.replace(tzinfo=timezone.utc)
        if isinstance(candidate, (int, float)):
            return datetime.fromtimestamp(float(candidate), tz=timezone.utc)
        if isinstance(candidate, str):
            try:
                dt = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return datetime.now(tz=timezone.utc)


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_coordinates(
    payload: Mapping[str, Any]
) -> tuple[float | None, float | None]:
    lon = payload.get("longitude")
    lat = payload.get("latitude")
    if lon is not None and lat is not None:
        return float(lon), float(lat)
    geo = payload.get("geoLocation") or payload.get("coordinates")
    if isinstance(geo, (list, tuple)) and len(geo) >= 2:
        return float(geo[0]), float(geo[1])
    if isinstance(geo, dict):
        lon = geo.get("lon") or geo.get("lng") or geo.get("x")
        lat = geo.get("lat") or geo.get("y")
        if lon is not None and lat is not None:
            return float(lon), float(lat)
    return None, None


__all__ = ["IncidentEvent", "load_incident_history"]

