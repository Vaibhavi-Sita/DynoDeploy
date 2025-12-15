"""Factory that converts planning specs into incident records."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence

from .location_repository import LocationRecord
from .description_picker import DescriptionPicker


INCIDENT_TYPE_DESCRIPTIONS = {
    1: "BLS TRANSPORT REQUEST",
    2: "MEDICAL EMERGENCY",
    3: "VEHICLE ACCIDENT-CLASS 1",
}


@dataclass
class IncidentSpec:
    timestamp: datetime
    incident_type: int
    zone: str
    location: LocationRecord


class RecordFactory:
    def __init__(
        self,
        schema_template: Dict,
        timezone_name: str = "America/New_York",
        description_picker: DescriptionPicker | None = None,
    ) -> None:
        self.schema_template = copy.deepcopy(schema_template)
        self.timezone_name = timezone_name
        self.description_picker = description_picker

    def build_records(self, specs: Sequence[IncidentSpec]) -> List[Dict]:
        ordered_specs = sorted(specs, key=lambda spec: spec.timestamp)
        records: List[Dict] = []
        for index, spec in enumerate(ordered_specs, start=1):
            records.append(self._build_single(spec, index))
        return records

    def _build_single(self, spec: IncidentSpec, sequence: int) -> Dict:
        record = copy.deepcopy(self.schema_template)
        tz = spec.timestamp.tzinfo or timezone.utc
        timestamp = spec.timestamp.astimezone(tz)
        tz_name = getattr(tz, "key", timestamp.tzname() or self.timezone_name)
        record["_id"] = f"SIM{sequence:06d}"
        record["incidentID"] = f"SIM-{timestamp.strftime('%Y%m%d')}-{sequence:05d}"
        record["incidentType"] = spec.incident_type
        if self.description_picker:
            record["description"] = self.description_picker.pick(spec.incident_type)
        else:
            record["description"] = INCIDENT_TYPE_DESCRIPTIONS.get(
                spec.incident_type, "MEDICAL EMERGENCY"
            )
        record["incidentTime"] = timestamp.strftime("%a, %d %b %Y %H:%M:%S %Z")
        record["incidentTime_timezone"] = tz_name
        record["_created_at"] = {"$date": timestamp.isoformat(), "timezone": tz_name}
        record["_updated_at"] = record["_created_at"]
        record["pushSent"] = False
        record["active"] = False
        record["unitsInactive"] = None
        record["numberOfUnits"] = None
        record["unitsString"] = None
        record["longitude"] = spec.location.longitude
        record["latitude"] = spec.location.latitude
        record["geoLocation"] = [spec.location.longitude, spec.location.latitude]
        record["street"] = spec.location.address or spec.location.name
        record["municipality"] = spec.location.municipality or spec.location.name
        record["county"] = spec.location.county or record.get("county")
        record["state"] = spec.location.state or record.get("state")
        record["type"] = "urban" if spec.location.is_urban else "rural"
        record["growth_zone_type"] = spec.location.metadata.get("growth_zone_type")
        record["growth_zone_name"] = spec.location.metadata.get("growth_zone_name")
        record["growth_zone_municipality"] = spec.location.metadata.get(
            "growth_zone_municipality"
        )
        record["growth_zone_source"] = spec.location.metadata.get(
            "growth_zone_source", record.get("growth_zone_source")
        )
        cross_street = spec.location.metadata.get("cross_street")
        record["crossStreet"] = cross_street or record.get("crossStreet")
        record["municipalID"] = spec.location.metadata.get(
            "municipal_id", record.get("municipalID")
        )
        if "incident_guid" in spec.location.metadata:
            record["incidentID"] = spec.location.metadata["incident_guid"]

        record.setdefault("incidentTime_timezone", self.timezone_name)
        record.setdefault("county", "Lancaster County")
        record.setdefault("state", "PA")

        return record


__all__ = ["IncidentSpec", "RecordFactory"]

