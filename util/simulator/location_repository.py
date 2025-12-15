"""Loads incident/base/hospital locations from the database (with a local cache)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd
from psycopg2 import sql
from psycopg2.errors import UndefinedTable
from psycopg2.extras import RealDictCursor

from util.simulator.database.db_connection import get_connection
from util.helper.paths import simulator_root


LOGGER = logging.getLogger(__name__)
CACHE_PATH = simulator_root() / "cache" / "location_snapshot.json"
TRAVEL_CACHE_DIR = simulator_root() / "cache" / "travel_matrices"
POPULAR_TABLE_NAMES = {"popular_incident_locations"}


@dataclass
class LocationRecord:
    """A location row in a shape the simulator can use."""

    id: str
    name: str
    longitude: float
    latitude: float
    address: str | None
    municipality: str | None
    county: str | None
    state: str | None
    location_type: str
    is_urban: bool
    source_table: str
    metadata: Dict[str, Any]

    def to_geojson_point(self) -> Dict[str, Any]:
        return {"type": "Point", "coordinates": [self.longitude, self.latitude]}

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["geojson"] = self.to_geojson_point()
        return data


@dataclass(frozen=True)
class BaseStationSummary:
    """Minimal base-station representation for dispatch logic."""

    id: str
    name: str
    longitude: float
    latitude: float
    is_urban: bool
    station_number: str | None = None
    capabilities: Sequence[str] | None = None
    units: Sequence[Mapping[str, Any]] | None = None
    number_of_units: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HospitalSummary:
    """Hospital representation with capability metadata."""

    id: str
    name: str
    longitude: float
    latitude: float
    facility_code: str | None = None
    units: Sequence[Mapping[str, Any]] | None = None
    number_of_units: int | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IncidentSiteSummary:
    """Incident or hotspot geometry plus descriptors."""

    id: str
    name: str
    longitude: float
    latitude: float
    location_type: str
    is_urban: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocationDataset:
    """Grouped location assets keyed by their identifier."""

    bases: Dict[str, BaseStationSummary]
    hospitals: Dict[str, HospitalSummary]
    incident_locations: Dict[str, IncidentSiteSummary]
    popular_locations: Dict[str, IncidentSiteSummary]

    def require_base(self, base_id: str) -> BaseStationSummary:
        return self._require(self.bases, base_id, "base station")

    def require_hospital(self, hospital_id: str) -> HospitalSummary:
        return self._require(self.hospitals, hospital_id, "hospital")

    def require_incident(self, incident_id: str) -> IncidentSiteSummary:
        if incident_id in self.incident_locations:
            return self.incident_locations[incident_id]
        return self._require(self.popular_locations, incident_id, "popular location")

    @staticmethod
    def _require(container: Mapping[str, Any], identifier: str, label: str) -> Any:
        try:
            return container[identifier]
        except KeyError as exc:
            raise KeyError(f"Unknown {label} id '{identifier}'") from exc


@dataclass
class TravelMatrixBundle:
    """In-memory copy of the relevant travel time matrices."""

    base_base: pd.DataFrame
    base_hospital: pd.DataFrame
    base_incident: pd.DataFrame

    def __post_init__(self) -> None:
        self._base_base_lookup = _build_lookup(
            self.base_base,
            "origin_base_id",
            "destination_base_id",
            symmetric=True,
        )
        self._base_hospital_lookup = _build_lookup(
            self.base_hospital,
            "origin_base_id",
            "destination_hospital_id",
            mirror=True,
        )
        self._base_incident_lookup = _build_lookup(
            self.base_incident,
            "origin_base_id",
            "destination_incident_id",
            mirror=True,
        )

    def travel_time(
        self, origin_id: str, destination_id: str, matrix_type: str
    ) -> float | None:
        matrix_type = matrix_type.lower()
        if matrix_type == "base_base":
            origin, destination = sorted((str(origin_id), str(destination_id)))
            return self._base_base_lookup.get((origin, destination))
        if matrix_type == "base_hospital":
            return self._base_hospital_lookup.get((str(origin_id), str(destination_id)))
        if matrix_type == "base_incident":
            return self._base_incident_lookup.get((str(origin_id), str(destination_id)))
        raise ValueError(f"Unsupported matrix_type '{matrix_type}'.")


class LocationRepository:
    """Fetches and caches candidate incident locations from multiple tables."""

    def __init__(
        self,
        cache_path: Path = CACHE_PATH,
        sample_limits: Dict[str, int | None] | None = None,
    ) -> None:
        self.cache_path = cache_path
        self.sample_limits = sample_limits or {}
        self._dataset_cache: LocationDataset | None = None

    def load_all(self, force_refresh: bool = False) -> List[LocationRecord]:
        if not force_refresh:
            cached = self._try_load_cache()
            if cached is not None:
                return cached
        try:
            records = list(self._fetch_from_database())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning(
                "Unable to refresh locations from the database (%s); "
                "falling back to cached snapshot.",
                exc,
            )
            cached = self._try_load_cache()
            if cached is not None:
                return cached
            raise
        if not records:
            raise RuntimeError(
                "No locations were fetched; verify the source tables contain data."
            )
        self._write_cache(records)
        return records

    def load_dataset(self, force_refresh: bool = False) -> LocationDataset:
        if self._dataset_cache and not force_refresh:
            return self._dataset_cache
        records = self.load_all(force_refresh=force_refresh)
        bases: Dict[str, BaseStationSummary] = {}
        hospitals: Dict[str, HospitalSummary] = {}
        incidents: Dict[str, IncidentSiteSummary] = {}
        populars: Dict[str, IncidentSiteSummary] = {}

        for record in records:
            if record.source_table == "base_stations":
                bases[record.id] = _record_to_base_station(record)
            elif record.source_table == "hospitals":
                hospitals[record.id] = _record_to_hospital(record)
            elif record.source_table == "incident_locations":
                incidents[record.id] = _record_to_incident(record)
            elif record.source_table in POPULAR_TABLE_NAMES:
                populars[record.id] = _record_to_incident(record)

        dataset = LocationDataset(
            bases=bases,
            hospitals=hospitals,
            incident_locations=incidents,
            popular_locations=populars,
        )
        self._dataset_cache = dataset
        return dataset

    def _fetch_from_database(self) -> Iterable[LocationRecord]:
        table_specs = [
            ("base_stations", "base_station"),
            ("hospitals", "hospital"),
            ("incident_locations", "incident_location"),
            ("popular_incident_locations", "popular_location"),
        ]
        with get_connection() as conn:
            for table_name, source_kind in table_specs:
                limit = self.sample_limits.get(table_name)
                query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
                if limit:
                    query += sql.SQL(" LIMIT {}").format(sql.Literal(limit))
                try:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(query)
                        rows = cur.fetchall()
                except UndefinedTable:
                    LOGGER.warning("Table %s not found; skipping.", table_name)
                    if table_name == "popular_incident_locations":
                        continue
                    continue
                except Exception as exc:  # pragma: no cover
                    LOGGER.error("Failed to query %s: %s", table_name, exc)
                    continue
                for row in rows:
                    record = self._row_to_location(row, source_kind, table_name)
                    if record:
                        yield record

    def _row_to_location(
        self,
        row: Dict[str, Any],
        source_kind: str,
        table_name: str,
    ) -> LocationRecord | None:
        identifier = _string_coalesce(
            row,
            ("id", "incident_id", "location_id", "external_id"),
        )
        name = _string_coalesce(
            row,
            ("name", "location_name", "description", "municipality"),
        )
        lon, lat = _extract_lon_lat(row)
        if not identifier or not name or lon is None or lat is None:
            return None

        address = _string_coalesce(row, ("address", "street", "location_address"))
        municipality = _string_coalesce(row, ("municipality", "city", "township"))
        county = _string_coalesce(row, ("county",))
        state = _string_coalesce(row, ("state", "state_code"))

        classification = (
            row.get("type")
            or row.get("location_type")
            or row.get("growth_zone_type")
            or source_kind
        )
        location_type = str(classification or source_kind)
        is_urban = _infer_is_urban(row, default="urban" in location_type.lower())

        metadata = dict(row)
        metadata["source_table"] = table_name

        return LocationRecord(
            id=str(identifier),
            name=str(name),
            longitude=float(lon),
            latitude=float(lat),
            address=address,
            municipality=municipality,
            county=county,
            state=state,
            location_type=location_type,
            is_urban=is_urban,
            source_table=table_name,
            metadata=metadata,
        )

    def _try_load_cache(self) -> List[LocationRecord] | None:
        if not self.cache_path.exists():
            return None
        with self.cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        valid_keys = {field.name for field in LocationRecord.__dataclass_fields__.values()}  # type: ignore
        cleaned = []
        for entry in payload:
            filtered = {key: value for key, value in entry.items() if key in valid_keys}
            cleaned.append(LocationRecord(**filtered))
        return cleaned

    def _write_cache(self, records: Sequence[LocationRecord]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [record.to_dict() for record in records],
                handle,
                indent=2,
                default=str,
            )


def _string_coalesce(row: Dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _extract_lon_lat(row: Dict[str, Any]) -> tuple[float | None, float | None]:
    lon_keys = ("longitude", "lon", "lng")
    lat_keys = ("latitude", "lat")
    lon = _first_float(row, lon_keys)
    lat = _first_float(row, lat_keys)
    if lon is not None and lat is not None:
        return lon, lat

    coordinates = row.get("coordinates")
    lon, lat = _parse_coordinates(coordinates)
    if lon is not None and lat is not None:
        return lon, lat

    geo = row.get("geo_location") or row.get("geolocation") or row.get("geoLocation")
    lon, lat = _parse_coordinates(geo)
    return lon, lat


def _first_float(row: Dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        number = _coerce_float(value)
        if number is not None:
            return number
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _parse_coordinates(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return (None, None)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lon = _coerce_float(value[0])
        lat = _coerce_float(value[1])
        return lon, lat
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
                return _parse_coordinates(parsed)
            except json.JSONDecodeError:
                pass
        if text.upper().startswith("POINT"):
            inner = text[text.find("(") + 1 : text.find(")")]
            parts = inner.replace(",", " ").split()
            if len(parts) >= 2:
                lon = _coerce_float(parts[0])
                lat = _coerce_float(parts[1])
                return lon, lat
        if "," in text:
            lon_str, lat_str, *_ = text.split(",")
            return _coerce_float(lon_str), _coerce_float(lat_str)
    if isinstance(value, dict):
        lon = _coerce_float(value.get("lon") or value.get("lng") or value.get("x"))
        lat = _coerce_float(value.get("lat") or value.get("y"))
        if lon is not None and lat is not None:
            return lon, lat
    return (None, None)


def _infer_is_urban(row: Dict[str, Any], default: bool = True) -> bool:
    indicators = [
        str(row.get("type", "")),
        str(row.get("location_type", "")),
        str(row.get("growth_zone_type", "")),
        str(row.get("growth_zone_name", "")),
    ]
    for indicator in indicators:
        indicator_lower = indicator.lower()
        if "rural" in indicator_lower or "vga" in indicator_lower:
            return False
        if "urban" in indicator_lower or "ugb" in indicator_lower:
            return True
    return default


def _record_to_base_station(record: LocationRecord) -> BaseStationSummary:
    metadata = dict(record.metadata)
    number_of_units = (
        _coerce_int(
            metadata.get("number_of_units")
            or metadata.get("unit_count")
            or metadata.get("units_available")
        )
        or _infer_units_from_collection(metadata.get("units"))
        or 1
    )
    return BaseStationSummary(
        id=record.id,
        name=record.name,
        longitude=record.longitude,
        latitude=record.latitude,
        is_urban=record.is_urban,
        station_number=str(metadata.get("station_number") or metadata.get("name")),
        capabilities=_as_list(metadata.get("capabilities")),
        units=_as_list(metadata.get("units")),
        number_of_units=number_of_units,
        metadata=metadata,
    )


def _record_to_hospital(record: LocationRecord) -> HospitalSummary:
    metadata = dict(record.metadata)
    coords = metadata.get("coordinates") or {}
    longitude = record.longitude or coords.get("lon")
    latitude = record.latitude or coords.get("lat")
    return HospitalSummary(
        id=record.id,
        name=record.name,
        longitude=float(longitude),
        latitude=float(latitude),
        facility_code=metadata.get("facility_code"),
        units=_as_list(metadata.get("units")),
        number_of_units=_coerce_int(metadata.get("number_of_units")),
        metadata=metadata,
    )


def _record_to_incident(record: LocationRecord) -> IncidentSiteSummary:
    metadata = dict(record.metadata)
    location_type = metadata.get("location_type") or record.location_type
    return IncidentSiteSummary(
        id=record.id,
        name=record.name,
        longitude=record.longitude,
        latitude=record.latitude,
        location_type=str(location_type),
        is_urban=record.is_urban,
        metadata=metadata,
    )


def _as_list(value: Any) -> List[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _infer_units_from_collection(units_field: Any) -> int | None:
    if units_field is None:
        return None
    if isinstance(units_field, (list, tuple)):
        return len(units_field)
    try:
        parsed = json.loads(units_field) if isinstance(units_field, str) else None
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return len(parsed)
    return None


def _build_lookup(
    dataframe: pd.DataFrame,
    origin_column: str,
    destination_column: str,
    *,
    symmetric: bool = False,
    mirror: bool = False,
) -> Dict[tuple[str, str], float]:
    if dataframe.empty:
        return {}
    required_columns = {
        origin_column,
        destination_column,
        "travel_time_minutes",
    }
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise KeyError(
            f"Missing expected columns {missing} in travel matrix table "
            f"({origin_column}->{destination_column})"
        )
    lookup: Dict[tuple[str, str], float] = {}
    for _, row in dataframe.iterrows():
        origin = str(row[origin_column])
        destination = str(row[destination_column])
        travel_time = float(row["travel_time_minutes"])
        key = (origin, destination)
        if symmetric and origin > destination:
            key = (destination, origin)
        lookup[key] = travel_time
        if mirror:
            lookup[(destination, origin)] = travel_time
    return lookup


class TravelMatrixRepository:
    """Loads and caches travel matrices as pandas DataFrames."""

    DEFAULT_TABLES = {
        "base_base": "base_base_travel_matrix",
        "base_hospital": "base_hospital_travel_matrix",
        "base_incident": "base_incident_travel_matrix",
    }

    def __init__(self, table_names: Mapping[str, str] | None = None) -> None:
        self.table_names = dict(self.DEFAULT_TABLES)
        if table_names:
            self.table_names.update(table_names)
        self._cache: TravelMatrixBundle | None = None

    def load_all(self, force_refresh: bool = False) -> TravelMatrixBundle:
        if self._cache and not force_refresh:
            return self._cache
        base_base = self._fetch_dataframe(self.table_names["base_base"])
        base_hospital = self._fetch_dataframe(self.table_names["base_hospital"])
        base_incident = self._fetch_dataframe(self.table_names["base_incident"])
        self._cache = TravelMatrixBundle(
            base_base=base_base,
            base_hospital=base_hospital,
            base_incident=base_incident,
        )
        return self._cache

    def travel_time(
        self, origin_id: str, destination_id: str, matrix_type: str
    ) -> float | None:
        bundle = self.load_all()
        return bundle.travel_time(origin_id, destination_id, matrix_type)

    @staticmethod
    def _fetch_dataframe(table_name: str) -> pd.DataFrame:
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
        try:
            with get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
        except UndefinedTable:
            LOGGER.warning("Travel matrix table %s is missing.", table_name)
            return TravelMatrixRepository._load_cached_dataframe(table_name)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning(
                "Failed to pull travel matrix %s from the database (%s); "
                "using cached copy if available.",
                table_name,
                exc,
            )
            cached = TravelMatrixRepository._load_cached_dataframe(table_name)
            if cached is not None:
                return cached
            raise
        if not rows:
            cached = TravelMatrixRepository._load_cached_dataframe(table_name)
            return cached if cached is not None else pd.DataFrame()
        TravelMatrixRepository._write_cache(table_name, rows)
        return pd.DataFrame(rows)

    @staticmethod
    def _load_cached_dataframe(table_name: str) -> pd.DataFrame | None:
        path = TRAVEL_CACHE_DIR / f"{table_name}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return pd.DataFrame(payload)

    @staticmethod
    def _write_cache(table_name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        TRAVEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRAVEL_CACHE_DIR / f"{table_name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(list(rows), handle, indent=2, default=str)


__all__ = [
    "BaseStationSummary",
    "HospitalSummary",
    "IncidentSiteSummary",
    "LocationDataset",
    "LocationRecord",
    "LocationRepository",
    "TravelMatrixBundle",
    "TravelMatrixRepository",
]
