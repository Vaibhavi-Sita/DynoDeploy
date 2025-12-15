"""Configuration helpers for the incident record generator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from util.helper.paths import simulator_root

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


PACKAGE_ROOT = simulator_root()
DEFAULT_CONFIG_PATHS = [PACKAGE_ROOT / "config.yaml", PACKAGE_ROOT / "config.json"]


@dataclass(frozen=True)
class OverlapConfig:
    probability: float = 0.05
    same_location_probability: float = 0.4
    time_offset_minutes_min: int = 1
    time_offset_minutes_max: int = 10
    max_simultaneous: int = 3


@dataclass(frozen=True)
class SimulationConfig:
    start_datetime: datetime
    horizon_hours: int = 24
    incident_count: int = 200
    location_sample_size: int = 200
    incident_type_shares: Dict[str, float] = field(
        default_factory=lambda: {"3": 0.54, "2": 0.41, "1": 0.05}
    )
    urban_rural_split: Dict[str, float] = field(
        default_factory=lambda: {"urban": 0.8, "rural": 0.2}
    )
    peak_hour_multipliers: Dict[int, float] = field(default_factory=dict)
    peak_weekday_multipliers: Dict[int, float] = field(default_factory=dict)
    peak_month_multipliers: Dict[int, float] = field(default_factory=dict)
    hotspot_weight_multiplier: float = 1.5
    overlap: OverlapConfig = field(default_factory=OverlapConfig)
    allow_duplicate_timestamps: bool = True
    max_incidents_per_location: int | None = None
    enforce_peak_hours: bool = False
    peak_enforcement_margin: int = 5
    high_demand_municipalities: Tuple[str, ...] = (
        "lancaster",
        "manheim township",
        "east hempfield township",
        "lancaster township",
        "east lampeter township",
        "manor township",
    )
    urban_hotspot_weight: float = 2.0
    rural_demand_weight: float = 0.6
    peak_hour_start: int = 11
    peak_hour_end: int = 14
    primary_location_share: float = 0.65
    primary_location_count: int = 3

    def validate(self) -> "SimulationConfig":
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive.")
        if self.incident_count <= 0:
            raise ValueError("incident_count must be positive.")
        if self.location_sample_size <= 0:
            raise ValueError("location_sample_size must be positive.")
        _validate_distribution(self.incident_type_shares, "incident_type_shares")
        _validate_distribution(self.urban_rural_split, "urban_rural_split")
        if (
            self.max_incidents_per_location is not None
            and self.max_incidents_per_location <= 0
        ):
            raise ValueError(
                "max_incidents_per_location must be positive when provided."
            )
        if self.peak_enforcement_margin < 0:
            raise ValueError("peak_enforcement_margin cannot be negative.")
        return self

    @property
    def horizon(self) -> timedelta:
        return timedelta(hours=self.horizon_hours)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "start_datetime": self.start_datetime.isoformat(),
            "horizon_hours": self.horizon_hours,
            "incident_count": self.incident_count,
            "location_sample_size": self.location_sample_size,
            "incident_type_shares": self.incident_type_shares,
            "urban_rural_split": self.urban_rural_split,
            "peak_hour_multipliers": self.peak_hour_multipliers,
            "peak_weekday_multipliers": self.peak_weekday_multipliers,
            "peak_month_multipliers": self.peak_month_multipliers,
            "hotspot_weight_multiplier": self.hotspot_weight_multiplier,
            "overlap": self.overlap.__dict__,
            "allow_duplicate_timestamps": self.allow_duplicate_timestamps,
            "max_incidents_per_location": self.max_incidents_per_location,
            "enforce_peak_hours": self.enforce_peak_hours,
            "peak_enforcement_margin": self.peak_enforcement_margin,
            "high_demand_municipalities": list(self.high_demand_municipalities),
            "urban_hotspot_weight": self.urban_hotspot_weight,
            "rural_demand_weight": self.rural_demand_weight,
            "peak_hour_start": self.peak_hour_start,
            "peak_hour_end": self.peak_hour_end,
            "primary_location_share": self.primary_location_share,
            "primary_location_count": self.primary_location_count,
        }
        return payload


def load_simulation_config(
    config_path: Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> SimulationConfig:
    """Load config from a file/env/overrides, falling back to defaults."""
    data = {}
    path = config_path or _find_default_config()
    if path and path.exists():
        data.update(_load_file(path))
    if overrides:
        data.update(overrides)

    start_dt = _parse_datetime(
        data.get("start_datetime") or os.getenv("SIM_START_DATETIME")
    )
    cfg = SimulationConfig(
        start_datetime=start_dt,
        horizon_hours=int(data.get("horizon_hours", 24)),
        incident_count=int(data.get("incident_count", 200)),
        location_sample_size=int(data.get("location_sample_size", 200)),
        incident_type_shares=_normalize_distribution(
            data.get("incident_type_shares")
            or _env_as_json("SIM_INCIDENT_TYPE_SHARES")
            or {"3": 0.54, "2": 0.41, "1": 0.05}
        ),
        urban_rural_split=_normalize_distribution(
            data.get("urban_rural_split")
            or _env_as_json("SIM_URBAN_RURAL_SPLIT")
            or {"urban": 0.8, "rural": 0.2}
        ),
        peak_hour_multipliers=_coerce_int_float_mapping(
            data.get("peak_hour_multipliers", {})
        ),
        peak_weekday_multipliers=_coerce_int_float_mapping(
            data.get("peak_weekday_multipliers", {})
        ),
        peak_month_multipliers=_coerce_int_float_mapping(
            data.get("peak_month_multipliers", {})
        ),
        hotspot_weight_multiplier=float(data.get("hotspot_weight_multiplier", 1.5)),
        overlap=_build_overlap_config(data.get("overlap", {})),
        allow_duplicate_timestamps=bool(data.get("allow_duplicate_timestamps", True)),
        max_incidents_per_location=_maybe_int(data.get("max_incidents_per_location")),
        enforce_peak_hours=bool(data.get("enforce_peak_hours", False)),
        peak_enforcement_margin=int(data.get("peak_enforcement_margin", 5)),
        high_demand_municipalities=tuple(
            value.lower().strip()
            for value in data.get("high_demand_municipalities", [])
        )
        or (
            "lancaster",
            "manheim township",
            "east hempfield township",
            "lancaster township",
        ),
        urban_hotspot_weight=float(data.get("urban_hotspot_weight", 2.0)),
        rural_demand_weight=float(data.get("rural_demand_weight", 0.6)),
        peak_hour_start=int(data.get("peak_hour_start", 10)),
        peak_hour_end=int(data.get("peak_hour_end", 19)),
        primary_location_share=float(data.get("primary_location_share", 0.65)),
        primary_location_count=int(data.get("primary_location_count", 3)),
    )
    return cfg.validate()


def _find_default_config() -> Path | None:
    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return path
    return None


def _load_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yml", ".yaml"}:
        if not yaml:
            raise RuntimeError("PyYAML is required to load YAML config files.")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=_default_timezone())
        except ValueError as exc:
            raise ValueError(f"Unable to parse start_datetime: {value}") from exc
    # Default: start of current day in ET.
    now = datetime.now(tz=_default_timezone())
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _default_timezone():
    if ZoneInfo:
        try:
            return ZoneInfo("America/New_York")
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def _normalize_distribution(mapping: Dict[str, Any]) -> Dict[str, float]:
    if not mapping:
        raise ValueError("Distribution mapping cannot be empty.")
    total = sum(float(value) for value in mapping.values())
    if total <= 0:
        raise ValueError("Distribution weights must sum to a positive number.")
    return {str(key): float(value) / total for key, value in mapping.items()}


def _coerce_int_float_mapping(mapping: Dict[Any, Any]) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for key, value in mapping.items():
        try:
            idx = int(key)
            result[idx] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _build_overlap_config(data: Dict[str, Any]) -> OverlapConfig:
    base = OverlapConfig()
    if not data:
        return base
    return replace(
        base,
        probability=float(data.get("probability", base.probability)),
        same_location_probability=float(
            data.get("same_location_probability", base.same_location_probability)
        ),
        time_offset_minutes_min=int(
            data.get("time_offset_minutes_min", base.time_offset_minutes_min)
        ),
        time_offset_minutes_max=int(
            data.get("time_offset_minutes_max", base.time_offset_minutes_max)
        ),
        max_simultaneous=int(data.get("max_simultaneous", base.max_simultaneous)),
    )


def _env_as_json(key: str) -> Dict[str, Any] | None:
    raw = os.getenv(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - safety
        raise ValueError(
            f"Environment variable {key} must contain valid JSON."
        ) from exc


def _validate_distribution(mapping: Dict[str, float], name: str) -> None:
    if not mapping:
        raise ValueError(f"{name} cannot be empty.")
    total = round(sum(mapping.values()), 6)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1.0; got {total}.")


__all__ = ["SimulationConfig", "OverlapConfig", "load_simulation_config"]
