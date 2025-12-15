"""CLI script to build Poisson-driven simulated incident records."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.helper.paths import generator_records_root, generator_root
from util.simulator.config import load_simulation_config
from util.simulator.description_picker import DescriptionPicker
from util.simulator.exporter import write_json_log, write_ndjson_log
from util.simulator.incident_type_picker import IncidentAttributePicker
from util.simulator.insight_loader import get_insight_summary
from util.simulator.location_repository import LocationRepository
from util.simulator.location_selector import LocationSelector
from util.simulator.overlap_manager import apply_overlaps
from util.simulator.poisson_sampler import generate_timestamps
from util.simulator.record_factory import IncidentSpec, RecordFactory

PACKAGE_DIR = generator_root()
LOGGER = logging.getLogger("incident_record_generator")
DEFAULT_OUTPUT = generator_records_root() / "simulated_incidents.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated incident records.")
    parser.add_argument("--config", type=Path, help="Path to config JSON/YAML file.")
    parser.add_argument("--start", help="Start datetime (ISO 8601).")
    parser.add_argument("--hours", type=int, help="Horizon duration in hours.")
    parser.add_argument(
        "--days", type=int, help="Alternative to --hours; converted to hours x24."
    )
    parser.add_argument("--incidents", type=int, help="Target number of incidents.")
    parser.add_argument(
        "--location-sample-size",
        type=int,
        help="Number of unique locations to consider when sampling.",
    )
    parser.add_argument(
        "--incident-type-shares",
        help='JSON mapping of incident type weights, e.g. \'{"3":0.54,"2":0.41,"1":0.05}\'.',
    )
    parser.add_argument(
        "--urban-rural-split",
        help='JSON mapping for zone split, e.g. \'{"urban":0.8,"rural":0.2}\'.',
    )
    parser.add_argument(
        "--peak-hour",
        action="append",
        default=[],
        help="Hour multiplier in the form 'HOUR:VALUE' (can repeat).",
    )
    parser.add_argument(
        "--peak-weekday",
        action="append",
        default=[],
        help="Weekday multiplier in the form 'WEEKDAY:VALUE' (Mon=0).",
    )
    parser.add_argument(
        "--peak-month",
        action="append",
        default=[],
        help="Month multiplier in the form 'MONTH:VALUE' (Jan=1).",
    )
    parser.add_argument(
        "--overlap-probability",
        type=float,
        help="Fraction of incidents that should be overlapped.",
    )
    parser.add_argument(
        "--overlap-same-location",
        type=float,
        help="Probability that overlapping incidents share the exact location.",
    )
    parser.add_argument(
        "--overlap-window",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="Overlap jitter window in minutes (min max).",
    )
    parser.add_argument(
        "--max-incidents-per-location",
        type=int,
        help="Cap per-location usage before forcing a new site.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--ndjson",
        action="store_true",
        help="Also emit newline-delimited JSON alongside the structured file.",
    )
    parser.add_argument(
        "--refresh-insights",
        action="store_true",
        help="Force recomputation of distributions from the source dataset.",
    )
    parser.add_argument(
        "--refresh-locations",
        action="store_true",
        help="Bypass cached location snapshot and re-query the DB.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate records but only print the first few instead of writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    overrides = build_overrides(args)
    config = load_simulation_config(args.config, overrides)
    insights = get_insight_summary(force_recompute=args.refresh_insights)
    LOGGER.info(
        "Loaded insight distributions (records=%s).",
        insights.total_source_incidents,
    )

    repo = LocationRepository()
    locations = repo.load_all(force_refresh=args.refresh_locations)
    selector = LocationSelector(locations, config)
    picker = IncidentAttributePicker(config)

    timestamps = generate_timestamps(
        config, insights.distributions, config.incident_count
    )
    timestamps = _normalize_horizon_coverage(
        timestamps,
        start_dt=config.start_datetime,
        horizon_days=max(int(config.horizon_hours // 24), 1),
    )
    specs: List[IncidentSpec] = []
    for timestamp in timestamps:
        incident_type = picker.pick_incident_type()
        zone = picker.pick_zone()
        location = selector.select(zone)
        zone = "urban" if location.is_urban else "rural"
        specs.append(
            IncidentSpec(
                timestamp=timestamp,
                incident_type=incident_type,
                zone=zone,
                location=location,
            )
        )

    specs = apply_overlaps(specs, config)
    description_picker = DescriptionPicker.from_source()
    factory = RecordFactory(
        insights.schema_template,
        description_picker=description_picker,
    )
    records = factory.build_records(specs)

    if args.dry_run:
        LOGGER.info("Dry run generated %s incidents.", len(records))
        for preview in records[:5]:
            LOGGER.info(
                "%s | type=%s | %s",
                preview["incidentTime"],
                preview["incidentType"],
                preview["street"],
            )
        return

    run_validations(records, config)
    summary = write_json_log(records, config, args.output)
    LOGGER.info("Wrote %s incidents to %s", len(records), args.output)
    if args.ndjson:
        ndjson_path = args.output.with_suffix(".ndjson")
        write_ndjson_log(records, ndjson_path)
        LOGGER.info("Wrote NDJSON to %s", ndjson_path)
    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))


def _normalize_horizon_coverage(
    timestamps: List, start_dt, horizon_days: int
) -> List:
    """
    Make sure each day in the horizon has some incidents by cycling timestamps across days.

    This prevents long gaps (e.g., missing day ranges) that can arise when Poisson
    sampling under-allocates certain days. Time-of-day is preserved; only the day
    offset is adjusted to wrap within [0, horizon_days).
    """
    if not timestamps or horizon_days <= 0:
        return timestamps
    normalized = []
    base = start_dt
    for idx, ts in enumerate(sorted(timestamps)):
        day_offset = idx % horizon_days
        tod = ts.timetz()
        new_ts = (base + timedelta(days=day_offset)).replace(
            hour=tod.hour,
            minute=tod.minute,
            second=tod.second,
            microsecond=tod.microsecond,
            tzinfo=tod.tzinfo or base.tzinfo,
        )
        normalized.append(new_ts)
    return normalized


def build_overrides(args: argparse.Namespace) -> Dict:
    overrides: Dict[str, object] = {}
    if args.start:
        overrides["start_datetime"] = args.start
    if args.hours:
        overrides["horizon_hours"] = args.hours
    elif args.days:
        overrides["horizon_hours"] = args.days * 24
    if args.incidents:
        overrides["incident_count"] = args.incidents
    if args.location_sample_size:
        overrides["location_sample_size"] = args.location_sample_size
    if args.incident_type_shares:
        overrides["incident_type_shares"] = _parse_mapping_arg(args.incident_type_shares)
    if args.urban_rural_split:
        overrides["urban_rural_split"] = _parse_mapping_arg(args.urban_rural_split)

    peak_hours = _parse_multiplier_list(args.peak_hour)
    if peak_hours:
        overrides["peak_hour_multipliers"] = peak_hours
    peak_weekdays = _parse_multiplier_list(args.peak_weekday)
    if peak_weekdays:
        overrides["peak_weekday_multipliers"] = peak_weekdays
    peak_months = _parse_multiplier_list(args.peak_month)
    if peak_months:
        overrides["peak_month_multipliers"] = peak_months

    overlap: Dict[str, object] = {}
    if args.overlap_probability is not None:
        overlap["probability"] = args.overlap_probability
    if args.overlap_same_location is not None:
        overlap["same_location_probability"] = args.overlap_same_location
    if args.overlap_window:
        overlap["time_offset_minutes_min"] = args.overlap_window[0]
        overlap["time_offset_minutes_max"] = args.overlap_window[1]
    if args.max_incidents_per_location:
        overrides["max_incidents_per_location"] = args.max_incidents_per_location
    if overlap:
        overrides["overlap"] = overlap
    return overrides


def _parse_multiplier_list(pairs: Sequence[str]) -> Dict[int, float]:
    mapping: Dict[int, float] = {}
    for raw in pairs:
        try:
            key, value = raw.split(":")
            mapping[int(key.strip())] = float(value.strip())
        except ValueError:
            LOGGER.warning(
                "Unable to parse multiplier %s; expected format key:value.", raw
            )
    return mapping


def _parse_mapping_arg(raw: str) -> Dict[str, float]:
    """Accept JSON dictionaries or simple key:value,key:value strings."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return {str(key): float(value) for key, value in data.items()}

    cleaned = raw.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace("\\", "")
    mapping: Dict[str, float] = {}
    for chunk in cleaned.split(","):
        part = chunk.strip()
        if not part:
            continue
        if ":" in part:
            key, value = part.split(":", 1)
        elif "=" in part:
            key, value = part.split("=", 1)
        else:
            raise ValueError(
                f"Unable to parse mapping segment '{part}'. "
                "Use JSON (e.g. {\"urban\":0.8}) or key:value pairs."
            )
        key = key.strip().strip("'\"")
        mapping[key] = float(value.strip().strip("'\""))
    if not mapping:
        raise ValueError(f"Could not parse mapping from '{raw}'.")
    return mapping


def run_validations(records: Sequence[Dict[str, object]], config) -> None:
    if not records:
        raise ValueError("No records generated.")
    if len(records) != config.incident_count:
        LOGGER.warning(
            "Record count (%s) differs from requested incident_count (%s).",
            len(records),
            config.incident_count,
        )
    type_counts = Counter(str(record.get("incidentType")) for record in records)
    zone_counts = Counter(str(record.get("type")) for record in records)
    _compare_distribution("incident types", type_counts, config.incident_type_shares)
    _compare_distribution("zone split", zone_counts, config.urban_rural_split)


def _compare_distribution(
    name: str, actual: Counter, target: Dict[str, float], tolerance: float = 0.15
) -> None:
    total = sum(actual.values())
    if total == 0:
        return
    for key, expected in target.items():
        actual_ratio = actual.get(str(key), 0) / total
        delta = abs(actual_ratio - expected)
        if delta > tolerance:
            LOGGER.warning(
                "Distribution mismatch for %s (%s): expected %.2f, observed %.2f.",
                name,
                key,
                expected,
                actual_ratio,
            )


if __name__ == "__main__":
    main()
