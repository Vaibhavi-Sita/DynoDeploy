#!/usr/bin/env python
"""
Populate base→base, base→hospital, and base→incident travel matrices using the
OpenRouteService matrix API. Distances are stored in miles and travel times in
minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import requests
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor, execute_values

from util.simulator.database.db_connection import get_connection

MILES_PER_METER = 0.000621371
MINUTES_PER_SECOND = 1 / 60
MAX_ORIGIN_BATCH = 50  # ORS matrix limit: up to 50 origins per request
MAX_DESTINATION_BATCH = 70  # ORS matrix limit: up to 70 destinations per request
MAX_MATRIX_PRODUCT = MAX_ORIGIN_BATCH * MAX_DESTINATION_BATCH


def _load_env_file() -> None:
    """
    Load the first `.env` file found while walking up the directory tree,
    allowing OPENROUTESERVICE_API_KEY to be sourced automatically.
    """
    script_path = Path(__file__).resolve()
    for directory in (script_path,) + tuple(script_path.parents):
        env_path = directory / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return
    # Fallback to default behavior (use python-dotenv's search heuristic).
    load_dotenv(override=False)


_load_env_file()

DEFAULT_PROFILE = os.getenv("ORS_PROFILE", "driving-hgv")
DEFAULT_BASE_URL = os.getenv("ORS_BASE_URL", "https://api.openrouteservice.org")
DEFAULT_ORIGIN_BATCH_SIZE = int(
    os.getenv(
        "ORS_ORIGIN_BATCH",
        os.getenv("ORS_MATRIX_BATCH", MAX_ORIGIN_BATCH),
    )
)
DEFAULT_DESTINATION_BATCH_SIZE = int(
    os.getenv(
        "ORS_DESTINATION_BATCH",
        os.getenv("ORS_MATRIX_BATCH", MAX_DESTINATION_BATCH),
    )
)
DEFAULT_MAX_RETRIES = int(os.getenv("ORS_MAX_RETRIES", 3))
DEFAULT_BACKOFF = float(os.getenv("ORS_RETRY_BACKOFF", 2.0))


@dataclass(frozen=True)
class Location:
    """Lightweight value object for origin/destination lookups."""

    id: str
    lat: float
    lon: float

    def to_lonlat(self) -> List[float]:
        return [self.lon, self.lat]


@dataclass
class MatrixResult:
    origin_id: str
    destination_id: str
    distance_miles: float
    travel_time_minutes: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch coordinates from base_stations, incident_locations, and hospitals "
            "tables, call the OpenRouteService matrix API, and persist the resulting "
            "distance (miles) and travel time (minutes) values."
        )
    )
    parser.add_argument(
        "--api-key",
        help="ORS API key (fallback env vars: ORS_API_KEY or OPENROUTESERVICE_API_KEY).",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="ORS routing profile (driving-car, driving-hgv, etc.).",
    )
    parser.add_argument(
        "--origin-batch-size",
        type=int,
        default=DEFAULT_ORIGIN_BATCH_SIZE,
        help=(
            "Maximum origins per ORS matrix call (up to 50 per "
            "https://openrouteservice.org/restrictions/)."
        ),
    )
    parser.add_argument(
        "--destination-batch-size",
        type=int,
        default=DEFAULT_DESTINATION_BATCH_SIZE,
        help=(
            "Maximum destinations per ORS matrix call (up to 70 per "
            "https://openrouteservice.org/restrictions/)."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for OpenRouteService (default: https://api.openrouteservice.org).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum attempts per ORS request.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_BACKOFF,
        help="Seconds to wait before retrying, multiplied exponentially.",
    )
    parser.add_argument(
        "--base-table",
        default="base_stations",
        help="Source table for base station coordinates.",
    )
    parser.add_argument(
        "--hospital-table",
        default="hospitals",
        help="Source table for hospital coordinates.",
    )
    parser.add_argument(
        "--incident-table",
        default="incident_locations",
        help="Source table for incident location coordinates.",
    )
    parser.add_argument(
        "--base-base-table",
        default="base_base_travel_matrix",
        help="Destination table for base→base travel entries.",
    )
    parser.add_argument(
        "--base-hospital-table",
        default="base_hospital_travel_matrix",
        help="Destination table for base→hospital travel entries.",
    )
    parser.add_argument(
        "--base-incident-table",
        default="base_incident_travel_matrix",
        help="Destination table for base→incident travel entries.",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Recompute and overwrite existing pairs instead of skipping them.",
    )
    parser.add_argument(
        "--skip-base-base",
        action="store_true",
        help="Skip computing base→base pairs.",
    )
    parser.add_argument(
        "--skip-base-hospital",
        action="store_true",
        help="Skip computing base→hospital pairs.",
    )
    parser.add_argument(
        "--skip-base-incident",
        action="store_true",
        help="Skip computing base→incident pairs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional progress details.",
    )
    return parser.parse_args()


def chunked(sequence: Sequence[Location], size: int) -> Iterable[List[Location]]:
    for idx in range(0, len(sequence), size):
        yield list(sequence[idx : idx + size])


def canonical_pair(id_a: str, id_b: str) -> Tuple[str, str]:
    """Return ids ordered so (left <= right)."""
    a = str(id_a)
    b = str(id_b)
    return (a, b) if a <= b else (b, a)


def resolve_api_key(cli_value: str | None) -> str | None:
    """Return the API key from CLI input or the supported environment variables."""
    if cli_value:
        return cli_value
    for var_name in ("ORS_API_KEY", "OPENROUTESERVICE_API_KEY"):
        value = os.getenv(var_name)
        if value:
            return value
    return None


def parse_coordinates(raw_value: object) -> Tuple[float, float]:
    """
    Convert the `coordinates` column to (lat, lon).

    Supports JSON objects, JSON strings, ["lat", "lon"] lists, or WKT POINT strings.
    """
    if raw_value is None:
        raise ValueError("Missing coordinates.")
    if isinstance(raw_value, dict):
        lat = raw_value.get("lat") or raw_value.get("latitude")
        lon = raw_value.get("lon") or raw_value.get("lng") or raw_value.get("longitude")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
        return float(raw_value[0]), float(raw_value[1])
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if raw_value.upper().startswith("POINT"):
            contents = raw_value[5:].strip(" ()")
            lon_str, lat_str = contents.split()
            return float(lat_str), float(lon_str)
        try:
            parsed = json.loads(raw_value)
            return parse_coordinates(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unable to parse coordinate JSON: {raw_value}") from exc
    raise ValueError(f"Unsupported coordinate format: {raw_value!r}")


ID_COLUMN_CANDIDATES = (
    "id",
    "station_number",
    "base_id",
    "incident_id",
    "location_id",
    "node_id",
)

COORDINATE_JSON_COLUMNS = ("coordinates", "location_coordinates")

LATITUDE_COLUMN_CANDIDATES = (
    "latitude",
    "lat",
    "origin_latitude",
    "location_latitude",
)

LONGITUDE_COLUMN_CANDIDATES = (
    "longitude",
    "lon",
    "lng",
    "origin_longitude",
    "location_longitude",
)


def _split_table_reference(table_name: str) -> Tuple[str | None, str]:
    if "." in table_name:
        schema, table = table_name.split(".", 1)
        return schema.strip('"'), table.strip('"')
    return None, table_name.strip('"')


def _list_table_columns(conn, table_name: str) -> List[str]:
    schema, table = _split_table_reference(table_name)
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
    """
    params: List[str] = [table]
    if schema:
        query += " AND table_schema = %s"
        params.append(schema)
    with conn.cursor() as cur:
        cur.execute(query, params)
        return [row[0] for row in cur.fetchall()]


def resolve_id_column(
    conn, table_name: str, *, available_columns: Iterable[str] | None = None
) -> str:
    columns = set(available_columns or _list_table_columns(conn, table_name))
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Unable to find identifier column for {table_name}. "
        f"Available columns: {sorted(columns)}"
    )


def resolve_coordinate_columns(
    conn, table_name: str, *, available_columns: Iterable[str] | None = None
) -> Tuple[str | None, str | None, str | None]:
    columns = set(available_columns or _list_table_columns(conn, table_name))
    for candidate in COORDINATE_JSON_COLUMNS:
        if candidate in columns:
            return candidate, None, None
    lat_col = next((col for col in LATITUDE_COLUMN_CANDIDATES if col in columns), None)
    lon_col = next((col for col in LONGITUDE_COLUMN_CANDIDATES if col in columns), None)
    if lat_col and lon_col:
        return None, lat_col, lon_col
    raise ValueError(
        f"Unable to find coordinates column(s) for {table_name}. "
        "Expected a JSON 'coordinates' column or latitude/longitude columns."
    )


def _quote_ident(identifier: str) -> str:
    return f'"{identifier.replace("\"", "\"\"")}"'


def fetch_locations(
    conn, table_name: str, *, id_column: str | None = None
) -> List[Location]:
    columns = set(_list_table_columns(conn, table_name))
    effective_id_column = id_column or resolve_id_column(
        conn, table_name, available_columns=columns
    )
    coord_json, lat_col, lon_col = resolve_coordinate_columns(
        conn, table_name, available_columns=columns
    )
    id_expr = f"{_quote_ident(effective_id_column)}::text AS id"

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if coord_json:
            query = f"""
                SELECT {id_expr}, {_quote_ident(coord_json)} AS coordinates
                FROM {table_name}
                WHERE {_quote_ident(coord_json)} IS NOT NULL;
            """
            cur.execute(query)
            rows = cur.fetchall()
            locations: List[Location] = []
            for row in rows:
                lat, lon = parse_coordinates(row["coordinates"])
                locations.append(Location(id=row["id"], lat=lat, lon=lon))
            return locations

        query = f"""
            SELECT
                {id_expr},
                {_quote_ident(lat_col)} AS latitude,
                {_quote_ident(lon_col)} AS longitude
            FROM {table_name}
            WHERE {_quote_ident(lat_col)} IS NOT NULL
              AND {_quote_ident(lon_col)} IS NOT NULL;
        """
        cur.execute(query)
        rows = cur.fetchall()
        locations = [
            Location(
                id=row["id"], lat=float(row["latitude"]), lon=float(row["longitude"])
            )
            for row in rows
        ]
        return locations


def fetch_existing_pairs(
    conn,
    table_name: str,
    origin_column: str,
    destination_column: str,
    symmetric_pairs: bool,
) -> set[Tuple[str, str]]:
    query = f"""
        SELECT {origin_column}::text AS origin_id, {destination_column}::text AS destination_id
        FROM {table_name};
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        try:
            cur.execute(query)
        except Exception:
            # Table might not exist yet; treat as empty.
            conn.rollback()
            return set()
        rows = cur.fetchall()
    pairs: set[Tuple[str, str]] = set()
    for row in rows:
        left, right = row["origin_id"], row["destination_id"]
        if symmetric_pairs:
            left, right = canonical_pair(left, right)
            if left == right:
                continue
        pairs.add((left, right))
    return pairs


class ORSMatrixClient:
    def __init__(
        self,
        api_key: str,
        profile: str,
        base_url: str,
        max_retries: int,
        retry_backoff: float,
    ) -> None:
        self.api_key = api_key
        self.profile = profile
        self.url = f"{base_url.rstrip('/')}/v2/matrix/{profile}"
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.session = requests.Session()

    def compute(
        self, origins: Sequence[Location], destinations: Sequence[Location]
    ) -> List[MatrixResult]:
        if not origins or not destinations:
            return []

        coords = [loc.to_lonlat() for loc in origins + destinations]
        sources = list(range(len(origins)))
        dest_offset = len(origins)
        dest_indices = list(range(dest_offset, dest_offset + len(destinations)))

        payload = {
            "locations": coords,
            "sources": sources,
            "destinations": dest_indices,
            "metrics": ["distance", "duration"],
        }
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

        delay = self.retry_backoff
        for attempt in range(1, self.max_retries + 1):
            response = self.session.post(
                self.url, headers=headers, json=payload, timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                return self._parse_matrix(data, origins, destinations)
            if response.status_code in (429, 500, 503):
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()

        raise RuntimeError(
            f"Unable to fetch ORS matrix after {self.max_retries} attempts (status {response.status_code})."
        )

    @staticmethod
    def _parse_matrix(
        data: Dict[str, List[List[float]]],
        origins: Sequence[Location],
        destinations: Sequence[Location],
    ) -> List[MatrixResult]:
        distances = data.get("distances") or []
        durations = data.get("durations") or []
        results: List[MatrixResult] = []
        for row_idx, origin in enumerate(origins):
            if row_idx >= len(distances) or row_idx >= len(durations):
                continue
            for col_idx, destination in enumerate(destinations):
                if col_idx >= len(distances[row_idx]) or col_idx >= len(
                    durations[row_idx]
                ):
                    continue
                distance_meters = distances[row_idx][col_idx]
                duration_seconds = durations[row_idx][col_idx]
                if distance_meters is None or duration_seconds is None:
                    continue
                results.append(
                    MatrixResult(
                        origin_id=origin.id,
                        destination_id=destination.id,
                        distance_miles=distance_meters * MILES_PER_METER,
                        travel_time_minutes=duration_seconds * MINUTES_PER_SECOND,
                    )
                )
        return results


def insert_results(
    conn,
    table_name: str,
    origin_column: str,
    destination_column: str,
    rows: List[MatrixResult],
    profile: str,
) -> int:
    if not rows:
        return 0

    fetched_at = datetime.now(timezone.utc)
    provider = "openrouteservice"
    records = [
        (
            row.origin_id,
            row.destination_id,
            row.distance_miles,
            row.travel_time_minutes,
            provider,
            profile,
            fetched_at,
        )
        for row in rows
    ]

    columns = (
        f"{origin_column}, {destination_column}, distance_miles, travel_time_minutes, "
        "provider, profile, fetched_at"
    )
    conflict_cols = f"{origin_column}, {destination_column}"
    sql = f"""
        INSERT INTO {table_name} ({columns})
        VALUES %s
        ON CONFLICT ({conflict_cols}) DO UPDATE SET
            distance_miles = EXCLUDED.distance_miles,
            travel_time_minutes = EXCLUDED.travel_time_minutes,
            provider = EXCLUDED.provider,
            profile = EXCLUDED.profile,
            fetched_at = EXCLUDED.fetched_at;
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, records)
    return len(rows)


def compute_pairs(
    args: argparse.Namespace,
    conn,
    client: ORSMatrixClient,
    origins: List[Location],
    destinations: List[Location],
    table_name: str,
    origin_column: str,
    destination_column: str,
    symmetric_pairs: bool,
) -> int:
    if symmetric_pairs:
        return _compute_symmetric_pairs(
            args,
            conn,
            client,
            origins,
            table_name,
            origin_column,
            destination_column,
        )
    return _compute_asymmetric_pairs(
        args,
        conn,
        client,
        origins,
        destinations,
        table_name,
        origin_column,
        destination_column,
    )


def _compute_symmetric_pairs(
    args: argparse.Namespace,
    conn,
    client: ORSMatrixClient,
    locations: List[Location],
    table_name: str,
    origin_column: str,
    destination_column: str,
) -> int:
    existing_pairs = set()
    if not args.refresh_existing:
        existing_pairs = fetch_existing_pairs(
            conn, table_name, origin_column, destination_column, True
        )

    inserted_total = 0
    for idx, origin in enumerate(locations):
        remaining = locations[idx + 1 :]
        if not remaining:
            continue
        for dest_chunk in chunked(remaining, args.destination_batch_size):
            pending_destinations: List[Location] = []
            pending_pairs: List[Tuple[str, str]] = []
            for destination in dest_chunk:
                pair = canonical_pair(origin.id, destination.id)
                if not args.refresh_existing and pair in existing_pairs:
                    continue
                pending_destinations.append(destination)
                pending_pairs.append(pair)
            if not pending_destinations:
                continue

            results = client.compute([origin], pending_destinations)
            filtered_results: List[MatrixResult] = []
            for row in results:
                left, right = canonical_pair(row.origin_id, row.destination_id)
                filtered_results.append(
                    MatrixResult(
                        origin_id=left,
                        destination_id=right,
                        distance_miles=row.distance_miles,
                        travel_time_minutes=row.travel_time_minutes,
                    )
                )
            if not args.refresh_existing:
                filtered_results = [
                    row
                    for row in filtered_results
                    if (row.origin_id, row.destination_id) not in existing_pairs
                ]
            inserted = insert_results(
                conn,
                table_name,
                origin_column,
                destination_column,
                filtered_results,
                args.profile,
            )
            if inserted:
                inserted_total += inserted
                if not args.refresh_existing:
                    existing_pairs.update(
                        (row.origin_id, row.destination_id) for row in filtered_results
                    )
            if args.verbose:
                print(
                    f"Processed {len(filtered_results)} symmetric pairs for "
                    f"{table_name} (+{inserted} rows)."
                )
    return inserted_total


def _compute_asymmetric_pairs(
    args: argparse.Namespace,
    conn,
    client: ORSMatrixClient,
    origins: List[Location],
    destinations: List[Location],
    table_name: str,
    origin_column: str,
    destination_column: str,
) -> int:
    existing_pairs = set()
    if not args.refresh_existing:
        existing_pairs = fetch_existing_pairs(
            conn, table_name, origin_column, destination_column, False
        )
    inserted_total = 0
    for origin_chunk in chunked(origins, args.origin_batch_size):
        for destination_chunk in chunked(destinations, args.destination_batch_size):
            pending_pairs = [
                (origin.id, destination.id)
                for origin in origin_chunk
                for destination in destination_chunk
            ]
            if not args.refresh_existing:
                pending_pairs = [
                    pair for pair in pending_pairs if pair not in existing_pairs
                ]
            if not pending_pairs:
                continue

            results = client.compute(origin_chunk, destination_chunk)
            filtered_results = [
                row
                for row in results
                if (row.origin_id, row.destination_id) in pending_pairs
            ]
            if not args.refresh_existing:
                filtered_results = [
                    row
                    for row in filtered_results
                    if (row.origin_id, row.destination_id) not in existing_pairs
                ]
            inserted = insert_results(
                conn,
                table_name,
                origin_column,
                destination_column,
                filtered_results,
                args.profile,
            )
            if inserted:
                inserted_total += inserted
                if not args.refresh_existing:
                    existing_pairs.update(
                        (row.origin_id, row.destination_id) for row in filtered_results
                    )
            if args.verbose:
                print(
                    f"Processed {len(origin_chunk)}x{len(destination_chunk)} "
                    f"pairs for {table_name} (+{inserted} rows)."
                )
    return inserted_total


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise SystemExit(
            "Missing OpenRouteService API key. Provide --api-key or set "
            "ORS_API_KEY / OPENROUTESERVICE_API_KEY."
        )
    if not 1 <= args.origin_batch_size <= MAX_ORIGIN_BATCH:
        raise SystemExit(
            f"Origin batch size must be between 1 and {MAX_ORIGIN_BATCH} "
            "(OpenRouteService origin limit)."
        )
    if not 1 <= args.destination_batch_size <= MAX_DESTINATION_BATCH:
        raise SystemExit(
            f"Destination batch size must be between 1 and {MAX_DESTINATION_BATCH} "
            "(OpenRouteService destination limit)."
        )
    if args.origin_batch_size * args.destination_batch_size > MAX_MATRIX_PRODUCT:
        raise SystemExit(
            "Origin and destination batch sizes would exceed the ORS 3,500-cell limit "
            f"({MAX_MATRIX_PRODUCT}). Reduce one of the batch sizes."
        )

    client = ORSMatrixClient(
        api_key=api_key,
        profile=args.profile,
        base_url=args.base_url,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )

    with get_connection() as conn:
        bases = fetch_locations(conn, args.base_table)
        hospitals: List[Location] = []
        if not args.skip_base_hospital:
            hospitals = fetch_locations(conn, args.hospital_table)
        incidents: List[Location] = []
        if not args.skip_base_incident:
            incidents = fetch_locations(conn, args.incident_table)
        if not bases:
            raise SystemExit(f"No base stations found in {args.base_table}.")
        if not hospitals and not args.skip_base_hospital:
            print(f"Warning: no hospital records found in {args.hospital_table}.")
        if not incidents and not args.skip_base_incident:
            print(f"Warning: no incident records found in {args.incident_table}.")

        total_rows = 0
        if not args.skip_base_base:
            total_rows += compute_pairs(
                args,
                conn,
                client,
                bases,
                bases,
                args.base_base_table,
                "origin_base_id",
                "destination_base_id",
                symmetric_pairs=True,
            )
        if not args.skip_base_hospital and hospitals:
            total_rows += compute_pairs(
                args,
                conn,
                client,
                bases,
                hospitals,
                args.base_hospital_table,
                "origin_base_id",
                "destination_hospital_id",
                symmetric_pairs=False,
            )
        if not args.skip_base_incident and incidents:
            total_rows += compute_pairs(
                args,
                conn,
                client,
                bases,
                incidents,
                args.base_incident_table,
                "origin_base_id",
                "destination_incident_id",
                symmetric_pairs=False,
            )

        conn.commit()
        print(f"Stored/updated {total_rows} travel matrix rows.")


if __name__ == "__main__":
    main()
