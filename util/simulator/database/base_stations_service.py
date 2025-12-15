"""
Helpers for interacting with the `base_stations` table.

Usage:
    from util.simulator.database.base_stations_service import fetch_base_stations
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from psycopg2.extras import RealDictCursor

from .db_connection import get_connection


def fetch_base_stations(limit: int = 25) -> List[Dict[str, Any]]:
    """Return base station rows as dictionaries ordered by station_number."""
    query = """
        SELECT
            id,
            station_number,
            name,
            agency,
            address,
            phone,
            phones,
            coordinates,
            capabilities,
            units,
            number_of_units
        FROM base_stations
        ORDER BY station_number
        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (limit,))
            return cur.fetchall()


def upsert_base_station(record: Dict[str, Any]) -> None:
    """
    Insert or update a base station row.

    The `record` keys must line up with the base_stations table columns.
    """
    query = """
        INSERT INTO base_stations (
            id,
            station_number,
            name,
            agency,
            address,
            phone,
            phones,
            coordinates,
            capabilities,
            units,
            number_of_units
        ) VALUES (
            %(id)s,
            %(station_number)s,
            %(name)s,
            %(agency)s,
            %(address)s,
            %(phone)s,
            %(phones)s,
            %(coordinates)s,
            %(capabilities)s,
            %(units)s,
            %(number_of_units)s
        )
        ON CONFLICT (id) DO UPDATE SET
            station_number = EXCLUDED.station_number,
            name = EXCLUDED.name,
            agency = EXCLUDED.agency,
            address = EXCLUDED.address,
            phone = EXCLUDED.phone,
            phones = EXCLUDED.phones,
            coordinates = EXCLUDED.coordinates,
            capabilities = EXCLUDED.capabilities,
            units = EXCLUDED.units,
            number_of_units = EXCLUDED.number_of_units;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, record)


if __name__ == "__main__":
    stations = fetch_base_stations()
    print(json.dumps(stations, indent=2, default=str))

