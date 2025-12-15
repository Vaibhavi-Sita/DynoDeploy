"""
Helpers for interacting with the `hospitals` table.

Usage:
    from util.simulator.database.hospitals_service import fetch_hospitals
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from psycopg2.extras import RealDictCursor

from .db_connection import get_connection


def fetch_hospitals(limit: int = 25) -> List[Dict[str, Any]]:
    """Return hospital rows as dictionaries ordered by name."""
    query = """
        SELECT
            id,
            name,
            facility_code,
            address,
            phone,
            agency,
            coordinates,
            units,
            number_of_units
        FROM hospitals
        ORDER BY name
        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (limit,))
            return cur.fetchall()


def upsert_hospital(record: Dict[str, Any]) -> None:
    """
    Insert or update a hospital row.

    The `record` keys must match the column names defined in the table.
    """
    query = """
        INSERT INTO hospitals (
            id,
            name,
            facility_code,
            address,
            phone,
            agency,
            coordinates,
            units,
            number_of_units
        ) VALUES (
            %(id)s,
            %(name)s,
            %(facility_code)s,
            %(address)s,
            %(phone)s,
            %(agency)s,
            %(coordinates)s,
            %(units)s,
            %(number_of_units)s
        )
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            facility_code = EXCLUDED.facility_code,
            address = EXCLUDED.address,
            phone = EXCLUDED.phone,
            agency = EXCLUDED.agency,
            coordinates = EXCLUDED.coordinates,
            units = EXCLUDED.units,
            number_of_units = EXCLUDED.number_of_units;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, record)


if __name__ == "__main__":
    hospitals = fetch_hospitals()
    print(json.dumps(hospitals, indent=2, default=str))

