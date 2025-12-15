"""
Convenience helpers for reading travel times/distances stored in the
base_base_travel_matrix and base_hospital_travel_matrix tables.

All base→base entries are stored in a fixed ordering so that the
smaller base ID is always in the `origin_base_id` column.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from psycopg2.extras import RealDictCursor

from .db_connection import get_connection


def _canonical_base_pair(base_a_id: str, base_b_id: str) -> Tuple[str, str]:
    a = str(base_a_id)
    b = str(base_b_id)
    return (a, b) if a <= b else (b, a)


def fetch_base_base_travel(
    base_a_id: str,
    base_b_id: str,
    *,
    table_name: str = "base_base_travel_matrix",
) -> Optional[Dict[str, object]]:
    """
    Return a distance/time row for the unordered pair (base_a_id, base_b_id).
    """
    if base_a_id == base_b_id:
        return None
    left, right = _canonical_base_pair(base_a_id, base_b_id)
    query = f"""
        SELECT *
        FROM {table_name}
        WHERE origin_base_id = %s
          AND destination_base_id = %s;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (left, right))
            return cur.fetchone()


def fetch_base_hospital_travel(
    base_id: str,
    hospital_id: str,
    *,
    table_name: str = "base_hospital_travel_matrix",
) -> Optional[Dict[str, object]]:
    """
    Return a travel entry for the directed pair (base_id, hospital_id).
    """
    query = f"""
        SELECT *
        FROM {table_name}
        WHERE origin_base_id = %s
          AND destination_hospital_id = %s;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (base_id, hospital_id))
            return cur.fetchone()


