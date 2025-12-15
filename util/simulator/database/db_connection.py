from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import connection as PGConnection


def _load_env_file() -> None:
    """Load the nearest .env file so database credentials are sourced early."""
    file_path = Path(__file__).resolve()
    for directory in (file_path,) + tuple(file_path.parents):
        env_path = directory / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return
    load_dotenv(override=False)


_load_env_file()


@dataclass
class DbConfig:
    """Holds the connection parameters that psycopg2 expects."""

    dbname: str = os.getenv("COMP594_DB_NAME", "comp594")
    user: str = os.getenv("COMP594_DB_USER", "postgres")
    password: str = os.getenv("COMP594_DB_PASS", "postgres")
    host: str = os.getenv("COMP594_DB_HOST", "localhost")
    port: int = int(os.getenv("COMP594_DB_PORT", 5432))


def get_connection(config: DbConfig | None = None) -> PGConnection:
    """Create a psycopg2 connection using the provided configuration."""
    cfg = config or DbConfig()
    return psycopg2.connect(
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        host=cfg.host,
        port=cfg.port,
    )


if __name__ == "__main__":
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user;")
            dbname, user = cur.fetchone()
            print(f"Connected to database '{dbname}' as '{user}'.")
