"""Path helpers for the repo (expects `util/` and `resources/` at the repo root)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository root (the directory that contains `util/`)."""
    # paths.py lives at: <repo>/util/helper/paths.py
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Back-compat alias for the repo root."""
    return repo_root()


@lru_cache(maxsize=1)
def util_root() -> Path:
    """Return the `util/` directory."""
    return repo_root() / "util"


@lru_cache(maxsize=1)
def simulator_root() -> Path:
    """Return the simulator package directory."""
    return util_root() / "simulator"


@lru_cache(maxsize=1)
def generator_root() -> Path:
    """Return the incident record generator package directory (not shipped)."""
    return util_root() / "incident_record_generator"


@lru_cache(maxsize=1)
def simulated_records_root() -> Path:
    """Return the directory where simulated incident datasets live."""
    return repo_root() / "resources" / "simulated_records"


@lru_cache(maxsize=1)
def generator_records_root() -> Path:
    """Return the directory for generator-produced simulated records."""
    return simulated_records_root()


@lru_cache(maxsize=1)
def simulator_records_root() -> Path:
    """Return the directory for simulator-ready demand profiles."""
    return simulated_records_root()


__all__ = [
    "generator_records_root",
    "generator_root",
    "project_root",
    "repo_root",
    "simulated_records_root",
    "simulator_records_root",
    "simulator_root",
    "util_root",
]
