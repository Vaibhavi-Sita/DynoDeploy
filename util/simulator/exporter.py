"""Output helpers for simulated incident logs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from .config import SimulationConfig


def write_json_log(
    records: Sequence[Dict[str, Any]],
    config: SimulationConfig,
    output_path: Path,
) -> Dict[str, Any]:
    summary = _build_summary(records, config)
    payload = {"metadata": summary, "incidents": list(records)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return summary


def write_ndjson_log(
    records: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str))
            handle.write("\n")


def _build_summary(records: Sequence[Dict[str, Any]], config: SimulationConfig) -> Dict[str, Any]:
    type_counts = Counter(record.get("incidentType") for record in records)
    zone_counts = Counter(record.get("type") for record in records)
    return {
        "total_incidents": len(records),
        "incident_type_counts": dict(type_counts),
        "zone_counts": dict(zone_counts),
        "config": config.to_dict(),
    }


__all__ = ["write_json_log", "write_ndjson_log"]

