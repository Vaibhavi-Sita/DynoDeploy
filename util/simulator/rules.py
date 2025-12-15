"""Rule catalog + configuration objects for redeployment scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from util.helper.paths import simulator_root

try:  # PyYAML is optional at runtime but required for CLI/catalog usage.
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - dependency guard
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

RULES_PATH = simulator_root() / "rules.yaml"


@dataclass(frozen=True)
class RuleConfig:
    """Toggle set describing how redeployment & downtime logic should behave."""

    id: str
    description: str
    first_come_first_serve: bool = True
    base_random_redeploy: bool = False
    popular_random_redeploy: bool = False
    base_peak_urban_redeploy: bool = False
    popular_peak_urban_redeploy: bool = False
    base_peak_redeploy: bool = False
    popular_peak_redeploy: bool = False
    base_perfect_redeploy: bool = False
    popular_perfect_redeploy: bool = False
    hospital_redeploy: bool = False
    hospital_downtime_hours: float | None = None
    popular_location_downtime_hours: float | None = None
    priority_incident_types: Sequence[int] = field(default_factory=tuple)
    random_success_rate: float = 0.25
    peak_window_start: int = 9
    peak_window_end: int = 22

    @classmethod
    def from_dict(cls, rule_id: str, payload: Mapping[str, Any]) -> "RuleConfig":
        data = dict(payload)
        priority_values = data.get("priority_incident_types", [])
        if isinstance(priority_values, Iterable) and not isinstance(
            priority_values, str
        ):
            priority = tuple(int(value) for value in priority_values)
        elif priority_values:
            priority = tuple(int(part) for part in str(priority_values).split(","))
        else:
            priority = tuple()
        return cls(
            id=rule_id,
            description=str(data.get("description", rule_id)),
            first_come_first_serve=bool(data.get("first_come_first_serve", True)),
            base_random_redeploy=bool(data.get("base_random_redeploy", False)),
            popular_random_redeploy=bool(data.get("popular_random_redeploy", False)),
            base_peak_urban_redeploy=bool(data.get("base_peak_urban_redeploy", False)),
            popular_peak_urban_redeploy=bool(
                data.get("popular_peak_urban_redeploy", False)
            ),
            base_peak_redeploy=bool(data.get("base_peak_redeploy", False)),
            popular_peak_redeploy=bool(data.get("popular_peak_redeploy", False)),
            base_perfect_redeploy=bool(data.get("base_perfect_redeploy", False)),
            popular_perfect_redeploy=bool(data.get("popular_perfect_redeploy", False)),
            hospital_redeploy=bool(data.get("hospital_redeploy", False)),
            hospital_downtime_hours=_maybe_float(data.get("hospital_downtime_hours")),
            popular_location_downtime_hours=_maybe_float(
                data.get("popular_location_downtime_hours")
            ),
            priority_incident_types=priority,
            random_success_rate=float(data.get("random_success_rate", 0.25)),
            peak_window_start=_maybe_int(data.get("peak_window_start")) or 10,
            peak_window_end=_maybe_int(data.get("peak_window_end")) or 19,
        )


class RuleCatalog:
    """Loads rules from YAML and exposes them via dictionary-style access."""

    def __init__(self, path: Path = RULES_PATH) -> None:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load rules. Install `pyyaml`."
            ) from YAML_IMPORT_ERROR
        self.path = path
        self._rules: Dict[str, RuleConfig] = {}
        self.reload()

    def reload(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        rules_section = payload.get("rules") if isinstance(payload, dict) else payload
        if not isinstance(rules_section, Mapping):
            raise ValueError(f"Rules file {self.path} did not contain a 'rules' map.")
        self._rules = {}
        for rule_id, data in rules_section.items():
            self._rules[rule_id] = RuleConfig.from_dict(rule_id, data)

    def get(self, rule_id: str) -> RuleConfig:
        try:
            return self._rules[rule_id]
        except KeyError as exc:
            raise KeyError(f"Unknown rule id '{rule_id}'.") from exc

    def ids(self) -> Sequence[str]:
        return tuple(self._rules.keys())

    def values(self) -> Sequence[RuleConfig]:
        return tuple(self._rules.values())


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric.is_integer():
            return int(numeric)
        return None


__all__ = ["RuleCatalog", "RuleConfig", "RULES_PATH"]
