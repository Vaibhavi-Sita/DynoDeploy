"""Scenario templates + builder utilities for feeding the simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from util.helper.paths import generator_records_root

from .incident_history import IncidentEvent, load_incident_history
from .rules import RuleCatalog, RuleConfig, RULES_PATH

DEFAULT_HISTORY = [generator_records_root() / "sim_day_peak1pm.json"]


@dataclass(frozen=True)
class ScenarioTemplate:
    name: str
    horizon_hours: int
    incident_count: int
    source: str = "historical_json"
    description: str = ""
    default_history_paths: Sequence[Path] = ()
    coverage_threshold_minutes: int = 20


@dataclass
class ScenarioContext:
    rule: RuleConfig
    template: ScenarioTemplate
    events: List[IncidentEvent]
    metadata: Dict[str, Any]


SCENARIO_TEMPLATES: Dict[str, ScenarioTemplate] = {
    "day": ScenarioTemplate(
        name="day",
        horizon_hours=24,
        incident_count=300,
        source="historical_json",
        description="Replay a single synthetic day with peak-at-1pm demand.",
        default_history_paths=tuple(DEFAULT_HISTORY),
    ),
    "week": ScenarioTemplate(
        name="week",
        horizon_hours=24 * 7,
        incident_count=300 * 7,
        source="historical_json",
        description="Repeat the synthetic day to mimic a week of demand.",
        default_history_paths=tuple(DEFAULT_HISTORY),
    ),
    "month": ScenarioTemplate(
        name="month",
        horizon_hours=24 * 30,
        incident_count=300 * 30,
        source="historical_json",
        description="Thirty-day planning horizon with replay-based incidents.",
        default_history_paths=tuple(DEFAULT_HISTORY),
    ),
    "quarter": ScenarioTemplate(
        name="quarter",
        horizon_hours=24 * 90,
        incident_count=300 * 90,
        source="historical_json",
        description="Ninety-day scenario for seasonal planning.",
        default_history_paths=tuple(DEFAULT_HISTORY),
    ),
    "year": ScenarioTemplate(
        name="year",
        horizon_hours=24 * 365,
        incident_count=300 * 365,
        source="historical_json",
        description="Full-year horizon stitched from synthetic peak days.",
        default_history_paths=tuple(DEFAULT_HISTORY),
    ),
}


class ScenarioBuilder:
    """Materializes incidents for a template + rule combination."""

    def __init__(
        self,
        rule_catalog: RuleCatalog | None = None,
        *,
        templates: Mapping[str, ScenarioTemplate] | None = None,
    ) -> None:
        self.rule_catalog = rule_catalog or RuleCatalog(RULES_PATH)
        self.templates = dict(templates or SCENARIO_TEMPLATES)

    def available_templates(self) -> Sequence[str]:
        return tuple(self.templates.keys())

    def build(
        self,
        template_name: str,
        rule_id: str,
        *,
        seed: int | None = None,
        history_paths: Sequence[Path] | None = None,
        incident_limit: int | None = None,
    ) -> ScenarioContext:
        template = self.templates[template_name]
        rule = self.rule_catalog.get(rule_id)
        path_list = (
            tuple(history_paths)
            if history_paths
            else tuple(template.default_history_paths)
        )
        events = self._load_events(template, path_list, incident_limit)
        metadata = {
            "seed": seed,
            "rule_id": rule.id,
            "template": template.name,
            "history_sources": [str(path) for path in path_list],
        }
        return ScenarioContext(
            rule=rule, template=template, events=events, metadata=metadata
        )

    def _load_events(
        self,
        template: ScenarioTemplate,
        history_paths: Sequence[Path],
        incident_limit: int | None,
    ) -> List[IncidentEvent]:
        limit = incident_limit or template.incident_count
        if template.source != "historical_json":
            raise NotImplementedError(
                f"Scenario source '{template.source}' is not implemented yet."
            )
        events = load_incident_history(history_paths, limit=limit)
        tiled = self._tile_events(events, template.horizon_hours, limit)
        return tiled[:limit]

    @staticmethod
    def _tile_events(
        events: Sequence[IncidentEvent],
        horizon_hours: int,
        incident_limit: int,
    ) -> List[IncidentEvent]:
        if not events:
            return []
        ordered = sorted(events, key=lambda evt: evt.occurred_at)
        base_start = ordered[0].occurred_at
        base_end = ordered[-1].occurred_at
        base_duration = base_end - base_start
        if base_duration <= timedelta(0):
            base_duration = timedelta(hours=24)
        horizon = timedelta(hours=horizon_hours)
        tiled: List[IncidentEvent] = []
        iteration = 0
        while len(tiled) < incident_limit and iteration * base_duration < horizon:
            offset = iteration * base_duration
            for event in ordered:
                new_time = event.occurred_at + offset
                if new_time - base_start >= horizon:
                    break
                copied = replace(
                    event,
                    occurred_at=new_time,
                    metadata={
                        **event.metadata,
                        "replication_index": iteration,
                    },
                )
                tiled.append(copied)
                if len(tiled) >= incident_limit:
                    break
            iteration += 1
        return tiled


__all__ = [
    "ScenarioBuilder",
    "ScenarioContext",
    "ScenarioTemplate",
    "SCENARIO_TEMPLATES",
]
