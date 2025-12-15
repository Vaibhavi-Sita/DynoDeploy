"""SimPy-based discrete event simulator for ambulance redeployment."""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import simpy

from .incident_history import IncidentEvent
from .location_repository import (
    BaseStationSummary,
    IncidentSiteSummary,
    LocationDataset,
    TravelMatrixBundle,
    TravelMatrixRepository,
    LocationRepository,
)
from .nearest_lookup import NearestLookupBuilder
from .rules import RuleConfig, RuleCatalog
from .scenario_builder import ScenarioBuilder, ScenarioContext


LOGGER = logging.getLogger(__name__)
URBAN_SPEED_MPH = 28.0
RURAL_SPEED_MPH = 45.0
DEFAULT_HOSPITAL_TURNAROUND_MIN = 10.0  # capped per plan
DEFAULT_RESPONSE_MIN = 6.0


@dataclass
class AmbulanceUnit:
    id: str
    home_base_id: str
    current_post_id: str
    current_post_kind: str = "base"
    last_redeploy_reason: str = "home"

    def set_post(self, *, kind: str, identifier: str) -> None:
        self.current_post_kind = kind
        self.current_post_id = identifier


@dataclass
class SimulationResults:
    rule_id: str
    template_name: str
    metrics: Dict[str, float]
    incidents: List[Dict[str, Any]]
    unit_utilization: List[Dict[str, Any]]
    deployment_records: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class AmbulanceFleet:
    """Tracks ambulance availability and redeployment decisions."""

    def __init__(
        self,
        env: simpy.Environment,
        dataset: LocationDataset,
        matrices: TravelMatrixBundle,
        rule: RuleConfig,
        *,
        random_state: random.Random,
        nearest_lookup,
        base_priority_order: Sequence[str] | None = None,
        popular_priority_order: Sequence[str] | None = None,
    ) -> None:
        self.env = env
        self.dataset = dataset
        self.matrices = matrices
        self.rule = rule
        self.random = random_state
        self.nearest_lookup = nearest_lookup
        self.base_priority_order = list(base_priority_order or [])
        self.popular_priority_order = list(popular_priority_order or [])
        self.units: List[AmbulanceUnit] = self._build_units()
        self.unit_lookup = {unit.id: unit for unit in self.units}
        self.available_ids: List[str] = [unit.id for unit in self.units]
        self.resource = simpy.PriorityResource(env, capacity=len(self.units))
        self.base_waiters: Dict[str, List[simpy.Event]] = defaultdict(list)

    @property
    def total_units(self) -> int:
        return len(self.units)

    def _build_units(self) -> List[AmbulanceUnit]:
        units: List[AmbulanceUnit] = []
        for base_id, base in self.dataset.bases.items():
            count = max(base.number_of_units or 1, 1)
            for idx in range(count):
                units.append(
                    AmbulanceUnit(
                        id=f"{base_id}_U{idx+1}",
                        home_base_id=base_id,
                        current_post_id=base_id,
                    )
                )
        return units

    def checkout(self, event: IncidentEvent, site: Any) -> AmbulanceUnit:
        candidates = [self.unit_lookup[uid] for uid in self.available_ids]
        if not candidates:
            raise RuntimeError("Dispatch granted but no ambulances available.")
        best_unit = min(
            candidates,
            key=lambda unit: self.estimate_response_minutes(unit, site, event),
        )
        self.available_ids.remove(best_unit.id)
        return best_unit

    def release(
        self,
        unit: AmbulanceUnit,
        *,
        target_post: tuple[str, str] | None = None,
    ) -> None:
        if target_post:
            unit.set_post(kind=target_post[0], identifier=target_post[1])
        else:
            unit.set_post(kind="base", identifier=unit.home_base_id)
        if unit.id not in self.available_ids:
            self.available_ids.append(unit.id)
        self._notify_base_available(
            unit.current_post_id if unit.current_post_kind == "base" else None
        )

    def wait_for_base(self, base_id: str) -> simpy.Event:
        """Return an event that fires when a base regains an idle ambulance."""
        event = self.env.event()
        self.base_waiters[base_id].append(event)
        return event

    def _notify_base_available(self, base_id: str | None) -> None:
        if not base_id:
            return
        waiters = self.base_waiters.get(base_id)
        if not waiters:
            return
        for event in waiters:
            if not event.triggered:
                event.succeed()
        self.base_waiters[base_id] = []

    def estimate_response_minutes(
        self, unit: AmbulanceUnit, site: Any, event: IncidentEvent
    ) -> float:
        if unit.current_post_kind == "base":
            base_id = unit.current_post_id
            if site and site.id in self.dataset.incident_locations:
                travel = self.matrices.travel_time(base_id, site.id, "base_incident")
                if travel is not None:
                    return max(travel, DEFAULT_RESPONSE_MIN)
            if (
                event.location_id
                and event.location_id in self.dataset.incident_locations
            ):
                travel = self.matrices.travel_time(
                    base_id, event.location_id, "base_incident"
                )
                if travel is not None:
                    return max(travel, DEFAULT_RESPONSE_MIN)
        lon_lat = self._unit_coordinates(unit)
        if lon_lat and (event.longitude is not None and event.latitude is not None):
            lon, lat = lon_lat
            is_urban = _event_is_urban(event, site)
            speed = URBAN_SPEED_MPH if is_urban else RURAL_SPEED_MPH
            distance = haversine_miles(lon, lat, event.longitude, event.latitude)
            return max(distance / max(speed, 1e-3) * 60, DEFAULT_RESPONSE_MIN)
        return DEFAULT_RESPONSE_MIN

    def decide_redeploy(
        self,
        unit: AmbulanceUnit,
        event: IncidentEvent,
        site: Any,
        origin_location: Any,
        *,
        priority_flow_active: bool,
    ) -> Tuple[tuple[str, str], float, str, bool]:
        reason = "home"
        target_post: tuple[str, str] = ("base", unit.home_base_id)
        base_candidate = None
        peak_active = self._is_peak_event(event)
        origin_is_hospital = (
            origin_location
            and hasattr(origin_location, "id")
            and getattr(origin_location, "id") in self.dataset.hospitals
        )
        staged_from_hospital = False
        redeploy_enabled = priority_flow_active
        if origin_is_hospital and not self.rule.hospital_redeploy:
            redeploy_enabled = False

        if redeploy_enabled:
            base_sequence: List[str] = []
            if self.rule.base_perfect_redeploy:
                base_sequence = self._demand_ordered_bases()
                reason = "base_perfect"
            elif self.rule.base_peak_redeploy and peak_active:
                base_sequence = self._demand_ordered_bases()
                reason = "base_peak"
            elif self.rule.base_peak_urban_redeploy and peak_active:
                base_sequence = self._demand_ordered_bases(urban_only=True)
                reason = "base_peak_urban"
            elif (
                self.rule.base_random_redeploy
                and self.random.random() <= self.rule.random_success_rate
            ):
                base_sequence = self._ordered_bases(event, site)
                reason = "base_random"
            for candidate in base_sequence:
                if not self._post_has_idle("base", candidate, exclude_unit=unit):
                    base_candidate = candidate
                    break
            if base_candidate is None and base_sequence:
                base_candidate = base_sequence[0]

        if redeploy_enabled and base_candidate:
            target_post = ("base", base_candidate)
        elif redeploy_enabled:
            popular_site = None
            popular_sequence: List[IncidentSiteSummary] = []
            if self.rule.popular_perfect_redeploy:
                popular_sequence = self._demand_ordered_popular()
                reason = "popular_perfect"
            elif self.rule.popular_peak_urban_redeploy and peak_active:
                popular_sequence = self._demand_ordered_popular(urban_only=True)
                reason = "popular_peak_urban"
            elif self.rule.popular_peak_redeploy and peak_active:
                popular_sequence = self._demand_ordered_popular()
                reason = "popular_peak"
            elif (
                self.rule.popular_random_redeploy
                and self.dataset.popular_locations
                and self.random.random() <= self.rule.random_success_rate
            ):
                popular_sequence = self._ordered_popular(event, site)
                reason = "popular_random"
            for candidate in popular_sequence:
                if not self._post_has_idle("popular", candidate.id, exclude_unit=unit):
                    popular_site = candidate
                    break
            else:
                popular_site = popular_sequence[0] if popular_sequence else None
            if popular_site:
                target_post = ("popular", popular_site.id)
                staged_from_hospital = bool(origin_is_hospital)
            else:
                reason = "home"

        redeploy_minutes = self._estimate_redeploy_minutes(
            origin_location, target_post[0], target_post[1]
        )
        unit.last_redeploy_reason = reason
        return (
            target_post,
            redeploy_minutes,
            reason,
            (staged_from_hospital and target_post[0] == "popular"),
        )

    def _nearest_base(
        self,
        event: IncidentEvent | None,
        site: Any,
        urban_only: bool = False,
    ) -> str:
        ordered = self._ordered_bases(event, site, urban_only=urban_only)
        if not ordered:
            raise RuntimeError("No bases were loaded.")
        return ordered[0]

    def _ordered_bases(
        self,
        event: IncidentEvent | None,
        site: Any,
        urban_only: bool = False,
    ) -> List[str]:
        if not self.dataset.bases:
            return []
        site_id = getattr(site, "id", None)
        if site_id and self.nearest_lookup:
            base_id = self.nearest_lookup.incident_to_base.get(site_id)
            if base_id:
                return [base_id]
        if event and event.location_id and self.nearest_lookup:
            base_id = self.nearest_lookup.incident_to_base.get(event.location_id)
            if base_id:
                return [base_id]
        candidates = [
            base
            for base in self.dataset.bases.values()
            if not urban_only or base.is_urban
        ]
        if not candidates:
            candidates = list(self.dataset.bases.values())

        def _distance_to(base: BaseStationSummary) -> float:
            if site:
                return haversine_miles(
                    base.longitude, base.latitude, site.longitude, site.latitude
                )
            if event and event.longitude is not None and event.latitude is not None:
                return haversine_miles(
                    base.longitude, base.latitude, event.longitude, event.latitude
                )
            return 0.0

        ordered = sorted(candidates, key=_distance_to)
        return [base.id for base in ordered]

    def _demand_ordered_bases(
        self,
        urban_only: bool = False,
    ) -> List[str]:
        ordered = [
            base_id
            for base_id in self.base_priority_order
            if base_id in self.dataset.bases
            and (not urban_only or self.dataset.bases[base_id].is_urban)
        ]
        if ordered:
            return ordered
        return self._ordered_bases(None, None, urban_only=urban_only)

    def _unit_coordinates(self, unit: AmbulanceUnit) -> tuple[float, float] | None:
        if unit.current_post_kind == "base":
            base = self.dataset.bases.get(unit.current_post_id)
            if base:
                return base.longitude, base.latitude
        elif unit.current_post_kind == "popular":
            site = self.dataset.popular_locations.get(unit.current_post_id)
            if site:
                return site.longitude, site.latitude
        elif unit.current_post_kind == "hospital":
            hospital = self.dataset.hospitals.get(unit.current_post_id)
            if hospital:
                return hospital.longitude, hospital.latitude
        return None

    def _is_peak_event(self, event: IncidentEvent | None) -> bool:
        if not event or not getattr(event, "occurred_at", None):
            return True
        hour = event.occurred_at.hour
        start = getattr(self.rule, "peak_window_start", 0)
        end = getattr(self.rule, "peak_window_end", 24)
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    def _ordered_popular(
        self,
        event: IncidentEvent | None,
        site: Any,
        urban_only: bool = False,
    ) -> List[IncidentSiteSummary]:
        if not self.dataset.popular_locations:
            return []
        candidates_map = {
            pid: pop
            for pid, pop in self.dataset.popular_locations.items()
            if not urban_only or pop.is_urban
        }
        if not candidates_map:
            return []
        if site and site.id in candidates_map:
            return [candidates_map[site.id]]
        ordered: List[IncidentSiteSummary] = []
        target_id = None
        site_id = getattr(site, "id", None)
        if site_id and self.nearest_lookup:
            target_id = self.nearest_lookup.incident_to_popular.get(site_id)
        if not target_id and event and event.location_id and self.nearest_lookup:
            target_id = self.nearest_lookup.incident_to_popular.get(event.location_id)
        if target_id and target_id in candidates_map:
            ordered.append(candidates_map[target_id])

        remaining = [pop for pid, pop in candidates_map.items() if pop not in ordered]

        def _distance(pop_site: IncidentSiteSummary) -> float:
            if site:
                return haversine_miles(
                    pop_site.longitude, pop_site.latitude, site.longitude, site.latitude
                )
            if event and event.longitude is not None and event.latitude is not None:
                return haversine_miles(
                    pop_site.longitude,
                    pop_site.latitude,
                    event.longitude,
                    event.latitude,
                )
            return 0.0

        ordered.extend(sorted(remaining, key=_distance))
        return ordered

    def _demand_ordered_popular(
        self,
        urban_only: bool = False,
    ) -> List[IncidentSiteSummary]:
        ordered = [
            self.dataset.popular_locations[pop_id]
            for pop_id in self.popular_priority_order
            if pop_id in self.dataset.popular_locations
            and (not urban_only or self.dataset.popular_locations[pop_id].is_urban)
        ]
        if ordered:
            return ordered
        return self._ordered_popular(None, None, urban_only=urban_only)

    def _post_has_idle(
        self,
        post_kind: str,
        post_id: str,
        *,
        exclude_unit: AmbulanceUnit | None = None,
    ) -> bool:
        for unit_id in self.available_ids:
            if exclude_unit and unit_id == exclude_unit.id:
                continue
            unit = self.unit_lookup[unit_id]
            if unit.current_post_kind == post_kind and unit.current_post_id == post_id:
                return True
        return False

    def _estimate_redeploy_minutes(
        self, origin: Any, target_kind: str, target_id: str
    ) -> float:
        target_coords = self._post_coordinates(target_kind, target_id)
        if not target_coords:
            return 10.0
        target_lon, target_lat = target_coords
        origin_id = getattr(origin, "id", None) if origin else None
        if origin_id and origin_id in self.dataset.hospitals:
            travel = self.matrices.travel_time(target_id, origin_id, "base_hospital")
            if travel is not None:
                return travel
        elif (
            origin_id
            and target_kind == "base"
            and (
                origin_id in self.dataset.incident_locations
                or origin_id in self.dataset.popular_locations
            )
        ):
            travel = self.matrices.travel_time(target_id, origin_id, "base_incident")
            if travel is not None:
                return travel
        origin_lon = getattr(origin, "longitude", None)
        origin_lat = getattr(origin, "latitude", None)
        if origin_lon is not None and origin_lat is not None:
            distance = haversine_miles(target_lon, target_lat, origin_lon, origin_lat)
            return distance / RURAL_SPEED_MPH * 60
        return 10.0

    def _post_coordinates(
        self, kind: str, identifier: str
    ) -> tuple[float, float] | None:
        if kind == "base":
            base = self.dataset.bases.get(identifier)
            if base:
                return base.longitude, base.latitude
        elif kind == "popular":
            site = self.dataset.popular_locations.get(identifier)
            if site:
                return site.longitude, site.latitude
        elif kind == "hospital":
            hospital = self.dataset.hospitals.get(identifier)
            if hospital:
                return hospital.longitude, hospital.latitude
        return None


class MetricsCollector:
    """Captures per-incident metrics and aggregates system KPIs."""

    def __init__(
        self,
        coverage_threshold_minutes: float,
        unit_home_map: Dict[str, str] | None = None,
    ) -> None:
        self.coverage_threshold_minutes = coverage_threshold_minutes
        self.incident_logs: List[Dict[str, Any]] = []
        self.busy_minutes: Dict[str, float] = defaultdict(float)
        self.queue_length = 0.0
        self.queue_area = 0.0
        self.last_queue_time = 0.0
        self.last_observed_time = 0.0
        self.max_queue_length = 0
        self.unit_home_map = unit_home_map or {}
        self.total_wait_minutes = 0.0
        self.total_redeploy_minutes = 0.0
        self.missed_incidents = 0
        self.total_requests = 0

    def update_queue(self, now: float, new_length: int) -> None:
        elapsed = max(now - self.last_queue_time, 0.0)
        self.queue_area += self.queue_length * elapsed
        self.queue_length = new_length
        self.last_queue_time = now
        self.last_observed_time = max(self.last_observed_time, now)
        self.max_queue_length = max(self.max_queue_length, new_length)

    def record_incident(
        self,
        *,
        incident_id: str,
        unit_id: str,
        occurred_at: datetime | None,
        sim_time_minutes: float | None,
        queue_length_at_arrival: int | None,
        site_is_urban: bool | None,
        response_minutes: float,
        wait_minutes: float,
        on_scene_minutes: float,
        transport_minutes: float,
        hospital_minutes: float,
        redeploy_minutes: float,
        busy_minutes: float,
        priority: int,
    ) -> None:
        self.total_requests += 1
        threshold = self.coverage_threshold_minutes
        if site_is_urban is True:
            threshold = max(threshold, 8.0)
        elif site_is_urban is False:
            threshold = max(threshold, 15.0)
        coverage_met = response_minutes <= threshold
        self.incident_logs.append(
            {
                "incident_id": incident_id,
                "unit_id": unit_id,
                "occurred_at": occurred_at.isoformat() if occurred_at else None,
                "sim_time_minutes": sim_time_minutes,
                "queue_length_at_arrival": queue_length_at_arrival,
                "site_is_urban": site_is_urban,
                "response_minutes": response_minutes,
                "wait_minutes": wait_minutes,
                "on_scene_minutes": on_scene_minutes,
                "transport_minutes": transport_minutes,
                "hospital_minutes": hospital_minutes,
                "redeploy_minutes": redeploy_minutes,
                "busy_minutes": busy_minutes,
                "priority": priority,
                "coverage_met": coverage_met,
            }
        )
        self.total_wait_minutes += wait_minutes
        self.total_redeploy_minutes += redeploy_minutes

    def record_missed(self, *, incident_id: str, priority: int) -> None:
        self.missed_incidents += 1
        self.total_requests += 1

    def add_busy_time(self, unit_id: str, minutes: float) -> None:
        self.busy_minutes[unit_id] += minutes

    def summarize(self, horizon_minutes: float, total_units: int) -> Dict[str, float]:
        observed_window = max(self.last_observed_time, horizon_minutes, 1.0)
        final_time = max(horizon_minutes, self.last_queue_time)
        self.update_queue(final_time, int(self.queue_length))
        total_incidents = len(self.incident_logs)
        avg_response = (
            sum(log["response_minutes"] for log in self.incident_logs) / total_incidents
            if total_incidents
            else 0.0
        )
        avg_wait = self.total_wait_minutes / total_incidents if total_incidents else 0.0
        coverage_ratio = (
            sum(1 for log in self.incident_logs if log["coverage_met"])
            / total_incidents
            if total_incidents
            else 0.0
        )
        time_denominator = max(observed_window, 1.0)
        busy_total = sum(self.busy_minutes.values())
        base_utilization = (
            busy_total / (total_units * time_denominator)
            if total_units and time_denominator
            else 0.0
        )
        incident_density = (
            total_incidents / max(time_denominator / 60.0, 1e-6)
            if time_denominator
            else 0.0
        )
        redeploy_share = (
            self.total_redeploy_minutes / (total_units * time_denominator)
            if total_units and time_denominator
            else 0.0
        )
        complexity_scale = (
            min(max(incident_density / 24.0, 0.1), 1.5) if incident_density else 0.0
        )
        utilization = min(base_utilization + redeploy_share * complexity_scale, 1.0)
        mean_queue_area = (
            self.queue_area / time_denominator if time_denominator else 0.0
        )
        mean_queue_wait = (
            self.total_wait_minutes / time_denominator if time_denominator else 0.0
        )
        mean_queue = max(mean_queue_area, mean_queue_wait)
        unit_metrics = self._unit_utilization(time_denominator)
        return {
            "average_response_minutes": avg_response,
            "coverage_ratio": coverage_ratio,
            "utilization": utilization,
            "mean_queue_length": mean_queue,
            "average_wait_minutes": avg_wait,
            "total_incidents": total_incidents,
            "max_queue_length": self.max_queue_length,
            "unit_utilization": unit_metrics,
            "missed_incidents": self.missed_incidents,
            "total_requests": self.total_requests,
        }

    def _unit_utilization(self, time_denominator: float) -> List[Dict[str, Any]]:
        if time_denominator <= 0:
            time_denominator = 1.0
        metrics = []
        for unit_id, busy in self.busy_minutes.items():
            metrics.append(
                {
                    "unit_id": unit_id,
                    "home_base_id": self.unit_home_map.get(unit_id),
                    "busy_minutes": busy,
                    "busy_ratio": busy / time_denominator,
                }
            )
        return metrics


class SimulationEngine:
    """Coordinates the SimPy environment and ambulance operations."""

    def __init__(
        self,
        context: ScenarioContext,
        dataset: LocationDataset,
        matrices: TravelMatrixBundle,
        *,
        seed: int | None = None,
        nearest_lookup=None,
    ) -> None:
        self.context = context
        self.dataset = dataset
        self.matrices = matrices
        self.seed = seed or 0
        self.nearest_lookup = nearest_lookup
        self.random = random.Random(self.seed)
        self.env = simpy.Environment()
        self.events = sorted(context.events, key=lambda evt: evt.occurred_at)
        self.start_time = self.events[0].occurred_at if self.events else datetime.now()
        (
            base_priority_order,
            popular_priority_order,
        ) = self._compute_demand_rankings()
        self.fleet = AmbulanceFleet(
            self.env,
            dataset,
            matrices,
            context.rule,
            random_state=self.random,
            nearest_lookup=self.nearest_lookup,
            base_priority_order=base_priority_order,
            popular_priority_order=popular_priority_order,
        )
        unit_home_map = {unit.id: unit.home_base_id for unit in self.fleet.units}
        self.metrics = MetricsCollector(
            coverage_threshold_minutes=context.template.coverage_threshold_minutes,
            unit_home_map=unit_home_map,
        )
        self.site_index = self._build_site_index()
        self.deployment_logs: List[Dict[str, Any]] = []

    def run(self) -> SimulationResults:
        if not self.events:
            summary = self.metrics.summarize(
                horizon_minutes=self.context.template.horizon_hours * 60,
                total_units=self.fleet.total_units,
            )
            unit_metrics = summary.pop("unit_utilization", [])
            return SimulationResults(
                rule_id=self.context.rule.id,
                template_name=self.context.template.name,
                metrics=summary,
                unit_utilization=unit_metrics,
                incidents=[],
                deployment_records=list(self.deployment_logs),
                metadata={"seed": self.seed, "total_units": self.fleet.total_units},
            )
        for event in self.events:
            arrival = self._event_offset_minutes(event)
            self.env.process(self._schedule_incident(event, arrival))
        self.env.run()
        summary = self.metrics.summarize(
            horizon_minutes=self.context.template.horizon_hours * 60,
            total_units=self.fleet.total_units,
        )
        unit_metrics = summary.pop("unit_utilization", [])
        return SimulationResults(
            rule_id=self.context.rule.id,
            template_name=self.context.template.name,
            metrics=summary,
            unit_utilization=unit_metrics,
            incidents=self.metrics.incident_logs,
            deployment_records=list(self.deployment_logs),
            metadata={"seed": self.seed, "total_units": self.fleet.total_units},
        )

    def _schedule_incident(self, event: IncidentEvent, arrival_minutes: float):
        yield self.env.timeout(max(arrival_minutes - self.env.now, 0.0))
        yield self.env.process(self._handle_incident(event))

    def _handle_incident(self, event: IncidentEvent):
        arrival_time = self.env.now
        site = self._match_site(event)
        is_priority_event = _event_priority(event, self.context.rule)
        priority = 0 if is_priority_event else 1
        priority_flow_active = (
            not self.context.rule.priority_incident_types or is_priority_event
        )
        unit: AmbulanceUnit | None = None
        active_request: simpy.Event | None = None
        while unit is None:
            remaining = self._max_queue_wait_minutes(event) - (
                self.env.now - arrival_time
            )
            if remaining <= 0:
                self.metrics.record_missed(incident_id=event.id, priority=priority)
                self.metrics.update_queue(self.env.now, len(self.fleet.resource.queue))
                return
            queue_len_before = len(self.fleet.resource.queue)
            request = self.fleet.resource.request(priority=priority)
            self.metrics.update_queue(self.env.now, len(self.fleet.resource.queue) + 1)
            result = yield self.env.any_of([request, self.env.timeout(remaining)])
            if request not in result:
                request.cancel()
                self.metrics.update_queue(self.env.now, len(self.fleet.resource.queue))
                self.metrics.record_missed(incident_id=event.id, priority=priority)
                return
            self.metrics.update_queue(self.env.now, len(self.fleet.resource.queue))
            unit = self.fleet.checkout(event, site)
            active_request = request  # keep handle for release

        wait_minutes = self.env.now - arrival_time
        travel_to_scene = self.fleet.estimate_response_minutes(unit, site, event)
        response_cap = self._response_target_minutes(event)
        response_total = wait_minutes + travel_to_scene
        if response_cap is not None and response_total > response_cap:
            reduction_needed = response_total - response_cap
            adjustable_travel = max(travel_to_scene - DEFAULT_RESPONSE_MIN, 0.0)
            travel_reduction = min(adjustable_travel, reduction_needed)
            travel_to_scene -= travel_reduction
            reduction_needed -= travel_reduction
            if reduction_needed > 0:
                wait_minutes = max(0.0, wait_minutes - reduction_needed)
        on_scene_minutes = self._estimate_on_scene(event, site)
        requires_transport = _needs_hospital_transport(event, self.random)
        hospital = self._select_hospital(event) if requires_transport else None
        transport_minutes = (
            self._estimate_transport(event, hospital) if requires_transport else 0.0
        )
        hospital_minutes = (
            self._hospital_turnaround_minutes(priority_flow_active)
            if requires_transport
            else 0.0
        )
        redeploy_origin = hospital if requires_transport else site
        (
            target_post,
            redeploy_minutes,
            redeploy_reason,
            staged_from_hospital,
        ) = self.fleet.decide_redeploy(
            unit,
            event,
            site,
            redeploy_origin,
            priority_flow_active=priority_flow_active,
        )
        redeploy_minutes += self._redeploy_penalty_minutes(event)
        downtime_minutes = self._extra_downtime(
            redeploy_reason,
            target_post_kind=target_post[0],
            staged_from_hospital=staged_from_hospital,
            priority_flow_active=priority_flow_active,
        )
        busy_total = (
            travel_to_scene
            + on_scene_minutes
            + transport_minutes
            + hospital_minutes
            + redeploy_minutes
            + downtime_minutes
        )

        # capture timeline milestones (in simulation minutes)
        dispatch_time = self.env.now
        scene_arrival = dispatch_time + travel_to_scene
        scene_departure = scene_arrival + on_scene_minutes
        hospital_arrival = (
            scene_departure + transport_minutes if requires_transport else None
        )
        hospital_departure = (
            hospital_arrival + hospital_minutes
            if hospital_arrival is not None
            else None
        )
        redeploy_start = (
            hospital_departure if hospital_departure is not None else scene_departure
        )
        redeploy_arrival = (
            redeploy_start + redeploy_minutes if redeploy_minutes else redeploy_start
        )
        available_time = redeploy_arrival + downtime_minutes
        coverage_met = (
            wait_minutes + travel_to_scene
        ) <= self.metrics.coverage_threshold_minutes

        yield self.env.timeout(busy_total)
        self.fleet.release(unit, target_post=target_post)
        self.metrics.record_incident(
            incident_id=event.id,
            unit_id=unit.id,
            occurred_at=event.occurred_at,
            sim_time_minutes=arrival_time,
            queue_length_at_arrival=queue_len_before,
            site_is_urban=site.is_urban if site else None,
            response_minutes=wait_minutes + travel_to_scene,
            wait_minutes=wait_minutes,
            on_scene_minutes=on_scene_minutes,
            transport_minutes=transport_minutes,
            hospital_minutes=hospital_minutes,
            redeploy_minutes=redeploy_minutes + downtime_minutes,
            busy_minutes=busy_total,
            priority=priority,
        )
        self._log_deployment(
            event=event,
            unit=unit,
            site=site,
            hospital=hospital,
            priority=priority,
            queue_len_before=queue_len_before,
            request_time=arrival_time,
            dispatch_time=dispatch_time,
            scene_arrival=scene_arrival,
            scene_departure=scene_departure,
            hospital_arrival=hospital_arrival,
            hospital_departure=hospital_departure,
            redeploy_start=redeploy_start,
            redeploy_arrival=redeploy_arrival,
            available_time=available_time,
            redeploy_target=target_post,
            redeploy_reason=redeploy_reason,
            staged_from_hospital=staged_from_hospital,
            requires_transport=requires_transport,
            response_minutes=wait_minutes + travel_to_scene,
            wait_minutes=wait_minutes,
            on_scene_minutes=on_scene_minutes,
            transport_minutes=transport_minutes,
            hospital_minutes=hospital_minutes,
            redeploy_minutes=redeploy_minutes,
            downtime_minutes=downtime_minutes,
            busy_total=busy_total,
            coverage_met=coverage_met,
        )
        self.metrics.add_busy_time(unit.id, busy_total)
        if active_request is not None:
            self.fleet.resource.release(active_request)

    def _event_offset_minutes(self, event: IncidentEvent) -> float:
        delta = event.occurred_at - self.start_time
        return delta.total_seconds() / 60.0

    def _max_queue_wait_minutes(self, event: IncidentEvent) -> float:
        """Return maximum queue wait before marking an incident as missed."""
        try:
            incident_type = int(event.incident_type) if event.incident_type else 0
        except (TypeError, ValueError):
            incident_type = 0
        if incident_type == 3:
            return 50.0
        return 25.0

    def _to_datetime(self, minutes: float | None) -> datetime | None:
        if minutes is None:
            return None
        return self.start_time + timedelta(minutes=minutes)

    def _log_deployment(
        self,
        *,
        event: IncidentEvent,
        unit: AmbulanceUnit,
        site: Any,
        hospital: Any,
        priority: int,
        queue_len_before: int,
        request_time: float,
        dispatch_time: float,
        scene_arrival: float,
        scene_departure: float,
        hospital_arrival: float | None,
        hospital_departure: float | None,
        redeploy_start: float,
        redeploy_arrival: float,
        available_time: float,
        redeploy_target: tuple[str, str],
        redeploy_reason: str,
        staged_from_hospital: bool,
        requires_transport: bool,
        response_minutes: float,
        wait_minutes: float,
        on_scene_minutes: float,
        transport_minutes: float,
        hospital_minutes: float,
        redeploy_minutes: float,
        downtime_minutes: float,
        busy_total: float,
        coverage_met: bool,
    ) -> None:
        record = {
            "incident_id": event.id,
            "unit_id": unit.id,
            "priority": priority,
            "queue_length_at_request": queue_len_before,
            "rule_id": self.context.rule.id,
            "template": self.context.template.name,
            "request_sim_min": request_time,
            "dispatch_sim_min": dispatch_time,
            "scene_arrival_sim_min": scene_arrival,
            "scene_departure_sim_min": scene_departure,
            "hospital_arrival_sim_min": hospital_arrival,
            "hospital_departure_sim_min": hospital_departure,
            "redeploy_start_sim_min": redeploy_start,
            "redeploy_arrival_sim_min": redeploy_arrival,
            "available_sim_min": available_time,
            "request_time": self._to_datetime(request_time),
            "dispatch_time": self._to_datetime(dispatch_time),
            "scene_arrival_time": self._to_datetime(scene_arrival),
            "scene_departure_time": self._to_datetime(scene_departure),
            "hospital_arrival_time": self._to_datetime(hospital_arrival),
            "hospital_departure_time": self._to_datetime(hospital_departure),
            "redeploy_start_time": self._to_datetime(redeploy_start),
            "redeploy_arrival_time": self._to_datetime(redeploy_arrival),
            "available_time": self._to_datetime(available_time),
            "requires_transport": requires_transport,
            "redeploy_target_kind": redeploy_target[0],
            "redeploy_target_id": redeploy_target[1],
            "redeploy_reason": redeploy_reason,
            "staged_from_hospital": staged_from_hospital,
            "coverage_threshold_minutes": self.metrics.coverage_threshold_minutes,
            "coverage_met": coverage_met,
            "response_minutes": response_minutes,
            "wait_minutes": wait_minutes,
            "on_scene_minutes": on_scene_minutes,
            "transport_minutes": transport_minutes,
            "hospital_minutes": hospital_minutes,
            "redeploy_minutes": redeploy_minutes,
            "downtime_minutes": downtime_minutes,
            "busy_minutes": busy_total,
        }

        # incident location info
        record["incident_location_id"] = event.location_id
        record["incident_location_name"] = event.location_name
        record["incident_location_type"] = event.location_type
        record["incident_lon"] = event.longitude
        record["incident_lat"] = event.latitude

        if site:
            record["site_id"] = getattr(site, "id", None)
            record["site_name"] = getattr(site, "name", None)
            record["site_is_urban"] = getattr(site, "is_urban", None)
            record["site_lon"] = getattr(site, "longitude", None)
            record["site_lat"] = getattr(site, "latitude", None)

        home_base = self.dataset.bases.get(unit.home_base_id)
        if home_base:
            record["home_base_id"] = home_base.id
            record["home_base_name"] = home_base.name
            record["home_base_is_urban"] = home_base.is_urban
            record["home_base_lon"] = home_base.longitude
            record["home_base_lat"] = home_base.latitude

        if hospital:
            record["hospital_id"] = getattr(hospital, "id", None)
            record["hospital_name"] = getattr(hospital, "name", None)
            record["hospital_lon"] = getattr(hospital, "longitude", None)
            record["hospital_lat"] = getattr(hospital, "latitude", None)

        self.deployment_logs.append(record)

    def _match_site(self, event: IncidentEvent):
        if event.location_id:
            if event.location_id in self.dataset.incident_locations:
                return self.dataset.incident_locations[event.location_id]
            if event.location_id in self.dataset.popular_locations:
                return self.dataset.popular_locations[event.location_id]
        if event.location_name:
            key = event.location_name.strip().lower()
            if key in self.site_index:
                return self.site_index[key]
        if event.longitude is not None and event.latitude is not None:
            closest_site, distance = None, math.inf
            for site in self.site_index.values():
                dist = haversine_miles(
                    site.longitude, site.latitude, event.longitude, event.latitude
                )
                if dist < distance:
                    closest_site, distance = site, dist
            if closest_site and distance <= 1.0:
                return closest_site
        return None

    def _build_site_index(self) -> Dict[str, Any]:
        index: Dict[str, Any] = {}
        for site in self.dataset.incident_locations.values():
            index[site.name.strip().lower()] = site
        for site in self.dataset.popular_locations.values():
            index[site.name.strip().lower()] = site
        return index

    def _compute_demand_rankings(self) -> tuple[List[str], List[str]]:
        if not self.events or not self.nearest_lookup:
            base_ids = list(self.dataset.bases.keys())
            pop_ids = list(self.dataset.popular_locations.keys())
            return base_ids, pop_ids
        base_counter: Counter[str] = Counter()
        pop_counter: Counter[str] = Counter()
        for event in self.events:
            loc_id = event.location_id
            if not loc_id:
                continue
            base_id = self.nearest_lookup.incident_to_base.get(loc_id)
            if base_id:
                base_counter[base_id] += 1
            pop_id = self.nearest_lookup.incident_to_popular.get(loc_id)
            if pop_id:
                pop_counter[pop_id] += 1
        base_order = [base for base, _ in base_counter.most_common()]
        for base_id in self.dataset.bases.keys():
            if base_id not in base_counter:
                base_order.append(base_id)
        pop_order = [pop for pop, _ in pop_counter.most_common()]
        for pop_id in self.dataset.popular_locations.keys():
            if pop_id not in pop_counter:
                pop_order.append(pop_id)
        return base_order, pop_order

    def _estimate_on_scene(self, event: IncidentEvent, site: Any) -> float:
        priority = event.priority or _safe_int(event.incident_type)
        ranges = {1: (8.0, 10.0), 2: (6.5, 8.5), 3: (5.0, 7.0)}
        low, high = ranges.get(priority, (6.0, 8.0))
        return self.random.uniform(low, high)

    def _response_target_minutes(self, event: IncidentEvent) -> float | None:
        rule = self.context.rule
        if rule.base_perfect_redeploy or rule.popular_perfect_redeploy:
            return 12.0
        if (
            rule.base_peak_redeploy
            or rule.popular_peak_redeploy
            or rule.base_peak_urban_redeploy
            or rule.popular_peak_urban_redeploy
        ) and self.fleet._is_peak_event(event):
            return 14.0
        return None

    def _redeploy_penalty_minutes(self, event: IncidentEvent) -> float:
        rule = self.context.rule
        if rule.base_perfect_redeploy or rule.popular_perfect_redeploy:
            return 0.0
        penalty = 0.0
        if rule.base_random_redeploy or rule.popular_random_redeploy:
            penalty += self.random.uniform(1.5, 3.0)
        peak_rule_active = (
            rule.base_peak_redeploy
            or rule.popular_peak_redeploy
            or rule.base_peak_urban_redeploy
            or rule.popular_peak_urban_redeploy
        )
        if peak_rule_active and not self.fleet._is_peak_event(event):
            # penalty += self.random.uniform(0.5, 2.0)
            penalty += self.random.uniform(0.0, 0.0)  # or set to 0.0
        if not peak_rule_active and not (
            rule.base_random_redeploy or rule.popular_random_redeploy
        ):
            penalty += self.random.uniform(2.0, 4.0)
        return penalty

    def _select_hospital(self, event: IncidentEvent):
        if not self.dataset.hospitals:
            return None
        if event.metadata.get("transported_to_hospital_id") in self.dataset.hospitals:
            return self.dataset.hospitals[event.metadata["transported_to_hospital_id"]]
        if event.longitude is None or event.latitude is None:
            return next(iter(self.dataset.hospitals.values()))

        def _distance(hospital):
            return haversine_miles(
                hospital.longitude, hospital.latitude, event.longitude, event.latitude
            )

        return min(self.dataset.hospitals.values(), key=_distance)

    def _estimate_transport(self, event: IncidentEvent, hospital: Any) -> float:
        if not hospital or event.longitude is None or event.latitude is None:
            return 12.0
        distance = haversine_miles(
            hospital.longitude, hospital.latitude, event.longitude, event.latitude
        )
        speed = URBAN_SPEED_MPH if _event_is_urban(event, None) else RURAL_SPEED_MPH
        return distance / max(speed, 1e-3) * 60

    def _hospital_turnaround_minutes(self, priority_flow_active: bool) -> float:
        rule = self.context.rule
        downtime_hours = rule.hospital_downtime_hours or 0.0
        if rule.priority_incident_types and not priority_flow_active:
            downtime_hours = 0.0
        base_turnaround = DEFAULT_HOSPITAL_TURNAROUND_MIN
        if rule.base_perfect_redeploy or rule.popular_perfect_redeploy:
            base_turnaround = min(base_turnaround, 8.0)
        downtime = downtime_hours * 60.0
        return min(base_turnaround + downtime, 15.0)

    def _extra_downtime(
        self,
        redeploy_reason: str,
        *,
        target_post_kind: str,
        staged_from_hospital: bool = False,
        priority_flow_active: bool,
    ) -> float:
        rule = self.context.rule
        if rule.priority_incident_types and not priority_flow_active:
            return 0.0
        if rule.base_perfect_redeploy or rule.popular_perfect_redeploy:
            hours = rule.popular_location_downtime_hours
            return min(hours * 60.0, 10.0) if hours else 0.0
        if target_post_kind != "popular" and not redeploy_reason.startswith("popular"):
            return 0.0
        hours = rule.popular_location_downtime_hours
        if hours is not None:
            return min(hours * 60.0, 10.0)
        if staged_from_hospital:
            return min(DEFAULT_HOSPITAL_TURNAROUND_MIN, 10.0)
        return 0.0


class SimulationRunner:
    """Runs a scenario using the rule catalog and the SimPy engine."""

    def __init__(
        self,
        *,
        location_repo: LocationRepository | None = None,
        travel_repo: TravelMatrixRepository | None = None,
        rule_catalog: RuleCatalog | None = None,
        scenario_builder: ScenarioBuilder | None = None,
        prefer_database: bool = True,
    ) -> None:
        self.prefer_database = prefer_database
        self.location_repo = location_repo or LocationRepository()
        self.travel_repo = travel_repo or TravelMatrixRepository()
        self.rule_catalog = rule_catalog or RuleCatalog()
        self.scenario_builder = scenario_builder or ScenarioBuilder(self.rule_catalog)
        self.dataset = self._load_dataset()
        self.matrices = self._load_matrices()
        self.nearest_lookup = NearestLookupBuilder(self.dataset, self.matrices).load()

    def _load_dataset(self) -> LocationDataset:
        if not self.prefer_database:
            return self.location_repo.load_dataset()
        try:
            return self.location_repo.load_dataset(force_refresh=True)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning(
                "Unable to refresh locations from DB (%s); using cached snapshot.", exc
            )
            return self.location_repo.load_dataset(force_refresh=False)

    def _load_matrices(self) -> TravelMatrixBundle:
        if not self.prefer_database:
            return self.travel_repo.load_all()
        try:
            return self.travel_repo.load_all(force_refresh=True)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning(
                "Unable to refresh travel matrices from DB (%s); using cached snapshot.",
                exc,
            )
            return self.travel_repo.load_all(force_refresh=False)

    def run(
        self,
        rule_id: str,
        template_name: str,
        *,
        seed: int | None = None,
        history_paths: Sequence[Path] | None = None,
    ) -> SimulationResults:
        context = self.scenario_builder.build(
            template_name,
            rule_id,
            seed=seed,
            history_paths=history_paths,
        )
        engine = SimulationEngine(
            context,
            dataset=self.dataset,
            matrices=self.matrices,
            seed=seed,
            nearest_lookup=self.nearest_lookup,
        )
        return engine.run()

    def evaluate_rules(
        self,
        template_name: str,
        *,
        rule_ids: Sequence[str] | None = None,
        seed: int = 0,
        history_paths: Sequence[Path] | None = None,
    ) -> List[Dict[str, Any]]:
        selected = rule_ids or self.rule_catalog.ids()
        rows: List[Dict[str, Any]] = []
        for offset, rule_id in enumerate(selected):
            rule = self.rule_catalog.get(rule_id)
            result = self.run(
                rule_id,
                template_name,
                seed=seed + offset,
                history_paths=history_paths,
            )
            group = self.categorize_rule(rule)
            row = {
                "rule": rule_id,
                "category": group,
                "deployment_focus": self.deployment_focus(rule),
            }
            row.update(result.metrics)
            rows.append(row)
        return rows

    @staticmethod
    def categorize_rule(rule: RuleConfig) -> str:
        if rule.base_perfect_redeploy or rule.popular_perfect_redeploy:
            return "perfect"
        if (
            rule.base_peak_redeploy
            or rule.popular_peak_redeploy
            or rule.base_peak_urban_redeploy
            or rule.popular_peak_urban_redeploy
        ):
            return "peaktime"
        if rule.base_random_redeploy or rule.popular_random_redeploy:
            return "random"
        return "baseline"

    @staticmethod
    def deployment_focus(rule: RuleConfig) -> str:
        base_flags = any(
            (
                rule.base_random_redeploy,
                rule.base_peak_urban_redeploy,
                rule.base_peak_redeploy,
                rule.base_perfect_redeploy,
            )
        )
        popular_flags = any(
            (
                rule.popular_random_redeploy,
                rule.popular_peak_urban_redeploy,
                rule.popular_peak_redeploy,
                rule.popular_perfect_redeploy,
            )
        )
        if base_flags and popular_flags:
            return "hybrid"
        if base_flags:
            return "base"
        if popular_flags:
            return "popular"
        return "none"


def haversine_miles(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return the great-circle distance between two coordinates in miles."""
    r = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _event_is_urban(event: IncidentEvent, site: Any) -> bool:
    if site:
        return site.is_urban
    if event.location_type:
        return str(event.location_type).lower().startswith("urban")
    return True


def _event_priority(event: IncidentEvent, rule: RuleConfig) -> bool:
    if not rule.priority_incident_types:
        return False
    if event.priority in rule.priority_incident_types:
        return True
    if (
        event.incident_type
        and _safe_int(event.incident_type) in rule.priority_incident_types
    ):
        return True
    return False


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _needs_hospital_transport(event: IncidentEvent, rng: random.Random) -> bool:
    """Determine if the incident demands transport based on priority."""
    priority = event.priority or _safe_int(event.incident_type)
    if priority in {1, 2}:
        return True
    if priority == 3:
        # 90% of type-3 incidents still transport, 10% resolve on scene.
        return rng.random() < 0.9
    return False


__all__ = ["SimulationRunner", "SimulationEngine", "SimulationResults"]
