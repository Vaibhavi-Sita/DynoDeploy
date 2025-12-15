"""Sanity checks for ensuring data inputs are simulation-ready."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .location_repository import LocationDataset, TravelMatrixBundle


@dataclass(frozen=True)
class ValidationIssue:
    message: str
    severity: str = "ERROR"
    context: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    issues: List[ValidationIssue]


class TravelMatrixValidator:
    """Validates that travel matrices cover the available locations."""

    MAX_EXPECTED_MINUTES = 180.0

    def __init__(
        self,
        dataset: LocationDataset,
        matrices: TravelMatrixBundle,
        *,
        coverage_sample: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.matrices = matrices
        self.coverage_sample = coverage_sample

    def validate(self) -> ValidationReport:
        issues: List[ValidationIssue] = []
        issues.extend(self._check_incident_coverage())
        issues.extend(self._check_extreme_values())
        return ValidationReport(ok=not issues, issues=issues)

    def _check_incident_coverage(self) -> Iterable[ValidationIssue]:
        missing: List[ValidationIssue] = []
        base_ids = list(self.dataset.bases.keys())
        incident_ids = list(self.dataset.incident_locations.keys())
        if self.coverage_sample:
            incident_ids = incident_ids[: self.coverage_sample]
        for incident_id in incident_ids:
            for base_id in base_ids:
                travel_time = self.matrices.travel_time(
                    base_id, incident_id, "base_incident"
                )
                if travel_time is None:
                    missing.append(
                        ValidationIssue(
                            message="Missing base→incident travel time",
                            context={"base_id": base_id, "incident_id": incident_id},
                        )
                    )
        return missing

    def _check_extreme_values(self) -> Iterable[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for matrix_type in ("base_incident", "base_hospital", "base_base"):
            dataframe = getattr(self.matrices, matrix_type)
            if dataframe.empty or "travel_time_minutes" not in dataframe.columns:
                continue
            extreme_rows = dataframe[
                dataframe["travel_time_minutes"] > self.MAX_EXPECTED_MINUTES
            ]
            for _, row in extreme_rows.iterrows():
                issues.append(
                    ValidationIssue(
                        message=f"{matrix_type} travel time exceeds expectation",
                        severity="WARNING",
                        context={
                            "origin": row.get("origin_base_id"),
                            "destination": row.get("destination_incident_id")
                            or row.get("destination_hospital_id")
                            or row.get("destination_base_id"),
                            "minutes": float(row["travel_time_minutes"]),
                        },
                    )
                )
        return issues


def run_default_validations(
    dataset: LocationDataset, matrices: TravelMatrixBundle
) -> ValidationReport:
    """Run the built-in validation suite."""
    validator = TravelMatrixValidator(dataset, matrices)
    return validator.validate()


__all__ = [
    "run_default_validations",
    "TravelMatrixValidator",
    "ValidationIssue",
    "ValidationReport",
]

