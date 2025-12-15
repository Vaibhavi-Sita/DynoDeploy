"""
Recalculate peak-rule top-5 using cached quarter evaluation outputs, treating
"downtime" as a fairness benefit and adding priority-gating as a fairness credit.

This script does not rerun the simulator.
It reads the cached CSVs produced by `peak_rules_quarter_tradeoff.ipynb`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def minmax(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    lo = float(s.min())
    hi = float(s.max())
    if np.isclose(hi, lo):
        out = pd.Series(np.ones(len(s)) * 0.5, index=s.index)
        out[s.isna()] = np.nan
        return out
    scaled = (s - lo) / (hi - lo)
    return scaled if higher_is_better else 1.0 - scaled


def gap(a: pd.Series, b: pd.Series) -> float:
    a_mean = float(pd.to_numeric(a, errors="coerce").mean())
    b_mean = float(pd.to_numeric(b, errors="coerce").mean())
    if np.isnan(a_mean) or np.isnan(b_mean):
        return np.nan
    return abs(a_mean - b_mean)


def main() -> None:
    cache_dir = Path(
        "util/simulator/outputs/peak_rules_quarter_tradeoff/cache"
    )
    # Read only the columns we need (these files can be large).
    results_df = pd.read_csv(
        cache_dir / "results_df.csv.gz",
        usecols=[
            "scenario",
            "rule",
            "coverage_ratio",
            "average_response_minutes",
            "average_wait_minutes",
            "mean_queue_length",
            "missed_incidents",
            "total_requests",
        ],
    )
    incidents_df = pd.read_csv(
        cache_dir / "incidents_df.csv.gz",
        usecols=[
            "scenario",
            "rule",
            "site_is_urban",
            "response_minutes",
            "coverage_met",
            "priority",
        ],
    )
    deployments_df = pd.read_csv(
        cache_dir / "deployments_df.csv.gz",
        usecols=[
            "scenario",
            "rule",
            "redeploy_reason",
            "redeploy_minutes",
            "downtime_minutes",
        ],
    )

    results_df["miss_rate"] = results_df["missed_incidents"] / results_df[
        "total_requests"
    ].replace({0: np.nan})

    # Ops aggregates.
    if not deployments_df.empty:
        deployments_df["redeploy_active"] = (
            deployments_df["redeploy_reason"] != "home"
        ).astype(float)
        ops_df = deployments_df.groupby(["scenario", "rule"], as_index=False).agg(
            ops_mean_redeploy_minutes=("redeploy_minutes", "mean"),
            ops_mean_downtime_minutes=("downtime_minutes", "mean"),
            ops_redeploy_rate=("redeploy_active", "mean"),
        )
    else:
        ops_df = pd.DataFrame(
            columns=[
                "scenario",
                "rule",
                "ops_mean_redeploy_minutes",
                "ops_mean_downtime_minutes",
                "ops_redeploy_rate",
            ]
        )

    # Fairness equity gaps from incident logs.
    inc = incidents_df.copy()
    inc["coverage_met"] = inc["coverage_met"].astype(float)
    fair_rows: list[dict] = []
    for (scenario, rule), g in inc.groupby(["scenario", "rule"]):
        urban = g[g["site_is_urban"] == True]
        rural = g[g["site_is_urban"] == False]
        fair_rows.append(
            {
                "scenario": scenario,
                "rule": rule,
                "fair_response_gap_urban": gap(
                    urban["response_minutes"], rural["response_minutes"]
                ),
                "fair_coverage_gap_urban": gap(
                    urban["coverage_met"], rural["coverage_met"]
                ),
                "fair_response_gap_priority": gap(
                    g[g["priority"] == 0]["response_minutes"],
                    g[g["priority"] == 1]["response_minutes"],
                ),
                "fair_coverage_gap_priority": gap(
                    g[g["priority"] == 0]["coverage_met"],
                    g[g["priority"] == 1]["coverage_met"],
                ),
            }
        )
    fair_df = pd.DataFrame(fair_rows)

    metric_df = (
        results_df.merge(ops_df, on=["scenario", "rule"], how="left")
        .merge(fair_df, on=["scenario", "rule"], how="left")
        .copy()
    )

    # Workforce fairness lens: downtime is beneficial (higher is better).
    metric_df["fair_mean_downtime_minutes"] = metric_df["ops_mean_downtime_minutes"]

    # Priority-gating fairness credit (rule policy flag).
    rules_yaml = yaml.safe_load(
        Path("util/simulator/rules.yaml").read_text(encoding="utf-8")
    )["rules"]
    metric_df["fair_priority_gating"] = metric_df["rule"].map(
        lambda rid: int(bool(rules_yaml.get(rid, {}).get("priority_incident_types")))
    )

    # Peak-rule filter (same definition as SimulationRunner.categorize_rule == "peaktime").
    from util.simulator.simulator import SimulationRunner

    runner = SimulationRunner(prefer_database=False)
    all_peak_rules = [
        rid
        for rid in runner.rule_catalog.ids()
        if runner.categorize_rule(runner.rule_catalog.get(rid)) == "peaktime"
    ]

    # STRICT ELIGIBILITY FILTER (matches notebook):
    # - Base redeploy + Popular redeploy (hybrid peak)
    # - Hospital downtime + Popular downtime (both)
    # - Mandatory priority gating
    def is_strictly_eligible(rule_id: str) -> bool:
        r = runner.rule_catalog.get(rule_id)
        has_base_redeploy = bool(r.base_peak_redeploy or r.base_peak_urban_redeploy)
        has_popular_redeploy = bool(
            r.popular_peak_redeploy or r.popular_peak_urban_redeploy
        )
        has_hospital_downtime = r.hospital_downtime_hours is not None
        has_popular_downtime = r.popular_location_downtime_hours is not None
        has_priority_gating = bool(r.priority_incident_types)
        return (
            has_base_redeploy
            and has_popular_redeploy
            and has_hospital_downtime
            and has_popular_downtime
            and has_priority_gating
        )

    peak_rules = [rid for rid in all_peak_rules if is_strictly_eligible(rid)]
    metric_df = metric_df[metric_df["rule"].isin(peak_rules)]

    # Interpretability: complexity count (same toggles used in the notebook).
    def complexity(rule_id: str) -> int:
        rule = runner.rule_catalog.get(rule_id)
        flags = [
            rule.base_peak_urban_redeploy,
            rule.popular_peak_urban_redeploy,
            rule.base_peak_redeploy,
            rule.popular_peak_redeploy,
            rule.hospital_redeploy,
            bool(rule.priority_incident_types),
            rule.hospital_downtime_hours is not None,
            rule.popular_location_downtime_hours is not None,
            (rule.peak_window_start, rule.peak_window_end) != (10, 19),
        ]
        return int(sum(bool(v) for v in flags))

    metric_df["complexity"] = metric_df["rule"].map(complexity)

    # Updated weights: downtime gets high weight inside fairness.
    RULE_WEIGHTS = {
        # Performance (0.40)
        "coverage_ratio": 0.15,
        "average_response_minutes": 0.15,
        "average_wait_minutes": 0.05,
        "mean_queue_length": 0.03,
        "miss_rate": 0.02,
        # Ops (0.15)
        "ops_mean_redeploy_minutes": 0.09,
        "ops_redeploy_rate": 0.06,
        # Fairness (0.25)
        "fair_mean_downtime_minutes": 0.12,
        "fair_priority_gating": 0.05,
        "fair_response_gap_urban": 0.03,
        "fair_response_gap_priority": 0.03,
        "fair_coverage_gap_urban": 0.01,
        "fair_coverage_gap_priority": 0.01,
        # Interpretability (0.20)
        "complexity": 0.20,
    }
    HIB = {
        "coverage_ratio": True,
        "average_response_minutes": False,
        "average_wait_minutes": False,
        "mean_queue_length": False,
        "miss_rate": False,
        "ops_mean_redeploy_minutes": False,
        "ops_redeploy_rate": False,
        "fair_mean_downtime_minutes": True,
        "fair_priority_gating": True,
        "fair_response_gap_urban": False,
        "fair_response_gap_priority": False,
        "fair_coverage_gap_urban": False,
        "fair_coverage_gap_priority": False,
        "complexity": False,
    }

    rule_agg = metric_df.groupby("rule", as_index=False).agg(
        {k: "mean" for k in RULE_WEIGHTS}
    )
    for metric in RULE_WEIGHTS:
        rule_agg[f"score_{metric}"] = minmax(
            rule_agg[metric], higher_is_better=HIB[metric]
        )
    rule_agg["composite_score"] = 0.0
    for metric, w in RULE_WEIGHTS.items():
        col = f"score_{metric}"
        rule_agg["composite_score"] += w * rule_agg[col].fillna(rule_agg[col].median())

    ranked = rule_agg.sort_values("composite_score", ascending=False).reset_index(
        drop=True
    )

    # HARD FILTER (realized downtime): require non-trivial downtime on average.
    if "fair_mean_downtime_minutes" in ranked.columns:
        ranked = ranked[
            ranked["fair_mean_downtime_minutes"].fillna(0) >= 0.5
        ].reset_index(drop=True)

    print("New top-10 peak rules (downtime treated as fairness):")
    print(ranked[["rule", "composite_score"]].head(10).to_string(index=False))
    print("New top-5:", ", ".join(ranked["rule"].head(5).tolist()))


if __name__ == "__main__":
    main()
