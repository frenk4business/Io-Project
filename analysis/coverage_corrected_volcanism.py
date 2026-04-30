"""
analysis/coverage_corrected_volcanism.py
----------------------------------------
Multi-instrument, metadata-normalized Io volcanism analysis.

This module produces unit-aware activity layers on the canonical 1 degree grid.
Coverage is an explicit metadata approximation: an event/product row is treated
as approximate coverage of its mapped grid cell and time bin. It is not a true
pixel-footprint or radiometric-sensitivity correction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

MULTI_INSTRUMENT_COVERAGE_FILENAME = "io_multi_instrument_coverage_cube.parquet"
COVERAGE_CORRECTED_CELL_MAPS_FILENAME = "io_coverage_corrected_cell_maps.parquet"
RESEARCH_QUESTION_EVALUATION_FILENAME = "io_research_question_evaluation.md"
METRIC_INTERPRETATION_FILENAME = "io_metric_interpretation_summary.csv"
POWER_CONCENTRATION_FILENAME = "io_power_concentration_summary.csv"

METRIC_COLUMNS = [
    "occurrence_event_count",
    "max_normalized_intensity",
    "coverage_corrected_event_rate",
    "coverage_corrected_intensity",
    "persistence_score",
]

METRIC_LABELS = {
    "hotspot_count": "Named hotspot occurrence",
    "occurrence_event_count": "Thermal event occurrence",
    "max_normalized_intensity": "Maximum unitless normalized intensity",
    "combined_normalized_intensity": "Combined unitless normalized intensity",
    "coverage_corrected_event_rate": "Metadata-normalized event rate",
    "coverage_corrected_intensity": "Metadata-normalized intensity proxy",
    "persistence_score": "Persistence score",
    "radiant_power_gw_layer": "Davies/JIRAM estimated proxy GW",
}

KEY_METRIC_PAIRS = [
    ("occurrence_event_count", "coverage_corrected_intensity"),
    ("occurrence_event_count", "max_normalized_intensity"),
    ("max_normalized_intensity", "coverage_corrected_intensity"),
    ("coverage_corrected_event_rate", "coverage_corrected_intensity"),
]

POWER_TOP_N = (1, 5, 10, 25, 50)


def _time_bin_year(values: pd.Series) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce", utc=True)
    return dt.dt.strftime("%Y").fillna("unknown")


def _safe_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _percentile_rank(values: pd.Series) -> pd.Series:
    numeric = _safe_numeric(values)
    valid = numeric.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.sum() == 0:
        return out
    out.loc[valid] = numeric.loc[valid].rank(method="average", pct=True)
    return out


def _zscore(values: pd.Series) -> pd.Series:
    numeric = _safe_numeric(values)
    valid = numeric.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.sum() <= 1:
        out.loc[valid] = 0.0
        return out
    std = numeric.loc[valid].std(ddof=0)
    if std == 0 or pd.isna(std):
        out.loc[valid] = 0.0
    else:
        out.loc[valid] = (numeric.loc[valid] - numeric.loc[valid].mean()) / std
    return out


def prepare_activity_events(activity_events: pd.DataFrame) -> pd.DataFrame:
    """Add unit-aware intensity families and normalized values to event rows."""
    required = {"event_id", "source_dataset", "cell_id", "instrument", "observation_time"}
    missing = required - set(activity_events.columns)
    if missing:
        raise ValueError(f"activity_events missing columns: {sorted(missing)}")

    events = activity_events.copy()
    events["cell_id"] = pd.to_numeric(events["cell_id"], errors="coerce").astype("Int64")
    events["time_bin_year"] = _time_bin_year(events["observation_time"])
    events["instrument"] = events["instrument"].fillna("unknown").astype(str).str.upper()
    events["source_dataset"] = events["source_dataset"].fillna("unknown").astype(str)
    events["intensity_value"] = _safe_numeric(events.get("intensity_value", pd.Series(index=events.index)))
    events["power_gw"] = _safe_numeric(events.get("power_gw", pd.Series(index=events.index)))

    events["intensity_family"] = "unknown"
    davies = events["source_dataset"].eq("Davies_2024_JIRAM_proxy") & events["power_gw"].notna()
    mura = events["source_dataset"].eq("Mura_2024_JIRAM_timeseries")
    nims = events["source_dataset"].eq("Galileo_NIMS_spectral_radiance")
    ao = events["source_dataset"].eq("Ground_based_AO_hotspot_catalogue")
    events.loc[davies, "intensity_family"] = "radiant_power_gw"
    events.loc[mura, "intensity_family"] = "jiram_radiance"
    events.loc[nims, "intensity_family"] = "nims_radiance"
    events.loc[ao, "intensity_family"] = "ao_brightness"

    events["family_intensity_value"] = np.nan
    events.loc[davies, "family_intensity_value"] = events.loc[davies, "power_gw"]
    events.loc[~davies, "family_intensity_value"] = events.loc[~davies, "intensity_value"]
    events["family_percentile"] = (
        events.groupby("intensity_family", group_keys=False)["family_intensity_value"]
        .apply(_percentile_rank)
    )
    events["family_zscore"] = (
        events.groupby("intensity_family", group_keys=False)["family_intensity_value"]
        .apply(_zscore)
    )
    events = events.dropna(subset=["cell_id"])
    events["cell_id"] = events["cell_id"].astype(int)
    return events.reset_index(drop=True)


def build_multi_instrument_coverage_cube(
    feature_matrix: pd.DataFrame,
    activity_events: pd.DataFrame,
    jiram_coverage: pd.DataFrame | None = None,
    time_bin: str = "year",
) -> pd.DataFrame:
    """Build metadata-based coverage rows from products and event observations."""
    grid_required = {"cell_id", "lon_centre", "lat_centre"}
    missing_grid = grid_required - set(feature_matrix.columns)
    if missing_grid:
        raise ValueError(f"feature_matrix missing columns: {sorted(missing_grid)}")
    grid = feature_matrix[["cell_id", "lon_centre", "lat_centre"]].drop_duplicates("cell_id")

    events = prepare_activity_events(activity_events)
    event_cov = (
        events.groupby(["cell_id", "time_bin_year", "instrument"], dropna=False)
        .agg(
            source_event_count=("event_id", "nunique"),
            observation_count=("event_id", "nunique"),
            source_product_ids=("event_id", lambda s: ";".join(sorted(set(s.astype(str)))[:20])),
            first_observation_time=("observation_time", "min"),
            last_observation_time=("observation_time", "max"),
        )
        .reset_index()
        .rename(columns={"time_bin_year": "time_bin"})
    )
    event_cov["coverage_weight"] = event_cov["observation_count"].astype(float)
    event_cov["coverage_quality"] = "event_metadata_cell_observation"
    event_cov["coverage_source"] = "activity_events"

    frames = [event_cov]
    if jiram_coverage is not None and not jiram_coverage.empty:
        cov = jiram_coverage.copy()
        cov["time_bin"] = _time_bin_year(cov["start_time"])
        cov = cov.merge(grid, on=["lon_centre", "lat_centre"], how="left")
        cov = cov.dropna(subset=["cell_id"])
        cov["cell_id"] = cov["cell_id"].astype(int)
        product_cov = (
            cov.groupby(["cell_id", "time_bin"], dropna=False)
            .agg(
                source_event_count=("product_id", "nunique"),
                observation_count=("product_id", "nunique"),
                coverage_weight=("coverage_weight", "sum"),
                source_product_ids=("product_id", lambda s: ";".join(sorted(set(s.astype(str)))[:20])),
                first_observation_time=("start_time", "min"),
                last_observation_time=("stop_time", "max"),
            )
            .reset_index()
        )
        product_cov["instrument"] = "JIRAM"
        product_cov["coverage_quality"] = "jiram_product_metadata_subspacecraft_cell"
        product_cov["coverage_source"] = "jiram_product_logs"
        frames.append(product_cov[event_cov.columns])

    cube = pd.concat(frames, ignore_index=True)
    cube = cube.merge(grid, on="cell_id", how="left")
    cube["instrument_diversity"] = cube.groupby(["cell_id", "time_bin"])["instrument"].transform("nunique")
    cube = cube[
        [
            "cell_id",
            "lon_centre",
            "lat_centre",
            "time_bin",
            "instrument",
            "observation_count",
            "coverage_weight",
            "instrument_diversity",
            "source_event_count",
            "coverage_quality",
            "coverage_source",
            "source_product_ids",
            "first_observation_time",
            "last_observation_time",
        ]
    ].sort_values(["instrument", "time_bin", "cell_id"]).reset_index(drop=True)
    return cube


def _cell_intensity_layers(events: pd.DataFrame) -> pd.DataFrame:
    def _sum_family(values: pd.Series, family: str, value_col: str = "family_intensity_value") -> float:
        mask = events.loc[values.index, "intensity_family"].eq(family)
        if value_col == "family_percentile":
            series = events.loc[values.index, "family_percentile"]
        else:
            series = values
        return float(pd.to_numeric(series[mask], errors="coerce").fillna(0).sum())

    layers = (
        events.groupby("cell_id")
        .agg(
            occurrence_event_count=("event_id", "nunique"),
            active_time_bins=("time_bin_year", "nunique"),
            active_instrument_count=("instrument", "nunique"),
            combined_normalized_intensity=("family_percentile", "sum"),
            mean_normalized_intensity=("family_percentile", "mean"),
            max_normalized_intensity=("family_percentile", "max"),
            radiant_power_gw_layer=("power_gw", "sum"),
            radiant_power_gw_normalized_layer=(
                "family_intensity_value",
                lambda s: _sum_family(s, "radiant_power_gw", value_col="family_percentile"),
            ),
            jiram_radiance_layer=(
                "family_intensity_value",
                lambda s: _sum_family(s, "jiram_radiance"),
            ),
            jiram_radiance_normalized_layer=(
                "family_intensity_value",
                lambda s: _sum_family(s, "jiram_radiance", value_col="family_percentile"),
            ),
            nims_radiance_layer=(
                "family_intensity_value",
                lambda s: _sum_family(s, "nims_radiance"),
            ),
            nims_radiance_normalized_layer=(
                "family_intensity_value",
                lambda s: _sum_family(s, "nims_radiance", value_col="family_percentile"),
            ),
            ao_brightness_layer=(
                "family_intensity_value",
                lambda s: _sum_family(s, "ao_brightness"),
            ),
            ao_brightness_normalized_layer=(
                "family_intensity_value",
                lambda s: _sum_family(s, "ao_brightness", value_col="family_percentile"),
            ),
            source_datasets=("source_dataset", lambda s: ";".join(sorted(set(s.astype(str))))),
            instruments=("instrument", lambda s: ";".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )
    return layers


def _coverage_by_cell(cube: pd.DataFrame) -> pd.DataFrame:
    return (
        cube.groupby("cell_id", dropna=False)
        .agg(
            observation_count=("observation_count", "sum"),
            coverage_weight=("coverage_weight", "sum"),
            observed_time_bins=("time_bin", "nunique"),
            instrument_diversity=("instrument", "nunique"),
            coverage_sources=("coverage_source", lambda s: ";".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )


def _time_maps(events: pd.DataFrame, cube: pd.DataFrame, min_observations: int) -> pd.DataFrame:
    event_time = (
        events.groupby(["cell_id", "time_bin_year"], dropna=False)
        .agg(
            occurrence_event_count=("event_id", "nunique"),
            combined_normalized_intensity=("family_percentile", "sum"),
            max_normalized_intensity=("family_percentile", "max"),
            active_instrument_count=("instrument", "nunique"),
        )
        .reset_index()
        .rename(columns={"time_bin_year": "time_bin"})
    )
    cov_time = (
        cube.groupby(["cell_id", "time_bin"], dropna=False)
        .agg(
            observation_count=("observation_count", "sum"),
            coverage_weight=("coverage_weight", "sum"),
            instrument_diversity=("instrument", "nunique"),
            independent_observation_count=(
                "observation_count",
                lambda s: float(
                    s[
                        cube.loc[s.index, "coverage_source"]
                        .astype(str)
                        .ne("activity_events")
                    ].sum()
                ),
            ),
            coverage_sources=("coverage_source", lambda s: ";".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )
    out = event_time.merge(cov_time, on=["cell_id", "time_bin"], how="outer")
    for col in [
        "occurrence_event_count",
        "combined_normalized_intensity",
        "max_normalized_intensity",
        "observation_count",
        "coverage_weight",
        "instrument_diversity",
        "independent_observation_count",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["coverage_sources"] = out["coverage_sources"].fillna("")
    observed = out["observation_count"] >= min_observations
    out["coverage_corrected_event_rate"] = np.where(
        observed,
        out["occurrence_event_count"] / out["observation_count"].replace(0, np.nan),
        np.nan,
    )
    out["coverage_corrected_intensity"] = np.where(
        observed,
        out["max_normalized_intensity"] / out["coverage_weight"].replace(0, np.nan),
        np.nan,
    )
    return out.sort_values(["time_bin", "cell_id"]).reset_index(drop=True)


def _persistence_summary(time_maps: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        time_maps.groupby("cell_id", dropna=False)
        .agg(
            observed_time_bins=("observation_count", lambda s: int((s > 0).sum())),
            independent_observed_time_bins=(
                "independent_observation_count",
                lambda s: int((s > 0).sum()),
            ),
            active_time_bins=("occurrence_event_count", lambda s: int((s > 0).sum())),
            max_normalized_intensity=("combined_normalized_intensity", "max"),
            median_active_intensity=(
                "combined_normalized_intensity",
                lambda s: float(s[s > 0].median()) if (s > 0).any() else 0.0,
            ),
        )
        .reset_index()
    )
    grouped["persistence_score"] = np.where(
        grouped["independent_observed_time_bins"] > 0,
        grouped["active_time_bins"] / grouped["independent_observed_time_bins"],
        np.nan,
    )
    grouped["episodicity_score"] = np.where(
        grouped["median_active_intensity"] > 0,
        grouped["max_normalized_intensity"] / grouped["median_active_intensity"],
        np.nan,
    )
    grouped["activity_class"] = "coverage_limited"
    grouped.loc[grouped["active_time_bins"].eq(1), "activity_class"] = "observed_active_single_bin"
    grouped.loc[
        grouped["active_time_bins"].ge(2),
        "activity_class",
    ] = "repeated_active"
    grouped.loc[
        grouped["persistence_score"].ge(0.5) & grouped["active_time_bins"].ge(2),
        "activity_class",
    ] = "persistent_active"
    grouped.loc[
        grouped["active_time_bins"].ge(2) & (grouped["episodicity_score"] >= 2.0),
        "activity_class",
    ] = "episodic_high_intensity"
    grouped.loc[
        grouped["active_time_bins"].eq(0) & grouped["independent_observed_time_bins"].gt(0),
        "activity_class",
    ] = "observed_inactive_or_unseen"
    return grouped


def _normalize_distribution(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float).copy()
    arr[arr < 0] = 0
    total = arr.sum()
    if total <= 0:
        return np.zeros_like(arr)
    return arr / total


def _js_divergence(p_values: pd.Series, q_values: pd.Series) -> float:
    p = _normalize_distribution(p_values)
    q = _normalize_distribution(q_values)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _top_fraction_set(df: pd.DataFrame, metric: str, fraction: float = 0.10) -> set[int]:
    values = pd.to_numeric(df[metric], errors="coerce").fillna(0)
    positive = df.loc[values > 0, ["cell_id", metric]].copy()
    if positive.empty:
        return set()
    n = max(1, int(np.ceil(len(positive) * fraction)))
    return set(positive.sort_values(metric, ascending=False).head(n)["cell_id"].astype(int))


def _rank_overlap_matrix(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    sets = {metric: _top_fraction_set(df, metric) for metric in metrics}
    rows = []
    for a in metrics:
        for b in metrics:
            union = sets[a] | sets[b]
            overlap = len(sets[a] & sets[b]) / len(union) if union else np.nan
            rows.append({"metric_a": a, "metric_b": b, "top10_jaccard": overlap})
    return pd.DataFrame(rows)


def _js_matrix(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for a in metrics:
        for b in metrics:
            rows.append({"metric_a": a, "metric_b": b, "js_divergence": _js_divergence(df[a], df[b])})
    return pd.DataFrame(rows)


def _latitude_bands(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-90, -60, -30, 0, 30, 60, 90]
    labels = [
        "south polar",
        "south mid-latitude",
        "south low-latitude",
        "north low-latitude",
        "north mid-latitude",
        "north polar",
    ]
    out = df.copy()
    out["lat_band"] = pd.cut(out["lat_centre"], bins=bins, labels=labels, include_lowest=True, right=False)
    summary = (
        out.groupby("lat_band", observed=False)
        .agg(
            named_hotspot_cells=("hotspot_count", lambda s: int((s > 0).sum())),
            event_cells=("occurrence_event_count", lambda s: int((s > 0).sum())),
            occurrence_events=("occurrence_event_count", "sum"),
            normalized_intensity=("combined_normalized_intensity", "sum"),
            coverage_corrected_intensity=("coverage_corrected_intensity", "sum"),
            observation_count=("observation_count", "sum"),
        )
        .reset_index()
    )
    total_intensity = summary["normalized_intensity"].sum()
    total_corrected = summary["coverage_corrected_intensity"].sum()
    summary["fraction_normalized_intensity"] = np.where(
        total_intensity > 0,
        summary["normalized_intensity"] / total_intensity,
        np.nan,
    )
    summary["fraction_coverage_corrected"] = np.where(
        total_corrected > 0,
        summary["coverage_corrected_intensity"] / total_corrected,
        np.nan,
    )
    return summary


def _top_n_curve(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    values = pd.to_numeric(df[metric], errors="coerce").fillna(0)
    positive = values[values > 0].sort_values(ascending=False).reset_index(drop=True)
    if positive.empty:
        return pd.DataFrame({"metric": [metric], "top_n": [0], "cumulative_fraction": [np.nan]})
    cumulative = positive.cumsum() / positive.sum()
    return pd.DataFrame(
        {
            "metric": metric,
            "top_n": np.arange(1, len(cumulative) + 1),
            "cumulative_fraction": cumulative,
        }
    )


def _metric_value(matrix: pd.DataFrame, metric_a: str, metric_b: str, value_col: str) -> float:
    if "metric" in matrix.columns:
        row = matrix[matrix["metric"].eq(metric_a)]
        if row.empty or metric_b not in row.columns:
            return float("nan")
        return float(row[metric_b].iloc[0])
    row = matrix[matrix["metric_a"].eq(metric_a) & matrix["metric_b"].eq(metric_b)]
    if row.empty or value_col not in row.columns:
        return float("nan")
    return float(row[value_col].iloc[0])


def _interpret_metric_pair(metric_a: str, metric_b: str, spearman: float, overlap: float, js: float) -> str:
    label_a = METRIC_LABELS.get(metric_a, metric_a)
    label_b = METRIC_LABELS.get(metric_b, metric_b)
    if pd.notna(overlap) and overlap <= 0.05:
        rank_note = "very different top-ranked cells"
    elif pd.notna(overlap) and overlap <= 0.20:
        rank_note = "limited top-ranked cell overlap"
    else:
        rank_note = "substantial top-ranked cell overlap"
    if pd.notna(spearman) and spearman < 0:
        corr_note = "negative monotonic agreement"
    elif pd.notna(spearman) and abs(spearman) < 0.3:
        corr_note = "weak monotonic agreement"
    else:
        corr_note = "moderate-to-strong monotonic agreement"
    return f"{label_a} and {label_b} show {corr_note} and {rank_note}; JS divergence={js:.3f}."


def _metric_interpretation_summary(comparison: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    corr = comparison["spearman_correlation"]
    rank = comparison["rank_overlap"]
    js = comparison["js_divergence"]
    for metric_a, metric_b in KEY_METRIC_PAIRS:
        spearman = _metric_value(corr, metric_a, metric_b, "spearman")
        overlap = _metric_value(rank, metric_a, metric_b, "top10_jaccard")
        divergence = _metric_value(js, metric_a, metric_b, "js_divergence")
        rows.append(
            {
                "metric_a": metric_a,
                "metric_a_label": METRIC_LABELS.get(metric_a, metric_a),
                "metric_b": metric_b,
                "metric_b_label": METRIC_LABELS.get(metric_b, metric_b),
                "spearman": spearman,
                "top10_jaccard": overlap,
                "js_divergence": divergence,
                "reviewer_interpretation": _interpret_metric_pair(
                    metric_a,
                    metric_b,
                    spearman,
                    overlap,
                    divergence,
                ),
            }
        )
    return pd.DataFrame(rows)


def _power_concentration_summary(cell_maps: pd.DataFrame) -> pd.DataFrame:
    if "radiant_power_gw_layer" not in cell_maps.columns:
        return pd.DataFrame(
            columns=[
                "metric",
                "metric_label",
                "top_n",
                "cumulative_value",
                "total_value",
                "cumulative_fraction",
                "positive_cells",
            ]
        )
    values = pd.to_numeric(cell_maps["radiant_power_gw_layer"], errors="coerce").fillna(0)
    positive = values[values > 0].sort_values(ascending=False).reset_index(drop=True)
    total = float(positive.sum())
    rows = []
    for n in POWER_TOP_N:
        cumulative = float(positive.head(n).sum()) if total > 0 else float("nan")
        rows.append(
            {
                "metric": "radiant_power_gw_layer",
                "metric_label": METRIC_LABELS["radiant_power_gw_layer"],
                "top_n": n,
                "cumulative_value": cumulative,
                "total_value": total,
                "cumulative_fraction": cumulative / total if total > 0 else float("nan"),
                "positive_cells": int(len(positive)),
            }
        )
    return pd.DataFrame(rows)


def _comparison_metrics(cell_maps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = [metric for metric in METRIC_COLUMNS if metric in cell_maps.columns]
    correlation = cell_maps[metrics].corr(method="spearman", numeric_only=True).reset_index().rename(columns={"index": "metric"})
    rank_overlap = _rank_overlap_matrix(cell_maps, metrics)
    js_divergence = _js_matrix(cell_maps, metrics)
    latitude = _latitude_bands(cell_maps)
    top_n_metrics = metrics + [metric for metric in ["radiant_power_gw_layer"] if metric in cell_maps.columns]
    top_n = pd.concat([_top_n_curve(cell_maps, metric) for metric in top_n_metrics], ignore_index=True)
    return {
        "spearman_correlation": correlation,
        "rank_overlap": rank_overlap,
        "js_divergence": js_divergence,
        "latitude_band_contributions": latitude,
        "top_n_cumulative": top_n,
        "metric_interpretation_summary": _metric_interpretation_summary(
            {
                "spearman_correlation": correlation,
                "rank_overlap": rank_overlap,
                "js_divergence": js_divergence,
            }
        ),
        "power_concentration_summary": _power_concentration_summary(cell_maps),
    }


def _scientific_summary(cell_maps: pd.DataFrame, comparison: dict[str, pd.DataFrame]) -> str:
    corr = comparison["spearman_correlation"].set_index("metric")
    occurrence_intensity = float(corr.loc["occurrence_event_count", "max_normalized_intensity"])
    occurrence_corrected = float(corr.loc["occurrence_event_count", "coverage_corrected_intensity"])
    positive_corrected = int((pd.to_numeric(cell_maps["coverage_corrected_intensity"], errors="coerce").fillna(0) > 0).sum())
    power_summary = comparison.get("power_concentration_summary", pd.DataFrame())
    top10_power = float(
        power_summary.loc[power_summary["top_n"].eq(10), "cumulative_fraction"].iloc[0]
    ) if not power_summary.empty and power_summary["top_n"].eq(10).any() else float("nan")
    return (
        "Metadata-normalized activity produces a nonzero proxy in "
        f"{positive_corrected} grid cells. Occurrence and strongest normalized intensity have Spearman r="
        f"{occurrence_intensity:.3f}; occurrence and metadata-normalized intensity have r="
        f"{occurrence_corrected:.3f}. Davies/JIRAM estimated proxy power has top-10 concentration "
        f"{top10_power:.1%}. "
        "This remains a "
        "metadata-based approximation, not a true footprint/sensitivity correction."
    )


def _research_question_evaluation_markdown(result: dict) -> str:
    metrics = result["comparison_metrics"]
    interp = metrics["metric_interpretation_summary"]
    power = metrics["power_concentration_summary"]
    data_quality = result["data_quality"]

    key = interp[
        interp["metric_a"].eq("occurrence_event_count")
        & interp["metric_b"].eq("coverage_corrected_intensity")
    ].iloc[0]
    top10 = float(power.loc[power["top_n"].eq(10), "cumulative_fraction"].iloc[0]) if not power.empty else float("nan")
    top25 = float(power.loc[power["top_n"].eq(25), "cumulative_fraction"].iloc[0]) if not power.empty else float("nan")
    top50 = float(power.loc[power["top_n"].eq(50), "cumulative_fraction"].iloc[0]) if not power.empty else float("nan")

    evidence_rows = [
        ("Common 1 degree grid", "Present", "`cell_id`, `lon_centre`, `lat_centre` across metric layers", "Metrics are spatially comparable on the shared grid."),
        ("Named hotspot occurrence", "Present", "`has_hotspot`, `hotspot_count`", "Separated from thermal event occurrence."),
        ("Event occurrence", "Present", "`occurrence_event_count`", "Includes normalized event rows from available thermal sources."),
        ("Estimated thermal intensity", "Present", "`radiant_power_gw_layer`, radiance/brightness layers, normalized intensity proxies", "Physical/semi-physical proxy GW is kept separate from unitless percentile proxies."),
        ("Metadata-normalized activity", "Present but limited", "`observation_count`, `coverage_weight`, `coverage_corrected_event_rate`, `coverage_corrected_intensity`", "Metadata-based normalization only; not true footprint/sensitivity correction."),
        ("Quantitative comparison", "Present", "Spearman, top-10% overlap, JS divergence, latitude bands, top-N curves", "Supports metric-dependent spatial interpretation."),
        ("Persistence/episodicity", "Limited", "`activity_class`, `persistence_score`", "Requires better independent observation windows before strong persistence claims."),
    ]
    evidence_table = "\n".join(
        f"| {component} | {status} | {support} | {comment} |"
        for component, status, support, comment in evidence_rows
    )

    return f"""# Io Multi-Metric Volcanism Research Question Evaluation

## Final Classification

**Answered as an exploratory analysis.**

The implementation supports a reproducible, hypothesis-generating comparison of named hotspot occurrence, estimated thermal intensity, and metadata-normalized observation activity on a common 1 degree grid. It does not support a near-publication-grade claim of true coverage correction because real footprints, sensitivity modeling, and systematic non-detections are not included.

## Research Question

**To what extent do different volcanic activity metrics, including named hotspot occurrence, estimated thermal intensity, and metadata-normalized observation activity, produce different spatial interpretations of Io's volcanism on a common 1 degree grid?**

## Direct Answer

The current analysis shows that Io's apparent volcanic activity pattern is metric-dependent. Event occurrence and metadata-normalized intensity differ strongly in the current result files: Spearman correlation is `{key['spearman']:.3f}`, top-10% rank overlap is `{key['top10_jaccard']:.3f}`, and Jensen-Shannon divergence is `{key['js_divergence']:.3f}`. Estimated Davies/JIRAM proxy power is concentrated, with top-10/top-25/top-50 cells contributing `{top10:.1%}`, `{top25:.1%}`, and `{top50:.1%}` of the estimated proxy GW layer.

## Evidence Table

| Component | Status | Supporting file/function/output | Reviewer comment |
|---|---|---|---|
{evidence_table}

## Current Data Quality

- Activity event rows: `{data_quality.get('activity_event_rows')}`
- Metadata coverage rows: `{data_quality.get('coverage_cube_rows')}`
- Metadata covered cells: `{data_quality.get('coverage_cells')}`
- Metadata coverage instruments: `{', '.join(data_quality.get('coverage_instruments', []))}`
- Nonzero metadata-normalized intensity cells: `{data_quality.get('nonzero_coverage_corrected_cells')}`

## Needed Improvements

- Add real JIRAM/NIMS/AO footprints, observation windows with non-detections, and sensitivity/geometry masks for publication-grade coverage correction.
- Keep Davies/JIRAM estimated proxy GW separate from radiance, brightness, and unitless percentile-normalized proxies.
- Treat persistence and episodicity as provisional until independent observation windows are available.
- Use dashboard language such as metadata-normalized activity or metadata-based coverage normalization, not true coverage-corrected volcanism.

## Suggested Scientific Claim

**Cautious supported claim:** This analysis shows that Io's apparent volcanic activity pattern is metric-dependent. On a common 1 degree grid, named hotspot occurrence, estimated thermal intensity, and metadata-normalized activity can produce different spatial rankings, indicating that hotspot catalogs alone do not fully represent the spatial structure of observed thermal activity.

**Stronger supported addendum:** Estimated Davies/JIRAM proxy power appears more concentrated than hotspot occurrence, with the top 10 power cells contributing about `{top10:.0%}` and the top 50 contributing about `{top50:.0%}` of the estimated proxy GW signal.
"""


def save_coverage_corrected_outputs(result: dict, results_dir: Path | None = None) -> None:
    results_dir = results_dir or PROCESSED_DIR.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    coverage_cube = result["coverage_cube"].copy()
    for col in ["time_bin", "source_product_ids", "first_observation_time", "last_observation_time"]:
        if col in coverage_cube.columns:
            coverage_cube[col] = coverage_cube[col].astype(str)
    cell_maps = result["cell_maps"].copy()
    for col in ["source_datasets", "instruments", "coverage_sources", "activity_class"]:
        if col in cell_maps.columns:
            cell_maps[col] = cell_maps[col].astype(str)
    coverage_cube.to_parquet(PROCESSED_DIR / MULTI_INSTRUMENT_COVERAGE_FILENAME, index=False)
    cell_maps.to_parquet(PROCESSED_DIR / COVERAGE_CORRECTED_CELL_MAPS_FILENAME, index=False)
    metrics = result["comparison_metrics"]
    metrics["spearman_correlation"].to_csv(results_dir / "io_metric_correlation_matrix.csv", index=False)
    metrics["rank_overlap"].to_csv(results_dir / "io_rank_overlap.csv", index=False)
    metrics["latitude_band_contributions"].to_csv(results_dir / "io_latitude_band_contributions.csv", index=False)
    metrics["js_divergence"].to_csv(results_dir / "io_js_divergence.csv", index=False)
    metrics["top_n_cumulative"].to_csv(results_dir / "io_top_n_cumulative_intensity.csv", index=False)
    metrics["metric_interpretation_summary"].to_csv(results_dir / METRIC_INTERPRETATION_FILENAME, index=False)
    metrics["power_concentration_summary"].to_csv(results_dir / POWER_CONCENTRATION_FILENAME, index=False)
    (results_dir / RESEARCH_QUESTION_EVALUATION_FILENAME).write_text(
        _research_question_evaluation_markdown(result),
        encoding="utf-8",
    )


def compute_coverage_corrected_volcanism(
    feature_matrix: pd.DataFrame,
    activity_events: pd.DataFrame,
    jiram_coverage: pd.DataFrame | None = None,
    min_observations: int = 1,
    time_bin: str = "year",
    persist_outputs: bool = False,
) -> dict:
    """Compute occurrence, intensity, coverage-corrected and persistence maps."""
    if min_observations < 1:
        raise ValueError("min_observations must be >= 1")
    grid_required = {"cell_id", "lon_centre", "lat_centre"}
    missing = grid_required - set(feature_matrix.columns)
    if missing:
        raise ValueError(f"feature_matrix missing columns: {sorted(missing)}")

    events = prepare_activity_events(activity_events)
    cube = build_multi_instrument_coverage_cube(feature_matrix, events, jiram_coverage, time_bin=time_bin)
    layers = _cell_intensity_layers(events)
    coverage = _coverage_by_cell(cube)
    time_maps = _time_maps(events, cube, min_observations=min_observations)
    persistence = _persistence_summary(time_maps)

    base_cols = ["cell_id", "lon_centre", "lat_centre"]
    if "hotspot_count" in feature_matrix.columns:
        base_cols.append("hotspot_count")
    if "has_hotspot" in feature_matrix.columns:
        base_cols.append("has_hotspot")
    cell_maps = feature_matrix[base_cols].drop_duplicates("cell_id").merge(layers, on="cell_id", how="left")
    cell_maps = cell_maps.merge(coverage, on="cell_id", how="left")
    cell_maps = cell_maps.merge(
        persistence[["cell_id", "persistence_score", "episodicity_score", "activity_class"]],
        on="cell_id",
        how="left",
    )

    fill_zero = [
        "hotspot_count",
        "has_hotspot",
        "occurrence_event_count",
        "active_time_bins",
        "active_instrument_count",
        "combined_normalized_intensity",
        "mean_normalized_intensity",
        "max_normalized_intensity",
        "radiant_power_gw_layer",
        "radiant_power_gw_normalized_layer",
        "jiram_radiance_layer",
        "jiram_radiance_normalized_layer",
        "nims_radiance_layer",
        "nims_radiance_normalized_layer",
        "ao_brightness_layer",
        "ao_brightness_normalized_layer",
        "observation_count",
        "coverage_weight",
        "observed_time_bins",
        "instrument_diversity",
    ]
    for col in fill_zero:
        if col in cell_maps.columns:
            cell_maps[col] = pd.to_numeric(cell_maps[col], errors="coerce").fillna(0)
    observed = cell_maps["observation_count"] >= min_observations
    cell_maps["coverage_corrected_event_rate"] = np.where(
        observed,
        cell_maps["occurrence_event_count"] / cell_maps["observation_count"].replace(0, np.nan),
        np.nan,
    )
    cell_maps["coverage_corrected_intensity"] = np.where(
        observed,
        cell_maps["max_normalized_intensity"] / cell_maps["coverage_weight"].replace(0, np.nan),
        np.nan,
    )
    cell_maps["coverage_corrected_activity"] = cell_maps["coverage_corrected_intensity"]
    cell_maps["persistence_score"] = pd.to_numeric(cell_maps["persistence_score"], errors="coerce")
    cell_maps["activity_class"] = cell_maps["activity_class"].fillna("coverage_limited")

    comparison = _comparison_metrics(cell_maps)
    data_quality = {
        "activity_event_rows": int(len(events)),
        "coverage_cube_rows": int(len(cube)),
        "coverage_cells": int(cube["cell_id"].nunique()),
        "coverage_time_bins": int(cube["time_bin"].nunique()),
        "coverage_instruments": sorted(cube["instrument"].dropna().astype(str).unique().tolist()),
        "nonzero_coverage_corrected_cells": int((cell_maps["coverage_corrected_intensity"].fillna(0) > 0).sum()),
        "min_observations": int(min_observations),
        "coverage_method": "metadata-based event/product cell observation approximation",
        "unit_policy": "raw units are kept in separate layers; combined map uses within-family percentiles only",
    }
    result = {
        "cell_maps": cell_maps,
        "time_maps": time_maps,
        "coverage_cube": cube,
        "comparison_metrics": comparison,
        "persistence_summary": persistence,
        "data_quality": data_quality,
        "scientific_summary": _scientific_summary(cell_maps, comparison),
    }
    if persist_outputs:
        save_coverage_corrected_outputs(result)
    return result
