"""
analysis/time_resolved_activity.py
----------------------------------
Compare Io hotspot occurrence, estimated JIRAM thermal-emission proxy intensity,
and approximate time-resolved JIRAM product coverage.

The coverage layer is product-centre metadata, not a full pixel-footprint or
radiometric sensitivity model. Treat outputs as hypothesis-generating.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import RAW_DIR
from ingest.observation_coverage_cube import build_coverage_cube
from ingest.jiram_coverage import REQUIRED_COVERAGE_COLUMNS
from ingest.thermal_activity_events import assign_cell_id, normalize_davies_power_events

MURA_TIMESERIES_FILENAME = "mura_2024_hotspot_timeseries.csv"
MURA_REQUIRED_COLUMNS: list[str] = [
    "source_id",
    "name",
    "longitude",
    "latitude",
    "orbit",
    "observation_time",
    "power_gw",
    "instrument",
    "source",
]

PERSISTENCE_CLASSES: tuple[str, ...] = (
    "named_only",
    "thermal_only",
    "persistent_active",
    "episodic_high_power",
    "named_inactive_or_unseen",
    "coverage_limited",
)

LATITUDE_BANDS: list[tuple[float, float, str]] = [
    (-90.0, -60.0, "south polar"),
    (-60.0, -30.0, "south mid-latitude"),
    (-30.0, 0.0, "south low-latitude"),
    (0.0, 30.0, "north low-latitude"),
    (30.0, 60.0, "north mid-latitude"),
    (60.0, 90.0, "north polar"),
]


def load_mura_hotspot_timeseries(path: Path | None = None) -> pd.DataFrame:
    """Load optional Mura et al. 2024 hotspot time-series table.

    The PDF is not parsed automatically because table extraction quality can
    change the science result. A curated CSV must be provided explicitly.
    """
    path = path or RAW_DIR / MURA_TIMESERIES_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Mura time-series table not found at {path}. Create a curated CSV "
            f"with columns: {', '.join(MURA_REQUIRED_COLUMNS)}."
        )
    df = pd.read_csv(path)
    missing = set(MURA_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Mura time-series table missing columns: {sorted(missing)}")
    out = df.copy()
    out["longitude"] = ((pd.to_numeric(out["longitude"], errors="coerce") + 180.0) % 360.0) - 180.0
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["power_gw"] = pd.to_numeric(out["power_gw"], errors="coerce")
    out["orbit"] = pd.to_numeric(out["orbit"], errors="coerce")
    out["observation_time"] = pd.to_datetime(out["observation_time"], errors="coerce", utc=True)
    out = out.dropna(subset=["longitude", "latitude", "power_gw", "orbit", "observation_time"])
    return out[out["power_gw"] > 0].reset_index(drop=True)


def _validate_inputs(
    feature_matrix: pd.DataFrame,
    power_grid: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    fm_required = {"cell_id", "lon_centre", "lat_centre", "has_hotspot", "hotspot_count"}
    pg_required = {"cell_id", "power_count", "sum_power_gw", "primary_power_gw"}
    missing_fm = fm_required - set(feature_matrix.columns)
    missing_pg = pg_required - set(power_grid.columns)
    missing_cov = set(REQUIRED_COVERAGE_COLUMNS) - set(coverage.columns)
    if missing_fm:
        raise ValueError(f"feature_matrix missing columns: {sorted(missing_fm)}")
    if missing_pg:
        raise ValueError(f"power_grid missing columns: {sorted(missing_pg)}")
    if missing_cov:
        raise ValueError(f"coverage missing columns: {sorted(missing_cov)}")


def _coverage_by_cell(coverage: pd.DataFrame) -> pd.DataFrame:
    cov = coverage.copy()
    cov["coverage_weight"] = pd.to_numeric(cov["coverage_weight"], errors="coerce").fillna(1.0)
    return (
        cov.groupby(["lon_centre", "lat_centre"], dropna=False)
        .agg(
            observed_product_count=("product_id", "nunique"),
            coverage_weight=("coverage_weight", "sum"),
            observed_time_bins=("time_bin", "nunique"),
            first_observed=("start_time", "min"),
            last_observed=("stop_time", "max"),
        )
        .reset_index()
    )


def _coverage_cube_by_cell(coverage_cube: pd.DataFrame) -> pd.DataFrame:
    if coverage_cube.empty:
        return pd.DataFrame(
            columns=[
                "cell_id",
                "coverage_cube_observation_count",
                "coverage_weight",
                "observed_time_bins",
                "covered_instruments",
                "best_resolution_km_px",
                "mean_emission_angle_deg",
                "coverage_quality",
            ]
        )
    cube = coverage_cube.copy()
    cube["coverage_weight"] = pd.to_numeric(cube["coverage_weight"], errors="coerce").fillna(0.0)
    cube["observation_count"] = pd.to_numeric(cube["observation_count"], errors="coerce").fillna(0)
    return (
        cube.groupby("cell_id", dropna=False)
        .agg(
            coverage_cube_observation_count=("observation_count", "sum"),
            coverage_weight=("coverage_weight", "sum"),
            observed_time_bins=("time_bin", "nunique"),
            covered_instruments=("instrument", lambda s: ";".join(sorted(set(s.astype(str))))),
            best_resolution_km_px=("best_resolution_km_px", "min"),
            mean_emission_angle_deg=("mean_emission_angle_deg", "mean"),
            coverage_quality=("coverage_quality", lambda s: ";".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )


def _event_summary_by_cell(activity_events: pd.DataFrame) -> pd.DataFrame:
    if activity_events.empty:
        return pd.DataFrame(
            columns=[
                "cell_id",
                "event_count",
                "event_time_bins",
                "event_instruments",
                "source_datasets",
                "sum_event_power_gw",
                "primary_event_power_gw",
                "event_names",
            ]
        )
    events = activity_events.copy()
    events["power_gw"] = pd.to_numeric(events["power_gw"], errors="coerce")
    events["intensity_value"] = pd.to_numeric(events["intensity_value"], errors="coerce")
    return (
        events.groupby("cell_id", dropna=False)
        .agg(
            event_count=("event_id", "nunique"),
            event_time_bins=("time_bin", "nunique"),
            event_instruments=("instrument", lambda s: ";".join(sorted(set(s.astype(str))))),
            source_datasets=("source_dataset", lambda s: ";".join(sorted(set(s.astype(str))))),
            sum_event_power_gw=("power_gw", "sum"),
            primary_event_power_gw=("power_gw", "max"),
            max_intensity_value=("intensity_value", "max"),
            event_names=("name", lambda s: ";".join(sorted(set(s.dropna().astype(str))))[:500]),
        )
        .reset_index()
    )


def _assign_regions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hemisphere_ns"] = np.where(out["lat_centre"] >= 0, "north", "south")
    out["hemisphere_jovian"] = np.where(
        out["lon_centre"].abs() < 90.0,
        "sub-Jovian",
        "anti-Jovian",
    )
    labels = []
    for lat in out["lat_centre"].to_numpy(dtype=float):
        label = "unknown"
        for lat_min, lat_max, band_label in LATITUDE_BANDS:
            if lat >= lat_min and lat < lat_max:
                label = band_label
                break
        labels.append(label)
    out["lat_band"] = labels
    return out


def _classify_activity(row: pd.Series, high_power_threshold: float) -> str:
    has_named = bool(row.get("has_hotspot", 0))
    has_thermal = row.get("power_count", 0) > 0
    covered = row.get("observed_product_count", 0) > 0
    time_bins = row.get("observed_time_bins", 0)
    primary_power = row.get("primary_power_gw", 0.0)

    if has_thermal and time_bins >= 2:
        return "persistent_thermal"
    if has_thermal and primary_power >= high_power_threshold:
        return "episodic_high_power"
    if has_thermal and not has_named:
        return "thermal_only"
    if has_named and not has_thermal:
        return "named_only"
    if not covered and not has_named and not has_thermal:
        return "coverage_limited"
    if has_thermal:
        return "thermal_only"
    return "coverage_limited"


def _classify_activity_v2(row: pd.Series, high_power_threshold: float) -> str:
    has_named = bool(row.get("has_hotspot", 0))
    event_count = row.get("event_count", 0)
    has_thermal = event_count > 0 or row.get("power_count", 0) > 0
    covered = row.get("coverage_cube_observation_count", 0) > 0 or row.get("observed_product_count", 0) > 0
    time_bins = max(row.get("event_time_bins", 0), row.get("observed_time_bins", 0))
    instruments = str(row.get("event_instruments", ""))
    instrument_count = len([x for x in instruments.split(";") if x])
    primary_power = max(
        row.get("primary_event_power_gw", 0.0) if pd.notna(row.get("primary_event_power_gw", 0.0)) else 0.0,
        row.get("primary_power_gw", 0.0),
    )

    if has_thermal and (time_bins >= 2 or instrument_count >= 2):
        return "persistent_active"
    if has_thermal and primary_power >= high_power_threshold:
        return "episodic_high_power"
    if has_thermal and not has_named:
        return "thermal_only"
    if has_named and not has_thermal and covered:
        return "named_inactive_or_unseen"
    if has_named and not has_thermal:
        return "named_only"
    if not covered:
        return "coverage_limited"
    return "coverage_limited"


def compute_time_resolved_activity(
    feature_matrix: pd.DataFrame,
    power_grid: pd.DataFrame,
    coverage: pd.DataFrame,
    mura_timeseries: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Compute occurrence/intensity/coverage comparison tables."""
    _validate_inputs(feature_matrix, power_grid, coverage)

    cols = [
        "cell_id",
        "lon_centre",
        "lat_centre",
        "has_hotspot",
        "hotspot_count",
        "hotspot_names",
        "geology_unit",
    ]
    available_cols = [c for c in cols if c in feature_matrix.columns]
    cell = feature_matrix[available_cols].merge(
        power_grid[["cell_id", "power_count", "sum_power_gw", "primary_power_gw", "power_names"]],
        on="cell_id",
        how="left",
    )
    cell = cell.merge(_coverage_by_cell(coverage), on=["lon_centre", "lat_centre"], how="left")
    for col in ["power_count", "sum_power_gw", "primary_power_gw", "observed_product_count", "coverage_weight", "observed_time_bins"]:
        cell[col] = pd.to_numeric(cell[col], errors="coerce").fillna(0)
    cell["coverage_normalized_proxy"] = np.where(
        cell["observed_product_count"] > 0,
        cell["sum_power_gw"] / cell["observed_product_count"],
        np.nan,
    )

    high_power_threshold = (
        float(cell.loc[cell["power_count"] > 0, "primary_power_gw"].quantile(0.9))
        if (cell["power_count"] > 0).any()
        else float("inf")
    )
    cell["persistence_class"] = cell.apply(
        lambda row: _classify_activity(row, high_power_threshold),
        axis=1,
    )
    cell = _assign_regions(cell)

    if mura_timeseries is not None and not mura_timeseries.empty:
        # Future-ready hook: the curated Mura table can increase persistence
        # evidence when multiple orbits map to the same 1 degree cell.
        mt = mura_timeseries.copy()
        lon_idx = np.floor((mt["longitude"].to_numpy(dtype=float) + 180.0))
        lat_idx = np.floor((mt["latitude"].to_numpy(dtype=float) + 90.0))
        mt["lon_centre"] = -180.0 + lon_idx + 0.5
        mt["lat_centre"] = -90.0 + lat_idx + 0.5
        mt_summary = (
            mt.groupby(["lon_centre", "lat_centre"])
            .agg(mura_orbits=("orbit", "nunique"), mura_max_power_gw=("power_gw", "max"))
            .reset_index()
        )
        cell = cell.merge(mt_summary, on=["lon_centre", "lat_centre"], how="left")
        cell["mura_orbits"] = cell["mura_orbits"].fillna(0)
        cell.loc[(cell["mura_orbits"] >= 2) & (cell["power_count"] > 0), "persistence_class"] = "persistent_thermal"
    else:
        cell["mura_orbits"] = 0

    regional_summary = (
        cell.groupby("lat_band", dropna=False)
        .agg(
            named_hotspot_cells=("has_hotspot", "sum"),
            thermal_proxy_cells=("power_count", lambda s: int((s > 0).sum())),
            total_proxy_gw=("sum_power_gw", "sum"),
            observed_product_count=("observed_product_count", "sum"),
            mean_coverage_normalized_proxy=("coverage_normalized_proxy", "mean"),
        )
        .reset_index()
    )

    comparison_summary = (
        cell.groupby(["hemisphere_ns", "hemisphere_jovian"], dropna=False)
        .agg(
            named_hotspot_cells=("has_hotspot", "sum"),
            thermal_proxy_cells=("power_count", lambda s: int((s > 0).sum())),
            total_proxy_gw=("sum_power_gw", "sum"),
            observed_product_count=("observed_product_count", "sum"),
            coverage_normalized_proxy_sum=("coverage_normalized_proxy", "sum"),
        )
        .reset_index()
    )

    data_quality = {
        "coverage_rows": int(len(coverage)),
        "coverage_cells": int((cell["observed_product_count"] > 0).sum()),
        "coverage_orbits": int(pd.Series(coverage["orbit"]).nunique()),
        "thermal_cells": int((cell["power_count"] > 0).sum()),
        "named_hotspot_cells": int(cell["has_hotspot"].sum()),
        "mura_timeseries_rows": int(len(mura_timeseries)) if mura_timeseries is not None else 0,
        "coverage_method": "JIRAM product-centre count; not full pixel footprint sensitivity",
        "power_definition": "estimated thermal-emission proxy from Davies/JIRAM 4.8 micron spectral radiance",
    }

    return {
        "cell_activity": cell,
        "regional_summary": regional_summary,
        "comparison_summary": comparison_summary,
        "data_quality": data_quality,
    }


def filter_activity_events(
    activity_events: pd.DataFrame,
    instrument: str = "combined",
    time_bin: str = "all",
) -> pd.DataFrame:
    """Filter normalized events by dashboard instrument/time controls."""
    events = activity_events.copy()
    if instrument and instrument.lower() not in {"combined", "all"}:
        instrument_upper = instrument.upper()
        if instrument_upper == "AO":
            events = events[events["instrument"].astype(str).str.upper().str.contains("AO", na=False)]
        else:
            events = events[events["instrument"].astype(str).str.upper().eq(instrument_upper)]
    if time_bin and time_bin != "all":
        events = events[events["time_bin"].astype(str).eq(str(time_bin))]
    return events.reset_index(drop=True)


def filter_coverage_cube(
    coverage_cube: pd.DataFrame,
    instrument: str = "combined",
    time_bin: str = "all",
) -> pd.DataFrame:
    cube = coverage_cube.copy()
    if instrument and instrument.lower() not in {"combined", "all", "sim3168"}:
        cube = cube[cube["instrument"].astype(str).str.upper().eq(instrument.upper())]
    if time_bin and time_bin != "all":
        cube = cube[cube["time_bin"].astype(str).eq(str(time_bin))]
    return cube.reset_index(drop=True)


def compute_time_resolved_activity_v2(
    feature_matrix: pd.DataFrame,
    power_grid: pd.DataFrame,
    jiram_coverage: pd.DataFrame,
    coverage_cube: pd.DataFrame | None = None,
    activity_events: pd.DataFrame | None = None,
    instrument: str = "combined",
    time_bin: str = "all",
) -> dict[str, pd.DataFrame | dict | list[str]]:
    """Compute v2 activity products from events plus a coverage cube."""
    _validate_inputs(feature_matrix, power_grid, jiram_coverage)
    if coverage_cube is None:
        coverage_cube = build_coverage_cube(feature_matrix, jiram_coverage)
    if activity_events is None:
        activity_events = normalize_davies_power_events()

    filtered_events = filter_activity_events(activity_events, instrument, time_bin)
    filtered_cube = filter_coverage_cube(coverage_cube, instrument, time_bin)

    base = compute_time_resolved_activity(feature_matrix, power_grid, jiram_coverage)
    cell = base["cell_activity"].drop(
        columns=[
            c
            for c in [
                "coverage_weight",
                "observed_time_bins",
                "first_observed",
                "last_observed",
                "persistence_class",
            ]
            if c in base["cell_activity"].columns
        ],
        errors="ignore",
    )
    cube_summary = _coverage_cube_by_cell(filtered_cube)
    event_summary = _event_summary_by_cell(filtered_events)
    cell = cell.merge(cube_summary, on="cell_id", how="left")
    cell = cell.merge(event_summary, on="cell_id", how="left")

    numeric_fill = [
        "coverage_cube_observation_count",
        "coverage_weight",
        "observed_time_bins",
        "event_count",
        "event_time_bins",
        "sum_event_power_gw",
        "primary_event_power_gw",
        "max_intensity_value",
    ]
    for col in numeric_fill:
        if col in cell.columns:
            cell[col] = pd.to_numeric(cell[col], errors="coerce").fillna(0)
    for col in ["covered_instruments", "coverage_quality", "event_instruments", "source_datasets", "event_names"]:
        if col in cell.columns:
            cell[col] = cell[col].fillna("")
    cell["coverage_corrected_activity"] = np.where(
        cell["coverage_weight"] > 0,
        cell["sum_event_power_gw"].fillna(0) / cell["coverage_weight"],
        np.nan,
    )
    cell["event_or_proxy_power_gw"] = np.where(
        cell["sum_event_power_gw"] > 0,
        cell["sum_event_power_gw"],
        cell["sum_power_gw"],
    )

    high_power_threshold = (
        float(cell.loc[cell["event_or_proxy_power_gw"] > 0, "event_or_proxy_power_gw"].quantile(0.9))
        if (cell["event_or_proxy_power_gw"] > 0).any()
        else float("inf")
    )
    cell["persistence_class"] = cell.apply(
        lambda row: _classify_activity_v2(row, high_power_threshold),
        axis=1,
    )
    cell = _assign_regions(cell)

    regional_summary = (
        cell.groupby("lat_band", dropna=False)
        .agg(
            named_hotspot_cells=("has_hotspot", "sum"),
            thermal_event_cells=("event_count", lambda s: int((s > 0).sum())),
            total_event_or_proxy_gw=("event_or_proxy_power_gw", "sum"),
            coverage_observation_count=("coverage_cube_observation_count", "sum"),
            coverage_weight=("coverage_weight", "sum"),
            mean_coverage_corrected_activity=("coverage_corrected_activity", "mean"),
        )
        .reset_index()
    )
    comparison_summary = (
        cell.groupby(["hemisphere_ns", "hemisphere_jovian"], dropna=False)
        .agg(
            named_hotspot_cells=("has_hotspot", "sum"),
            thermal_event_cells=("event_count", lambda s: int((s > 0).sum())),
            total_event_or_proxy_gw=("event_or_proxy_power_gw", "sum"),
            coverage_weight=("coverage_weight", "sum"),
            coverage_corrected_activity_sum=("coverage_corrected_activity", "sum"),
        )
        .reset_index()
    )
    available_instruments = sorted(set(activity_events["instrument"].dropna().astype(str).str.upper()))
    available_time_bins = sorted(set(activity_events["time_bin"].dropna().astype(str)))
    coverage_time_bins = sorted(set(coverage_cube["time_bin"].dropna().astype(str))) if not coverage_cube.empty else []
    data_quality = {
        **base["data_quality"],
        "coverage_cube_rows": int(len(coverage_cube)),
        "filtered_coverage_cube_rows": int(len(filtered_cube)),
        "activity_event_rows": int(len(activity_events)),
        "filtered_activity_event_rows": int(len(filtered_events)),
        "available_instruments": available_instruments,
        "activity_time_bins": available_time_bins,
        "coverage_time_bins": coverage_time_bins,
        "coverage_method": "metadata-based product count plus resolution/emission-angle quality weighting",
        "footprint_status": "true pixel footprints and radiometric sensitivity are not reconstructed yet",
        "optional_dataset_status": "Mura/NIMS/AO events are included only when curated structured files are present",
    }
    return {
        "cell_activity": cell,
        "regional_summary": regional_summary,
        "comparison_summary": comparison_summary,
        "coverage_cube": coverage_cube,
        "activity_events": activity_events,
        "data_quality": data_quality,
        "available_instruments": available_instruments,
        "available_time_bins": sorted(set(available_time_bins + coverage_time_bins)),
    }
