"""
ingest/observation_coverage_cube.py
-----------------------------------
Build a gridded, time-resolved observation coverage cube for Io.

The current implementation is metadata-based. It converts available product
logs such as JIRAM PDS rows into 1 degree grid/time-bin/instrument counts and
geometry-quality weights. It does not reconstruct true pixel footprints or
radiometric sensitivity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

COVERAGE_CUBE_FILENAME = "io_observation_coverage_cube.parquet"

REQUIRED_COVERAGE_CUBE_COLUMNS: list[str] = [
    "cell_id",
    "lon_centre",
    "lat_centre",
    "time_bin",
    "instrument",
    "orbit_or_epoch",
    "observation_count",
    "coverage_weight",
    "best_resolution_km_px",
    "mean_emission_angle_deg",
    "coverage_quality",
    "source_product_ids",
]


def _cell_id_from_centres(grid: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    required = {"cell_id", "lon_centre", "lat_centre"}
    missing_grid = required - set(grid.columns)
    missing_rows = {"lon_centre", "lat_centre"} - set(rows.columns)
    if missing_grid:
        raise ValueError(f"Grid missing columns: {sorted(missing_grid)}")
    if missing_rows:
        raise ValueError(f"Coverage rows missing columns: {sorted(missing_rows)}")
    return rows.merge(
        grid[["cell_id", "lon_centre", "lat_centre"]],
        on=["lon_centre", "lat_centre"],
        how="left",
    )


def geometry_coverage_weight(
    spatial_resolution_km_pixel: pd.Series,
    emission_angle_deg: pd.Series,
    base_weight: pd.Series | float = 1.0,
) -> pd.Series:
    """Compute a conservative metadata-quality coverage weight.

    Resolution and emission-angle metadata are useful but incomplete proxies.
    Missing values are downweighted rather than discarded.
    """
    resolution = pd.to_numeric(spatial_resolution_km_pixel, errors="coerce")
    emission = pd.to_numeric(emission_angle_deg, errors="coerce").clip(lower=0, upper=89)
    if not isinstance(base_weight, pd.Series):
        base = pd.Series(float(base_weight), index=resolution.index)
    else:
        base = pd.to_numeric(base_weight, errors="coerce").fillna(1.0)

    resolution_factor = pd.Series(0.75, index=resolution.index, dtype=float)
    resolution_factor.loc[resolution <= 5.0] = 1.0
    resolution_factor.loc[(resolution > 5.0) & (resolution <= 25.0)] = 0.85
    resolution_factor.loc[resolution > 25.0] = 0.65

    emission_factor = np.cos(np.deg2rad(emission)).clip(0.15, 1.0)
    emission_factor = pd.Series(emission_factor, index=resolution.index).fillna(0.75)
    return (base * resolution_factor * emission_factor).clip(lower=0.05)


def coverage_quality_label(row: pd.Series) -> str:
    """Label the dominant limitation of one coverage row."""
    resolution = row.get("spatial_resolution_km_pixel")
    emission = row.get("emission_angle_deg")
    if pd.isna(resolution) and pd.isna(emission):
        return "metadata_only_missing_geometry"
    if pd.isna(resolution):
        return "metadata_only_missing_resolution"
    if pd.isna(emission):
        return "metadata_only_missing_emission_angle"
    if float(emission) >= 70.0:
        return "metadata_geometry_high_emission_angle"
    return "metadata_geometry"


def build_coverage_cube(
    grid: pd.DataFrame,
    jiram_coverage: pd.DataFrame | None = None,
    extra_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate available observation metadata into coverage-cube rows."""
    frames: list[pd.DataFrame] = []
    if jiram_coverage is not None and not jiram_coverage.empty:
        j = jiram_coverage.copy()
        j["instrument"] = "JIRAM"
        j["orbit_or_epoch"] = j["orbit"].astype(str)
        frames.append(j)
    if extra_coverage is not None and not extra_coverage.empty:
        frames.append(extra_coverage.copy())
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COVERAGE_CUBE_COLUMNS)

    rows = pd.concat(frames, ignore_index=True)
    required = {"product_id", "time_bin", "lon_centre", "lat_centre", "instrument", "orbit_or_epoch"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Coverage metadata missing columns: {sorted(missing)}")

    rows = _cell_id_from_centres(grid, rows)
    rows = rows.dropna(subset=["cell_id", "time_bin", "instrument"])
    rows["cell_id"] = rows["cell_id"].astype(int)
    rows["spatial_resolution_km_pixel"] = pd.to_numeric(
        rows.get("spatial_resolution_km_pixel"), errors="coerce"
    )
    rows["emission_angle_deg"] = pd.to_numeric(rows.get("emission_angle_deg"), errors="coerce")
    rows["coverage_weight"] = geometry_coverage_weight(
        rows["spatial_resolution_km_pixel"],
        rows["emission_angle_deg"],
        rows.get("coverage_weight", 1.0),
    )
    rows["coverage_quality"] = rows.apply(coverage_quality_label, axis=1)

    cube = (
        rows.groupby(
            ["cell_id", "lon_centre", "lat_centre", "time_bin", "instrument", "orbit_or_epoch"],
            dropna=False,
        )
        .agg(
            observation_count=("product_id", "nunique"),
            coverage_weight=("coverage_weight", "sum"),
            best_resolution_km_px=("spatial_resolution_km_pixel", "min"),
            mean_emission_angle_deg=("emission_angle_deg", "mean"),
            coverage_quality=("coverage_quality", lambda s: ";".join(sorted(set(s.astype(str))))),
            source_product_ids=("product_id", lambda s: ";".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )
    return cube[REQUIRED_COVERAGE_CUBE_COLUMNS].sort_values(
        ["instrument", "time_bin", "cell_id"]
    ).reset_index(drop=True)


def save_coverage_cube(cube: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or PROCESSED_DIR / COVERAGE_CUBE_FILENAME
    missing = set(REQUIRED_COVERAGE_CUBE_COLUMNS) - set(cube.columns)
    if missing:
        raise ValueError(f"Coverage cube missing required columns: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cube.to_parquet(path, index=False)
    return path


def load_coverage_cube(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_DIR / COVERAGE_CUBE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Coverage cube not found at {path}.")
    return pd.read_parquet(path)
