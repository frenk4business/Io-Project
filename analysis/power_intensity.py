"""
analysis/power_intensity.py
---------------------------
Radiance-aware hotspot intensity summaries for Io.

All ``power_gw`` values are estimated thermal-emission proxies derived from
Davies/JIRAM 4.8 micron spectral radiance. They are not directly measured
bolometric radiant powers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
POWER_BY_LATITUDE_CSV: Path = RESULTS_DIR / "power_by_latitude.csv"
POWER_BY_GEOLOGY_CSV: Path = RESULTS_DIR / "power_by_geology.csv"
POWER_OUTLIER_SENSITIVITY_CSV: Path = RESULTS_DIR / "power_outlier_sensitivity.csv"
POWER_POLAR_SENSITIVITY_CSV: Path = RESULTS_DIR / "power_polar_sensitivity.csv"

LATITUDE_BINS: list[tuple[float, float, str]] = [
    (-90.0, -60.0, "south polar"),
    (-60.0, -30.0, "south mid-latitude"),
    (-30.0, 0.0, "south low-latitude"),
    (0.0, 30.0, "north low-latitude"),
    (30.0, 60.0, "north mid-latitude"),
    (60.0, 90.0, "north polar"),
]


def _observed_power_cells(power_grid: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cell_id",
        "lat_centre",
        "lon_centre",
        "power_count",
        "primary_power_gw",
        "mean_power_gw",
        "sum_power_gw",
    }
    missing = required - set(power_grid.columns)
    if missing:
        raise ValueError(f"Power grid missing required columns: {sorted(missing)}")
    return power_grid[power_grid["power_count"] > 0].copy()


def summarize_power_by_latitude(power_grid: pd.DataFrame) -> pd.DataFrame:
    """Summarise estimated thermal-emission proxy by latitude band."""
    obs = _observed_power_cells(power_grid)
    rows = []
    total_proxy = float(obs["sum_power_gw"].sum())
    for lat_min, lat_max, label in LATITUDE_BINS:
        sub = obs[(obs["lat_centre"] >= lat_min) & (obs["lat_centre"] < lat_max)]
        proxy_sum = float(sub["sum_power_gw"].sum())
        rows.append(
            {
                "lat_band": label,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "n_power_cells": int(len(sub)),
                "power_count": int(sub["power_count"].sum()),
                "sum_power_gw": proxy_sum,
                "mean_primary_power_gw": float(sub["primary_power_gw"].mean()) if len(sub) else np.nan,
                "max_primary_power_gw": float(sub["primary_power_gw"].max()) if len(sub) else np.nan,
                "fraction_total_power": proxy_sum / total_proxy if total_proxy > 0 else np.nan,
                "power_definition": "estimated thermal-emission proxy from Davies/JIRAM 4.8 micron spectral radiance",
            }
        )
    return pd.DataFrame(rows)


def summarize_power_by_geology(
    feature_matrix: pd.DataFrame,
    power_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise estimated thermal-emission proxy by geology unit."""
    if "geology_unit" not in feature_matrix.columns:
        raise KeyError("feature_matrix must contain geology_unit")

    cols = ["cell_id", "geology_unit"]
    merged = power_grid.merge(feature_matrix[cols], on="cell_id", how="left")
    obs = _observed_power_cells(merged)
    total_proxy = float(obs["sum_power_gw"].sum())

    rows = []
    for unit, sub in obs.groupby("geology_unit", dropna=False):
        proxy_sum = float(sub["sum_power_gw"].sum())
        rows.append(
            {
                "geology_unit": str(unit),
                "n_power_cells": int(len(sub)),
                "power_count": int(sub["power_count"].sum()),
                "sum_power_gw": proxy_sum,
                "mean_primary_power_gw": float(sub["primary_power_gw"].mean()),
                "max_primary_power_gw": float(sub["primary_power_gw"].max()),
                "fraction_total_power": proxy_sum / total_proxy if total_proxy > 0 else np.nan,
                "power_definition": "estimated thermal-emission proxy from Davies/JIRAM 4.8 micron spectral radiance",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("sum_power_gw", ascending=False)
        .reset_index(drop=True)
    )


def polar_threshold_sensitivity(
    power_grid: pd.DataFrame,
    thresholds: tuple[int, ...] = (55, 60, 65),
) -> pd.DataFrame:
    """Compare polar and non-polar estimated power for several thresholds."""
    obs = _observed_power_cells(power_grid)
    rows = []
    for threshold in thresholds:
        polar = obs[obs["lat_centre"].abs() >= threshold]
        nonpolar = obs[obs["lat_centre"].abs() < threshold]
        polar_sum = float(polar["sum_power_gw"].sum())
        nonpolar_sum = float(nonpolar["sum_power_gw"].sum())
        total = polar_sum + nonpolar_sum
        rows.append(
            {
                "polar_threshold_abs_lat": threshold,
                "polar_power_cells": int(len(polar)),
                "nonpolar_power_cells": int(len(nonpolar)),
                "polar_sum_power_gw": polar_sum,
                "nonpolar_sum_power_gw": nonpolar_sum,
                "polar_fraction_total_power": polar_sum / total if total > 0 else np.nan,
                "polar_mean_primary_power_gw": float(polar["primary_power_gw"].mean()) if len(polar) else np.nan,
                "nonpolar_mean_primary_power_gw": float(nonpolar["primary_power_gw"].mean()) if len(nonpolar) else np.nan,
                "power_definition": "estimated thermal-emission proxy from Davies/JIRAM 4.8 micron spectral radiance",
            }
        )
    return pd.DataFrame(rows)


def outlier_sensitivity(
    power_grid: pd.DataFrame,
    remove_top_n: tuple[int, ...] = (0, 1, 5, 10),
    polar_threshold: float = 60.0,
) -> pd.DataFrame:
    """Quantify how top-power hotspots/cells affect polar/non-polar conclusions."""
    obs = _observed_power_cells(power_grid).sort_values(
        "primary_power_gw", ascending=False
    )
    rows = []
    for n_remove in remove_top_n:
        sub = obs.iloc[n_remove:].copy()
        polar = sub[sub["lat_centre"].abs() >= polar_threshold]
        nonpolar = sub[sub["lat_centre"].abs() < polar_threshold]
        total = float(sub["sum_power_gw"].sum())
        polar_sum = float(polar["sum_power_gw"].sum())
        nonpolar_sum = float(nonpolar["sum_power_gw"].sum())
        rows.append(
            {
                "removed_top_n_cells": int(n_remove),
                "remaining_power_cells": int(len(sub)),
                "sum_power_gw": total,
                "polar_sum_power_gw": polar_sum,
                "nonpolar_sum_power_gw": nonpolar_sum,
                "polar_fraction_total_power": polar_sum / total if total > 0 else np.nan,
                "max_remaining_primary_power_gw": float(sub["primary_power_gw"].max()) if len(sub) else np.nan,
                "power_definition": "estimated thermal-emission proxy from Davies/JIRAM 4.8 micron spectral radiance",
            }
        )
    return pd.DataFrame(rows)


def compute_power_intensity_suite(
    feature_matrix: pd.DataFrame,
    power_grid: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute all intensity-analysis tables."""
    return {
        "by_latitude": summarize_power_by_latitude(power_grid),
        "by_geology": summarize_power_by_geology(feature_matrix, power_grid),
        "polar_sensitivity": polar_threshold_sensitivity(power_grid),
        "outlier_sensitivity": outlier_sensitivity(power_grid),
    }


def save_power_intensity_suite(results: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Persist intensity-analysis tables to data/results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "by_latitude": POWER_BY_LATITUDE_CSV,
        "by_geology": POWER_BY_GEOLOGY_CSV,
        "polar_sensitivity": POWER_POLAR_SENSITIVITY_CSV,
        "outlier_sensitivity": POWER_OUTLIER_SENSITIVITY_CSV,
    }
    paths = {}
    for key, path in mapping.items():
        results[key].to_csv(path, index=False)
        paths[key] = path
        logger.info("Saved %s to %s", key, path)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from features.build import load_feature_matrix
    from preprocess.power_grid import load_power_grid

    suite = compute_power_intensity_suite(load_feature_matrix(), load_power_grid())
    save_power_intensity_suite(suite)
    print(suite["by_latitude"].to_string(index=False))
