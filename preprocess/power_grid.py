"""
preprocess/power_grid.py
------------------------
Aggregate Davies/JIRAM estimated thermal-emission proxy observations onto the
project's canonical 1 degree Io grid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import GRID_RESOLUTION_DEG, PROCESSED_DIR
from ingest.power_catalog import load_power_catalog
from preprocess.grid import load_base_grid

logger = logging.getLogger(__name__)

POWER_GRID_FILENAME = "power_grid_1deg.parquet"


def _cell_centres(
    longitude: pd.Series | np.ndarray,
    latitude: pd.Series | np.ndarray,
    resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    half = resolution_deg / 2.0
    lon_idx = np.floor((np.asarray(longitude, dtype=float) + 180.0) / resolution_deg)
    lat_idx = np.floor((np.asarray(latitude, dtype=float) + 90.0) / resolution_deg)
    lon = -180.0 + lon_idx * resolution_deg + half
    lat = -90.0 + lat_idx * resolution_deg + half
    return (
        np.clip(lon, -180.0 + half, 180.0 - half),
        np.clip(lat, -90.0 + half, 90.0 - half),
    )


def assign_power_to_grid(
    grid: pd.DataFrame,
    power_catalog: pd.DataFrame,
    resolution_deg: float = GRID_RESOLUTION_DEG,
) -> pd.DataFrame:
    """Assign estimated thermal-emission proxy records to grid cells.

    Aggregates multiple records per cell as count, max, mean, and sum.
    """
    required = {"longitude", "latitude", "power_gw", "name"}
    missing = required - set(power_catalog.columns)
    if missing:
        raise ValueError(f"Power catalog missing required columns: {sorted(missing)}")

    cat = power_catalog.copy()
    cat["power_gw"] = pd.to_numeric(cat["power_gw"], errors="coerce")
    cat = cat.dropna(subset=["longitude", "latitude", "power_gw"])
    cat = cat[cat["power_gw"] > 0]
    lon_centres, lat_centres = _cell_centres(
        cat["longitude"], cat["latitude"], resolution_deg
    )
    cat["lon_centre"] = lon_centres
    cat["lat_centre"] = lat_centres

    agg = (
        cat.groupby(["lon_centre", "lat_centre"])
        .agg(
            power_count=("power_gw", "count"),
            primary_power_gw=("power_gw", "max"),
            mean_power_gw=("power_gw", "mean"),
            sum_power_gw=("power_gw", "sum"),
            power_names=("name", lambda x: ";".join(x.astype(str))),
        )
        .reset_index()
    )
    agg["log_primary_power"] = np.log1p(agg["primary_power_gw"])

    out = grid.merge(agg, on=["lon_centre", "lat_centre"], how="left")
    out["power_count"] = out["power_count"].fillna(0).astype(int)
    for col in ["primary_power_gw", "mean_power_gw", "sum_power_gw", "log_primary_power"]:
        out[col] = out[col].fillna(0.0).astype(float)
    out["power_names"] = out["power_names"].fillna("")

    logger.info(
        "Power-grid assignment complete: %d cells with estimated proxy data, "
        "%.1f GW total proxy.",
        int((out["power_count"] > 0).sum()),
        float(out["sum_power_gw"].sum()),
    )
    return out


def save_power_grid(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or PROCESSED_DIR / POWER_GRID_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved power grid to %s.", path)
    return path


def load_power_grid(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_DIR / POWER_GRID_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Power grid not found at {path}. Run: python -m preprocess.power_grid"
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    grid_df = load_base_grid()
    power_df = load_power_catalog()
    out_df = assign_power_to_grid(grid_df, power_df)
    save_power_grid(out_df)
    print(out_df[out_df["power_count"] > 0].head())
