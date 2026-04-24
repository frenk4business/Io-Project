"""
preprocess/grid.py
------------------
Build the canonical 1°×1° planetary grid for Io.

This module creates the base grid that all features and predictions are
computed on. Every downstream module imports this grid and joins to it.

Grid convention
---------------
- Cell centres are at (lon + 0.5°, lat + 0.5°) for 1° cells
- Longitude range: -180 to +180
- Latitude range: -90 to +90
- Total cells: 360 × 180 = 64,800
- Cell IDs are stable row indices (0-indexed, row-major: lon varies fastest)

Output
------
data/processed/base_grid_1deg.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    GRID_LAT_MAX,
    GRID_LAT_MIN,
    GRID_LON_MAX,
    GRID_LON_MIN,
    GRID_RESOLUTION_DEG,
    PROCESSED_DIR,
    BASE_GRID_FILENAME,
)

logger = logging.getLogger(__name__)


def build_base_grid(resolution_deg: float = GRID_RESOLUTION_DEG) -> pd.DataFrame:
    """Build the base 1°×1° planetary grid.

    Creates a DataFrame where each row represents one grid cell, identified
    by its centre longitude and latitude.

    Args:
        resolution_deg: Grid resolution in degrees. Defaults to project constant.

    Returns:
        DataFrame with columns:
            cell_id   : int, stable row index
            lon_centre: float, cell centre longitude [-180, 180)
            lat_centre: float, cell centre latitude [-90, 90)
            lon_min   : float, cell west edge
            lon_max   : float, cell east edge
            lat_min   : float, cell south edge
            lat_max   : float, cell north edge
    """
    half = resolution_deg / 2.0

    lons = np.arange(
        GRID_LON_MIN + half,
        GRID_LON_MAX,
        resolution_deg,
    )
    lats = np.arange(
        GRID_LAT_MIN + half,
        GRID_LAT_MAX,
        resolution_deg,
    )

    # meshgrid: lat varies slowest (row), lon varies fastest (col)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    n_cells = lat_grid.size
    df = pd.DataFrame(
        {
            "cell_id": np.arange(n_cells),
            "lon_centre": lon_grid.ravel(),
            "lat_centre": lat_grid.ravel(),
            "lon_min": lon_grid.ravel() - half,
            "lon_max": lon_grid.ravel() + half,
            "lat_min": lat_grid.ravel() - half,
            "lat_max": lat_grid.ravel() + half,
        }
    )

    logger.info(
        "Built base grid: %d cells at %.1f° resolution.", n_cells, resolution_deg
    )
    return df


def save_base_grid(
    df: pd.DataFrame, path: Path | None = None
) -> Path:
    """Save the base grid to parquet.

    Args:
        df: Base grid DataFrame from build_base_grid().
        path: Output path. Defaults to data/processed/base_grid_1deg.parquet.

    Returns:
        Path to the saved file.
    """
    if path is None:
        path = PROCESSED_DIR / BASE_GRID_FILENAME

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved base grid to %s.", path)
    return path


def load_base_grid(path: Path | None = None) -> pd.DataFrame:
    """Load the base grid from parquet.

    Args:
        path: Path to parquet file. Defaults to data/processed/base_grid_1deg.parquet.

    Returns:
        Base grid DataFrame.

    Raises:
        FileNotFoundError: If grid has not been built yet.
    """
    if path is None:
        path = PROCESSED_DIR / BASE_GRID_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"Base grid not found at {path}. "
            "Run: python -m preprocess.grid"
        )

    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    grid = build_base_grid()
    print(grid.head())
    print(f"\nShape: {grid.shape}")
    print(f"Lon range: {grid['lon_centre'].min():.1f} to {grid['lon_centre'].max():.1f}")
    print(f"Lat range: {grid['lat_centre'].min():.1f} to {grid['lat_centre'].max():.1f}")
    save_base_grid(grid)
