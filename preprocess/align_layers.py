"""
preprocess/align_layers.py
--------------------------
Align all input layers to the base 1°×1° grid.

For each grid cell, this module determines:
  - Whether the cell contains at least one known hotspot (binary label)
  - The count of hotspots in the cell
  - The geology unit code at the cell centre
  - The tidal heating flux at the cell centre (by nearest-neighbour join)

Output
------
data/processed/hotspots_1deg_grid.parquet   (label + hotspot count per cell)

Notes
-----
Geology and tidal heating alignment is handled in features/geology.py and
features/tidal_heating.py respectively. This module handles the hotspot label.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    GRID_RESOLUTION_DEG,
    PROCESSED_DIR,
    HOTSPOT_GRID_FILENAME,
)
from ingest.hotspot_catalog import load_hotspot_catalog
from preprocess.grid import load_base_grid

logger = logging.getLogger(__name__)


def assign_hotspots_to_grid(
    grid: pd.DataFrame,
    catalog: pd.DataFrame,
    resolution_deg: float = GRID_RESOLUTION_DEG,
) -> pd.DataFrame:
    """Assign hotspot catalog records to grid cells.

    For each hotspot, finds the grid cell whose centre is closest to the
    hotspot's coordinates (using floor assignment, i.e. which cell bin
    the coordinate falls into).

    Args:
        grid: Base grid DataFrame from preprocess.grid.
        catalog: Hotspot catalog DataFrame from ingest.hotspot_catalog.
        resolution_deg: Grid resolution in degrees.

    Returns:
        Grid DataFrame with additional columns:
            hotspot_count    : int, number of hotspots in cell
            has_hotspot      : int (0/1), binary label
            hotspot_names    : str, semicolon-separated names (for audit)
    """
    half = resolution_deg / 2.0

    # Assign each hotspot to its grid cell by binning coordinates
    def _bin(coord: float, min_val: float, half: float, res: float) -> float:
        """Return cell centre for a coordinate."""
        cell_idx = np.floor((coord - min_val) / res).astype(int)
        return min_val + cell_idx * res + half

    lon_centres = _bin(catalog["longitude"].values, -180.0, half, resolution_deg)
    lat_centres = _bin(catalog["latitude"].values, -90.0, half, resolution_deg)

    catalog_assigned = catalog.copy()
    catalog_assigned["lon_centre"] = np.clip(lon_centres, -180 + half, 180 - half)
    catalog_assigned["lat_centre"] = np.clip(lat_centres, -90 + half, 90 - half)

    # Aggregate by cell
    agg = (
        catalog_assigned.groupby(["lon_centre", "lat_centre"])
        .agg(
            hotspot_count=("name", "count"),
            hotspot_names=("name", lambda x: ";".join(x.astype(str))),
        )
        .reset_index()
    )

    # Merge onto base grid
    grid_out = grid.merge(agg, on=["lon_centre", "lat_centre"], how="left")
    grid_out["hotspot_count"] = grid_out["hotspot_count"].fillna(0).astype(int)
    grid_out["has_hotspot"] = (grid_out["hotspot_count"] > 0).astype(int)
    grid_out["hotspot_names"] = grid_out["hotspot_names"].fillna("")

    n_positive = grid_out["has_hotspot"].sum()
    n_total = len(grid_out)
    logger.info(
        "Hotspot assignment complete: %d / %d cells positive (%.2f%%).",
        n_positive,
        n_total,
        100 * n_positive / n_total,
    )

    return grid_out


def save_hotspot_grid(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save the hotspot-labelled grid to parquet.

    Args:
        df: Grid DataFrame from assign_hotspots_to_grid().
        path: Output path. Defaults to data/processed/hotspots_1deg_grid.parquet.

    Returns:
        Path to the saved file.
    """
    if path is None:
        path = PROCESSED_DIR / HOTSPOT_GRID_FILENAME

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved hotspot grid to %s.", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    grid = load_base_grid()
    catalog = load_hotspot_catalog()
    labeled_grid = assign_hotspots_to_grid(grid, catalog)
    print(labeled_grid[labeled_grid["has_hotspot"] == 1].head())
    save_hotspot_grid(labeled_grid)
