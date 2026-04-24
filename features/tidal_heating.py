"""
features/tidal_heating.py
-------------------------
Interpolate tidal heating flux onto the base grid.

For each grid cell centre, assigns the nearest-available tidal heating
flux value from the loaded model grid.

Assumption: We use nearest-neighbour interpolation, which is honest about
the limited resolution of the source model. Do not use bicubic or spline
interpolation — it would imply more precision than the source data supports.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def add_tidal_heating_feature(
    grid: pd.DataFrame,
    tidal_df: pd.DataFrame,
    flux_col: str = "flux_w_m2",
) -> pd.DataFrame:
    """Assign tidal heating flux to grid cells via nearest-neighbour lookup.

    Args:
        grid: Base grid DataFrame with lon_centre, lat_centre columns.
        tidal_df: Tidal heating DataFrame with longitude, latitude, flux_w_m2.
        flux_col: Column name in tidal_df containing flux values.

    Returns:
        Grid DataFrame with new column: tidal_heating_flux (float).
    """
    if flux_col not in tidal_df.columns:
        raise ValueError(
            f"Column '{flux_col}' not found in tidal_df. "
            f"Available: {list(tidal_df.columns)}"
        )

    # Build KD-tree on tidal model points
    tidal_coords = np.column_stack(
        [tidal_df["longitude"].values, tidal_df["latitude"].values]
    )
    tree = cKDTree(tidal_coords)

    # Query for each grid cell centre
    grid_coords = np.column_stack(
        [grid["lon_centre"].values, grid["lat_centre"].values]
    )
    _, indices = tree.query(grid_coords, k=1)

    grid = grid.copy()
    grid["tidal_heating_flux"] = tidal_df[flux_col].values[indices]

    logger.info(
        "Added tidal_heating_flux feature. "
        "Min=%.4f, Max=%.4f, Mean=%.4f.",
        grid["tidal_heating_flux"].min(),
        grid["tidal_heating_flux"].max(),
        grid["tidal_heating_flux"].mean(),
    )
    return grid
