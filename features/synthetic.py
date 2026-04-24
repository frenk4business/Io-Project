"""
features/synthetic.py
---------------------
Compute synthetic (derived) spatial features for each grid cell.

These features require no external data beyond what is already in the pipeline.

Feature 1: dist_nearest_hotspot_km
    Great-circle distance (km) from each grid cell centre to the nearest
    known hotspot in the catalog. This is a proxy for spatial clustering
    of volcanic activity.

Feature 2: dist_tidal_stress_max_km
    Great-circle distance (km) from each grid cell centre to the nearest
    theoretical tidal stress maximum. Tidal stress is maximum at four
    points: (0°N 0°E), (0°N 180°E), and to a lesser extent the poles.
    This is an analytical approximation — not a full stress tensor model.

Uncertainty note
----------------
dist_nearest_hotspot_km reflects observational coverage bias. Cells far
from known hotspots may simply be underobserved, not actually hotspot-free.
This feature may introduce observational bias into the model. Use with
awareness.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import IO_RADIUS_KM

logger = logging.getLogger(__name__)

# Theoretical tidal stress maxima on Io (sub-Jupiter and anti-Jupiter points)
# These are the points of maximum tidal flexing on a synchronously rotating body.
TIDAL_STRESS_MAXIMA: list[tuple[float, float]] = [
    (0.0, 0.0),    # sub-Jupiter point
    (0.0, 180.0),  # anti-Jupiter point
    (90.0, 0.0),   # leading hemisphere apex
    (-90.0, 0.0),  # trailing hemisphere apex
]


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
    radius_km: float = IO_RADIUS_KM,
) -> np.ndarray:
    """Compute great-circle distance in km using the haversine formula.

    Args:
        lat1: Latitudes of query points (degrees).
        lon1: Longitudes of query points (degrees).
        lat2: Latitude of reference point (degrees).
        lon2: Longitude of reference point (degrees).
        radius_km: Planetary radius in km.

    Returns:
        Array of distances in km.
    """
    lat1_r = np.deg2rad(lat1)
    lon1_r = np.deg2rad(lon1)
    lat2_r = np.deg2rad(lat2)
    lon2_r = np.deg2rad(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return radius_km * c


def add_distance_to_nearest_hotspot(
    grid: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Add distance (km) from each grid cell to the nearest known hotspot.

    Args:
        grid: Grid DataFrame with lon_centre, lat_centre.
        catalog: Hotspot catalog DataFrame with longitude, latitude.

    Returns:
        Grid DataFrame with new column: dist_nearest_hotspot_km (float).
    """
    lats = grid["lat_centre"].values
    lons = grid["lon_centre"].values

    hotspot_lats = catalog["latitude"].values
    hotspot_lons = catalog["longitude"].values

    # Vectorised: compute distance to all hotspots, take minimum
    min_distances = np.full(len(grid), np.inf)
    for h_lat, h_lon in zip(hotspot_lats, hotspot_lons):
        d = _haversine_km(lats, lons, h_lat, h_lon)
        min_distances = np.minimum(min_distances, d)

    grid = grid.copy()
    grid["dist_nearest_hotspot_km"] = min_distances

    logger.info(
        "Added dist_nearest_hotspot_km. "
        "Min=%.1f km, Max=%.1f km, Mean=%.1f km.",
        grid["dist_nearest_hotspot_km"].min(),
        grid["dist_nearest_hotspot_km"].max(),
        grid["dist_nearest_hotspot_km"].mean(),
    )
    return grid


def add_distance_to_tidal_stress_max(
    grid: pd.DataFrame,
    stress_maxima: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Add distance (km) from each grid cell to the nearest tidal stress maximum.

    Args:
        grid: Grid DataFrame with lon_centre, lat_centre.
        stress_maxima: List of (lat, lon) tuples for stress maxima.
                       Defaults to TIDAL_STRESS_MAXIMA module constant.

    Returns:
        Grid DataFrame with new column: dist_tidal_stress_max_km (float).
    """
    if stress_maxima is None:
        stress_maxima = TIDAL_STRESS_MAXIMA

    lats = grid["lat_centre"].values
    lons = grid["lon_centre"].values

    min_distances = np.full(len(grid), np.inf)
    for s_lat, s_lon in stress_maxima:
        d = _haversine_km(lats, lons, s_lat, s_lon)
        min_distances = np.minimum(min_distances, d)

    grid = grid.copy()
    grid["dist_tidal_stress_max_km"] = min_distances

    logger.info(
        "Added dist_tidal_stress_max_km using %d stress maxima. "
        "Min=%.1f km, Max=%.1f km.",
        len(stress_maxima),
        grid["dist_tidal_stress_max_km"].min(),
        grid["dist_tidal_stress_max_km"].max(),
    )
    return grid
