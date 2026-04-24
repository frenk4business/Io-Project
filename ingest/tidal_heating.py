"""
ingest/tidal_heating.py
-----------------------
Load the tidal heating flux grid for Io.

Data source
-----------
The tidal heating flux model is derived from published work. Two common sources:
  - Tackley, P.J. et al. (2001). Three-dimensional simulations of mantle
    convection in Io. Icarus, 149(1), 79–93.
  - Hamilton, C.W. et al. (2013). Spatial distribution of volcanoes on Io:
    Implications for tidal heating models. Earth and Planetary Science Letters.

See data/external/SOURCES.md for the specific model version used and download URL.

Expected raw file
-----------------
data/raw/io_tidal_heating_flux.csv

Expected columns:
  - longitude : degrees, -180 to 180 (float)
  - latitude  : degrees, -90 to 90 (float)
  - flux_w_m2 : tidal heating flux in W/m² (float)

Alternatively, if you have a GeoTIFF:
  data/raw/io_tidal_heating_flux.tif
  Use load_tidal_heating_raster() instead.

Notes
-----
This is a TIME-AVERAGED model. Real tidal heating varies with Io's orbital
position. The model assumes a homogeneous interior structure. Both assumptions
introduce uncertainty that must be acknowledged in analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import RAW_DIR, TIDAL_HEATING_FILENAME

logger = logging.getLogger(__name__)

TIDAL_HEATING_PROVENANCE_NOTE = (
    "Time-averaged tidal heating model. "
    "Actual flux varies with orbital phase and interior structure. "
    "See data/external/SOURCES.md for model citation."
)


def load_tidal_heating_csv(path: Path | None = None) -> pd.DataFrame:
    """Load a tidal heating flux grid from a CSV file.

    Args:
        path: Path to CSV. Defaults to data/raw/io_tidal_heating_flux.csv.

    Returns:
        DataFrame with columns: longitude, latitude, flux_w_m2, provenance.

    Raises:
        FileNotFoundError: If the raw file is missing.
        ValueError: If required columns are absent.
    """
    if path is None:
        path = RAW_DIR / TIDAL_HEATING_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"Tidal heating file not found at {path}.\n"
            "See data/external/SOURCES.md for the download URL.\n"
            "Alternatively, use load_synthetic_tidal_proxy() for a placeholder."
        )

    logger.info("Loading tidal heating grid from %s", path)
    df = pd.read_csv(path)

    required = ["longitude", "latitude", "flux_w_m2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Tidal heating CSV is missing columns: {missing}\n"
            f"Available: {list(df.columns)}\n"
            "Rename your columns or update this loader."
        )

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["flux_w_m2"] = pd.to_numeric(df["flux_w_m2"], errors="coerce")
    df["provenance"] = TIDAL_HEATING_PROVENANCE_NOTE

    logger.info("Loaded tidal heating grid: %d points.", len(df))
    return df


def load_synthetic_tidal_proxy(resolution_deg: float = 1.0) -> pd.DataFrame:
    """Generate a synthetic tidal heating proxy for development/testing.

    Uses an analytical approximation of tidal dissipation pattern on Io.
    Tidal heating is maximum at the sub-Jupiter and anti-Jupiter points
    (0° and 180° longitude) and at mid-latitudes.

    THIS IS A SYNTHETIC PLACEHOLDER — not a physical model.
    Replace with real data before any analysis.

    Args:
        resolution_deg: Grid resolution in degrees.

    Returns:
        DataFrame with columns: longitude, latitude, flux_w_m2, provenance.
    """
    logger.warning(
        "Using SYNTHETIC tidal heating proxy. "
        "Replace with real data before analysis. "
        "See data/external/SOURCES.md."
    )

    lons = np.arange(-180 + resolution_deg / 2, 180, resolution_deg)
    lats = np.arange(-90 + resolution_deg / 2, 90, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Analytical proxy: max at sub-Jupiter (lon=0) and anti-Jupiter (lon=180)
    # and suppressed at poles (tidal torque geometry)
    lon_rad = np.deg2rad(lon_grid)
    lat_rad = np.deg2rad(lat_grid)
    flux_proxy = (
        np.cos(lon_rad) ** 2 * np.cos(lat_rad) ** 2
        + 0.3 * np.sin(lat_rad) ** 2
    )

    df = pd.DataFrame(
        {
            "longitude": lon_grid.ravel(),
            "latitude": lat_grid.ravel(),
            "flux_w_m2": flux_proxy.ravel(),
        }
    )
    df["provenance"] = "SYNTHETIC PROXY — analytical approximation, not a physical model"
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Use synthetic proxy if real data not yet available
    df = load_synthetic_tidal_proxy()
    print(df.describe())
    print(f"\nProvenance: {df['provenance'].iloc[0]}")
