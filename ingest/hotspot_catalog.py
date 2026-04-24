"""
ingest/hotspot_catalog.py
-------------------------
Load and validate the USGS Io Volcanic Hotspot catalog.

Data source
-----------
Lopes, R.M.C. & Williams, D.A. (2005). Io after Galileo.
USGS Astrogeology hotspot compilation.

Expected raw file
-----------------
data/raw/io_volcanic_hotspots.csv

The CSV should contain at minimum:
  - name        : hotspot name (string)
  - longitude   : degrees, 0–360 or -180–180 (float)
  - latitude    : degrees, -90–90 (float)
  - temperature : brightness temperature in K (float, may be NaN)
  - source      : originating observation (string)

If your source file uses different column names, update COLUMN_MAP below.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import RAW_DIR, HOTSPOT_CATALOG_FILENAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name mapping — adjust to match your actual CSV headers
# ---------------------------------------------------------------------------
COLUMN_MAP: dict[str, str] = {
    # raw_name : canonical_name
    "Name": "name",
    "Longitude": "longitude",
    "Latitude": "latitude",
    "Temperature_K": "temperature",
    "Source": "source",
}

REQUIRED_COLUMNS: list[str] = ["name", "longitude", "latitude"]


def load_hotspot_catalog(
    path: Path | None = None,
    normalize_longitude: bool = True,
) -> pd.DataFrame:
    """Load and validate the USGS Io Volcanic Hotspot catalog.

    Args:
        path: Path to raw CSV. Defaults to data/raw/io_volcanic_hotspots.csv.
        normalize_longitude: If True, converts 0–360 longitude to -180–180.

    Returns:
        DataFrame with canonical column names and validated dtypes.

    Raises:
        FileNotFoundError: If the raw catalog CSV is missing.
        ValueError: If required columns are absent after mapping.
    """
    if path is None:
        path = RAW_DIR / HOTSPOT_CATALOG_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"Hotspot catalog not found at {path}.\n"
            "Run the download step or place the file manually.\n"
            "See data/external/SOURCES.md for the download URL."
        )

    logger.info("Loading hotspot catalog from %s", path)
    df = pd.read_csv(path)

    # Rename columns to canonical names where mapping exists
    df = df.rename(columns=COLUMN_MAP)

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Catalog is missing required columns after mapping: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update COLUMN_MAP in ingest/hotspot_catalog.py to match your CSV."
        )

    # Coerce numeric columns
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "temperature" in df.columns:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

    # Drop rows with missing coordinates
    n_before = len(df)
    df = df.dropna(subset=["longitude", "latitude"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with missing coordinates.", n_dropped)

    # Normalize longitude to [-180, 180]
    if normalize_longitude:
        mask = df["longitude"] > 180
        if mask.any():
            logger.info(
                "Converting %d longitude values from 0–360 to -180–180.", mask.sum()
            )
            df.loc[mask, "longitude"] -= 360

    # Validate coordinate ranges
    out_of_range_lon = (~df["longitude"].between(-180, 180)).sum()
    out_of_range_lat = (~df["latitude"].between(-90, 90)).sum()
    if out_of_range_lon or out_of_range_lat:
        logger.warning(
            "Coordinate range issues: %d longitude, %d latitude values out of range.",
            out_of_range_lon,
            out_of_range_lat,
        )

    # Add provenance column
    df["provenance"] = str(path)

    logger.info("Loaded %d hotspot records.", len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    catalog = load_hotspot_catalog()
    print(catalog.head())
    print(f"\nShape: {catalog.shape}")
    print(f"Columns: {list(catalog.columns)}")
