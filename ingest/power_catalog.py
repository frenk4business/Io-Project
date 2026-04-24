"""
ingest/power_catalog.py
-----------------------
Load the Davies/JIRAM estimated thermal-emission proxy catalogue.

The canonical project file is ``data/raw/io_hotspot_power.csv``. Its
``power_gw`` column is an estimated thermal-emission proxy derived from
Davies et al. (2024) JIRAM 4.8 micron spectral radiance, not a directly
measured bolometric radiant power product.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import RAW_DIR

logger = logging.getLogger(__name__)

POWER_CATALOG_FILENAME = "io_hotspot_power.csv"

REQUIRED_COLUMNS: list[str] = ["name", "longitude", "latitude", "power_gw"]
OPTIONAL_COLUMNS: list[str] = [
    "source_id",
    "epoch",
    "instrument",
    "power_is_estimated",
    "basis_column",
    "source",
]


def load_power_catalog(
    path: Path | None = None,
    normalize_longitude: bool = True,
) -> pd.DataFrame:
    """Load and validate the estimated thermal-emission proxy catalogue.

    Args:
        path: Optional path to CSV. Defaults to ``data/raw/io_hotspot_power.csv``.
        normalize_longitude: Convert longitudes to [-180, 180].

    Returns:
        Clean DataFrame with required columns and valid positive ``power_gw``.

    Raises:
        FileNotFoundError: If the catalogue file is missing.
        ValueError: If required columns are absent.
    """
    path = path or RAW_DIR / POWER_CATALOG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Power catalogue not found at {path}. "
            "Expected Davies/JIRAM estimated thermal-emission proxy data."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Power catalogue is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}."
        )

    cat = df.copy()
    cat["name"] = cat["name"].fillna("").astype(str)
    cat["longitude"] = pd.to_numeric(cat["longitude"], errors="coerce")
    cat["latitude"] = pd.to_numeric(cat["latitude"], errors="coerce")
    cat["power_gw"] = pd.to_numeric(cat["power_gw"], errors="coerce")

    n_before = len(cat)
    cat = cat.dropna(subset=["longitude", "latitude", "power_gw"])
    cat = cat[cat["power_gw"] > 0]
    if normalize_longitude:
        cat["longitude"] = ((cat["longitude"] + 180.0) % 360.0) - 180.0
    cat = cat[cat["longitude"].between(-180, 180) & cat["latitude"].between(-90, 90)]

    if "power_is_estimated" not in cat.columns:
        cat["power_is_estimated"] = True
    if "basis_column" not in cat.columns:
        cat["basis_column"] = "Davies/JIRAM 4.8 micron spectral radiance"

    logger.info(
        "Loaded %d estimated thermal-emission proxy records from %s "
        "(dropped %d invalid rows).",
        len(cat),
        path,
        n_before - len(cat),
    )
    return cat.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    catalog = load_power_catalog()
    print(catalog.head())
    print(f"\nShape: {catalog.shape}")
