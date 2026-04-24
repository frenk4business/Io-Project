"""
features/build.py
-----------------
Orchestrate the full feature matrix build.

Loads all processed layers and computes all four features on the base grid,
producing the final feature matrix parquet file.

Output
------
data/processed/feature_matrix.parquet

Columns in output
-----------------
cell_id, lon_centre, lat_centre, lon_min, lon_max, lat_min, lat_max,
hotspot_count, has_hotspot, hotspot_names,
tidal_heating_flux, geology_unit, geology_encoded,
dist_nearest_hotspot_km, dist_tidal_stress_max_km
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR, FEATURE_MATRIX_FILENAME
from ingest.hotspot_catalog import load_hotspot_catalog
from ingest.tidal_heating import load_tidal_heating_csv, load_synthetic_tidal_proxy
from preprocess.grid import load_base_grid
from preprocess.align_layers import assign_hotspots_to_grid
from features.tidal_heating import add_tidal_heating_feature
from features.synthetic import (
    add_distance_to_nearest_hotspot,
    add_distance_to_tidal_stress_max,
)

logger = logging.getLogger(__name__)

# Feature column names — used by models to select features
FEATURE_COLUMNS: list[str] = [
    "tidal_heating_flux",
    "geology_encoded",
    "dist_nearest_hotspot_km",
    "dist_tidal_stress_max_km",
]
LABEL_COLUMN: str = "has_hotspot"


def build_feature_matrix(use_synthetic_tidal: bool = False) -> pd.DataFrame:
    """Build the complete feature matrix.

    Args:
        use_synthetic_tidal: If True, use synthetic tidal proxy instead of
                             real data. Only for development/testing.
                             All outputs will be marked as synthetic.

    Returns:
        DataFrame with all features and labels, one row per grid cell.
    """
    logger.info("=== Building feature matrix ===")

    # 1. Load base grid
    grid = load_base_grid()
    logger.info("Base grid: %d cells.", len(grid))

    # 2. Load hotspot catalog and assign to grid
    catalog = load_hotspot_catalog()
    grid = assign_hotspots_to_grid(grid, catalog)

    # 3. Load tidal heating and add feature
    if use_synthetic_tidal:
        logger.warning("Using SYNTHETIC tidal heating proxy.")
        tidal_df = load_synthetic_tidal_proxy()
    else:
        tidal_df = load_tidal_heating_csv()
    grid = add_tidal_heating_feature(grid, tidal_df)

    # 4. Add geology feature
    # Note: geology requires geopandas and the shapefile to be present.
    # If not available, we skip and log a warning.
    try:
        import geopandas as gpd
        from ingest.geology_map import load_geology_map
        from features.geology import add_geology_feature

        geology_gdf = load_geology_map()
        grid = add_geology_feature(grid, geology_gdf)
    except FileNotFoundError as exc:
        logger.warning(
            "Geology map not available (%s). "
            "Setting geology_unit='UNKNOWN' and geology_encoded=0 for all cells.",
            exc,
        )
        grid["geology_unit"] = "UNKNOWN"
        grid["geology_encoded"] = 0

    # 5. Add synthetic distance features
    grid = add_distance_to_nearest_hotspot(grid, catalog)
    grid = add_distance_to_tidal_stress_max(grid)

    # 6. Check for missing values in feature columns
    missing = grid[FEATURE_COLUMNS].isnull().sum()
    if missing.any():
        logger.warning("Missing values in features:\n%s", missing[missing > 0])

    logger.info(
        "Feature matrix complete: %d cells, %d features, %d positive labels (%.2f%%).",
        len(grid),
        len(FEATURE_COLUMNS),
        grid[LABEL_COLUMN].sum(),
        100 * grid[LABEL_COLUMN].mean(),
    )

    return grid


def save_feature_matrix(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save feature matrix to parquet.

    Args:
        df: Feature matrix DataFrame.
        path: Output path. Defaults to data/processed/feature_matrix.parquet.

    Returns:
        Path to saved file.
    """
    if path is None:
        path = PROCESSED_DIR / FEATURE_MATRIX_FILENAME

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved feature matrix to %s.", path)
    return path


def load_feature_matrix(path: Path | None = None) -> pd.DataFrame:
    """Load the feature matrix from parquet.

    Args:
        path: Path to parquet. Defaults to data/processed/feature_matrix.parquet.

    Returns:
        Feature matrix DataFrame.

    Raises:
        FileNotFoundError: If feature matrix has not been built.
    """
    if path is None:
        path = PROCESSED_DIR / FEATURE_MATRIX_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. "
            "Run: python -m features.build"
        )

    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_feature_matrix(use_synthetic_tidal=True)
    save_feature_matrix(df)
    print(df[FEATURE_COLUMNS + [LABEL_COLUMN]].describe())
