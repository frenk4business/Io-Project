"""
features/geology.py
-------------------
Assign geology unit codes to grid cells via spatial join.

For each grid cell centre (as a point), performs a spatial join to the
USGS Io geology map polygons to determine which geologic unit the cell
centre falls within.

Cells outside all polygons (coverage gaps) receive unit_code = "UNKNOWN".
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from config import TARGET_CRS

logger = logging.getLogger(__name__)


def add_geology_feature(
    grid: pd.DataFrame,
    geology_gdf: gpd.GeoDataFrame,
    unit_col: str = "unit_code",
) -> pd.DataFrame:
    """Assign geology unit to each grid cell via spatial point-in-polygon join.

    Args:
        grid: Base grid DataFrame with lon_centre, lat_centre columns.
        geology_gdf: GeoDataFrame of Io geology polygons.
        unit_col: Column in geology_gdf containing the unit code/name.

    Returns:
        Grid DataFrame with new columns:
            geology_unit     : str, geologic unit code (or 'UNKNOWN')
            geology_encoded  : int, label-encoded geology unit
    """
    if unit_col not in geology_gdf.columns:
        raise ValueError(
            f"Column '{unit_col}' not found in geology_gdf. "
            f"Available: {list(geology_gdf.columns)}"
        )

    # Build GeoDataFrame of grid cell centres
    points = gpd.GeoDataFrame(
        grid[["cell_id", "lon_centre", "lat_centre"]].copy(),
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(grid["lon_centre"], grid["lat_centre"])
        ],
        crs=TARGET_CRS,
    )

    # Ensure geology is in the same CRS
    if geology_gdf.crs.to_string() != TARGET_CRS:
        geology_gdf = geology_gdf.to_crs(TARGET_CRS)

    # Spatial join: left join preserves all grid cells
    joined = gpd.sjoin(
        points,
        geology_gdf[[unit_col, "geometry"]],
        how="left",
        predicate="within",
    )

    # Handle cells that fell outside all polygons (coverage gaps)
    joined[unit_col] = joined[unit_col].fillna("UNKNOWN")

    # Handle duplicate matches (cell centre on polygon boundary)
    joined = joined.drop_duplicates(subset="cell_id", keep="first")

    # Label-encode geology unit
    categories = sorted(joined[unit_col].unique())
    code_map = {cat: i for i, cat in enumerate(categories)}
    joined["geology_encoded"] = joined[unit_col].map(code_map).astype(int)

    n_unknown = (joined[unit_col] == "UNKNOWN").sum()
    if n_unknown > 0:
        logger.warning(
            "%d grid cells had no geology polygon match (assigned 'UNKNOWN'). "
            "This may indicate coverage gaps in the source map.",
            n_unknown,
        )

    logger.info(
        "Added geology feature: %d unique units, %d UNKNOWN cells.",
        len(categories),
        n_unknown,
    )

    # Merge back onto original grid
    grid = grid.merge(
        joined[["cell_id", unit_col, "geology_encoded"]].rename(
            columns={unit_col: "geology_unit"}
        ),
        on="cell_id",
        how="left",
    )
    grid["geology_unit"] = grid["geology_unit"].fillna("UNKNOWN")
    grid["geology_encoded"] = grid["geology_encoded"].fillna(code_map.get("UNKNOWN", -1)).astype(int)

    return grid


# Label-encoding limitation: the integer codes 0–15 assigned by add_geology_feature()
# imply a spurious ordinal relationship (e.g., Fb=0 < Fd=1 implies Fb < Fd in a linear
# model). This is geologically meaningless. One-hot encoding is the correct treatment for
# logistic regression; it is provided below for future use. The existing pipeline continues
# to use label encoding for backwards compatibility.
GEOLOGY_ENCODING_NOTE: str = (
    "geology_encoded uses label encoding (integers 0–15) for backwards compatibility. "
    "For logistic regression, one-hot encoding is preferred — use "
    "add_geology_onehot_features() and include the resulting binary columns in your "
    "feature set. The one-hot representation avoids spurious ordinal relationships "
    "between geologically unrelated units."
)


def add_geology_onehot_features(
    grid: pd.DataFrame,
    geology_gdf: gpd.GeoDataFrame,
    unit_col: str = "unit_code",
    prefix: str = "geology_",
) -> pd.DataFrame:
    """Add one-hot encoded geology columns to the grid (preferred over label encoding).

    Each unique geology unit becomes a binary column `geology_<unit_code>`.
    Use this for logistic regression instead of `geology_encoded`.

    Args:
        grid: Base grid DataFrame with lon_centre, lat_centre columns.
              Must already have `geology_unit` from add_geology_feature(),
              OR unit_col / unit_col data will be re-joined here.
        geology_gdf: GeoDataFrame of Io geology polygons.
        unit_col: Column in geology_gdf containing the unit code.
        prefix: Prefix for the one-hot column names.

    Returns:
        Grid DataFrame with new binary columns: geology_Fb, geology_Fd, ...
        (one per unique unit). UNKNOWN and NoData receive their own columns.
    """
    # If geology_unit column is already present, use it directly
    if "geology_unit" in grid.columns:
        unit_series = grid["geology_unit"]
    else:
        # Run the existing join to get the unit
        grid = add_geology_feature(grid, geology_gdf, unit_col)
        unit_series = grid["geology_unit"]

    # One-hot encode
    dummies = pd.get_dummies(unit_series, prefix=prefix.rstrip("_"), prefix_sep="_")
    dummies.index = grid.index

    n_cols = len(dummies.columns)
    logger.info(
        "Added %d one-hot geology columns: %s",
        n_cols,
        list(dummies.columns),
    )
    return pd.concat([grid, dummies], axis=1)
