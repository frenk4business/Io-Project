"""
ingest/geology_map.py
---------------------
Load the USGS Io geologic map shapefile.

Data source
-----------
Williams, D.A. et al. (2011). Geologic Map of Io.
USGS Scientific Investigations Map 3168.
https://pubs.usgs.gov/sim/3168/

Expected raw file
-----------------
data/raw/io_geology_map.shp  (and associated .dbf, .prj, .shx files)

The shapefile should contain polygon features with at minimum:
  - geometry   : polygon
  - UNIT_NAME  : geologic unit name (string)
  - UNIT_CODE  : short code for the unit (string)

If your shapefile uses different attribute names, update ATTRIBUTE_MAP below.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from config import RAW_DIR, GEOLOGY_MAP_FILENAME, TARGET_CRS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attribute name mapping
# ---------------------------------------------------------------------------
ATTRIBUTE_MAP: dict[str, str] = {
    # raw_name : canonical_name
    # USGS SIM3168 actual shapefile columns:
    "Unit": "unit_code",
    "UnitName": "unit_name",
    # Original expected names (kept for compatibility):
    "UNIT_NAME": "unit_name",
    "UNIT_CODE": "unit_code",
}


def load_geology_map(path: Path | None = None) -> gpd.GeoDataFrame:
    """Load and reproject the USGS Io geologic map shapefile.

    Args:
        path: Path to the .shp file. Defaults to data/raw/io_geology_map.shp.

    Returns:
        GeoDataFrame in TARGET_CRS with canonical attribute names.

    Raises:
        FileNotFoundError: If the shapefile is missing.
    """
    if path is None:
        path = RAW_DIR / GEOLOGY_MAP_FILENAME

    if not path.exists():
        raise FileNotFoundError(
            f"Geology map not found at {path}.\n"
            "See data/external/SOURCES.md for the download URL."
        )

    logger.info("Loading geology map from %s", path)
    gdf = gpd.read_file(path)

    # Rename attributes to canonical names where mapping exists
    gdf = gdf.rename(columns=ATTRIBUTE_MAP)

    # Reproject to target CRS
    if gdf.crs is None:
        logger.warning(
            "Geology map has no CRS defined. Assuming %s. Verify this is correct.",
            TARGET_CRS,
        )
        gdf = gdf.set_crs(TARGET_CRS)
    elif gdf.crs.to_string() != TARGET_CRS:
        logger.info("Reprojecting geology map from %s to %s.", gdf.crs, TARGET_CRS)
        try:
            gdf = gdf.to_crs(TARGET_CRS)
        except Exception as exc:
            # Planetary datums (e.g. GCS_Io_2000) are not in the EPSG registry.
            # Since coordinates are already in degrees (geographic CRS), we can
            # safely override the CRS without transforming coordinate values.
            logger.warning(
                "CRS reprojection failed (%s). "
                "Forcing assignment of %s — coordinates are already in degrees.",
                exc,
                TARGET_CRS,
            )
            gdf = gdf.set_crs(TARGET_CRS, allow_override=True)

    # Add provenance
    gdf["provenance"] = str(path)

    logger.info(
        "Loaded geology map: %d polygons, CRS=%s.", len(gdf), gdf.crs
    )
    return gdf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gdf = load_geology_map()
    print(gdf.head())
    print(f"\nShape: {gdf.shape}")
    print(f"CRS: {gdf.crs}")
    if "unit_name" in gdf.columns:
        print(f"Unique geology units: {gdf['unit_name'].nunique()}")
