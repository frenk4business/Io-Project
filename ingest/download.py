"""
ingest/download.py
------------------
Fetch all three raw datasets required by the pipeline.

Datasets
--------
A. Io Volcanic Hotspot Catalog (REAL DATA)
   Source: USGS SIM3168 Io_HotSpots_Locations.shp
   Williams, D.A. et al. (2011). Geologic Map of Io.
   USGS Scientific Investigations Map 3168.
   Extracted from the same GIS database as the geology map.

B. USGS Io Geology Map (REAL DATA)
   Source: https://pubs.usgs.gov/sim/3168/
   GIS database: Io_SIM3168_Database.zip (268 MB)
   Files extracted: Io_GeoUnits.shp and companions

C. Tidal Heating Flux Grid (SYNTHETIC PROXY)
   No freely available gridded tidal heating model was found.
   Using the analytical proxy from ingest.tidal_heating.
   See data/external/SOURCES.md for references to consult.

Usage
-----
    python -m ingest.download          # download if missing
    python -m ingest.download --force  # re-download even if present

All downloads are logged to data/raw/FETCH_LOG.txt.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config import RAW_DIR, HOTSPOT_CATALOG_FILENAME, GEOLOGY_MAP_FILENAME, TIDAL_HEATING_FILENAME
from ingest.tidal_heating import load_synthetic_tidal_proxy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
GEOLOGY_GIS_URL = (
    "https://asc-astropedia.s3.us-west-2.amazonaws.com/Io/Geology/Io_SIM3168_Database.zip"
)

FETCH_LOG_PATH = RAW_DIR / "FETCH_LOG.txt"

GEOLOGY_SHP_IN_ZIP = "Io_SIM3168_Database/Io_Geology_SIM3168_shapefiles/"
HOTSPOT_SHP_NAME = "Io_HotSpots_Locations"
GEOUNITS_SHP_NAME = "Io_GeoUnits"


def _log(message: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    line = f"[{now}] {message}\n"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(FETCH_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
    logger.info(message)


def _download_zip(url: str, description: str) -> bytes:
    """Download a zip file from URL, logging progress."""
    _log(f"Downloading {description} from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "io-hotspot-prediction/1.0"})
    chunks: list[bytes] = []
    downloaded = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        _log(f"  Total size: {total / 1024 / 1024:.1f} MB")
        while True:
            block = resp.read(65536)
            if not block:
                break
            chunks.append(block)
            downloaded += len(block)
    data = b"".join(chunks)
    _log(f"  Downloaded {len(data) / 1024 / 1024:.1f} MB")
    return data


def _extract_shp_files(
    zip_data: bytes,
    shp_prefix: str,
    dest_dir: Path,
    output_stem: str,
) -> Path:
    """Extract a specific shapefile (and companions) from a zip, renaming to output_stem.

    Args:
        zip_data: Raw zip bytes.
        shp_prefix: Path prefix inside the zip (e.g. "Io_SIM3168_Database/.../Io_GeoUnits").
        dest_dir: Destination directory.
        output_stem: Output filename stem (without extension).

    Returns:
        Path to the extracted .shp file.
    """
    shp_exts = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for item in zf.infolist():
            name = item.filename
            basename = os.path.basename(name)
            stem, ext = os.path.splitext(basename)
            ext = ext.lower()
            # Exact stem match: prevent "Io_GeoUnitsPoints" from matching "Io_GeoUnits"
            expected_basename_no_ext = shp_prefix.split("/")[-1]
            if ext in shp_exts and stem == expected_basename_no_ext:
                try:
                    raw = zf.read(item)
                    out_path = dest_dir / f"{output_stem}{ext}"
                    with open(out_path, "wb") as f:
                        f.write(raw)
                except Exception as exc:
                    logger.warning("Could not extract %s: %s", name, exc)

    out_shp = dest_dir / f"{output_stem}.shp"
    if not out_shp.exists():
        raise FileNotFoundError(
            f"Shapefile prefix '{shp_prefix}' not found in zip. "
            "The zip structure may have changed."
        )
    return out_shp


def download_hotspot_catalog(force: bool = False) -> Path:
    """Build io_volcanic_hotspots.csv from the USGS SIM3168 HotSpots shapefile.

    The hotspot locations are extracted from the same GIS database as the
    geology map (Io_SIM3168_Database.zip). This is REAL DATA from:
      Williams, D.A. et al. (2011). Geologic Map of Io.
      USGS Scientific Investigations Map 3168.

    Args:
        force: Re-download even if file already exists.

    Returns:
        Path to the written CSV.
    """
    out_path = RAW_DIR / HOTSPOT_CATALOG_FILENAME
    if out_path.exists() and not force:
        _log(f"Hotspot catalog already exists at {out_path}. Skipping (use --force to re-download).")
        return out_path

    _log("=== Dataset A: Io Volcanic Hotspot Catalog ===")
    _log("Source: USGS SIM3168 Io_HotSpots_Locations.shp (Williams et al. 2011)")

    zip_data = _download_zip(GEOLOGY_GIS_URL, "Io SIM3168 GIS database")

    # Extract hotspot shapefile to a temp directory
    shp_prefix = f"{GEOLOGY_SHP_IN_ZIP}{HOTSPOT_SHP_NAME}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shp_path = _extract_shp_files(
            zip_data,
            shp_prefix,
            tmp_path,
            HOTSPOT_SHP_NAME,
        )
        gdf = gpd.read_file(shp_path)

    logger.info("Loaded hotspot shapefile: %d records, columns: %s", len(gdf), list(gdf.columns))

    # Map shapefile columns to canonical CSV columns
    # Shapefile columns: Volcanic_C (name), Long_E (0-360 east longitude), Lat
    df = pd.DataFrame({
        "Name": gdf["Volcanic_C"].fillna("Unnamed").astype(str),
        "Longitude": pd.to_numeric(gdf["Long_E"], errors="coerce"),
        "Latitude": pd.to_numeric(gdf["Lat"], errors="coerce"),
        "Temperature_K": np.nan,
        "Source": "USGS SIM3168, Williams et al. 2011",
    })

    # Drop records with missing coordinates
    n_before = len(df)
    df = df.dropna(subset=["Longitude", "Latitude"])
    if len(df) < n_before:
        _log(f"  Dropped {n_before - len(df)} records with missing coordinates.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _log(
        f"  Wrote {len(df)} hotspot records to {out_path}  "
        f"  [REAL DATA: USGS SIM3168 Williams et al. 2011]"
    )
    return out_path


def download_geology_map(force: bool = False) -> Path:
    """Download and extract the USGS Io geology map shapefile.

    Source: Williams, D.A. et al. (2011). Geologic Map of Io.
            USGS Scientific Investigations Map 3168.
    URL:    https://pubs.usgs.gov/sim/3168/

    Args:
        force: Re-download even if file already exists.

    Returns:
        Path to the extracted .shp file.
    """
    out_shp = RAW_DIR / GEOLOGY_MAP_FILENAME
    if out_shp.exists() and not force:
        _log(f"Geology map already exists at {out_shp}. Skipping (use --force to re-download).")
        return out_shp

    _log("=== Dataset B: USGS Io Geology Map ===")
    _log("Source: USGS SIM3168 Io_GeoUnits.shp (Williams et al. 2011)")

    zip_data = _download_zip(GEOLOGY_GIS_URL, "Io SIM3168 GIS database")

    shp_prefix = f"{GEOLOGY_SHP_IN_ZIP}{GEOUNITS_SHP_NAME}"
    out_stem = Path(GEOLOGY_MAP_FILENAME).stem
    _extract_shp_files(zip_data, shp_prefix, RAW_DIR, out_stem)

    _log(
        f"  Extracted geology map to {RAW_DIR / GEOLOGY_MAP_FILENAME}  "
        f"  [REAL DATA: USGS SIM3168 Williams et al. 2011]"
    )
    return RAW_DIR / GEOLOGY_MAP_FILENAME


def generate_tidal_heating(force: bool = False) -> Path:
    """Write the synthetic tidal heating proxy to data/raw.

    No freely available tidal heating flux grid was found.
    Using the analytical proxy from ingest.tidal_heating.

    See data/external/SOURCES.md § 3 for references to real models.

    Args:
        force: Regenerate even if file already exists.

    Returns:
        Path to the written CSV.
    """
    out_path = RAW_DIR / TIDAL_HEATING_FILENAME
    if out_path.exists() and not force:
        _log(f"Tidal heating already exists at {out_path}. Skipping (use --force to re-generate).")
        return out_path

    _log("=== Dataset C: Tidal Heating Flux Grid ===")
    _log("WARNING: No real tidal heating grid was found. Using SYNTHETIC PROXY.")
    _log("See data/external/SOURCES.md § 3 for real data sources.")

    df = load_synthetic_tidal_proxy(resolution_deg=1.0)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _log(
        f"  Wrote {len(df)} synthetic tidal heating grid points to {out_path}  "
        f"  [SYNTHETIC PROXY — replace with real data before final analysis]"
    )
    return out_path


def main(force: bool = False) -> None:
    """Run all three download steps."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _log("=" * 60)
    _log("Starting data acquisition run")
    _log(f"force={force}")

    try:
        hotspot_path = download_hotspot_catalog(force=force)
        _log(f"OK: {hotspot_path}")
    except Exception as exc:
        _log(f"ERROR downloading hotspot catalog: {exc}")
        raise

    try:
        geology_path = download_geology_map(force=force)
        _log(f"OK: {geology_path}")
    except Exception as exc:
        _log(f"ERROR downloading geology map: {exc}")
        raise

    try:
        tidal_path = generate_tidal_heating(force=force)
        _log(f"OK: {tidal_path}")
    except Exception as exc:
        _log(f"ERROR generating tidal heating: {exc}")
        raise

    # Verify all files exist
    expected = [
        RAW_DIR / HOTSPOT_CATALOG_FILENAME,
        RAW_DIR / GEOLOGY_MAP_FILENAME,
        RAW_DIR / TIDAL_HEATING_FILENAME,
    ]
    _log("\n=== Verification ===")
    all_ok = True
    for p in expected:
        if p.exists():
            size_kb = p.stat().st_size / 1024
            _log(f"  [OK] {p.name} ({size_kb:.1f} KB)")
        else:
            _log(f"  [MISSING] {p}")
            all_ok = False

    if all_ok:
        _log("All datasets are in place. Pipeline is ready to run.")
    else:
        _log("Some datasets are missing. See errors above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Io hotspot pipeline datasets.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()
    main(force=args.force)
