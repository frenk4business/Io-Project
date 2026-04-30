"""
ingest/jiram_coverage.py
------------------------
Normalize Juno/JIRAM Io product metadata into an approximate observation
coverage table on the project's 1 degree grid.

This is not a full pixel-footprint reconstruction. The current project data
contain per-product sub-spacecraft Io longitude/latitude and geometry metadata,
so this module records product-centre coverage. It is a real PDS metadata layer,
but it should be replaced by SPICE/pixel footprint rasters for publication-grade
coverage correction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import GRID_RESOLUTION_DEG, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

JIRAM_PRODUCT_LOG_FILENAMES: tuple[str, ...] = (
    "jiram_io_calibrated_orbit57_products.csv",
    "jiram_io_calibrated_orbit58_products.csv",
)
JIRAM_COVERAGE_FILENAME = "jiram_observation_coverage.parquet"

REQUIRED_COVERAGE_COLUMNS: list[str] = [
    "orbit",
    "product_id",
    "start_time",
    "stop_time",
    "time_bin",
    "lon_centre",
    "lat_centre",
    "coverage_weight",
    "source_url",
]


def _cell_centres(
    longitude: pd.Series | np.ndarray,
    latitude: pd.Series | np.ndarray,
    resolution_deg: float = GRID_RESOLUTION_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    half = resolution_deg / 2.0
    lon = ((np.asarray(longitude, dtype=float) + 180.0) % 360.0) - 180.0
    lat = np.asarray(latitude, dtype=float)
    lon_idx = np.floor((lon + 180.0) / resolution_deg)
    lat_idx = np.floor((lat + 90.0) / resolution_deg)
    lon_centre = -180.0 + lon_idx * resolution_deg + half
    lat_centre = -90.0 + lat_idx * resolution_deg + half
    return (
        np.clip(lon_centre, -180.0 + half, 180.0 - half),
        np.clip(lat_centre, -90.0 + half, 90.0 - half),
    )


def _product_id(row: pd.Series) -> str:
    file_name = str(row.get("file_name") or "").strip()
    if file_name:
        return Path(file_name).stem
    urn = str(row.get("urn") or "").strip()
    return urn.rsplit(":", 1)[-1] if urn else "unknown_product"


def normalize_jiram_product_logs(
    paths: list[Path] | None = None,
    resolution_deg: float = GRID_RESOLUTION_DEG,
) -> pd.DataFrame:
    """Normalize JIRAM product logs into approximate 1 degree coverage rows."""
    if paths is None:
        paths = [RAW_DIR / name for name in JIRAM_PRODUCT_LOG_FILENAMES]

    frames = []
    for path in paths:
        if not path.exists():
            logger.warning("JIRAM product log missing: %s", path)
            continue
        df = pd.read_csv(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No JIRAM product logs found. Expected files under data/raw/: "
            + ", ".join(JIRAM_PRODUCT_LOG_FILENAMES)
        )

    raw = pd.concat(frames, ignore_index=True)
    required = {
        "urn",
        "start_date_time",
        "stop_date_time",
        "target_name",
        "orbit_number",
        "io_planetocentric_longitude_deg",
        "io_planetocentric_latitude_deg",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"JIRAM product logs missing columns: {sorted(missing)}")

    obs = raw.copy()
    obs = obs[obs["target_name"].astype(str).str.upper().eq("IO")]
    obs["longitude"] = pd.to_numeric(
        obs["io_planetocentric_longitude_deg"], errors="coerce"
    )
    obs["latitude"] = pd.to_numeric(
        obs["io_planetocentric_latitude_deg"], errors="coerce"
    )
    obs = obs.dropna(subset=["longitude", "latitude", "start_date_time"])
    obs = obs[obs["latitude"].between(-90.0, 90.0)]

    lon_centre, lat_centre = _cell_centres(
        obs["longitude"],
        obs["latitude"],
        resolution_deg,
    )
    obs["lon_centre"] = lon_centre
    obs["lat_centre"] = lat_centre
    start = pd.to_datetime(obs["start_date_time"], errors="coerce", utc=True)
    stop = pd.to_datetime(obs["stop_date_time"], errors="coerce", utc=True)

    out = pd.DataFrame(
        {
            "orbit": pd.to_numeric(obs["orbit_number"], errors="coerce").astype("Int64"),
            "product_id": obs.apply(_product_id, axis=1),
            "start_time": start.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stop_time": stop.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_bin": start.dt.strftime("%Y-%m-%d"),
            "lon_centre": obs["lon_centre"].astype(float),
            "lat_centre": obs["lat_centre"].astype(float),
            "coverage_weight": 1.0,
            "source_url": obs["urn"].astype(str),
            "spatial_resolution_km_pixel": pd.to_numeric(
                obs.get("spatial_resolution_km_pixel"), errors="coerce"
            ),
            "emission_angle_deg": pd.to_numeric(
                obs.get("emission_angle_deg"), errors="coerce"
            ),
            "phase_angle_deg": pd.to_numeric(obs.get("phase_angle_deg"), errors="coerce"),
            "instrument_mode_id": obs.get("instrument_mode_id", "").fillna("").astype(str),
        }
    )
    out = out.dropna(subset=["orbit", "start_time", "time_bin"])
    out = out.sort_values(["orbit", "start_time", "product_id"]).reset_index(drop=True)
    logger.info("Normalized %d JIRAM product coverage rows.", len(out))
    return out


def save_jiram_observation_coverage(
    coverage: pd.DataFrame,
    path: Path | None = None,
) -> Path:
    """Persist normalized JIRAM coverage metadata."""
    path = path or PROCESSED_DIR / JIRAM_COVERAGE_FILENAME
    missing = set(REQUIRED_COVERAGE_COLUMNS) - set(coverage.columns)
    if missing:
        raise ValueError(f"Coverage table missing required columns: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(path, index=False)
    return path


def load_jiram_observation_coverage(path: Path | None = None) -> pd.DataFrame:
    """Load saved coverage metadata, or build it from raw product logs."""
    path = path or PROCESSED_DIR / JIRAM_COVERAGE_FILENAME
    if path.exists():
        return pd.read_parquet(path)
    coverage = normalize_jiram_product_logs()
    save_jiram_observation_coverage(coverage, path)
    return coverage


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = normalize_jiram_product_logs()
    out_path = save_jiram_observation_coverage(df)
    print(f"Saved {len(df)} rows to {out_path}")
