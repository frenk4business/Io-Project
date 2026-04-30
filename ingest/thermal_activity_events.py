"""
ingest/thermal_activity_events.py
---------------------------------
Normalize Io thermal activity catalogues into one event table.

Large mission products are intentionally not downloaded here. This module uses
the small structured catalogues already present in the project and validates
optional curated tables for Mura 2024, Galileo NIMS, and ground-based AO data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import GRID_RESOLUTION_DEG, PROCESSED_DIR, RAW_DIR
from ingest.power_catalog import load_power_catalog

ACTIVITY_EVENTS_FILENAME = "io_thermal_activity_events.parquet"

REQUIRED_ACTIVITY_EVENT_COLUMNS: list[str] = [
    "event_id",
    "source_dataset",
    "source_id",
    "name",
    "longitude",
    "latitude",
    "cell_id",
    "observation_time",
    "time_bin",
    "instrument",
    "wavelength_um",
    "intensity_value",
    "intensity_unit",
    "power_gw",
    "is_power_estimated",
    "quality_flag",
]

MURA_MANUAL_SCHEMA: list[str] = [
    "source_id",
    "name",
    "longitude",
    "latitude",
    "orbit",
    "observation_time",
    "power_gw",
    "instrument",
    "source",
]

NIMS_SCHEMA: list[str] = [
    "source_id",
    "name",
    "longitude",
    "latitude",
    "observation_time",
    "wavelength_um",
    "spectral_radiance",
    "spectral_radiance_unit",
    "source",
]

AO_SCHEMA: list[str] = [
    "source_id",
    "name",
    "longitude",
    "latitude",
    "observation_time",
    "brightness_value",
    "brightness_unit",
    "instrument",
    "source",
]


def assign_cell_id(
    longitude: pd.Series | np.ndarray,
    latitude: pd.Series | np.ndarray,
    resolution_deg: float = GRID_RESOLUTION_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map lon/lat coordinates to the project's stable 1 degree cell ID."""
    lon = ((np.asarray(longitude, dtype=float) + 180.0) % 360.0) - 180.0
    lat = np.asarray(latitude, dtype=float)
    lon_idx = np.floor((lon + 180.0) / resolution_deg).clip(0, int(360 / resolution_deg) - 1)
    lat_idx = np.floor((lat + 90.0) / resolution_deg).clip(0, int(180 / resolution_deg) - 1)
    cell_id = lat_idx * int(360 / resolution_deg) + lon_idx
    half = resolution_deg / 2.0
    lon_centre = -180.0 + lon_idx * resolution_deg + half
    lat_centre = -90.0 + lat_idx * resolution_deg + half
    return cell_id.astype(int), lon_centre.astype(float), lat_centre.astype(float)


def _finalize_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUIRED_ACTIVITY_EVENT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["longitude"] = ((pd.to_numeric(out["longitude"], errors="coerce") + 180.0) % 360.0) - 180.0
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["observation_time"] = pd.to_datetime(out["observation_time"], errors="coerce", utc=True)
    out["time_bin"] = out["observation_time"].dt.strftime("%Y-%m")
    out["wavelength_um"] = pd.to_numeric(out["wavelength_um"], errors="coerce")
    out["intensity_value"] = pd.to_numeric(out["intensity_value"], errors="coerce")
    out["power_gw"] = pd.to_numeric(out["power_gw"], errors="coerce")
    out = out.dropna(subset=["longitude", "latitude"])
    out = out[out["latitude"].between(-90.0, 90.0)]
    cell_id, _, _ = assign_cell_id(out["longitude"], out["latitude"])
    out["cell_id"] = cell_id
    out["event_id"] = out["event_id"].fillna("").astype(str)
    missing_event_id = out["event_id"].eq("")
    out.loc[missing_event_id, "event_id"] = [
        f"event_{i:06d}" for i in range(int(missing_event_id.sum()))
    ]
    out["source_id"] = out["source_id"].fillna(out["event_id"]).astype(str)
    out["name"] = out["name"].fillna("").astype(str)
    out["instrument"] = out["instrument"].fillna("unknown").astype(str).str.upper()
    out.loc[out["instrument"].str.contains("JIRAM", na=False), "instrument"] = "JIRAM"
    out.loc[out["instrument"].str.contains("NIMS", na=False), "instrument"] = "NIMS"
    out["source_dataset"] = out["source_dataset"].fillna("unknown").astype(str)
    out["intensity_unit"] = out["intensity_unit"].fillna("").astype(str)
    out["quality_flag"] = out["quality_flag"].fillna("usable_metadata").astype(str)
    out["is_power_estimated"] = out["is_power_estimated"].fillna(True).astype(bool)
    return out[REQUIRED_ACTIVITY_EVENT_COLUMNS].reset_index(drop=True)


def normalize_davies_power_events(power_catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize the existing Davies/JIRAM estimated proxy catalogue."""
    cat = power_catalog if power_catalog is not None else load_power_catalog()
    rows = cat.copy()
    if "epoch" not in rows.columns:
        rows["epoch"] = "2024"
    rows["event_id"] = [
        f"davies_jiram_{str(source_id) if pd.notna(source_id) else i}"
        for i, source_id in enumerate(rows.get("source_id", rows.index))
    ]
    rows["source_dataset"] = "Davies_2024_JIRAM_proxy"
    rows["source_id"] = rows.get("source_id", rows["event_id"])
    rows["observation_time"] = rows["epoch"].astype(str).str.extract(r"(\d{4})", expand=False).fillna("2024") + "-01-01"
    rows["instrument"] = rows.get("instrument", "JIRAM")
    rows["wavelength_um"] = 4.8
    rows["intensity_value"] = rows["power_gw"]
    rows["intensity_unit"] = "estimated_proxy_GW"
    rows["is_power_estimated"] = rows.get("power_is_estimated", True)
    rows["quality_flag"] = "estimated_from_4p8um_spectral_radiance"
    return _finalize_events(rows)


def validate_manual_schema(path: Path, required_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")
    return df


def normalize_mura_events(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DIR / "mura_2024_hotspot_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Mura curated CSV not found at {path}. Required columns: {', '.join(MURA_MANUAL_SCHEMA)}."
        )
    rows = validate_manual_schema(path, MURA_MANUAL_SCHEMA)
    rows["event_id"] = "mura_2024_" + rows["source_id"].astype(str) + "_" + rows["orbit"].astype(str)
    rows["source_dataset"] = "Mura_2024_JIRAM_timeseries"
    rows["wavelength_um"] = pd.to_numeric(rows.get("wavelength_um", 4.8), errors="coerce")
    if "intensity_value" in rows.columns:
        rows["intensity_value"] = pd.to_numeric(rows["intensity_value"], errors="coerce")
        rows["intensity_unit"] = rows.get("intensity_unit", "reported_value")
        rows["is_power_estimated"] = rows["power_gw"].notna()
        rows["quality_flag"] = "curated_or_extracted_table_preserves_source_units"
    else:
        rows["intensity_value"] = pd.to_numeric(rows["power_gw"], errors="coerce")
        rows["intensity_unit"] = "reported_or_estimated_GW"
        rows["is_power_estimated"] = True
        rows["quality_flag"] = "curated_manual_table"
    return _finalize_events(rows)


def normalize_nims_events(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DIR / "galileo_nims_io_hotspot_spectral_radiance.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"NIMS derived table not found at {path}. Required columns: {', '.join(NIMS_SCHEMA)}."
        )
    rows = validate_manual_schema(path, NIMS_SCHEMA)
    rows["event_id"] = "nims_" + rows["source_id"].astype(str)
    rows["source_dataset"] = "Galileo_NIMS_spectral_radiance"
    rows["instrument"] = "NIMS"
    rows["intensity_value"] = pd.to_numeric(rows["spectral_radiance"], errors="coerce")
    rows["intensity_unit"] = rows["spectral_radiance_unit"]
    rows["power_gw"] = np.nan
    rows["is_power_estimated"] = False
    rows["quality_flag"] = "spectral_radiance_no_bolometric_conversion"
    return _finalize_events(rows)


def normalize_ao_events(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DIR / "ground_based_ao_io_hotspots.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"AO catalogue not found at {path}. Required columns: {', '.join(AO_SCHEMA)}."
        )
    rows = validate_manual_schema(path, AO_SCHEMA)
    rows["event_id"] = "ao_" + rows["source_id"].astype(str)
    rows["source_dataset"] = "Ground_based_AO_hotspot_catalogue"
    rows["intensity_value"] = pd.to_numeric(rows["brightness_value"], errors="coerce")
    rows["intensity_unit"] = rows["brightness_unit"]
    if "wavelength_um" in rows.columns:
        rows["wavelength_um"] = pd.to_numeric(rows["wavelength_um"], errors="coerce")
    rows["power_gw"] = np.nan
    rows["is_power_estimated"] = False
    rows["quality_flag"] = "brightness_without_unified_power_conversion"
    return _finalize_events(rows)


def load_activity_events(
    include_optional: bool = True,
    auto_fetch_external: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load all available normalized activity events.

    Missing optional datasets are reported in the status dictionary and do not
    fail the pipeline.
    """
    frames = [normalize_davies_power_events()]
    status = {"Davies/JIRAM": "loaded"}
    optional_loaders = {
        "Mura 2024": normalize_mura_events,
        "Galileo NIMS": normalize_nims_events,
        "Ground-based AO": normalize_ao_events,
    }
    if include_optional and auto_fetch_external:
        try:
            from ingest.external_activity_sources import fetch_all_external_activity_sources

            status.update(
                {f"external fetch {k}": v for k, v in fetch_all_external_activity_sources().items()}
            )
        except Exception as exc:
            status["external fetch"] = f"failed: {exc}"

    if include_optional:
        for label, loader in optional_loaders.items():
            try:
                events = loader()
                frames.append(events)
                status[label] = f"loaded {len(events)} rows from {events['source_dataset'].nunique()} dataset(s)"
            except FileNotFoundError as exc:
                status[label] = str(exc)
            except ValueError as exc:
                status[label] = f"schema error: {exc}"
    return pd.concat(frames, ignore_index=True), status


def save_activity_events(events: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or PROCESSED_DIR / ACTIVITY_EVENTS_FILENAME
    missing = set(REQUIRED_ACTIVITY_EVENT_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"Activity events missing required columns: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(path, index=False)
    return path


def load_saved_activity_events(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_DIR / ACTIVITY_EVENTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Activity events table not found at {path}.")
    return pd.read_parquet(path)
