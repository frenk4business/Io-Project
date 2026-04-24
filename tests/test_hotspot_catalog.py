"""
tests/test_hotspot_catalog.py
------------------------------
Unit tests for ingest/hotspot_catalog.py

Uses temporary CSV files to avoid requiring real data during testing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ingest.hotspot_catalog import (
    load_hotspot_catalog,
    REQUIRED_COLUMNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_catalog_csv(rows: list[dict], path: Path) -> Path:
    """Write a minimal hotspot catalog CSV to a temp path."""
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


VALID_ROWS = [
    {"Name": "Pele", "Longitude": 255.0, "Latitude": -19.0, "Temperature_K": 1700.0, "Source": "Galileo"},
    {"Name": "Loki", "Longitude": 308.8, "Latitude": 12.6, "Temperature_K": 600.0, "Source": "Voyager"},
    {"Name": "Prometheus", "Longitude": 153.5, "Latitude": -2.7, "Temperature_K": 450.0, "Source": "Galileo"},
]

VALID_ROWS_360 = [
    # Longitude in 0–360 range, should be converted to -180–180
    {"Name": "Pele", "Longitude": 255.0, "Latitude": -19.0, "Temperature_K": 1700.0, "Source": "Galileo"},
    {"Name": "Tvashtar", "Longitude": 360.0 - 124.0, "Latitude": 63.0, "Temperature_K": 800.0, "Source": "Galileo"},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadHotspotCatalog:

    def test_loads_valid_csv(self, tmp_path):
        path = _write_catalog_csv(VALID_ROWS, tmp_path / "hotspots.csv")
        df = load_hotspot_catalog(path)
        assert len(df) == 3

    def test_required_columns_present(self, tmp_path):
        path = _write_catalog_csv(VALID_ROWS, tmp_path / "hotspots.csv")
        df = load_hotspot_catalog(path)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing required column: {col}"

    def test_provenance_column_added(self, tmp_path):
        path = _write_catalog_csv(VALID_ROWS, tmp_path / "hotspots.csv")
        df = load_hotspot_catalog(path)
        assert "provenance" in df.columns
        assert df["provenance"].notna().all()

    def test_longitude_normalized_to_180(self, tmp_path):
        path = _write_catalog_csv(VALID_ROWS_360, tmp_path / "hotspots.csv")
        df = load_hotspot_catalog(path, normalize_longitude=True)
        assert df["longitude"].max() <= 180.0
        assert df["longitude"].min() >= -180.0

    def test_no_normalization_when_disabled(self, tmp_path):
        path = _write_catalog_csv(VALID_ROWS_360, tmp_path / "hotspots.csv")
        df = load_hotspot_catalog(path, normalize_longitude=False)
        # Original 0–360 values should be preserved
        assert df["longitude"].max() > 180.0

    def test_numeric_coercion(self, tmp_path):
        rows = [
            {"Name": "Pele", "Longitude": "255.0", "Latitude": "-19.0",
             "Temperature_K": "1700", "Source": "Galileo"},
        ]
        path = _write_catalog_csv(rows, tmp_path / "hotspots.csv")
        df = load_hotspot_catalog(path)
        assert df["longitude"].dtype == float
        assert df["latitude"].dtype == float

    def test_drops_rows_with_missing_coordinates(self, tmp_path):
        rows = VALID_ROWS + [
            {"Name": "Unknown", "Longitude": None, "Latitude": None,
             "Temperature_K": None, "Source": ""},
        ]
        path = _write_catalog_csv(rows, tmp_path / "hotspots.csv")
        df = load_hotspot_catalog(path)
        assert len(df) == 3  # missing-coordinate row dropped
        assert df["longitude"].notna().all()
        assert df["latitude"].notna().all()

    def test_raises_if_file_missing(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_hotspot_catalog(Path("/nonexistent/path/hotspots.csv"))

    def test_raises_if_required_columns_absent(self, tmp_path):
        # CSV with wrong column names and no matching COLUMN_MAP entry
        rows = [{"x": 1, "y": 2, "z": 3}]
        path = _write_catalog_csv(rows, tmp_path / "hotspots.csv")
        with pytest.raises(ValueError, match="missing required columns"):
            load_hotspot_catalog(path)

    def test_temperature_can_be_nan(self, tmp_path):
        rows = [
            {"Name": "Pele", "Longitude": 255.0, "Latitude": -19.0,
             "Temperature_K": None, "Source": "Galileo"},
        ]
        path = _write_catalog_csv(rows, tmp_path / "hotspots.csv")
        df = load_hotspot_catalog(path)
        assert len(df) == 1
        assert pd.isna(df["temperature"].iloc[0])

    def test_coordinate_range_valid(self, tmp_path):
        path = _write_catalog_csv(VALID_ROWS, tmp_path / "hotspots.csv")
        df = load_hotspot_catalog(path)
        assert df["longitude"].between(-180, 180).all()
        assert df["latitude"].between(-90, 90).all()
