"""
tests/test_tidal_heating_feature.py
------------------------------------
Unit tests for features/tidal_heating.py and ingest/tidal_heating.py (synthetic proxy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingest.tidal_heating import load_synthetic_tidal_proxy
from features.tidal_heating import add_tidal_heating_feature
from preprocess.grid import build_base_grid


class TestSyntheticTidalProxy:

    def test_returns_dataframe(self):
        df = load_synthetic_tidal_proxy()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = load_synthetic_tidal_proxy()
        for col in ["longitude", "latitude", "flux_w_m2", "provenance"]:
            assert col in df.columns

    def test_longitude_range(self):
        df = load_synthetic_tidal_proxy()
        assert df["longitude"].between(-180, 180).all()

    def test_latitude_range(self):
        df = load_synthetic_tidal_proxy()
        assert df["latitude"].between(-90, 90).all()

    def test_flux_non_negative(self):
        df = load_synthetic_tidal_proxy()
        assert (df["flux_w_m2"] >= 0).all()

    def test_provenance_marks_synthetic(self):
        df = load_synthetic_tidal_proxy()
        assert "SYNTHETIC" in df["provenance"].iloc[0].upper()

    def test_resolution_parameter(self):
        df1 = load_synthetic_tidal_proxy(resolution_deg=1.0)
        df2 = load_synthetic_tidal_proxy(resolution_deg=2.0)
        assert len(df1) > len(df2)


class TestAddTidalHeatingFeature:

    def test_adds_column(self):
        grid = build_base_grid()
        tidal = load_synthetic_tidal_proxy()
        result = add_tidal_heating_feature(grid, tidal)
        assert "tidal_heating_flux" in result.columns

    def test_no_missing_values(self):
        grid = build_base_grid()
        tidal = load_synthetic_tidal_proxy()
        result = add_tidal_heating_feature(grid, tidal)
        assert result["tidal_heating_flux"].notna().all()

    def test_values_are_numeric(self):
        grid = build_base_grid()
        tidal = load_synthetic_tidal_proxy()
        result = add_tidal_heating_feature(grid, tidal)
        assert pd.api.types.is_float_dtype(result["tidal_heating_flux"])

    def test_grid_length_preserved(self):
        grid = build_base_grid()
        tidal = load_synthetic_tidal_proxy()
        result = add_tidal_heating_feature(grid, tidal)
        assert len(result) == len(grid)

    def test_raises_if_flux_col_missing(self):
        grid = build_base_grid()
        tidal = pd.DataFrame({"longitude": [0.0], "latitude": [0.0], "wrong_col": [1.0]})
        with pytest.raises(ValueError, match="flux_w_m2"):
            add_tidal_heating_feature(grid, tidal)

    def test_original_grid_not_mutated(self):
        grid = build_base_grid()
        original_cols = set(grid.columns)
        tidal = load_synthetic_tidal_proxy()
        _ = add_tidal_heating_feature(grid, tidal)
        assert set(grid.columns) == original_cols
