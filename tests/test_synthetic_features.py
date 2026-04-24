"""
tests/test_synthetic_features.py
---------------------------------
Unit tests for features/synthetic.py
"""

import numpy as np
import pandas as pd
import pytest

from features.synthetic import (
    _haversine_km,
    add_distance_to_nearest_hotspot,
    add_distance_to_tidal_stress_max,
    TIDAL_STRESS_MAXIMA,
)
from config import IO_RADIUS_KM


class TestHaversine:
    def test_same_point_is_zero(self):
        d = _haversine_km(
            np.array([0.0]), np.array([0.0]), 0.0, 0.0
        )
        assert d[0] == pytest.approx(0.0, abs=1e-6)

    def test_antipodal_points(self):
        """Antipodal distance should be half the circumference."""
        d = _haversine_km(
            np.array([0.0]), np.array([0.0]), 0.0, 180.0
        )
        expected = np.pi * IO_RADIUS_KM
        assert d[0] == pytest.approx(expected, rel=1e-3)

    def test_pole_to_pole(self):
        d = _haversine_km(
            np.array([90.0]), np.array([0.0]), -90.0, 0.0
        )
        expected = np.pi * IO_RADIUS_KM
        assert d[0] == pytest.approx(expected, rel=1e-3)

    def test_output_shape_matches_input(self):
        lats = np.random.uniform(-90, 90, 50)
        lons = np.random.uniform(-180, 180, 50)
        d = _haversine_km(lats, lons, 0.0, 0.0)
        assert d.shape == (50,)

    def test_distances_non_negative(self):
        lats = np.random.uniform(-90, 90, 100)
        lons = np.random.uniform(-180, 180, 100)
        d = _haversine_km(lats, lons, 0.0, 0.0)
        assert np.all(d >= 0)


class TestDistanceFeatures:
    def _make_grid(self, n: int = 100) -> pd.DataFrame:
        return pd.DataFrame({
            "cell_id": range(n),
            "lon_centre": np.random.uniform(-180, 180, n),
            "lat_centre": np.random.uniform(-90, 90, n),
        })

    def _make_catalog(self, n: int = 20) -> pd.DataFrame:
        return pd.DataFrame({
            "name": [f"H{i}" for i in range(n)],
            "longitude": np.random.uniform(-180, 180, n),
            "latitude": np.random.uniform(-90, 90, n),
        })

    def test_nearest_hotspot_adds_column(self):
        grid = self._make_grid()
        catalog = self._make_catalog()
        result = add_distance_to_nearest_hotspot(grid, catalog)
        assert "dist_nearest_hotspot_km" in result.columns

    def test_nearest_hotspot_non_negative(self):
        grid = self._make_grid()
        catalog = self._make_catalog()
        result = add_distance_to_nearest_hotspot(grid, catalog)
        assert (result["dist_nearest_hotspot_km"] >= 0).all()

    def test_cell_at_hotspot_location_has_zero_distance(self):
        grid = pd.DataFrame({
            "cell_id": [0],
            "lon_centre": [45.0],
            "lat_centre": [30.0],
        })
        catalog = pd.DataFrame({
            "name": ["test"],
            "longitude": [45.0],
            "latitude": [30.0],
        })
        result = add_distance_to_nearest_hotspot(grid, catalog)
        assert result["dist_nearest_hotspot_km"].iloc[0] == pytest.approx(0.0, abs=1e-3)

    def test_tidal_stress_adds_column(self):
        grid = self._make_grid()
        result = add_distance_to_tidal_stress_max(grid)
        assert "dist_tidal_stress_max_km" in result.columns

    def test_sub_jupiter_point_near_zero(self):
        """The sub-Jupiter point (0°N, 0°E) should have near-zero tidal distance."""
        grid = pd.DataFrame({
            "cell_id": [0],
            "lon_centre": [0.0],
            "lat_centre": [0.0],
        })
        result = add_distance_to_tidal_stress_max(grid)
        assert result["dist_tidal_stress_max_km"].iloc[0] == pytest.approx(0.0, abs=1e-3)

    def test_original_grid_not_mutated(self):
        grid = self._make_grid()
        catalog = self._make_catalog()
        original_cols = set(grid.columns)
        _ = add_distance_to_nearest_hotspot(grid, catalog)
        assert set(grid.columns) == original_cols, "Input grid must not be mutated"
