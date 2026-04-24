"""
tests/test_grid.py
------------------
Unit tests for preprocess/grid.py
"""

import numpy as np
import pandas as pd
import pytest

from preprocess.grid import build_base_grid
from config import GRID_N_CELLS, GRID_N_LON, GRID_N_LAT, GRID_RESOLUTION_DEG


class TestBuildBaseGrid:
    def test_correct_number_of_cells(self):
        grid = build_base_grid()
        assert len(grid) == GRID_N_CELLS, (
            f"Expected {GRID_N_CELLS} cells, got {len(grid)}"
        )

    def test_correct_columns(self):
        grid = build_base_grid()
        required = {"cell_id", "lon_centre", "lat_centre", "lon_min", "lon_max", "lat_min", "lat_max"}
        assert required.issubset(set(grid.columns))

    def test_longitude_range(self):
        grid = build_base_grid()
        assert grid["lon_centre"].min() >= -180
        assert grid["lon_centre"].max() <= 180

    def test_latitude_range(self):
        grid = build_base_grid()
        assert grid["lat_centre"].min() >= -90
        assert grid["lat_centre"].max() <= 90

    def test_cell_id_unique(self):
        grid = build_base_grid()
        assert grid["cell_id"].nunique() == len(grid)

    def test_cell_widths_correct(self):
        grid = build_base_grid()
        lon_widths = grid["lon_max"] - grid["lon_min"]
        lat_widths = grid["lat_max"] - grid["lat_min"]
        np.testing.assert_allclose(lon_widths, GRID_RESOLUTION_DEG, atol=1e-6)
        np.testing.assert_allclose(lat_widths, GRID_RESOLUTION_DEG, atol=1e-6)

    def test_centres_are_midpoints(self):
        grid = build_base_grid()
        expected_lon = (grid["lon_min"] + grid["lon_max"]) / 2
        expected_lat = (grid["lat_min"] + grid["lat_max"]) / 2
        np.testing.assert_allclose(grid["lon_centre"], expected_lon, atol=1e-6)
        np.testing.assert_allclose(grid["lat_centre"], expected_lat, atol=1e-6)

    def test_no_missing_values(self):
        grid = build_base_grid()
        assert not grid.isnull().any().any()
