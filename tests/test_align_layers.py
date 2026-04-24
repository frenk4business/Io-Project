"""
tests/test_align_layers.py
--------------------------
Unit tests for preprocess/align_layers.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocess.grid import build_base_grid
from preprocess.align_layers import assign_hotspots_to_grid


def _make_catalog(**kwargs) -> pd.DataFrame:
    defaults = {
        "name": ["Pele", "Loki", "Prometheus"],
        "longitude": [75.0, -51.2, -26.5],
        "latitude": [-19.0, 12.6, -2.7],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestAssignHotspotsToGrid:

    def test_output_has_same_length_as_grid(self):
        grid = build_base_grid()
        catalog = _make_catalog()
        result = assign_hotspots_to_grid(grid, catalog)
        assert len(result) == len(grid)

    def test_has_hotspot_column_is_binary(self):
        grid = build_base_grid()
        catalog = _make_catalog()
        result = assign_hotspots_to_grid(grid, catalog)
        assert set(result["has_hotspot"].unique()).issubset({0, 1})

    def test_hotspot_count_non_negative(self):
        grid = build_base_grid()
        catalog = _make_catalog()
        result = assign_hotspots_to_grid(grid, catalog)
        assert (result["hotspot_count"] >= 0).all()

    def test_total_hotspot_count_equals_catalog_size(self):
        """Sum of hotspot_count across all cells must equal catalog length."""
        grid = build_base_grid()
        catalog = _make_catalog()
        result = assign_hotspots_to_grid(grid, catalog)
        assert result["hotspot_count"].sum() == len(catalog)

    def test_hotspot_at_known_location_is_assigned(self):
        """A hotspot exactly at a known coordinate should appear in a cell."""
        grid = build_base_grid()
        # Place hotspot dead-centre in a known 1° cell
        catalog = pd.DataFrame({
            "name": ["Test"],
            "longitude": [0.5],   # centre of cell lon [-180+0 to -180+1]... actually cell centre is 0.5
            "latitude": [0.5],
        })
        result = assign_hotspots_to_grid(grid, catalog)
        # The cell containing (0.5, 0.5) should have has_hotspot=1
        cell = result[
            (result["lon_min"] <= 0.5) & (result["lon_max"] > 0.5) &
            (result["lat_min"] <= 0.5) & (result["lat_max"] > 0.5)
        ]
        assert len(cell) == 1
        assert cell["has_hotspot"].iloc[0] == 1

    def test_multiple_hotspots_in_same_cell_counted(self):
        """Two hotspots in the same grid cell should have hotspot_count=2."""
        grid = build_base_grid()
        catalog = pd.DataFrame({
            "name": ["A", "B"],
            "longitude": [0.2, 0.7],  # both in [0, 1)° cell
            "latitude": [0.2, 0.7],
        })
        result = assign_hotspots_to_grid(grid, catalog)
        cell = result[
            (result["lon_min"] <= 0.2) & (result["lon_max"] > 0.7) &
            (result["lat_min"] <= 0.2) & (result["lat_max"] > 0.7)
        ]
        if len(cell) == 1:  # both in same cell
            assert cell["hotspot_count"].iloc[0] == 2

    def test_no_hotspot_cells_have_count_zero(self):
        grid = build_base_grid()
        catalog = _make_catalog()
        result = assign_hotspots_to_grid(grid, catalog)
        mask = result["has_hotspot"] == 0
        assert (result.loc[mask, "hotspot_count"] == 0).all()

    def test_positive_cells_have_hotspot_names(self):
        grid = build_base_grid()
        catalog = _make_catalog()
        result = assign_hotspots_to_grid(grid, catalog)
        positive = result[result["has_hotspot"] == 1]
        assert (positive["hotspot_names"] != "").all()

    def test_cell_id_preserved(self):
        """cell_id values should be unchanged after hotspot assignment."""
        grid = build_base_grid()
        catalog = _make_catalog()
        result = assign_hotspots_to_grid(grid, catalog)
        pd.testing.assert_series_equal(
            result["cell_id"].reset_index(drop=True),
            grid["cell_id"].reset_index(drop=True),
        )

    def test_empty_catalog_produces_all_negative(self):
        grid = build_base_grid()
        catalog = pd.DataFrame(columns=["name", "longitude", "latitude"])
        result = assign_hotspots_to_grid(grid, catalog)
        assert result["has_hotspot"].sum() == 0
        assert result["hotspot_count"].sum() == 0
