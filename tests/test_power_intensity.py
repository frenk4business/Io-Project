"""
tests/test_power_intensity.py
-----------------------------
Contract tests for Davies/JIRAM estimated thermal-emission proxy ingestion,
grid aggregation, regression, and intensity summaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_grid() -> pd.DataFrame:
    rows = []
    cell_id = 0
    for lat in [-75.5, -25.5, 25.5, 75.5]:
        for lon in [-10.5, -5.5, 5.5, 10.5]:
            rows.append(
                {
                    "cell_id": cell_id,
                    "lon_centre": lon,
                    "lat_centre": lat,
                    "lon_min": lon - 0.5,
                    "lon_max": lon + 0.5,
                    "lat_min": lat - 0.5,
                    "lat_max": lat + 0.5,
                }
            )
            cell_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_power_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["a", "b", "c", "bad_zero", "bad_null"],
            "longitude": [-10.4, -10.2, 5.1, 10.0, 20.0],
            "latitude": [-75.4, -75.2, 25.2, 25.0, 25.0],
            "power_gw": [10.0, 30.0, 5.0, 0.0, np.nan],
        }
    )


@pytest.fixture
def tiny_power_feature_matrix(tiny_grid) -> pd.DataFrame:
    fm = tiny_grid.copy()
    fm["tidal_heating_flux"] = np.linspace(0.1, 1.0, len(fm))
    fm["geology_encoded"] = np.arange(len(fm)) % 3
    fm["dist_nearest_hotspot_km"] = np.linspace(1, 100, len(fm))
    fm["dist_tidal_stress_max_km"] = np.linspace(50, 5, len(fm))
    fm["geology_unit"] = np.where(fm["cell_id"] % 2 == 0, "Fb", "UNKNOWN")
    return fm


def test_load_power_catalog_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "bad_power.csv"
    pd.DataFrame({"name": ["x"], "longitude": [1.0]}).to_csv(path, index=False)
    from ingest.power_catalog import load_power_catalog

    with pytest.raises(ValueError, match="missing required columns"):
        load_power_catalog(path)


def test_load_power_catalog_normalizes_longitude_and_drops_invalid(tmp_path):
    path = tmp_path / "power.csv"
    pd.DataFrame(
        {
            "name": ["x", "y", "z"],
            "longitude": [350.0, -181.0, 10.0],
            "latitude": [0.0, 0.0, 0.0],
            "power_gw": [1.0, 2.0, 0.0],
        }
    ).to_csv(path, index=False)
    from ingest.power_catalog import load_power_catalog

    df = load_power_catalog(path)
    assert len(df) == 2
    assert set(df["longitude"].round(6)) == {-10.0, 179.0}
    assert (df["power_gw"] > 0).all()


def test_assign_power_to_grid_aggregates_duplicate_cells(tiny_grid, tiny_power_catalog):
    from preprocess.power_grid import assign_power_to_grid

    out = assign_power_to_grid(tiny_grid, tiny_power_catalog)
    cell = out[(out["lon_centre"] == -10.5) & (out["lat_centre"] == -75.5)].iloc[0]
    assert cell["power_count"] == 2
    assert cell["primary_power_gw"] == pytest.approx(30.0)
    assert cell["mean_power_gw"] == pytest.approx(20.0)
    assert cell["sum_power_gw"] == pytest.approx(40.0)
    assert cell["log_primary_power"] == pytest.approx(np.log1p(30.0))


def test_power_regression_runs_and_excludes_leakage(
    tiny_grid,
    tiny_power_feature_matrix,
):
    from models.regression import train_power_regression
    from preprocess.power_grid import assign_power_to_grid

    catalog = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "longitude": [-10.4, -5.4, 5.1, 10.1],
            "latitude": [-75.4, -25.4, 25.4, 75.4],
            "power_gw": [10.0, 20.0, 30.0, 40.0],
        }
    )
    power_grid = assign_power_to_grid(tiny_grid, catalog)
    res = train_power_regression(tiny_power_feature_matrix, power_grid)

    assert "dist_nearest_hotspot_km" not in res["features"]
    assert {"r2", "rmse", "mae", "spearman"}.issubset(res["overall_oof"])
    assert not res["residuals"].empty


def test_power_regression_refuses_leakage_feature(
    tiny_grid,
    tiny_power_feature_matrix,
    tiny_power_catalog,
):
    from models.regression import train_power_regression
    from preprocess.power_grid import assign_power_to_grid

    power_grid = assign_power_to_grid(tiny_grid, tiny_power_catalog)
    with pytest.raises(ValueError, match="leakage"):
        train_power_regression(
            tiny_power_feature_matrix,
            power_grid,
            feature_cols=["dist_nearest_hotspot_km"],
        )


def test_power_intensity_tables_have_required_columns(
    tiny_grid,
    tiny_power_feature_matrix,
    tiny_power_catalog,
):
    from analysis.power_intensity import compute_power_intensity_suite
    from preprocess.power_grid import assign_power_to_grid

    power_grid = assign_power_to_grid(tiny_grid, tiny_power_catalog)
    suite = compute_power_intensity_suite(tiny_power_feature_matrix, power_grid)

    assert {"lat_band", "sum_power_gw", "fraction_total_power"}.issubset(
        suite["by_latitude"].columns
    )
    assert {"geology_unit", "sum_power_gw", "max_primary_power_gw"}.issubset(
        suite["by_geology"].columns
    )
    assert {"removed_top_n_cells", "polar_fraction_total_power"}.issubset(
        suite["outlier_sensitivity"].columns
    )
    assert {"polar_threshold_abs_lat", "polar_sum_power_gw"}.issubset(
        suite["polar_sensitivity"].columns
    )
