from __future__ import annotations

import pandas as pd


def test_missing_file_diagnostic_includes_path_purpose_policy_and_command():
    from dashboard.data_loader import FILE_SPECS, missing_file_diagnostic

    diagnostic = missing_file_diagnostic("feature_matrix")
    spec = FILE_SPECS["feature_matrix"]

    assert str(spec.path) in diagnostic
    assert spec.purpose in diagnostic
    assert spec.restore_policy in diagnostic
    assert "python -m features.build" in diagnostic


def test_cached_loaders_return_expected_shapes(tmp_path):
    from dashboard.data_loader import (
        load_feature_matrix_cached,
        load_hotspot_catalog_cached,
        load_power_grid_cached,
    )

    catalog_path = tmp_path / "hotspots.csv"
    pd.DataFrame(
        {
            "Name": ["Loki"],
            "Longitude": [310.0],
            "Latitude": [10.0],
            "Temperature_K": [500],
            "Source": ["test"],
        }
    ).to_csv(catalog_path, index=False)

    feature_path = tmp_path / "feature.parquet"
    pd.DataFrame(
        {
            "cell_id": [1, 2],
            "lon_centre": [0.5, 1.5],
            "lat_centre": [0.5, 1.5],
            "has_hotspot": [1, 0],
        }
    ).to_parquet(feature_path, index=False)

    power_path = tmp_path / "power.parquet"
    pd.DataFrame(
        {
            "cell_id": [1],
            "primary_power_gw": [42.0],
            "power_count": [1],
        }
    ).to_parquet(power_path, index=False)

    load_hotspot_catalog_cached.clear()
    load_feature_matrix_cached.clear()
    load_power_grid_cached.clear()

    catalog = load_hotspot_catalog_cached(catalog_path)
    feature = load_feature_matrix_cached(feature_path)
    power = load_power_grid_cached(power_path)

    assert catalog.shape[0] == 1
    assert feature.shape == (2, 4)
    assert power.shape == (1, 3)
    assert catalog.loc[0, "longitude"] == -50.0


def test_dashboard_app_import_does_not_run_router(monkeypatch):
    import importlib
    import sys

    import streamlit as st

    def fail_set_page_config(*args, **kwargs):
        raise AssertionError("st.set_page_config should only run from main()")

    monkeypatch.setattr(st, "set_page_config", fail_set_page_config)
    sys.modules.pop("dashboard.app", None)

    app = importlib.import_module("dashboard.app")

    assert callable(app.main)
    assert app.NAV_GROUPS["Explore Io"] == ["Io Experience", "2D Maps", "3D Globe"]


def test_check_dashboard_data_reports_missing_and_returns_nonzero(tmp_path, monkeypatch, capsys):
    from dashboard.data_loader import DashboardFileSpec
    from scripts import check_dashboard_data

    missing = tmp_path / "missing.parquet"
    spec = DashboardFileSpec(
        key="missing_fixture",
        path=missing,
        purpose="test purpose",
        restore_policy="restore for test",
        regenerate_command="python -m test.fixture",
    )
    monkeypatch.setattr(check_dashboard_data, "FILE_SPECS", {"missing_fixture": spec})
    monkeypatch.setattr(check_dashboard_data, "REQUIRED_DASHBOARD_FILE_KEYS", ("missing_fixture",))

    code = check_dashboard_data.main()
    out = capsys.readouterr().out

    assert code == 1
    assert "FAIL:" in out
    assert str(missing) in out
    assert "python -m test.fixture" in out
