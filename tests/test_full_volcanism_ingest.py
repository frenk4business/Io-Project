from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.test_time_resolved_activity import tiny_coverage, tiny_feature_matrix, tiny_power_grid


def test_coverage_cube_builds_weighted_rows():
    from ingest.observation_coverage_cube import REQUIRED_COVERAGE_CUBE_COLUMNS, build_coverage_cube

    coverage = tiny_coverage()
    coverage["spatial_resolution_km_pixel"] = [0.5, 10.0, 30.0]
    coverage["emission_angle_deg"] = [0.0, 60.0, 75.0]
    cube = build_coverage_cube(tiny_feature_matrix(), coverage)

    assert set(REQUIRED_COVERAGE_CUBE_COLUMNS).issubset(cube.columns)
    assert len(cube) == 2
    cell2 = cube[cube["cell_id"] == 2].iloc[0]
    assert cell2["observation_count"] == 2
    assert 0 < cell2["coverage_weight"] < 2
    assert cell2["instrument"] == "JIRAM"


def test_davies_power_catalog_normalizes_to_activity_events():
    from ingest.thermal_activity_events import (
        REQUIRED_ACTIVITY_EVENT_COLUMNS,
        normalize_davies_power_events,
    )

    power_catalog = pd.DataFrame(
        {
            "source_id": ["D1"],
            "name": ["Test Patera"],
            "longitude": [181.0],
            "latitude": [10.2],
            "power_gw": [42.0],
            "epoch": ["2024"],
            "instrument": ["JIRAM"],
            "power_is_estimated": [True],
        }
    )
    events = normalize_davies_power_events(power_catalog)
    assert set(REQUIRED_ACTIVITY_EVENT_COLUMNS).issubset(events.columns)
    assert events.loc[0, "longitude"] == pytest.approx(-179.0)
    assert events.loc[0, "instrument"] == "JIRAM"
    assert events.loc[0, "power_gw"] == pytest.approx(42.0)


def test_mura_manual_schema_validation(tmp_path: Path):
    from ingest.thermal_activity_events import normalize_mura_events

    path = tmp_path / "mura.csv"
    pd.DataFrame(
        {
            "source_id": ["M1"],
            "name": ["hotspot"],
            "longitude": [20.1],
            "latitude": [10.1],
            "orbit": [41],
            "observation_time": ["2022-07-05T00:00:00Z"],
            "power_gw": [12.0],
            "instrument": ["JIRAM"],
            "source": ["Mura et al. 2024"],
        }
    ).to_csv(path, index=False)

    events = normalize_mura_events(path)
    assert len(events) == 1
    assert events.loc[0, "source_dataset"] == "Mura_2024_JIRAM_timeseries"
    assert events.loc[0, "time_bin"] == "2022-07"


def test_nims_and_ao_missing_fail_clearly(tmp_path: Path):
    from ingest.thermal_activity_events import normalize_ao_events, normalize_nims_events

    with pytest.raises(FileNotFoundError, match="NIMS derived table"):
        normalize_nims_events(tmp_path / "missing_nims.csv")
    with pytest.raises(FileNotFoundError, match="AO catalogue"):
        normalize_ao_events(tmp_path / "missing_ao.csv")


def test_v2_activity_uses_coverage_cube_and_events():
    from analysis.time_resolved_activity import compute_time_resolved_activity_v2
    from ingest.observation_coverage_cube import build_coverage_cube
    from ingest.thermal_activity_events import assign_cell_id, normalize_davies_power_events

    event_cell_id = int(assign_cell_id(pd.Series([20.5]), pd.Series([10.5]))[0][0])
    feature_matrix = tiny_feature_matrix()
    power_grid = tiny_power_grid()
    feature_matrix.loc[feature_matrix["cell_id"] == 2, "cell_id"] = event_cell_id
    power_grid.loc[power_grid["cell_id"] == 2, "cell_id"] = event_cell_id
    coverage = tiny_coverage()
    coverage["spatial_resolution_km_pixel"] = [1.0, 1.0, 1.0]
    coverage["emission_angle_deg"] = [0.0, 0.0, 0.0]
    events = normalize_davies_power_events(
        pd.DataFrame(
            {
                "source_id": ["E1", "E2"],
                "name": ["thermal_hi", "thermal_hi_again"],
                "longitude": [20.5, 20.5],
                "latitude": [10.5, 10.5],
                "power_gw": [100.0, 120.0],
                "epoch": ["2023", "2024"],
                "instrument": ["JIRAM", "JIRAM"],
                "power_is_estimated": [True, True],
            }
        )
    )
    cube = build_coverage_cube(feature_matrix, coverage)
    result = compute_time_resolved_activity_v2(
        feature_matrix,
        power_grid,
        coverage,
        coverage_cube=cube,
        activity_events=events,
    )

    cell = result["cell_activity"]
    row = cell[cell["cell_id"] == event_cell_id].iloc[0]
    assert row["event_count"] == 2
    assert row["coverage_corrected_activity"] > 0
    assert row["persistence_class"] == "persistent_active"
    assert "JIRAM" in result["available_instruments"]


def test_ao_dashboard_filter_matches_keck_gemini_ao_label():
    from analysis.time_resolved_activity import filter_activity_events

    events = pd.DataFrame(
        {
            "instrument": ["KECK/GEMINI AO", "JIRAM"],
            "time_bin": ["2017-01", "2017-01"],
        }
    )
    filtered = filter_activity_events(events, instrument="AO")
    assert len(filtered) == 1
    assert filtered.iloc[0]["instrument"] == "KECK/GEMINI AO"
