from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def tiny_feature_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "lon_centre": [-10.5, 20.5, 40.5, 80.5],
            "lat_centre": [-60.5, 10.5, 30.5, 70.5],
            "has_hotspot": [1, 0, 0, 0],
            "hotspot_count": [1, 0, 0, 0],
            "hotspot_names": ["named", "", "", ""],
            "geology_unit": ["Fb", "UNKNOWN", "Fd", "Fd"],
        }
    )


def tiny_power_grid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "power_count": [0, 1, 1, 0],
            "sum_power_gw": [0.0, 100.0, 10.0, 0.0],
            "primary_power_gw": [0.0, 100.0, 10.0, 0.0],
            "power_names": ["", "thermal_hi", "thermal_low", ""],
        }
    )


def tiny_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "orbit": [57, 57, 58],
            "product_id": ["a", "b", "c"],
            "start_time": [
                "2023-12-30T00:00:00Z",
                "2023-12-30T00:01:00Z",
                "2024-02-03T00:00:00Z",
            ],
            "stop_time": [
                "2023-12-30T00:00:01Z",
                "2023-12-30T00:01:01Z",
                "2024-02-03T00:00:01Z",
            ],
            "time_bin": ["2023-12-30", "2023-12-30", "2024-02-03"],
            "lon_centre": [20.5, 20.5, 40.5],
            "lat_centre": [10.5, 10.5, 30.5],
            "coverage_weight": [1.0, 1.0, 1.0],
            "source_url": ["urn:a", "urn:b", "urn:c"],
        }
    )


def test_normalize_jiram_product_logs(tmp_path: Path):
    path = tmp_path / "orbit57.csv"
    pd.DataFrame(
        {
            "urn": ["urn:nasa:pds:test:prod1"],
            "start_date_time": ["2023-12-30T08:35:15.302Z"],
            "stop_date_time": ["2023-12-30T08:35:15.304Z"],
            "file_name": ["JIR_IMG_RDR_TEST.LBL"],
            "target_name": ["IO"],
            "orbit_number": [57],
            "io_planetocentric_longitude_deg": [-104.058],
            "io_planetocentric_latitude_deg": [57.673],
            "spatial_resolution_km_pixel": [0.4],
            "emission_angle_deg": [0.0],
            "phase_angle_deg": [110.0],
            "instrument_mode_id": ["SCI_I1_S3"],
        }
    ).to_csv(path, index=False)

    from ingest.jiram_coverage import REQUIRED_COVERAGE_COLUMNS, normalize_jiram_product_logs

    out = normalize_jiram_product_logs([path])
    assert set(REQUIRED_COVERAGE_COLUMNS).issubset(out.columns)
    assert out.loc[0, "orbit"] == 57
    assert out.loc[0, "product_id"] == "JIR_IMG_RDR_TEST"
    assert out.loc[0, "time_bin"] == "2023-12-30"


def test_mura_timeseries_missing_fails_clearly(tmp_path: Path):
    from analysis.time_resolved_activity import load_mura_hotspot_timeseries

    with pytest.raises(FileNotFoundError, match="curated CSV"):
        load_mura_hotspot_timeseries(tmp_path / "missing.csv")


def test_time_resolved_activity_metrics_and_classes():
    from analysis.time_resolved_activity import compute_time_resolved_activity

    result = compute_time_resolved_activity(
        tiny_feature_matrix(),
        tiny_power_grid(),
        tiny_coverage(),
    )
    cell = result["cell_activity"]
    assert not cell.empty
    high = cell[cell["cell_id"] == 2].iloc[0]
    assert high["coverage_normalized_proxy"] == pytest.approx(50.0)
    assert high["persistence_class"] in {"thermal_only", "episodic_high_power"}
    named = cell[cell["cell_id"] == 1].iloc[0]
    assert named["persistence_class"] == "named_only"
    limited = cell[cell["cell_id"] == 4].iloc[0]
    assert limited["persistence_class"] == "coverage_limited"
    assert result["data_quality"]["coverage_rows"] == 3


def test_persistent_class_from_multiple_time_bins():
    from analysis.time_resolved_activity import compute_time_resolved_activity

    cov = tiny_coverage()
    cov.loc[1, "time_bin"] = "2024-02-03"
    result = compute_time_resolved_activity(tiny_feature_matrix(), tiny_power_grid(), cov)
    cell = result["cell_activity"]
    assert cell[cell["cell_id"] == 2].iloc[0]["persistence_class"] == "persistent_thermal"

