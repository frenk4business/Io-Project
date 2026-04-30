from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def tiny_grid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [1, 2, 3],
            "lon_centre": [10.5, 20.5, 30.5],
            "lat_centre": [0.5, 10.5, -10.5],
            "hotspot_count": [1, 0, 1],
            "has_hotspot": [1, 0, 1],
        }
    )


def tiny_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["d1", "m1", "n1", "a1", "a2"],
            "source_dataset": [
                "Davies_2024_JIRAM_proxy",
                "Mura_2024_JIRAM_timeseries",
                "Galileo_NIMS_spectral_radiance",
                "Ground_based_AO_hotspot_catalogue",
                "Ground_based_AO_hotspot_catalogue",
            ],
            "source_id": ["d1", "m1", "n1", "a1", "a2"],
            "name": ["Davies", "Mura", "NIMS", "AO", "AO"],
            "longitude": [10.5, 10.5, 20.5, 30.5, 30.5],
            "latitude": [0.5, 0.5, 10.5, -10.5, -10.5],
            "cell_id": [1, 1, 2, 3, 3],
            "observation_time": [
                "2024-01-01",
                "2022-04-09",
                "1996-06-28",
                "2017-01-03",
                "2018-01-03",
            ],
            "time_bin": ["2024-01", "2022-04", "1996-06", "2017-01", "2018-01"],
            "instrument": ["JIRAM", "JIRAM", "NIMS", "KECK/GEMINI AO", "KECK/GEMINI AO"],
            "wavelength_um": [4.8, 4.78, 3.2, 3.78, 3.78],
            "intensity_value": [100.0, 2.0, 20.0, 4.0, 10.0],
            "intensity_unit": [
                "estimated_proxy_GW",
                "W m-2 sr-1 peak_radiance",
                "PDS converted spectral_radiance",
                "GW/um/sr filter_integrated_flux_density",
                "GW/um/sr filter_integrated_flux_density",
            ],
            "power_gw": [100.0, None, None, None, None],
            "is_power_estimated": [True, False, False, False, False],
            "quality_flag": ["ok", "ok", "ok", "ok", "ok"],
        }
    )


def test_prepare_activity_events_keeps_intensity_families_separate():
    from analysis.coverage_corrected_volcanism import prepare_activity_events

    events = prepare_activity_events(tiny_events())
    families = set(events["intensity_family"])
    assert {
        "radiant_power_gw",
        "jiram_radiance",
        "nims_radiance",
        "ao_brightness",
    }.issubset(families)
    assert events.loc[events["intensity_family"].eq("radiant_power_gw"), "family_intensity_value"].iloc[0] == pytest.approx(100.0)
    assert events["family_percentile"].between(0, 1).all()


def test_multi_instrument_coverage_cube_has_nonzero_rows_for_each_instrument():
    from analysis.coverage_corrected_volcanism import build_multi_instrument_coverage_cube

    cube = build_multi_instrument_coverage_cube(tiny_grid(), tiny_events())
    assert set(cube["instrument"]) == {"JIRAM", "NIMS", "KECK/GEMINI AO"}
    assert cube["coverage_weight"].sum() > 0
    assert cube["cell_id"].nunique() == 3


def test_coverage_corrected_maps_are_nonzero_and_mask_zero_coverage():
    from analysis.coverage_corrected_volcanism import compute_coverage_corrected_volcanism

    grid = pd.concat(
        [
            tiny_grid(),
            pd.DataFrame(
                {
                    "cell_id": [4],
                    "lon_centre": [40.5],
                    "lat_centre": [40.5],
                    "hotspot_count": [0],
                    "has_hotspot": [0],
                }
            ),
        ],
        ignore_index=True,
    )
    result = compute_coverage_corrected_volcanism(grid, tiny_events(), min_observations=1)
    cell = result["cell_maps"]
    assert (cell["coverage_corrected_intensity"].fillna(0) > 0).sum() == 3
    assert pd.isna(cell[cell["cell_id"] == 4].iloc[0]["coverage_corrected_intensity"])
    assert result["data_quality"]["nonzero_coverage_corrected_cells"] == 3


def test_comparison_metrics_are_returned():
    from analysis.coverage_corrected_volcanism import compute_coverage_corrected_volcanism

    result = compute_coverage_corrected_volcanism(tiny_grid(), tiny_events())
    metrics = result["comparison_metrics"]
    assert {"spearman_correlation", "rank_overlap", "js_divergence", "latitude_band_contributions", "top_n_cumulative"}.issubset(metrics)
    assert not metrics["spearman_correlation"].empty
    assert not metrics["rank_overlap"].empty
    assert "metadata-based" in result["scientific_summary"]


def test_persistence_and_episodicity_scores():
    from analysis.coverage_corrected_volcanism import compute_coverage_corrected_volcanism

    result = compute_coverage_corrected_volcanism(tiny_grid(), tiny_events())
    persistence = result["persistence_summary"]
    cell3 = persistence[persistence["cell_id"] == 3].iloc[0]
    assert cell3["active_time_bins"] == 2
    assert cell3["activity_class"] == "repeated_active"


def test_single_event_cells_are_not_marked_persistent_active():
    from analysis.coverage_corrected_volcanism import compute_coverage_corrected_volcanism

    result = compute_coverage_corrected_volcanism(tiny_grid(), tiny_events())
    cell = result["cell_maps"]
    active = cell[cell["occurrence_event_count"] > 0]
    single_event = active[active["occurrence_event_count"] == 1]
    assert not single_event.empty
    assert "persistent_active" not in set(single_event["activity_class"])


def test_interpretation_and_power_concentration_outputs_are_returned():
    from analysis.coverage_corrected_volcanism import compute_coverage_corrected_volcanism

    result = compute_coverage_corrected_volcanism(tiny_grid(), tiny_events())
    metrics = result["comparison_metrics"]
    assert "metric_interpretation_summary" in metrics
    assert "power_concentration_summary" in metrics
    interpretation = metrics["metric_interpretation_summary"]
    assert {
        "metric_a",
        "metric_b",
        "spearman",
        "top10_jaccard",
        "js_divergence",
        "reviewer_interpretation",
    }.issubset(interpretation.columns)
    power = metrics["power_concentration_summary"]
    assert set(power["top_n"]) == {1, 5, 10, 25, 50}
    top10 = power.loc[power["top_n"].eq(10), "cumulative_fraction"].iloc[0]
    assert top10 == pytest.approx(1.0)


def test_dashboard_copy_uses_metadata_normalized_language():
    text = (Path(__file__).parents[1] / "dashboard" / "i18n.py").read_text(encoding="utf-8")
    assert "Metric-Dependent Io Volcanism On A 1 deg Grid" in text
    assert "metadata-normalized observation activity" in text
    assert "true footprint-corrected coverage" in text
    assert "Coverage-Corrected, Time-Resolved Activity" not in text
