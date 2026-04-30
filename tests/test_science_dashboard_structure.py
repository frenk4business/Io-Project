from __future__ import annotations


FINAL_RESEARCH_QUESTION = (
    "To what extent do different volcanic activity metrics, including named hotspot "
    "occurrence, estimated thermal intensity, and metadata-normalized observation "
    "activity, produce different spatial interpretations of Io's volcanism on a "
    "common 1 deg grid?"
)


def test_science_copy_uses_final_research_question():
    from dashboard.i18n import TRANSLATIONS

    body = TRANSLATIONS["en"]["page.science.research_question.body"]
    limitations = TRANSLATIONS["en"]["page.science.limitations.body"]

    assert FINAL_RESEARCH_QUESTION in body
    assert "USGS geology map" not in body
    assert "proxy covariates" not in body
    assert "metadata-normalized observation activity" in body
    assert "not a true footprint/sensitivity correction" in limitations


def test_science_result_loader_missing_file_falls_back():
    from dashboard.app import _load_result_csv

    df = _load_result_csv("definitely_missing_science_result.csv", ["a", "b"])

    assert df.empty
    assert list(df.columns) == ["a", "b"]


def test_science_result_loaders_have_expected_columns():
    from dashboard.app import (
        get_metric_interpretation_summary,
        get_power_concentration_summary,
    )

    metric_df = get_metric_interpretation_summary()
    power_df = get_power_concentration_summary()

    assert {"spearman", "top10_jaccard", "js_divergence"}.issubset(metric_df.columns)
    assert "cumulative_fraction" in power_df.columns


def test_explore_io_navigation_group_is_unchanged():
    from dashboard.app import NAV_GROUPS

    assert NAV_GROUPS["Explore Io"] == ["Io Experience", "2D Maps", "3D Globe"]


def test_time_bin_copy_is_metadata_framed():
    from dashboard.i18n import TRANSLATIONS

    assert TRANSLATIONS["en"]["page.time.filter.time_bin"] == "Metadata time/product bin"
    assert "not complete observing windows" in TRANSLATIONS["en"]["page.time.filter.time_bin_help"]
    assert "continuous monitoring" in TRANSLATIONS["en"]["page.time.filter.time_bin_note"]


def test_about_page_info_matches_metric_comparison_framing():
    from dashboard.i18n import TRANSLATIONS

    en_body = TRANSLATIONS["en"]["page.about.body"]
    nl_body = TRANSLATIONS["nl"]["page.about.body"]

    assert FINAL_RESEARCH_QUESTION in en_body
    assert "metadata-normalized observation activity" in en_body
    assert "not a full footprint- or sensitivity-corrected coverage model" in en_body
    assert "common 1 deg grid" in en_body
    assert "metadata-normalized observation activity" in nl_body
    assert "not a full footprint- or sensitivity-corrected coverage model" in nl_body

    for old_phrase in ["Logistic Regression", "AUC-ROC", "class_weight", "predict"]:
        assert old_phrase not in en_body
        assert old_phrase not in nl_body
