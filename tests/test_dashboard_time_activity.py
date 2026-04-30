from __future__ import annotations

import pandas as pd


def test_activity_scatter_handles_missing_power_count():
    from dashboard.app import _activity_scatter

    df = pd.DataFrame(
        {
            "lon_centre": [1.5],
            "lat_centre": [2.5],
            "occurrence_event_count": [1],
        }
    )
    fig = _activity_scatter(
        df,
        "occurrence_event_count",
        "Occurrence",
        "events",
        filter_col="occurrence_event_count",
    )

    assert len(fig.data) == 1
    assert list(fig.data[0].marker.size) == [6]


def test_activity_class_scatter_handles_missing_power_names():
    from dashboard.app import _activity_class_scatter

    df = pd.DataFrame(
        {
            "lon_centre": [1.5],
            "lat_centre": [2.5],
            "persistence_class": ["observed_active_single_bin"],
        }
    )
    fig = _activity_class_scatter(df, "Activity class")

    assert len(fig.data) == 1
    assert list(fig.data[0].text) == [""]
