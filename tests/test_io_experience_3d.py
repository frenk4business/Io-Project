"""
tests/test_io_experience_3d.py
------------------------------
Contract tests for the public-facing 3D Io Experience.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _tiny_feature_matrix() -> pd.DataFrame:
    rows = []
    cell_id = 0
    for lat in [-60.5, -59.5, 0.5, 60.5]:
        for lon in [-1.5, -0.5, 0.5, 1.5]:
            rows.append(
                {
                    "cell_id": cell_id,
                    "lon_centre": lon,
                    "lat_centre": lat,
                    "geology_encoded": cell_id % 4,
                    "geology_unit": "UNKNOWN" if cell_id % 5 == 0 else "Fb",
                    "tidal_heating_flux": 0.5,
                }
            )
            cell_id += 1
    return pd.DataFrame(rows)


def _tiny_power_grid(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    pg = feature_matrix[["cell_id", "lon_centre", "lat_centre"]].copy()
    pg["power_count"] = 0
    pg["primary_power_gw"] = 0.0
    pg["mean_power_gw"] = 0.0
    pg["sum_power_gw"] = 0.0
    pg["log_primary_power"] = 0.0
    pg["power_names"] = ""
    powers = [1.0, 5.0, 10.0, 100.0, 300.0]
    for i, p in enumerate(powers):
        pg.loc[i, "power_count"] = 1
        pg.loc[i, "primary_power_gw"] = p
        pg.loc[i, "mean_power_gw"] = p
        pg.loc[i, "sum_power_gw"] = p
        pg.loc[i, "log_primary_power"] = np.log1p(p)
        pg.loc[i, "power_names"] = f"hotspot_{i}"
    return pg


def test_io_experience_builds_nonblank_figure():
    from visualization.io_experience_3d import build_io_experience_3d

    fm = _tiny_feature_matrix()
    pg = _tiny_power_grid(fm)
    fig, insights = build_io_experience_3d(fm, pg)

    assert len(fig.data) >= 2
    assert insights["visible_hotspots"] == 5
    assert "estimated thermal-emission proxy" in insights["power_definition"]


def test_natural_io_surface_is_finite_and_normalized():
    from visualization.io_experience_3d import _natural_io_surface

    surface = _natural_io_surface(_tiny_feature_matrix())
    assert np.isfinite(surface).all()
    assert surface.min() >= 0.0
    assert surface.max() <= 1.0
    assert surface.max() > surface.min()


def test_io_experience_defaults_to_natural_io_surface():
    from visualization.io_experience_3d import build_io_experience_3d

    fm = _tiny_feature_matrix()
    pg = _tiny_power_grid(fm)
    fig, _ = build_io_experience_3d(fm, pg)

    assert fig.data[0].name == "Io surface"
    assert fig.data[0].cmin == 0.0
    assert fig.data[0].cmax == 1.0


def test_io_experience_unknown_surface_falls_back_to_natural_io():
    from visualization.io_experience_3d import build_io_experience_3d

    fm = _tiny_feature_matrix()
    pg = _tiny_power_grid(fm)
    fig, _ = build_io_experience_3d(fm, pg, surface_mode="old_cached_key")

    assert fig.data[0].cmin == 0.0
    assert fig.data[0].cmax == 1.0


def test_io_experience_default_marker_sizes_are_bounded():
    from visualization.io_experience_3d import _marker_sizes

    sizes = _marker_sizes(pd.Series([1.0, 5.0, 10.0, 100.0, 300.0]))
    assert sizes.min() >= 3.0
    assert sizes.max() <= 14.5


def test_io_experience_top_10_limits_visible_points():
    from visualization.io_experience_3d import build_io_experience_3d

    fm = _tiny_feature_matrix()
    pg = _tiny_power_grid(fm)
    fig, insights = build_io_experience_3d(
        fm,
        pg,
        scene="The giants dominate",
        show_only_top_10=True,
    )

    assert insights["visible_hotspots"] <= 10
    assert any("Power towers" == trace.name for trace in fig.data)


def test_io_experience_min_power_filters_hotspots():
    from visualization.io_experience_3d import build_io_experience_3d

    fm = _tiny_feature_matrix()
    pg = _tiny_power_grid(fm)
    _, insights = build_io_experience_3d(fm, pg, min_power_gw=50.0)

    assert insights["visible_hotspots"] == 2
    assert insights["visible_power_gw"] == 400.0


def test_io_experience_polar_scene_adds_rings():
    from visualization.io_experience_3d import build_io_experience_3d

    fm = _tiny_feature_matrix()
    pg = _tiny_power_grid(fm)
    fig, _ = build_io_experience_3d(fm, pg, scene="Poles vs equator")

    names = {trace.name for trace in fig.data}
    assert "North polar cap" in names
    assert "South polar cap" in names
