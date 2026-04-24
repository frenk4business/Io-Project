"""
visualization/io_experience_3d.py
---------------------------------
Public-facing 3D Io experience.

This module builds a museum-style Plotly globe: recognizable Io surface,
thermal hotspots as glowing markers, optional power towers, polar rings, and
coverage-uncertainty dimming. All power values are estimated thermal-emission
proxies derived from Davies/JIRAM 4.8 micron spectral radiance.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dashboard.i18n import translate as t

from visualization.globe_3d import (
    GLOBE_BG_COLOR,
    IO_DIMMED_COLOR,
    IO_GEOLOGY_COLORS,
)

ExperienceSurface = Literal[
    "natural_io",
    "thermal_intensity",
    "geology",
    "model_probability",
]

SCENE_COPY: dict[str, str] = {
    "Meet Io": "Io is a small moon, but it is alive with volcanoes. Each glowing point is a detected hot spot.",
    "Not all volcanoes are equal": "Some volcanoes whisper. A few dominate Io's estimated thermal-emission proxy.",
    "The giants dominate": "The brightest ten sources carry a huge share of the heat signal. Big eruptions can shape the whole story.",
    "Poles vs equator": "The blue rings mark polar caps. This view asks whether Io's heat is mostly polar or closer to the equator.",
    "What we still do not know": "Darkened regions may mean missing observations, not a quiet surface. No detection is not the same as no volcano.",
}

SCENE_ORDER = list(SCENE_COPY.keys())


def _scene_copy(scene: str, language: str) -> str:
    keys = {
        "Meet Io": "scene.meet_io",
        "Not all volcanoes are equal": "scene.not_equal",
        "The giants dominate": "scene.giants",
        "Poles vs equator": "scene.poles",
        "What we still do not know": "scene.unknown",
    }
    return t(keys.get(scene, "scene.meet_io"), language)

_SURFACE_LABELS = {
    "natural_io": "Natural Io",
    "thermal_intensity": "Thermal intensity",
    "geology": "Geology",
    "model_probability": "Model probability",
}

NATURAL_IO_COLORSCALE: list[list] = [
    [0.00, "#2b2114"],
    [0.20, "#6f4a1f"],
    [0.45, "#b9822a"],
    [0.70, "#d8b64a"],
    [1.00, "#f2e6b0"],
]


def _io_colorscale(extra_dimmed: bool = False) -> list[list]:
    colors = IO_GEOLOGY_COLORS + ([IO_DIMMED_COLOR] if extra_dimmed else [])
    n = len(colors)
    return [[i / (n - 1), color] for i, color in enumerate(colors)]


def _sphere_grid(feature_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_centres = np.sort(feature_matrix["lon_centre"].unique())
    lat_centres = np.sort(feature_matrix["lat_centre"].unique())
    lon, lat = np.meshgrid(np.deg2rad(lon_centres), np.deg2rad(lat_centres))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def _natural_io_surface(feature_matrix: pd.DataFrame) -> np.ndarray:
    """Return a calm Google/NASA-inspired Io color field in [0, 1].

    The field is procedural and deliberately smooth: broad sulfur/ochre bands,
    soft darker volcanic regions, and pale frost-like patches. It avoids the
    hard cell boundaries of the geology classification layer.
    """
    lon = np.deg2rad(feature_matrix["lon_centre"].astype(float).to_numpy())
    lat = np.deg2rad(feature_matrix["lat_centre"].astype(float).to_numpy())

    base = (
        0.58
        + 0.16 * np.sin(1.7 * lon + 0.6 * np.cos(lat))
        + 0.10 * np.cos(2.4 * lat - 0.3 * np.sin(lon))
        + 0.06 * np.sin(3.0 * lon + 2.2 * lat)
    )

    # Soft rust/dark regions, inspired by Io's patchy volcanic deposits.
    dark_a = np.exp(-(((np.rad2deg(lon) - 25.0) / 34.0) ** 2 + ((np.rad2deg(lat) + 18.0) / 22.0) ** 2))
    dark_b = np.exp(-(((np.rad2deg(lon) + 118.0) / 40.0) ** 2 + ((np.rad2deg(lat) - 10.0) / 26.0) ** 2))
    frost_n = np.exp(-(((np.rad2deg(lat) - 54.0) / 24.0) ** 2)) * (0.65 + 0.35 * np.cos(1.8 * lon))
    frost_s = np.exp(-(((np.rad2deg(lat) + 58.0) / 25.0) ** 2)) * (0.65 + 0.35 * np.sin(1.6 * lon))

    values = base - 0.24 * dark_a - 0.20 * dark_b + 0.14 * frost_n + 0.10 * frost_s
    values = (values - values.min()) / max(float(values.max() - values.min()), 1e-9)
    return np.clip(values, 0.0, 1.0)


def _xyz_from_lonlat(
    lon_deg: np.ndarray | pd.Series,
    lat_deg: np.ndarray | pd.Series,
    radius: np.ndarray | float = 1.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    r = np.asarray(radius, dtype=float)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def _surface_values(
    feature_matrix: pd.DataFrame,
    power_grid: pd.DataFrame | None,
    probabilities: np.ndarray | None,
    surface_mode: ExperienceSurface,
    show_coverage_uncertainty: bool,
) -> tuple[np.ndarray, str | list[list], bool, float | None, float | None, str]:
    fm = feature_matrix.copy()
    surface_mode = surface_mode if surface_mode in _SURFACE_LABELS else "natural_io"
    colorbar_title = _SURFACE_LABELS.get(surface_mode, "Natural Io")

    if surface_mode == "model_probability" and probabilities is not None:
        fm["_experience_surface"] = probabilities
        colorscale: str | list[list] = "RdPu"
        showscale = True
        cmin, cmax = 0.0, 1.0
        colorbar_title = "P(hotspot)"
    elif surface_mode == "thermal_intensity" and power_grid is not None:
        pg = power_grid[["cell_id", "log_primary_power"]].copy()
        fm = fm.merge(pg, on="cell_id", how="left")
        fm["_experience_surface"] = fm["log_primary_power"].fillna(0.0)
        colorscale = [
            [0.00, "#21180b"],
            [0.25, "#5a240c"],
            [0.50, "#c14a13"],
            [0.75, "#ffb000"],
            [1.00, "#fff6bf"],
        ]
        showscale = True
        cmin, cmax = 0.0, None
        colorbar_title = "log estimated proxy"
    elif surface_mode == "geology":
        fm["_experience_surface"] = fm["geology_encoded"]
        colorscale = "Viridis"
        showscale = True
        cmin, cmax = None, None
        colorbar_title = "Geology"
    else:
        if surface_mode == "natural_io":
            values = _natural_io_surface(fm)
            if show_coverage_uncertainty and "geology_unit" in fm.columns:
                no_data = fm["geology_unit"].astype(str).str.upper().isin(["UNKNOWN", "NODATA"])
                values = values.copy()
                values[no_data.to_numpy()] *= 0.34
            fm["_experience_surface"] = values
            pivot = fm.pivot(index="lat_centre", columns="lon_centre", values="_experience_surface")
            return pivot.values, NATURAL_IO_COLORSCALE, False, 0.0, 1.0, "Natural Io"

        values = fm["geology_encoded"].astype(float).to_numpy()
        if show_coverage_uncertainty and "geology_unit" in fm.columns:
            no_data = fm["geology_unit"].astype(str).str.upper().isin(["UNKNOWN", "NODATA"])
            values = values.copy()
            values[no_data.to_numpy()] = 16.0
            colorscale = _io_colorscale(extra_dimmed=True)
            cmin, cmax = 0.0, 16.0
        else:
            colorscale = _io_colorscale(extra_dimmed=False)
            cmin, cmax = 0.0, 15.0
        fm["_experience_surface"] = values
        showscale = False

    pivot = fm.pivot(index="lat_centre", columns="lon_centre", values="_experience_surface")
    return pivot.values, colorscale, showscale, cmin, cmax, colorbar_title


def _power_points(power_grid: pd.DataFrame, min_power_gw: float, top_n: int | None) -> pd.DataFrame:
    obs = power_grid[power_grid["power_count"] > 0].copy()
    obs = obs[obs["primary_power_gw"] >= min_power_gw]
    obs = obs.sort_values("primary_power_gw", ascending=False)
    if top_n is not None:
        obs = obs.head(top_n)
    return obs.reset_index(drop=True)


def _marker_sizes(power: pd.Series) -> np.ndarray:
    vals = np.log1p(power.astype(float).to_numpy())
    if vals.size == 0:
        return vals
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.full(vals.shape, 5.5)
    return 3.2 + 10.8 * (vals - lo) / (hi - lo)


def _power_colors(power: pd.Series) -> np.ndarray:
    return np.log1p(power.astype(float).to_numpy())


def _make_power_tower_trace(points: pd.DataFrame) -> go.Scatter3d | None:
    if points.empty:
        return None
    p = points["primary_power_gw"].astype(float)
    heights = 0.08 + 0.35 * np.log1p(p) / max(float(np.log1p(p.max())), 1.0)

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for row, height in zip(points.itertuples(index=False), heights):
        x0, y0, z0 = _xyz_from_lonlat([row.lon_centre], [row.lat_centre], 1.02)
        x1, y1, z1 = _xyz_from_lonlat([row.lon_centre], [row.lat_centre], 1.02 + height)
        xs.extend([float(x0[0]), float(x1[0]), None])
        ys.extend([float(y0[0]), float(y1[0]), None])
        zs.extend([float(z0[0]), float(z1[0]), None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="#ffb000", width=5),
        opacity=0.74,
        hoverinfo="skip",
        name="Power towers",
    )


def _polar_ring_traces(threshold: float = 60.0) -> list[go.Scatter3d]:
    traces: list[go.Scatter3d] = []
    lons = np.linspace(-180, 180, 361)
    for lat, name in [(threshold, "North polar cap"), (-threshold, "South polar cap")]:
        x, y, z = _xyz_from_lonlat(lons, np.full_like(lons, lat), 1.055)
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color="#38c9ff", width=6),
                opacity=0.82,
                hoverinfo="skip",
                name=name,
            )
        )
    return traces


def build_io_experience_3d(
    feature_matrix: pd.DataFrame,
    power_grid: pd.DataFrame,
    probabilities: np.ndarray | None = None,
    scene: str = "Meet Io",
    surface_mode: ExperienceSurface = "natural_io",
    show_heat_glow: bool = True,
    show_power_towers: bool = False,
    show_only_top_10: bool = False,
    show_polar_bands: bool = False,
    show_coverage_uncertainty: bool = False,
    min_power_gw: float = 0.0,
    language: str = "en",
) -> tuple[go.Figure, dict]:
    """Build the public Io Experience figure and plain-English insight data."""
    scene = scene if scene in SCENE_COPY else "Meet Io"
    if scene == "Not all volcanoes are equal":
        show_heat_glow = True
    elif scene == "The giants dominate":
        show_heat_glow = True
        show_only_top_10 = True
        show_power_towers = True
    elif scene == "Poles vs equator":
        show_heat_glow = True
        show_polar_bands = True
        surface_mode = "natural_io"
    elif scene == "What we still do not know":
        show_coverage_uncertainty = True
        show_heat_glow = True

    x, y, z = _sphere_grid(feature_matrix)
    values, colorscale, showscale, cmin, cmax, colorbar_title = _surface_values(
        feature_matrix=feature_matrix,
        power_grid=power_grid,
        probabilities=probabilities,
        surface_mode=surface_mode,
        show_coverage_uncertainty=show_coverage_uncertainty,
    )

    surface = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=values,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        showscale=showscale,
        colorbar=dict(
            title=dict(text=colorbar_title, side="right", font=dict(color="white")),
            tickfont=dict(color="white"),
            bgcolor="rgba(0,0,0,0.35)",
            thickness=14,
            len=0.56,
        ),
        hoverinfo="skip",
        name="Io surface",
        lighting=dict(ambient=0.78, diffuse=0.58, roughness=0.54, specular=0.08),
        lightposition=dict(x=240, y=180, z=420),
    )

    points = _power_points(
        power_grid=power_grid,
        min_power_gw=min_power_gw,
        top_n=10 if show_only_top_10 else None,
    )
    px, py, pz = _xyz_from_lonlat(points["lon_centre"], points["lat_centre"], 1.055)
    marker_sizes = _marker_sizes(points["primary_power_gw"])
    marker_colors = _power_colors(points["primary_power_gw"])

    hover = [
        (
            f"<b>{row.power_names or t('viewer.hotspot.default_name', language)}</b><br>"
            f"{t('viewer.hover.lon', language)}: {row.lon_centre:.1f} deg  {t('viewer.hover.lat', language)}: {row.lat_centre:.1f} deg<br>"
            f"{t('viewer.hover.proxy', language)}: {row.primary_power_gw:,.1f} GW<br>"
            f"{t('viewer.hover.derived', language)}"
        )
        for row in points.itertuples(index=False)
    ]

    marker_trace = go.Scatter3d(
        x=px,
        y=py,
        z=pz,
        mode="markers",
        marker=dict(
            size=marker_sizes if show_heat_glow else np.full(len(points), 5.5),
            color=marker_colors if show_heat_glow else "#ff3311",
            colorscale=[
                [0.0, "#ff3b1f"],
                [0.5, "#ffb000"],
                [1.0, "#fff6bf"],
            ],
            opacity=0.88,
            line=dict(color="rgba(255,255,255,0.72)", width=0.55),
            showscale=False,
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name=f"{t('viewer.visible_heat_sources', language)} (n={len(points)})",
    )

    traces: list = [surface]
    if show_heat_glow and not points.empty:
        gx, gy, gz = _xyz_from_lonlat(points["lon_centre"], points["lat_centre"], 1.052)
        traces.append(
            go.Scatter3d(
                x=gx,
                y=gy,
                z=gz,
                mode="markers",
                marker=dict(
                    size=np.clip(marker_sizes * 1.45, 6, 24),
                    color="#ff7a00",
                    opacity=0.12,
                    line=dict(width=0),
                ),
                hoverinfo="skip",
                name=t("viewer.heat_glow", language),
            )
        )
    traces.append(marker_trace)

    tower_trace = _make_power_tower_trace(points) if show_power_towers else None
    if tower_trace is not None:
        traces.append(tower_trace)
    if show_polar_bands:
        traces.extend(_polar_ring_traces(60.0))

    polar = points[points["lat_centre"].abs() >= 60.0]
    total_power = float(points["sum_power_gw"].sum()) if not points.empty else 0.0
    polar_power = float(polar["sum_power_gw"].sum()) if not polar.empty else 0.0
    strongest = points.iloc[0] if not points.empty else None
    insights = {
        "scene_copy": _scene_copy(scene, language),
        "visible_hotspots": int(len(points)),
        "visible_power_gw": total_power,
        "strongest_name": str(strongest["power_names"] or t("viewer.hotspot.default_name", language)) if strongest is not None else t("viewer.none", language),
        "strongest_power_gw": float(strongest["primary_power_gw"]) if strongest is not None else 0.0,
        "polar_fraction": polar_power / total_power if total_power > 0 else 0.0,
        "power_definition": t("viewer.power_definition", language),
    }

    cameras = {
        "Meet Io": dict(eye=dict(x=1.45, y=1.10, z=0.58), up=dict(x=0, y=0, z=1)),
        "Not all volcanoes are equal": dict(eye=dict(x=1.2, y=1.45, z=0.72), up=dict(x=0, y=0, z=1)),
        "The giants dominate": dict(eye=dict(x=0.9, y=1.55, z=0.95), up=dict(x=0, y=0, z=1)),
        "Poles vs equator": dict(eye=dict(x=1.1, y=0.8, z=1.7), up=dict(x=0, y=0, z=1)),
        "What we still do not know": dict(eye=dict(x=1.55, y=0.7, z=0.75), up=dict(x=0, y=0, z=1)),
    }

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=760,
        paper_bgcolor=GLOBE_BG_COLOR,
        plot_bgcolor=GLOBE_BG_COLOR,
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(0,0,0,0.42)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
        ),
        scene=dict(
            bgcolor=GLOBE_BG_COLOR,
            xaxis=dict(showbackground=False, showgrid=False, showticklabels=False, title="", zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, showticklabels=False, title="", zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, showticklabels=False, title="", zeroline=False),
            aspectmode="data",
            camera=cameras[scene],
        ),
    )
    return fig, insights


