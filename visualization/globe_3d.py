"""
visualization/globe_3d.py
-------------------------
Build an interactive 3D Plotly globe of Io with hotspot markers
and selectable surface coloring layers.

Functions
---------
build_io_globe_3d : Returns a Plotly Figure — interactive 3D sphere of Io.

Architecture note
-----------------
This module does zero disk I/O. All data arrives as parameters.
Caller is responsible for loading catalog and feature_matrix.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------
GLOBE_MARKER_RADIUS: float = 1.02
GLOBE_MARKER_SIZE: int = 5
GLOBE_MARKER_SIZE_ON_BASEMAP: int = 6
GLOBE_MARKER_COLOR: str = "orange"
GLOBE_MARKER_COLOR_ON_BASEMAP: str = "#ff3311"
GLOBE_MARKER_EDGE_COLOR: str = "white"
GLOBE_MARKER_EDGE_WIDTH: float = 0.8
GLOBE_BG_COLOR: str = "black"
GLOBE_FIGURE_HEIGHT: int = 700
GLOBE_COLORSCALE_TIDAL: str = "YlOrRd"
GLOBE_COLORSCALE_PROBABILITY: str = "RdPu"
GLOBE_COLORSCALE_GEOLOGY: str = "Viridis"

# Color rendered for geological cells outside the active unit filter.
# Warm light gray — recedes visually while keeping the 3D globe shape readable.
IO_DIMMED_COLOR: str = "#c8c4be"

ColorBy = Literal["io_appearance", "tidal_heating_flux", "geology_encoded", "model_probability"]

# ---------------------------------------------------------------------------
# Io geology legend — single source of truth for unit codes, descriptions,
# and their Io-realistic surface colors.
#
# Order matches geology_encoded values 0–15 from the USGS SIM3168 map.
# Each entry: (unit_code, description, hex_color)
# ---------------------------------------------------------------------------
IO_GEOLOGY_LEGEND: list[tuple[str, str, str]] = [
    ("Fb",      "Basaltic flow (fresh lava)",         "#1a1a18"),
    ("Fd",      "Dark flow (older lava)",              "#2a2218"),
    ("Fu",      "Undivided flow",                      "#7a4820"),
    ("Ml",      "Mountain, layered",                   "#7a7268"),
    ("Mm",      "Mountain, massive",                   "#4a4038"),
    ("Mu",      "Mountain, undivided",                 "#3e3c38"),
    ("NoData",  "No data",                             "#988e84"),
    ("PFb",     "Bright plains + flow",                "#f0c050"),
    ("PFd",     "Dark plains + flow",                  "#b86020"),
    ("PFu",     "Undivided plains + flow",             "#d07830"),
    ("Pbw",     "Bright white plains (SO\u2082 frost)", "#faf2da"),
    ("Pby",     "Bright yellow plains (sulfur)",       "#f4df28"),
    ("Pl",      "Light plains",                        "#f0c860"),
    ("Prb",     "Red-brown plains",                    "#c43010"),
    ("Tb",      "Bright tholus",                       "#f0b040"),
    ("UNKNOWN", "Unclassified",                        "#a09888"),
]

# Derived lists — keep in sync with IO_GEOLOGY_LEGEND above.
IO_GEOLOGY_COLORS: list[str] = [color for _, _, color in IO_GEOLOGY_LEGEND]
IO_GEOLOGY_CODES: list[str]  = [code  for code, _, _  in IO_GEOLOGY_LEGEND]

_COLOR_META: dict[str, dict] = {
    "io_appearance": {
        "colorscale": None,      # built dynamically
        "colorbar_title": None,
        "showscale": False,
    },
    "tidal_heating_flux": {
        "colorscale": GLOBE_COLORSCALE_TIDAL,
        "colorbar_title": "Tidal Heating Flux",
        "showscale": True,
    },
    "geology_encoded": {
        "colorscale": GLOBE_COLORSCALE_GEOLOGY,
        "colorbar_title": "Geology Unit",
        "showscale": True,
    },
    "model_probability": {
        "colorscale": GLOBE_COLORSCALE_PROBABILITY,
        "colorbar_title": "P(hotspot)",
        "showscale": True,
    },
}


def _build_io_colorscale(extra_dimmed: bool = False) -> list[list]:
    """Build a smooth colorscale for io_appearance.

    Args:
        extra_dimmed: If True, appends IO_DIMMED_COLOR as index 16 so that
                      cells replaced with value 16 render as the dim color.
    """
    colors = IO_GEOLOGY_COLORS + ([IO_DIMMED_COLOR] if extra_dimmed else [])
    n = len(colors)
    return [[i / (n - 1), color] for i, color in enumerate(colors)]


def build_io_globe_3d(
    catalog: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    probabilities: np.ndarray | None = None,
    color_by: ColorBy = "io_appearance",
    unit_filter: frozenset[int] | None = None,
) -> go.Figure:
    """Build an interactive 3D Plotly globe of Io with hotspot markers.

    Args:
        catalog: Hotspot catalog with columns 'name', 'longitude', 'latitude'.
                 172 records from USGS SIM3168.
        feature_matrix: 1°×1° grid DataFrame (64,800 rows) with columns
                        lon_centre, lat_centre, tidal_heating_flux,
                        geology_encoded, has_hotspot.
        probabilities: Optional array of predicted P(hotspot) per grid cell,
                       same row order as feature_matrix. Required when
                       color_by='model_probability'.
        color_by: Which dataset to use as the sphere surface color.
                  'io_appearance' renders Io's recognizable sulfurous look.
        unit_filter: When color_by='io_appearance', only these geology_encoded
                     indices are shown in their original color; all others are
                     rendered as IO_DIMMED_COLOR. None or empty set = show all.

    Returns:
        Plotly Figure with go.Surface sphere and go.Scatter3d hotspot markers.
    """
    # ------------------------------------------------------------------
    # Resolve color column and effective mode
    # ------------------------------------------------------------------
    effective_color_by = color_by

    if color_by == "model_probability":
        if probabilities is None:
            logger.warning(
                "color_by='model_probability' requested but probabilities=None. "
                "Falling back to tidal_heating_flux."
            )
            effective_color_by = "tidal_heating_flux"
            color_column = "tidal_heating_flux"
        else:
            color_column = "_prob"
            feature_matrix = feature_matrix.copy()
            feature_matrix["_prob"] = probabilities
    elif color_by == "io_appearance":
        color_column = "geology_encoded"
    else:
        color_column = effective_color_by

    meta = _COLOR_META[effective_color_by]

    # ------------------------------------------------------------------
    # Build sphere coordinate meshgrid from grid cell centres
    # ------------------------------------------------------------------
    lon_centres = np.sort(feature_matrix["lon_centre"].unique())   # (360,)
    lat_centres = np.sort(feature_matrix["lat_centre"].unique())   # (180,)

    LON, LAT = np.meshgrid(np.deg2rad(lon_centres), np.deg2rad(lat_centres))

    X = np.cos(LAT) * np.cos(LON)   # (180, 360)
    Y = np.cos(LAT) * np.sin(LON)   # (180, 360)
    Z = np.sin(LAT)                  # (180, 360)

    # ------------------------------------------------------------------
    # Build surface color array (180×360) via pivot
    # ------------------------------------------------------------------
    color_pivot = feature_matrix.pivot(
        index="lat_centre",
        columns="lon_centre",
        values=color_column,
    )
    color_array = color_pivot.values   # shape (180, 360)

    # ------------------------------------------------------------------
    # Surface trace
    # ------------------------------------------------------------------
    if effective_color_by == "io_appearance":
        applying_filter = bool(unit_filter)
        if applying_filter:
            # Replace cells outside the filter with index 16 (IO_DIMMED_COLOR).
            # cmin/cmax are set on the Surface trace so Plotly does NOT re-normalize
            # based on the subset of values present — each index stays at its correct
            # color position regardless of which units are currently selected.
            filtered_array = color_array.copy().astype(float)
            mask_out = ~np.isin(color_array, list(unit_filter))
            filtered_array[mask_out] = 16
            surface_color_array = filtered_array
            colorscale = _build_io_colorscale(extra_dimmed=True)
        else:
            surface_color_array = color_array
            colorscale = _build_io_colorscale(extra_dimmed=False)

        # cmin/cmax pin the colorscale to the full 0–15 (or 0–16) range so
        # that each geology index always maps to its correct color regardless
        # of which values are actually present in the filtered array.
        cmax_val = 16 if applying_filter else 15
        surface_trace = go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=surface_color_array,
            colorscale=colorscale,
            cmin=0,
            cmax=cmax_val,
            showscale=False,
            hoverinfo="skip",
            name="Io surface",
            lighting=dict(
                ambient=0.80,
                diffuse=0.60,
                roughness=0.55,
                specular=0.06,
                fresnel=0.08,
            ),
            lightposition=dict(x=200, y=200, z=500),
        )
    else:
        surface_trace = go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=color_array,
            colorscale=meta["colorscale"],
            showscale=meta["showscale"],
            colorbar=dict(
                title=dict(
                    text=meta["colorbar_title"],
                    side="right",
                    font=dict(color="white"),
                ),
                thickness=14,
                len=0.55,
                x=1.0,
                bgcolor="rgba(0,0,0,0.4)",
                tickfont=dict(color="white"),
            ),
            hoverinfo="skip",
            name="Io surface",
            lighting=dict(
                ambient=0.72,
                diffuse=0.55,
                roughness=0.5,
                specular=0.08,
                fresnel=0.1,
            ),
            lightposition=dict(x=200, y=200, z=500),
        )

    # ------------------------------------------------------------------
    # Known hotspot marker trace
    # ------------------------------------------------------------------
    lat_h = np.deg2rad(catalog["latitude"].values)
    lon_h = np.deg2rad(catalog["longitude"].values)
    r = GLOBE_MARKER_RADIUS

    hx = r * np.cos(lat_h) * np.cos(lon_h)
    hy = r * np.cos(lat_h) * np.sin(lon_h)
    hz = r * np.sin(lat_h)

    hover_texts = [
        (
            f"<b>{row['name']}</b><br>"
            f"Lon: {row['longitude']:.1f}°  Lat: {row['latitude']:.1f}°<br>"
            f"Source: USGS SIM3168"
        )
        for _, row in catalog.iterrows()
    ]

    known_color = (
        GLOBE_MARKER_COLOR_ON_BASEMAP
        if effective_color_by == "io_appearance"
        else GLOBE_MARKER_COLOR
    )
    known_size = (
        GLOBE_MARKER_SIZE_ON_BASEMAP
        if effective_color_by == "io_appearance"
        else GLOBE_MARKER_SIZE
    )

    markers_trace = go.Scatter3d(
        x=hx,
        y=hy,
        z=hz,
        mode="markers",
        marker=dict(
            size=known_size,
            color=known_color,
            line=dict(color=GLOBE_MARKER_EDGE_COLOR, width=GLOBE_MARKER_EDGE_WIDTH),
            opacity=0.95,
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        name=f"Known hotspots (n={len(catalog)})",
    )

    # ------------------------------------------------------------------
    # Layout — dark space background, all axis chrome hidden
    # ------------------------------------------------------------------
    layout = go.Layout(
        height=GLOBE_FIGURE_HEIGHT,
        paper_bgcolor=GLOBE_BG_COLOR,
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(color="white"),
        ),
        scene=dict(
            bgcolor=GLOBE_BG_COLOR,
            xaxis=dict(
                showbackground=False,
                showgrid=False,
                showticklabels=False,
                title="",
                zeroline=False,
            ),
            yaxis=dict(
                showbackground=False,
                showgrid=False,
                showticklabels=False,
                title="",
                zeroline=False,
            ),
            zaxis=dict(
                showbackground=False,
                showgrid=False,
                showticklabels=False,
                title="",
                zeroline=False,
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.4, y=1.0, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
        ),
    )

    fig = go.Figure(data=[surface_trace, markers_trace], layout=layout)

    logger.info(
        "Built 3D globe: %d known volcano markers, surface='%s', unit_filter=%s.",
        len(catalog),
        effective_color_by,
        sorted(unit_filter) if unit_filter else "none",
    )
    return fig
