"""
visualization/hotspot_map.py
----------------------------
Generate publication-quality map figures for Io hotspot analysis.

Functions
---------
plot_hotspot_catalog       : Plot known hotspot locations on Io
plot_kde_heatmap           : KDE density heatmap (labelled as interpolated)
plot_prediction_surface    : Model probability surface on 1°×1° grid
plot_feature_map           : Visualise any feature column on the grid
plot_cv_fold_layout        : Show spatial CV fold boundaries

Notes on visualization honesty
-------------------------------
KDE heatmaps are rendered at higher resolution than the modeling grid.
Every KDE output MUST include the label "Kernel density estimate — spatially
interpolated. Not a model prediction." This is non-negotiable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from config import IO_RADIUS_KM, SPATIAL_CV_FOLDS

logger = logging.getLogger(__name__)

# Visualization constants
FIGURE_DPI: int = 150
CMAP_HOTSPOT: str = "YlOrRd"
CMAP_PROBABILITY: str = "RdPu"
CMAP_FEATURE: str = "viridis"
IO_BACKGROUND_COLOR: str = "#1a1a2e"

KDE_INTERPOLATION_DISCLAIMER = (
    "Kernel density estimate — spatially interpolated.\n"
    "Not a model prediction. Based on known hotspot catalog only."
)


def _setup_io_axes(ax: plt.Axes, title: str = "") -> None:
    """Configure axes for an Io global map.

    Args:
        ax: Matplotlib Axes object.
        title: Plot title.
    """
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_facecolor(IO_BACKGROUND_COLOR)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


def plot_hotspot_catalog(
    catalog: pd.DataFrame,
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot known hotspot locations on a simple lon/lat map.

    Args:
        catalog: Hotspot catalog DataFrame with longitude, latitude columns.
        ax: Optional existing Axes to plot on.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)

    _setup_io_axes(ax, title="Known Volcanic Hotspots on Io (USGS Catalog)")

    scatter = ax.scatter(
        catalog["longitude"],
        catalog["latitude"],
        c="orange",
        s=20,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.3,
        zorder=5,
        label=f"Hotspots (n={len(catalog)})",
    )

    ax.legend(loc="lower left", fontsize=9)
    ax.text(
        0.02, 0.02,
        "Source: USGS Io Volcanic Hotspot Catalog",
        transform=ax.transAxes,
        fontsize=7,
        color="white",
        alpha=0.7,
    )

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("Saved hotspot catalog map to %s.", save_path)

    return fig or ax.figure


def plot_kde_heatmap(
    catalog: pd.DataFrame,
    resolution: float = 0.25,
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot a KDE density heatmap of hotspot locations.

    Renders at higher resolution than the modeling grid.
    ALWAYS labelled as interpolated — not a model prediction.

    Args:
        catalog: Hotspot catalog DataFrame with longitude, latitude.
        resolution: KDE evaluation grid resolution in degrees.
        ax: Optional existing Axes.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)

    _setup_io_axes(ax, title="Hotspot Density — Io (KDE)")

    # Build KDE
    lons = catalog["longitude"].values
    lats = catalog["latitude"].values
    kde = gaussian_kde(np.vstack([lons, lats]), bw_method="scott")

    # Evaluation grid
    lon_range = np.arange(-180, 180 + resolution, resolution)
    lat_range = np.arange(-90, 90 + resolution, resolution)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    positions = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
    density = kde(positions).reshape(lon_grid.shape)

    # Plot
    im = ax.pcolormesh(
        lon_grid, lat_grid, density,
        cmap=CMAP_HOTSPOT,
        alpha=0.85,
        shading="auto",
    )
    plt.colorbar(im, ax=ax, label="KDE Density", shrink=0.6)

    # REQUIRED disclaimer
    ax.text(
        0.02, 0.98,
        KDE_INTERPOLATION_DISCLAIMER,
        transform=ax.transAxes,
        fontsize=7,
        color="white",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("Saved KDE heatmap to %s.", save_path)

    return fig or ax.figure


def plot_prediction_surface(
    feature_matrix: pd.DataFrame,
    probabilities: np.ndarray,
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the model's predicted probability surface on the 1°×1° grid.

    Args:
        feature_matrix: Grid DataFrame with lon_centre, lat_centre columns.
        probabilities: Predicted probabilities for the positive class.
        ax: Optional existing Axes.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)

    _setup_io_axes(ax, title="Predicted Hotspot Probability — Logistic Regression (1°×1° grid)")

    sc = ax.scatter(
        feature_matrix["lon_centre"],
        feature_matrix["lat_centre"],
        c=probabilities,
        cmap=CMAP_PROBABILITY,
        s=2,
        alpha=0.9,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="P(hotspot)", shrink=0.6)

    ax.text(
        0.02, 0.02,
        "Grid resolution: 1°×1°. Logistic regression with spatial CV.",
        transform=ax.transAxes,
        fontsize=7,
        color="white",
        alpha=0.7,
    )

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("Saved prediction surface to %s.", save_path)

    return fig or ax.figure


def plot_cv_fold_layout(
    feature_matrix: pd.DataFrame,
    ax: plt.Axes | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Visualise the spatial CV fold boundaries.

    Args:
        feature_matrix: Grid DataFrame with lat_centre column.
        ax: Optional existing Axes.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI)

    _setup_io_axes(ax, title="Spatial CV Fold Layout (4 latitude bands)")

    colors = plt.cm.Set2(np.linspace(0, 1, len(SPATIAL_CV_FOLDS)))
    for fold_idx, ((lat_min, lat_max), color) in enumerate(zip(SPATIAL_CV_FOLDS, colors)):
        ax.axhspan(
            lat_min, lat_max,
            alpha=0.3,
            color=color,
            label=f"Fold {fold_idx}: {lat_min:+.0f}° to {lat_max:+.0f}°",
        )
        ax.axhline(lat_min, color="white", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.legend(loc="lower left", fontsize=9)

    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("Saved CV fold layout to %s.", save_path)

    return fig or ax.figure
