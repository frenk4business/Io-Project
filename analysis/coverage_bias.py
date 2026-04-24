"""
analysis/coverage_bias.py
-------------------------
Approximate observational coverage bias assessment for the Io hotspot catalogue.

The fundamental problem
-----------------------
The USGS hotspot catalogue reflects where Galileo, Voyager, and ground-based
instruments *looked*, not where hotspots *exist*. A cell with no hotspot record
may be genuinely hotspot-free, or it may simply be unobserved. This distinction
cannot be resolved without explicit imaging-coverage data.

Proxy used here
---------------
We use the geology layer as a partial observational proxy: cells whose geology
unit is 'NoData' or 'UNKNOWN' are treated as **likely unobserved** (no geological
mapping was possible, which is itself a coverage artefact). This is a conservative
lower bound — cells with known geology may still have been observed with
insufficient thermal sensitivity to detect weak hotspots.

Per-band analysis
-----------------
We compute the following per latitude band (the same 4 bands as spatial CV):
  - total_cells:    all 1°×1° cells in the band
  - observed_cells: cells with known geology (not NoData / UNKNOWN)
  - unobserved_pct: 100 × (1 − observed/total)
  - raw_rate:       hotspots / total_cells
  - adjusted_rate:  hotspots / observed_cells
  - rate_ratio:     adjusted_rate / raw_rate  (> 1 means bias suppresses raw rate)

Interpretation
--------------
A high rate_ratio indicates that raw hotspot density in that band is substantially
depressed by lack of observational coverage. The adjusted rate is still biased (see
caveat) but is a better minimum estimate of intrinsic hotspot density.

Required data to replace this scaffold
---------------------------------------
For a rigorous correction, use:
  - Galileo SSI / NIMS imaging footprint rasters (available from USGS PDS)
  - Juno/JIRAM coverage maps (post-2016 flybys)
  These should be rasterised to the 1°×1° grid and used to define `observed_cells`
  instead of the geology proxy.

See docs/scientific_methods.md §Coverage-bias awareness.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, SPATIAL_CV_FOLDS
from features.build import LABEL_COLUMN

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
COVERAGE_BIAS_CSV: Path = RESULTS_DIR / "coverage_bias.csv"
COVERAGE_BIAS_PNG: Path = RESULTS_DIR / "coverage_bias.png"

# Geology unit values that indicate absent/incomplete mapping
UNOBSERVED_UNIT_VALUES: frozenset[str] = frozenset({"NoData", "UNKNOWN", "nan", ""})


def _is_observed(unit: object) -> bool:
    """Return True if the geology unit suggests the cell was actually mapped."""
    return str(unit).strip() not in UNOBSERVED_UNIT_VALUES


def compute_coverage_bias(
    feature_matrix: pd.DataFrame,
    unit_column: str = "geology_unit",
    label_column: str = LABEL_COLUMN,
    folds: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Compute per-latitude-band coverage bias metrics.

    Args:
        feature_matrix: Feature DataFrame with lat_centre, geology_unit, has_hotspot.
        unit_column: Column identifying the geological unit per cell.
        label_column: Binary hotspot label column.
        folds: Latitude-band boundaries. Defaults to SPATIAL_CV_FOLDS from config.

    Returns:
        DataFrame with one row per latitude band:
        lat_band, total_cells, observed_cells, n_hotspots, unobserved_pct,
        raw_rate, adjusted_rate, rate_ratio.
    """
    if folds is None:
        folds = SPATIAL_CV_FOLDS

    if unit_column not in feature_matrix.columns:
        logger.warning(
            "Column '%s' missing — cannot assess coverage bias. "
            "Returning empty DataFrame.",
            unit_column,
        )
        return pd.DataFrame()

    rows = []
    for lat_min, lat_max in folds:
        mask = (feature_matrix["lat_centre"] >= lat_min) & (
            feature_matrix["lat_centre"] < lat_max
        )
        band = feature_matrix[mask]

        total = len(band)
        observed = int(band[unit_column].apply(_is_observed).sum())
        n_hot = int(band[label_column].sum())
        unobserved_pct = 100.0 * (total - observed) / total if total > 0 else float("nan")
        raw_rate = n_hot / total if total > 0 else float("nan")
        adj_rate = n_hot / observed if observed > 0 else float("nan")
        rate_ratio = adj_rate / raw_rate if raw_rate and raw_rate > 0 else float("nan")

        rows.append(
            {
                "lat_band": f"{lat_min:+.0f}° to {lat_max:+.0f}°",
                "lat_min": lat_min,
                "lat_max": lat_max,
                "total_cells": total,
                "observed_cells": observed,
                "n_hotspots": n_hot,
                "unobserved_pct": round(unobserved_pct, 2),
                "raw_rate": raw_rate,
                "adjusted_rate": adj_rate,
                "rate_ratio": round(rate_ratio, 3) if not np.isnan(rate_ratio) else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "Coverage bias per band:\n%s",
        df[["lat_band", "unobserved_pct", "raw_rate", "adjusted_rate", "rate_ratio"]].to_string(index=False),
    )
    return df


def plot_coverage_bias(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Dual-panel figure: adjusted vs raw hotspot rate; unobserved cell fraction.

    Args:
        df: Output of compute_coverage_bias().
        out_path: Output PNG path.

    Returns:
        Path to saved figure.
    """
    out_path = out_path or COVERAGE_BIAS_PNG
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No coverage data available", ha="center", va="center")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    bands = df["lat_band"].values
    x = np.arange(len(bands))
    width = 0.35

    # Panel 1: raw vs adjusted hotspot rate
    ax1.bar(x - width / 2, df["raw_rate"] * 1000, width=width, label="Raw rate (per 1000 cells)", color="C0")
    ax1.bar(x + width / 2, df["adjusted_rate"] * 1000, width=width, label="Adjusted rate (per 1000 cells)", color="C1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("Hotspot rate (per 1000 grid cells)")
    ax1.set_title("Raw vs coverage-adjusted hotspot density")
    ax1.legend(fontsize=8)

    # Panel 2: unobserved cell percentage per band
    colors = ["#d62728" if p > 5.0 else "#1f77b4" for p in df["unobserved_pct"]]
    ax2.bar(x, df["unobserved_pct"], color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("% cells with absent/unknown geology")
    ax2.set_title("Proxy coverage gap per band\n(red > 5% unobserved)")
    ax2.axhline(5.0, linestyle="--", color="grey", lw=1)

    fig.suptitle(
        "Coverage bias assessment (geology proxy — not imaging coverage rasters)\n"
        "Adjusted rate = hotspots / geologically-mapped cells only",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved coverage bias figure to %s", out_path)
    return out_path


def save_coverage_bias(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist coverage bias table to CSV."""
    path = path or COVERAGE_BIAS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved coverage bias table to %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from features.build import load_feature_matrix

    fm = load_feature_matrix()
    df = compute_coverage_bias(fm)
    save_coverage_bias(df)
    plot_coverage_bias(df)
    print(df.to_string(index=False))
