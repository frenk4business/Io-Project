"""
analysis/geology_enrichment.py
------------------------------
Per-geological-unit chi-square enrichment test.

For each USGS map unit we compute:
  - observed hotspot count
  - cell area (count) labelled with the unit
  - expected count under Complete Spatial Randomness (CSR), proportional to area
  - enrichment ratio (obs / exp)
  - Wilson 95% confidence interval on the unit's hotspot probability
  - chi-square contribution
  - Bonferroni-corrected p-value

Inputs:
  feature_matrix: DataFrame with columns `geology_unit`, `has_hotspot`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from config import PROCESSED_DIR
from features.build import LABEL_COLUMN

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
ENRICHMENT_CSV: Path = RESULTS_DIR / "geology_enrichment.csv"
ENRICHMENT_PNG: Path = RESULTS_DIR / "geology_enrichment.png"


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def compute_geology_enrichment(
    feature_matrix: pd.DataFrame,
    unit_column: str = "geology_unit",
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Compute per-unit enrichment statistics.

    Returns:
        DataFrame sorted by enrichment ratio (descending), with columns:
        unit, n_cells, n_hotspots, expected, enrichment_ratio,
        wilson_lo, wilson_hi, chi2_contribution, p_bonf, significant.
    """
    if unit_column not in feature_matrix.columns:
        raise KeyError(f"Column '{unit_column}' missing from feature matrix.")

    total_cells = len(feature_matrix)
    total_hotspots = int(feature_matrix[label_column].sum())
    if total_hotspots == 0 or total_cells == 0:
        raise ValueError("Feature matrix has no hotspots or no cells.")
    base_rate = total_hotspots / total_cells

    rows = []
    grouped = feature_matrix.groupby(unit_column, dropna=False)
    for unit, sub in grouped:
        n_cells = int(len(sub))
        n_hot = int(sub[label_column].sum())
        expected = base_rate * n_cells
        enrichment = (n_hot / expected) if expected > 0 else np.nan
        lo, hi = _wilson_ci(n_hot, n_cells)
        chi2_contrib = ((n_hot - expected) ** 2 / expected) if expected > 0 else np.nan
        rows.append(
            {
                "unit": str(unit),
                "n_cells": n_cells,
                "n_hotspots": n_hot,
                "expected": float(expected),
                "enrichment_ratio": float(enrichment) if not np.isnan(enrichment) else np.nan,
                "wilson_lo": float(lo) if not np.isnan(lo) else np.nan,
                "wilson_hi": float(hi) if not np.isnan(hi) else np.nan,
                "chi2_contribution": float(chi2_contrib) if not np.isnan(chi2_contrib) else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    # Bonferroni-corrected p-values using chi-square with 1 df
    k = len(df)
    df["p_raw"] = df["chi2_contribution"].apply(
        lambda c: float(chi2.sf(c, df=1)) if pd.notna(c) else np.nan
    )
    df["p_bonf"] = (df["p_raw"] * k).clip(upper=1.0)
    df["significant"] = df["p_bonf"] < 0.05

    df = df.sort_values("enrichment_ratio", ascending=False, na_position="last").reset_index(drop=True)
    return df


def plot_enrichment_bar(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Forest-plot style bar chart of enrichment ratios with Wilson CIs."""
    out_path = out_path or ENRICHMENT_PNG
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df[df["n_cells"] > 0].copy()
    plot_df = plot_df.sort_values("enrichment_ratio", ascending=True, na_position="first")

    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(plot_df))))
    ax.axvline(1.0, linestyle="--", color="grey", lw=1, label="CSR (obs/exp = 1)")
    y = np.arange(len(plot_df))
    ratios = plot_df["enrichment_ratio"].fillna(0).values

    # CI on enrichment ratio via Wilson bounds normalised by base rate
    total_cells = plot_df["n_cells"].sum()
    total_hot = plot_df["n_hotspots"].sum()
    base_rate = (total_hot / total_cells) if total_cells else 1.0
    lo_ratio = (plot_df["wilson_lo"].fillna(0) / base_rate).values
    hi_ratio = (plot_df["wilson_hi"].fillna(0) / base_rate).values
    err = np.vstack([ratios - lo_ratio, hi_ratio - ratios])
    err = np.clip(err, 0, None)

    colours = ["#d62728" if s else "#1f77b4" for s in plot_df["significant"]]
    ax.errorbar(ratios, y, xerr=err, fmt="o", color="black", ecolor="grey", capsize=2)
    for y_i, c, r in zip(y, colours, ratios):
        ax.plot(r, y_i, "o", color=c, markersize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["unit"].values)
    ax.set_xlabel("Enrichment ratio (obs / exp under CSR)")
    ax.set_title("Per-unit hotspot enrichment (95% Wilson CI; red = Bonferroni sig.)")
    ax.set_xscale("symlog", linthresh=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_enrichment(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or ENRICHMENT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved geology enrichment table to %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from features.build import load_feature_matrix

    fm = load_feature_matrix()
    df = compute_geology_enrichment(fm)
    save_enrichment(df)
    plot_enrichment_bar(df)
    print(df.to_string(index=False))
