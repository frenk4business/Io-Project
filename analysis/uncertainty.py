"""
analysis/uncertainty.py
-----------------------
Bootstrap uncertainty quantification for the scientific analysis layer.

Two complementary approaches
-----------------------------
1. **Catalog bootstrap** (implemented here): resample the 172-hotspot catalogue
   with replacement, rebuild the binary grid label for each resample, and
   recompute geology enrichment ratios. This gives confidence intervals that
   account for the finite catalogue size. It does NOT account for spatial
   dependence of hotspot locations.

2. **Fold variability** (from existing ablation): the 4-fold spatial CV already
   produces per-fold metrics. The standard deviation across folds is the most
   appropriate uncertainty measure for spatially cross-validated estimates —
   it reflects genuine geographic variation, not just sampling noise.

What is NOT implemented
------------------------
- Spatial bootstrap (resampling the grid): requires re-running the full
  spatial join for each resample, which is too expensive for interactive use.
- Parametric bootstrap: would require a generative model for hotspot locations.

Usage
-----
The dashboard uses `fold_variability_summary` to extract ± std from ablation
results (fast, no recomputation). `bootstrap_enrichment_ci` is used to add
tighter uncertainty estimates to the geology enrichment table.

See docs/scientific_methods.md §Uncertainty for methodology.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from features.build import LABEL_COLUMN
from analysis.geology_enrichment import compute_geology_enrichment

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
UNCERTAINTY_CSV: Path = RESULTS_DIR / "enrichment_bootstrap_ci.csv"

DEFAULT_N_BOOT: int = 500
DEFAULT_SEED: int = 42


def bootstrap_enrichment_ci(
    feature_matrix: pd.DataFrame,
    catalog: pd.DataFrame,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    unit_column: str = "geology_unit",
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Bootstrap 95% CI for per-unit enrichment ratios via catalogue resampling.

    For each bootstrap iteration:
      1. Resample the hotspot catalogue with replacement (same n).
      2. Re-assign resampled hotspots to grid cells (floor-binning).
      3. Recompute `has_hotspot` label and enrichment ratios.

    This captures uncertainty from finite catalogue size. It does not capture
    spatial uncertainty (where exactly each hotspot sits within its 1°×1° cell).

    Args:
        feature_matrix: Feature DataFrame with lon_centre, lat_centre, geology_unit.
        catalog: Hotspot catalogue DataFrame with longitude, latitude.
        n_boot: Number of bootstrap iterations.
        seed: Random seed.
        unit_column: Geology unit column in feature_matrix.
        label_column: Binary label column (overwritten per iteration).

    Returns:
        DataFrame: unit, enrichment_ratio (observed), boot_lo (2.5%), boot_hi (97.5%),
        boot_mean, boot_std.
    """
    rng = np.random.default_rng(seed)
    n_catalog = len(catalog)

    # Observed enrichment (full catalog)
    obs_df = compute_geology_enrichment(feature_matrix, unit_column, label_column)
    obs_ratios = obs_df.set_index("unit")["enrichment_ratio"].to_dict()
    units = obs_df["unit"].tolist()

    # Bootstrap
    boot_matrix = np.full((n_boot, len(units)), fill_value=np.nan)

    for b in range(n_boot):
        # Resample catalog rows
        sample_idx = rng.integers(0, n_catalog, size=n_catalog)
        sample = catalog.iloc[sample_idx].copy()

        # Re-assign hotspots to grid cells
        fm_b = feature_matrix[
            ["lon_centre", "lat_centre", unit_column]
        ].copy()
        fm_b[label_column] = 0

        # Floor-bin to match preprocess/align_layers.py logic
        lon_bins = np.floor(sample["longitude"]).astype(int)
        lat_bins = np.floor(sample["latitude"]).astype(int)

        for lon_b, lat_b in zip(lon_bins, lat_bins):
            mask = (
                (np.floor(fm_b["lon_centre"]).astype(int) == lon_b)
                & (np.floor(fm_b["lat_centre"]).astype(int) == lat_b)
            )
            fm_b.loc[mask, label_column] = 1

        try:
            enr_b = compute_geology_enrichment(fm_b, unit_column, label_column)
            enr_dict = enr_b.set_index("unit")["enrichment_ratio"].to_dict()
            for j, u in enumerate(units):
                boot_matrix[b, j] = enr_dict.get(u, np.nan)
        except (ValueError, ZeroDivisionError):
            pass  # zero hotspots in this bootstrap draw — leave as nan

    lo = np.nanpercentile(boot_matrix, 2.5, axis=0)
    hi = np.nanpercentile(boot_matrix, 97.5, axis=0)
    mean = np.nanmean(boot_matrix, axis=0)
    std = np.nanstd(boot_matrix, axis=0)

    result = pd.DataFrame(
        {
            "unit": units,
            "enrichment_ratio": [obs_ratios.get(u, np.nan) for u in units],
            "boot_lo": lo,
            "boot_hi": hi,
            "boot_mean": mean,
            "boot_std": std,
        }
    )
    logger.info(
        "Bootstrap enrichment CI: %d units, %d iterations, seed=%d",
        len(units), n_boot, seed,
    )
    return result


def fold_variability_summary(ablation_results: dict) -> pd.DataFrame:
    """Extract per-fold std as the primary CV uncertainty measure.

    Args:
        ablation_results: Output dict from models.ablation.run_ablation().

    Returns:
        DataFrame with one row per feature set:
        name, is_leaky, auc_mean, auc_std, pr_auc_mean, pr_auc_std,
        f1_mean, f1_std, precision_mean, precision_std, recall_mean, recall_std.
    """
    rows = []
    for fs in ablation_results.get("feature_sets", []):
        s = fs.get("summary", {})
        rows.append(
            {
                "name": fs["name"],
                "is_leaky_baseline": fs.get("is_leaky_baseline", False),
                "is_null_sanity_check": fs.get("is_null_sanity_check", False),
                "auc_mean": s.get("auc_roc", {}).get("mean", float("nan")),
                "auc_std": s.get("auc_roc", {}).get("std", float("nan")),
                "pr_auc_mean": s.get("pr_auc", {}).get("mean", float("nan")),
                "pr_auc_std": s.get("pr_auc", {}).get("std", float("nan")),
                "f1_mean": s.get("f1", {}).get("mean", float("nan")),
                "f1_std": s.get("f1", {}).get("std", float("nan")),
                "precision_mean": s.get("precision", {}).get("mean", float("nan")),
                "precision_std": s.get("precision", {}).get("std", float("nan")),
                "recall_mean": s.get("recall", {}).get("mean", float("nan")),
                "recall_std": s.get("recall", {}).get("std", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def save_bootstrap_ci(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist bootstrap enrichment CI table to CSV."""
    path = path or UNCERTAINTY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved bootstrap enrichment CI to %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from features.build import load_feature_matrix
    from ingest.hotspot_catalog import load_hotspot_catalog

    fm = load_feature_matrix()
    cat = load_hotspot_catalog()
    df = bootstrap_enrichment_ci(fm, cat, n_boot=200)
    save_bootstrap_ci(df)
    print(df.to_string(index=False))
