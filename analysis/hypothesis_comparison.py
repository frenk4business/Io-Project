"""
analysis/hypothesis_comparison.py
---------------------------------
Inhomogeneous Poisson point-process comparison of candidate tidal-heating
hypotheses and a geology-based intensity field against the USGS hotspot
catalogue.

The question
------------
Treat the observed hotspot catalogue as a realisation of an inhomogeneous
Poisson point process on the sphere with cell-level intensity

    λ_i = α · h_i · A_i

where ``h_i`` is a candidate *spatial intensity template* (``tidal_templates``
or a geology-enrichment field), ``A_i`` is the cell area (area-weighting from
``cos(lat_centre)``), and ``α`` is a single global rate that we fit by maximum
likelihood. For count data ``y_i`` on the grid (``has_hotspot`` or
``hotspot_count``), the Poisson log-likelihood is

    log L(α, h) = Σ y_i log(α h_i A_i) - Σ α h_i A_i   (+ const).

The α̂ MLE is the global count / total intensity, so for each template we
report ``log L(α̂, h)``, AIC (parameters = 1), ΔAIC relative to the best
template, and McFadden pseudo-R² against the uniform null.

What this computes
------------------
For each candidate template:
    - log_lik
    - aic
    - delta_aic (vs. best template)
    - pseudo_r2 = 1 - log_lik / log_lik_uniform
    - lr_p_value — likelihood-ratio test vs uniform null (asymptotic χ² with
      1 d.o.f.; only meaningful when the model strictly nests the null, which
      is the case here since uniform corresponds to a constant template).

What this does NOT claim
------------------------
- Causality. A high-likelihood template does not prove the underlying physics.
- Completeness. Coverage bias, catalogue vintage, and detection thresholds
  are not modelled. The geology-enrichment template will usually win on this
  dataset simply because it is fit to the same labels it is scored against —
  it is included as an upper-bound reference, not as a fair competitor.
- Absolute flux. Templates are normalised to unit area-weighted mean; only
  the *shape* is being compared.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from config import PROCESSED_DIR
from analysis.tidal_templates import TEMPLATE_NAMES, attach_templates_to_grid

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
HYPOTHESIS_CSV: Path = RESULTS_DIR / "hypothesis_comparison.csv"


def _cell_areas(lats_deg: np.ndarray, resolution_deg: float = 1.0) -> np.ndarray:
    """Relative area of a 1° cell centred at given latitudes. Scale-free."""
    phi = np.radians(lats_deg)
    # Exact area of a lat-lon cell is |Δλ · (sin φ_top − sin φ_bot)|.
    half = np.radians(resolution_deg / 2.0)
    return np.abs(np.sin(phi + half) - np.sin(phi - half))


def _poisson_loglik(counts: np.ndarray, intensity: np.ndarray) -> float:
    """Poisson log-likelihood with α fit by MLE (α̂ = N / Σ intensity).

    Returns log L(α̂, h) up to a constant (Σ log y_i!) that does not affect
    relative comparisons or ΔAIC.
    """
    total_count = float(counts.sum())
    total_intensity = float(intensity.sum())
    if total_intensity <= 0:
        return -np.inf
    alpha_hat = total_count / total_intensity
    mu = alpha_hat * intensity
    # Guard against log(0). y_i = 0 contributes 0 to the y log μ term.
    positive = counts > 0
    ll = np.sum(counts[positive] * np.log(mu[positive])) - mu.sum()
    return float(ll)


def _geology_intensity(feature_matrix: pd.DataFrame) -> np.ndarray:
    """Per-cell intensity derived from observed per-unit hotspot rates.

    This is an in-sample template (fit to the same labels it will be scored
    against) and is included as an optimistic upper-bound reference.
    """
    rates = (
        feature_matrix.groupby("geology_unit")["has_hotspot"]
        .mean()
        .reindex(feature_matrix["geology_unit"].unique())
    )
    # Clip floor to avoid zero intensity in genuinely empty units.
    rates = rates.clip(lower=1e-6)
    rate_map = feature_matrix["geology_unit"].map(rates).to_numpy()
    # Normalise to unit area-weighted mean.
    lats = feature_matrix["lat_centre"].to_numpy()
    weights = np.cos(np.radians(lats))
    return rate_map / (np.sum(rate_map * weights) / np.sum(weights))


def compare_hypotheses(
    feature_matrix: pd.DataFrame,
    count_column: str = "has_hotspot",
    include_geology: bool = True,
) -> pd.DataFrame:
    """Compare tidal templates (+ optional geology field) as Poisson intensities.

    Args:
        feature_matrix: Grid DataFrame with ``lat_centre``, ``lon_centre``,
            ``has_hotspot`` (or ``hotspot_count``) and ``geology_unit``.
        count_column: Name of the Poisson count column.
        include_geology: If True, also score a geology-rate template.

    Returns:
        DataFrame (one row per template) with columns:
            template, log_lik, aic, delta_aic, pseudo_r2, lr_chi2, lr_p_value,
            reference.
    """
    required = {"lat_centre", "lon_centre", count_column}
    missing = required - set(feature_matrix.columns)
    if missing:
        raise ValueError(f"feature_matrix missing columns: {missing}")

    df = attach_templates_to_grid(feature_matrix)
    lats = df["lat_centre"].to_numpy()
    counts = df[count_column].to_numpy().astype(float)
    areas = _cell_areas(lats)

    # Uniform baseline log-lik is the null for the likelihood-ratio test
    uniform_intensity = areas * df["tidal_template_uniform"].to_numpy()
    ll_uniform = _poisson_loglik(counts, uniform_intensity)

    records = []
    from analysis.tidal_templates import TEMPLATE_REFERENCES

    for name in TEMPLATE_NAMES:
        h = df[f"tidal_template_{name}"].to_numpy()
        intensity = areas * h
        ll = _poisson_loglik(counts, intensity)
        records.append(
            {
                "template": name,
                "log_lik": ll,
                "aic": 2.0 - 2.0 * ll,  # k = 1 (α), AIC = 2k − 2 ln L
                "pseudo_r2": 1.0 - ll / ll_uniform if ll_uniform != 0 else np.nan,
                "lr_chi2": 2.0 * (ll - ll_uniform),
                "lr_p_value": 1.0 - chi2.cdf(max(0.0, 2.0 * (ll - ll_uniform)), df=1),
                "reference": TEMPLATE_REFERENCES.get(name, ""),
            }
        )

    if include_geology and "geology_unit" in df.columns:
        h = _geology_intensity(df)
        intensity = areas * h
        ll = _poisson_loglik(counts, intensity)
        records.append(
            {
                "template": "geology_insample",
                "log_lik": ll,
                "aic": 2.0 - 2.0 * ll,
                "pseudo_r2": 1.0 - ll / ll_uniform if ll_uniform != 0 else np.nan,
                "lr_chi2": 2.0 * (ll - ll_uniform),
                "lr_p_value": 1.0 - chi2.cdf(max(0.0, 2.0 * (ll - ll_uniform)), df=1),
                "reference": "In-sample geology rates (upper bound, not a fair competitor)",
            }
        )

    out = pd.DataFrame.from_records(records)
    out["delta_aic"] = out["aic"] - out["aic"].min()
    out = out.sort_values("aic").reset_index(drop=True)
    return out


def save_hypothesis_comparison(
    df: pd.DataFrame, path: Path | None = None
) -> Path:
    out_path = path or HYPOTHESIS_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote hypothesis comparison → %s", out_path)
    return out_path
