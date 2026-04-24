"""
analysis/leakage_audit.py
-------------------------
Audit each feature in the Io feature matrix for target leakage.

A feature is flagged as **suspected leakage** if any of:
  - It is derived from the target label itself (configurable allow/deny list)
  - Its Pearson / Spearman correlation with the label exceeds LEAKAGE_CORR_THRESHOLD
  - Its logistic-regression coefficient (absolute value, scaled features) exceeds
    LEAKAGE_COEF_THRESHOLD

This module does not re-fit models against production — it fits a small LR
on all data only to measure coefficient magnitude per feature, as a diagnostic.

See docs/scientific_methods.md §Leakage for methodology and rationale.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import LOGISTIC_REGRESSION_PARAMS
from features.build import FEATURE_COLUMNS, LABEL_COLUMN

logger = logging.getLogger(__name__)

# Thresholds are deliberately conservative — flag, don't gate
LEAKAGE_CORR_THRESHOLD: float = 0.6
LEAKAGE_COEF_THRESHOLD: float = 5.0

# Features whose construction references the target label directly
# (documented in features/synthetic.py). These are target-derived by design.
TARGET_DERIVED_FEATURES: frozenset[str] = frozenset(
    {
        "dist_nearest_hotspot_km",
    }
)

# Features whose construction references an auxiliary signal that is not the
# target but is hand-crafted rather than observed, so they cannot claim
# independent physical meaning.
ANALYTICAL_PROXY_FEATURES: frozenset[str] = frozenset(
    {
        "dist_tidal_stress_max_km",
        "tidal_heating_flux",
    }
)


def _feature_correlations(
    feature_matrix: pd.DataFrame,
    feature_columns: Iterable[str],
    label_column: str,
) -> pd.DataFrame:
    """Return point-biserial (Pearson) and Spearman correlations with the label."""
    rows = []
    y = feature_matrix[label_column].astype(float).values
    for col in feature_columns:
        x = pd.to_numeric(feature_matrix[col], errors="coerce").values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 2 or np.unique(y[mask]).size < 2:
            pb = np.nan
            sp = np.nan
        else:
            pb, _ = pointbiserialr(y[mask], x[mask])
            sp, _ = spearmanr(x[mask], y[mask])
        rows.append({"feature": col, "pearson_r": pb, "spearman_r": sp})
    return pd.DataFrame(rows)


def _feature_coefficients(
    feature_matrix: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    """Fit a logistic regression on standardised features; return coefficients.

    Note: this is a full-data fit and reports *in-sample* coefficient magnitudes.
    Not a performance metric — purely a diagnostic signal. Use the ablation
    module for leakage-free CV performance.
    """
    X = feature_matrix[feature_columns].astype(float).values
    y = feature_matrix[label_column].astype(int).values

    # Imputation for diagnostic purposes only — median-fill NaNs
    nan_cols = np.isnan(X).any(axis=0)
    if nan_cols.any():
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    model.fit(Xs, y)
    return pd.DataFrame(
        {"feature": feature_columns, "lr_coef": model.coef_[0]}
    )


def _derivation_source(feature: str) -> str:
    if feature in TARGET_DERIVED_FEATURES:
        return "target-derived (from hotspot catalogue)"
    if feature in ANALYTICAL_PROXY_FEATURES:
        return "analytical proxy (not observed)"
    return "observed / external"


def audit_feature_leakage(
    feature_matrix: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = LABEL_COLUMN,
    corr_threshold: float = LEAKAGE_CORR_THRESHOLD,
    coef_threshold: float = LEAKAGE_COEF_THRESHOLD,
) -> pd.DataFrame:
    """Audit each feature for suspected target leakage.

    Args:
        feature_matrix: DataFrame with features and a binary label column.
        feature_columns: Columns to audit. Defaults to FEATURE_COLUMNS.
        label_column: Name of binary label column.
        corr_threshold: |correlation| above which a feature is flagged.
        coef_threshold: |LR coefficient| above which a feature is flagged.

    Returns:
        DataFrame with one row per feature: derivation source, correlations,
        LR coefficient, and a boolean `suspected_leakage` flag with a
        `flag_reason` column explaining the trigger.
    """
    if feature_columns is None:
        feature_columns = list(FEATURE_COLUMNS)

    corr_df = _feature_correlations(feature_matrix, feature_columns, label_column)
    coef_df = _feature_coefficients(feature_matrix, feature_columns, label_column)
    audit = corr_df.merge(coef_df, on="feature", how="outer")

    audit["derivation_source"] = audit["feature"].map(_derivation_source)
    audit["abs_pearson"] = audit["pearson_r"].abs()
    audit["abs_coef"] = audit["lr_coef"].abs()

    reasons: list[str] = []
    for _, row in audit.iterrows():
        triggers = []
        if row["feature"] in TARGET_DERIVED_FEATURES:
            triggers.append("target-derived by construction")
        if row["abs_pearson"] >= corr_threshold:
            triggers.append(f"|r|={row['abs_pearson']:.2f} ≥ {corr_threshold}")
        if row["abs_coef"] >= coef_threshold:
            triggers.append(f"|coef|={row['abs_coef']:.2f} ≥ {coef_threshold}")
        reasons.append("; ".join(triggers) if triggers else "")
    audit["flag_reason"] = reasons
    audit["suspected_leakage"] = audit["flag_reason"].str.len() > 0

    ordered = [
        "feature",
        "derivation_source",
        "pearson_r",
        "spearman_r",
        "lr_coef",
        "abs_coef",
        "suspected_leakage",
        "flag_reason",
    ]
    return audit[ordered].sort_values("abs_coef", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from features.build import load_feature_matrix

    fm = load_feature_matrix()
    result = audit_feature_leakage(fm)
    print(result.to_string(index=False))
