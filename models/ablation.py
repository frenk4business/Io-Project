"""
models/ablation.py
------------------
Feature-set ablation under spatial cross-validation.

For each named feature set we train a logistic regression through the same
4-latitude-band spatial CV used by models/train.py and report precision,
recall, F1, AUC-ROC, PR-AUC per fold and aggregated mean±std.

Default feature sets
--------------------
- `all`          — current four features (LEAKY BASELINE; includes
                   dist_nearest_hotspot_km).
- `no_leakage`   — drops `dist_nearest_hotspot_km`. This is the project's
                   honest headline.
- `geology_only` — geology_encoded alone.
- `tidal_only`   — tidal_heating_flux alone (NOTE: synthetic proxy — not a
                   physical model; see docs/scientific_methods.md §Tidal).
- `null`         — labels permuted. AUC must come out ~0.5; a sanity check
                   on the CV implementation.

Results are returned as a dict and, when called via the run script, persisted
to data/results/ablation.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config import LOGISTIC_REGRESSION_PARAMS, PROCESSED_DIR
from features.build import FEATURE_COLUMNS, LABEL_COLUMN
from models.spatial_cv import SpatialLatitudeFoldCV

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
ABLATION_OUTPUT: Path = RESULTS_DIR / "ablation.json"

NULL_PERMUTATION_SEED: int = 42


def default_feature_sets() -> dict[str, list[str]]:
    """Return the default ablation feature-set dictionary.

    Keys are feature-set names; values are lists of feature column names.
    """
    return {
        "all": list(FEATURE_COLUMNS),
        "no_leakage": [c for c in FEATURE_COLUMNS if c != "dist_nearest_hotspot_km"],
        "geology_only": ["geology_encoded"],
        "tidal_only": ["tidal_heating_flux"],
        # null uses the same feature set as `no_leakage` but with permuted labels
        "null": [c for c in FEATURE_COLUMNS if c != "dist_nearest_hotspot_km"],
    }


def _fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": (
            float(roc_auc_score(y_true, y_prob))
            if np.unique(y_true).size > 1
            else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if np.unique(y_true).size > 1
            else float("nan")
        ),
        "n_test": int(len(y_true)),
        "n_positive_test": int(y_true.sum()),
    }


def _run_single_feature_set(
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    permute_labels: bool,
    cv: SpatialLatitudeFoldCV,
) -> list[dict]:
    """Return per-fold metrics for one feature set."""
    X = feature_matrix[feature_cols].astype(float).values
    y = feature_matrix[label_col].astype(int).values
    latitudes = feature_matrix["lat_centre"].values

    # NaN-fill (median) — diagnostic ablation, not production
    if np.isnan(X).any():
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])

    rng = np.random.default_rng(NULL_PERMUTATION_SEED)
    if permute_labels:
        y = rng.permutation(y)

    fold_metrics: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, groups=latitudes)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        m = _fold_metrics(y_test, y_pred, y_prob)
        m["fold"] = int(fold_idx)
        m["lat_band"] = (
            f"{cv.folds[fold_idx][0]:+.0f}° to {cv.folds[fold_idx][1]:+.0f}°"
        )
        fold_metrics.append(m)

    return fold_metrics


def _summarise(folds: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate mean±std of each numeric metric across folds."""
    numeric_keys = ["precision", "recall", "f1", "auc_roc", "pr_auc"]
    summary: dict[str, dict[str, float]] = {}
    for k in numeric_keys:
        vals = np.array([f[k] for f in folds], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            summary[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return summary


def run_ablation(
    feature_matrix: pd.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    label_column: str = LABEL_COLUMN,
) -> dict:
    """Run the full ablation and return the structured results dict.

    Args:
        feature_matrix: Feature DataFrame with the label column present.
        feature_sets: Mapping name -> list of columns. Defaults to
                      default_feature_sets(). The key "null" is treated specially:
                      labels are permuted with a fixed seed.
        label_column: Name of the binary label column.

    Returns:
        Dict with keys: generated_at (iso timestamp), feature_sets (list of
        {name, features, leaky, description, folds, summary} entries).
    """
    if feature_sets is None:
        feature_sets = default_feature_sets()

    cv = SpatialLatitudeFoldCV()
    set_reports: list[dict] = []

    for name, cols in feature_sets.items():
        is_null = name == "null"
        is_leaky = "dist_nearest_hotspot_km" in cols
        note = ""
        if name == "tidal_only":
            note = (
                "tidal_heating_flux is a synthetic analytical proxy in this project "
                "(see docs/scientific_methods.md §Tidal); swap in real M3/M4 grids "
                "via ingest/tidal_models.py when ingested."
            )
        if is_leaky:
            note = (
                "LEAKY BASELINE: includes dist_nearest_hotspot_km which is derived "
                "from the target label; AUC here is inflated by construction."
            )
        if is_null:
            note = (
                "Null-label-permutation sanity check: AUC must be ≈ 0.5 for the CV "
                "implementation to be trustworthy."
            )

        logger.info(
            "Ablation set '%s' (%d features, permute=%s): %s",
            name, len(cols), is_null, cols,
        )
        folds = _run_single_feature_set(
            feature_matrix=feature_matrix,
            feature_cols=cols,
            label_col=label_column,
            permute_labels=is_null,
            cv=cv,
        )
        summary = _summarise(folds)
        set_reports.append(
            {
                "name": name,
                "features": cols,
                "is_leaky_baseline": is_leaky,
                "is_null_sanity_check": is_null,
                "note": note,
                "folds": folds,
                "summary": summary,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_folds": cv.n_splits,
        "feature_sets": set_reports,
    }


def save_ablation(results: dict, path: Path | None = None) -> Path:
    """Persist ablation results to JSON."""
    path = path or ABLATION_OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved ablation results to %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from features.build import load_feature_matrix

    fm = load_feature_matrix()
    results = run_ablation(fm)
    save_ablation(results)
    for s in results["feature_sets"]:
        summ = s["summary"]
        print(
            f"{s['name']:14s}  AUC={summ['auc_roc']['mean']:.3f}±{summ['auc_roc']['std']:.3f}  "
            f"PR-AUC={summ['pr_auc']['mean']:.3f}±{summ['pr_auc']['std']:.3f}  "
            f"F1={summ['f1']['mean']:.3f}±{summ['f1']['std']:.3f}"
        )
