"""
models/validation_metrics.py
----------------------------
Calibration, PR, and Brier-score diagnostics per spatial CV fold.

Produces figures per fold and a summary JSON with Brier score and Matthews
correlation coefficient. Works on the `no_leakage` feature set by default so
reported calibration corresponds to the honest headline, not the leaky baseline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

from config import LOGISTIC_REGRESSION_PARAMS, PROCESSED_DIR
from features.build import FEATURE_COLUMNS, LABEL_COLUMN
from models.spatial_cv import SpatialLatitudeFoldCV

logger = logging.getLogger(__name__)

VALIDATION_DIR: Path = PROCESSED_DIR.parent / "results" / "validation"
VALIDATION_SUMMARY: Path = VALIDATION_DIR / "summary.json"

DEFAULT_NON_LEAKY_FEATURES: list[str] = [
    c for c in FEATURE_COLUMNS if c != "dist_nearest_hotspot_km"
]


def compute_validation_metrics(
    feature_matrix: pd.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = LABEL_COLUMN,
) -> dict:
    """Run spatial CV, compute calibration + PR + Brier + MCC per fold.

    Returns a dict ready for JSON persistence, plus a populated figure directory.
    """
    if feature_cols is None:
        feature_cols = list(DEFAULT_NON_LEAKY_FEATURES)

    X = feature_matrix[feature_cols].astype(float).values
    y = feature_matrix[label_col].astype(int).values
    latitudes = feature_matrix["lat_centre"].values

    if np.isnan(X).any():
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])

    cv = SpatialLatitudeFoldCV()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    fold_records: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, groups=latitudes)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        lat_band = f"{cv.folds[fold_idx][0]:+.0f}° to {cv.folds[fold_idx][1]:+.0f}°"

        if np.unique(y_test).size < 2:
            logger.warning(
                "Fold %d (%s) has a single class in test; skipping calibration/PR.",
                fold_idx, lat_band,
            )
            fold_records.append(
                {
                    "fold": fold_idx,
                    "lat_band": lat_band,
                    "brier": float("nan"),
                    "mcc": float("nan"),
                    "note": "single-class test fold",
                }
            )
            continue

        brier = float(brier_score_loss(y_test, y_prob))
        mcc = float(matthews_corrcoef(y_test, y_pred))

        # Calibration figure
        frac_pos, mean_pred = calibration_curve(
            y_test, y_prob, n_bins=10, strategy="quantile"
        )
        fig_cal, ax_cal = plt.subplots(figsize=(5, 5))
        ax_cal.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
        ax_cal.plot(mean_pred, frac_pos, "o-", color="C0")
        ax_cal.set_xlabel("Mean predicted probability")
        ax_cal.set_ylabel("Observed positive fraction")
        ax_cal.set_title(f"Reliability — fold {fold_idx} ({lat_band})")
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        ax_cal.legend(loc="upper left")
        fig_cal.tight_layout()
        cal_path = VALIDATION_DIR / f"calibration_fold{fold_idx}.png"
        fig_cal.savefig(cal_path, dpi=120)
        plt.close(fig_cal)

        # PR figure
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig_pr, ax_pr = plt.subplots(figsize=(5, 5))
        ax_pr.step(recall, precision, where="post", color="C3")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"PR curve — fold {fold_idx} ({lat_band})")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        fig_pr.tight_layout()
        pr_path = VALIDATION_DIR / f"pr_fold{fold_idx}.png"
        fig_pr.savefig(pr_path, dpi=120)
        plt.close(fig_pr)

        fold_records.append(
            {
                "fold": fold_idx,
                "lat_band": lat_band,
                "brier": brier,
                "mcc": mcc,
                "calibration_png": str(cal_path.relative_to(VALIDATION_DIR.parent.parent)),
                "pr_png": str(pr_path.relative_to(VALIDATION_DIR.parent.parent)),
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_cols": feature_cols,
        "folds": fold_records,
    }
    VALIDATION_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with VALIDATION_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved validation summary to %s", VALIDATION_SUMMARY)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from features.build import load_feature_matrix

    compute_validation_metrics(load_feature_matrix())
