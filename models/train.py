"""
models/train.py
---------------
Train a logistic regression classifier on the Io hotspot feature matrix
and evaluate it using spatial cross-validation.

Model choice rationale
----------------------
Logistic regression is the correct baseline here:
  - Interpretable coefficients (directly tells us which features matter)
  - No hyperparameter tuning required for a first pass
  - Handles class imbalance via class_weight='balanced'
  - Cannot overfit in complex nonlinear ways that would be hard to diagnose

Do not replace this with gradient boosting or neural networks until you can
demonstrate the logistic regression baseline is insufficient and you understand
exactly why.

Metrics
-------
We report precision, recall, F1, and AUC-ROC per fold and as mean ± std.
We do NOT report accuracy — a zero-rule classifier achieves 99.8% accuracy
on this dataset. Reporting accuracy would be misleading.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from config import (
    LOGISTIC_REGRESSION_PARAMS,
    PROCESSED_DIR,
    FEATURE_MATRIX_FILENAME,
)
from features.build import FEATURE_COLUMNS, LABEL_COLUMN, load_feature_matrix
from models.spatial_cv import SpatialLatitudeFoldCV

logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = PROCESSED_DIR.parent / "models" / "logistic_regression.pkl"
SCALER_OUTPUT_PATH = PROCESSED_DIR.parent / "models" / "scaler.pkl"
COEFFICIENTS_AUDIT_PATH = PROCESSED_DIR.parent / "results" / "coefficients.json"

# Coefficients above this magnitude (on standardised features) are flagged as
# suspicious — either the feature is extremely informative or it leaks the
# target. The leakage audit module is the authoritative check; this is a
# simple sentinel.
COEFFICIENT_SUSPICION_THRESHOLD: float = 5.0


def _compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute per-fold classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary of metric names to values.
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "n_test": len(y_true),
        "n_positive_test": int(y_true.sum()),
        "n_positive_pred": int(y_pred.sum()),
    }


def train_with_spatial_cv(
    feature_matrix: pd.DataFrame | None = None,
) -> tuple[LogisticRegression, StandardScaler, pd.DataFrame]:
    """Train logistic regression using spatial cross-validation.

    Args:
        feature_matrix: Pre-loaded feature matrix. If None, loads from disk.

    Returns:
        Tuple of:
            model: Final LogisticRegression trained on all data.
            scaler: StandardScaler fitted on all data.
            cv_results: DataFrame of per-fold metrics.
    """
    if feature_matrix is None:
        feature_matrix = load_feature_matrix()

    X = feature_matrix[FEATURE_COLUMNS].values
    y = feature_matrix[LABEL_COLUMN].values
    latitudes = feature_matrix["lat_centre"].values

    # Report class imbalance — this is required, not optional
    n_positive = y.sum()
    n_total = len(y)
    imbalance_ratio = n_total / max(n_positive, 1)
    logger.info(
        "Class imbalance: %d positive / %d total (ratio 1:%.0f). "
        "Using class_weight='balanced'.",
        n_positive, n_total, imbalance_ratio,
    )

    cv = SpatialLatitudeFoldCV()
    fold_results = []

    logger.info("Running %d-fold spatial CV...", cv.n_splits)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, groups=latitudes)):
        lat_band = cv.folds[fold_idx]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features (fit on train only — no leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        metrics = _compute_fold_metrics(y_test, y_pred, y_prob)
        metrics["fold"] = fold_idx
        metrics["lat_band"] = f"{lat_band[0]:+.0f}° to {lat_band[1]:+.0f}°"

        logger.info(
            "Fold %d (%s): precision=%.3f recall=%.3f f1=%.3f auc=%.3f",
            fold_idx, metrics["lat_band"],
            metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["auc_roc"],
        )

        fold_results.append(metrics)

    cv_results = pd.DataFrame(fold_results)

    # Summary statistics
    numeric_cols = ["precision", "recall", "f1", "auc_roc"]
    logger.info("\n=== Spatial CV Summary ===")
    for col in numeric_cols:
        vals = cv_results[col].dropna()
        logger.info(
            "%s: mean=%.3f ± %.3f", col, vals.mean(), vals.std()
        )

    # Train final model on all data
    logger.info("Training final model on all data...")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    final_model.fit(X_scaled, y)

    # Log coefficients
    coef_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "coefficient": final_model.coef_[0],
    }).sort_values("coefficient", ascending=False)
    logger.info("Feature coefficients (final model):\n%s", coef_df.to_string())

    # Emit coefficient audit to data/results/ — additive, never touches the pkl
    try:
        _write_coefficient_audit(coef_df)
    except OSError as exc:
        logger.warning("Could not write coefficient audit: %s", exc)

    return final_model, final_scaler, cv_results


def _write_coefficient_audit(coef_df: pd.DataFrame) -> Path:
    """Persist a coefficient audit JSON alongside model training.

    Fields per feature:
      - feature name
      - standardised coefficient
      - |coefficient|
      - suspicion flag (|coef| >= COEFFICIENT_SUSPICION_THRESHOLD)

    This output is consumed by the Scientific Analysis → Leakage Audit tab.
    """
    records = []
    for _, row in coef_df.iterrows():
        coef = float(row["coefficient"])
        abs_coef = abs(coef)
        records.append(
            {
                "feature": row["feature"],
                "coefficient": coef,
                "abs_coefficient": abs_coef,
                "suspected_leakage": abs_coef >= COEFFICIENT_SUSPICION_THRESHOLD,
            }
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": COEFFICIENT_SUSPICION_THRESHOLD,
        "features": records,
    }
    COEFFICIENTS_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COEFFICIENTS_AUDIT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved coefficient audit to %s", COEFFICIENTS_AUDIT_PATH)
    return COEFFICIENTS_AUDIT_PATH


def save_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
) -> None:
    """Persist trained model and scaler to disk.

    Args:
        model: Trained LogisticRegression.
        scaler: Fitted StandardScaler.
        model_path: Output path for model pickle.
        scaler_path: Output path for scaler pickle.
    """
    model_path = model_path or MODEL_OUTPUT_PATH
    scaler_path = scaler_path or SCALER_OUTPUT_PATH

    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Saved model to %s.", model_path)
    logger.info("Saved scaler to %s.", scaler_path)


def load_model(
    model_path: Path | None = None,
    scaler_path: Path | None = None,
) -> tuple[LogisticRegression, StandardScaler]:
    """Load trained model and scaler from disk.

    Args:
        model_path: Path to model pickle.
        scaler_path: Path to scaler pickle.

    Returns:
        Tuple of (model, scaler).

    Raises:
        FileNotFoundError: If model files are missing.
    """
    model_path = model_path or MODEL_OUTPUT_PATH
    scaler_path = scaler_path or SCALER_OUTPUT_PATH

    for p in [model_path, scaler_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Model file not found: {p}. Run: python -m models.train"
            )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, scaler, cv_results = train_with_spatial_cv()
    print("\nCV Results:")
    print(cv_results[["fold", "lat_band", "precision", "recall", "f1", "auc_roc"]].to_string())
    save_model(model, scaler)
