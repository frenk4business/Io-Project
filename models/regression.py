"""
models/regression.py
--------------------
Regression on Io hotspot thermal-activity intensity.

The target is ``log1p(primary_power_gw)`` where ``power_gw`` is the project's
estimated thermal-emission proxy derived from Davies et al. (2024) JIRAM
4.8 micron spectral radiance. It is not directly measured bolometric radiant
power.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR
from features.build import FEATURE_COLUMNS, load_feature_matrix
from models.spatial_cv import SpatialLatitudeFoldCV
from preprocess.power_grid import load_power_grid

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
POWER_REGRESSION_JSON: Path = RESULTS_DIR / "power_regression.json"
POWER_RESIDUALS_CSV: Path = RESULTS_DIR / "power_residuals.csv"

LEAKAGE_FEATURES: set[str] = {"dist_nearest_hotspot_km"}
DEFAULT_REGRESSION_FEATURES: list[str] = [
    c for c in FEATURE_COLUMNS if c not in LEAKAGE_FEATURES
]


def _prepare_regression_frame(
    feature_matrix: pd.DataFrame,
    power_grid: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    required_power = {
        "cell_id",
        "power_count",
        "primary_power_gw",
        "mean_power_gw",
        "sum_power_gw",
        "log_primary_power",
    }
    missing = required_power - set(power_grid.columns)
    if missing:
        raise ValueError(f"Power grid missing required columns: {sorted(missing)}")

    missing_features = [c for c in feature_cols if c not in feature_matrix.columns]
    if missing_features:
        raise ValueError(f"Feature matrix missing columns: {missing_features}")

    cols = [
        "cell_id",
        "lon_centre",
        "lat_centre",
        *feature_cols,
    ]
    if "geology_unit" in feature_matrix.columns:
        cols.append("geology_unit")

    merged = feature_matrix[cols].merge(
        power_grid[
            [
                "cell_id",
                "power_count",
                "primary_power_gw",
                "mean_power_gw",
                "sum_power_gw",
                "log_primary_power",
                "power_names",
            ]
        ],
        on="cell_id",
        how="inner",
    )
    observed = merged[merged["power_count"] > 0].copy()
    if observed.empty:
        raise ValueError(
            "No grid cells contain estimated thermal-emission proxy observations."
        )
    return observed.reset_index(drop=True)


def _fill_numeric_nan(X: np.ndarray) -> np.ndarray:
    if not np.isnan(X).any():
        return X
    X = X.copy()
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rho, p_val = spearmanr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "spearman_p": float(p_val) if np.isfinite(p_val) else float("nan"),
    }


def train_power_regression(
    feature_matrix: pd.DataFrame | None = None,
    power_grid: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    model_type: str = "ridge",
    alpha: float = 1.0,
) -> dict:
    """Train a leakage-aware spatial-CV regression on hotspot intensity.

    Args:
        feature_matrix: Optional preloaded feature matrix.
        power_grid: Optional preloaded power grid.
        feature_cols: Feature columns to use. Defaults to non-leaky features.
        model_type: ``ridge`` or ``elasticnet``.
        alpha: Regularisation strength.

    Returns:
        Structured dict with fold metrics, overall metrics, residuals, and notes.
    """
    feature_matrix = feature_matrix if feature_matrix is not None else load_feature_matrix()
    power_grid = power_grid if power_grid is not None else load_power_grid()
    feature_cols = feature_cols or list(DEFAULT_REGRESSION_FEATURES)

    leaky = sorted(set(feature_cols) & LEAKAGE_FEATURES)
    if leaky:
        raise ValueError(
            f"Refusing to train intensity regression with leakage feature(s): {leaky}"
        )

    data = _prepare_regression_frame(feature_matrix, power_grid, feature_cols)
    X = _fill_numeric_nan(data[feature_cols].astype(float).to_numpy())
    y = data["log_primary_power"].astype(float).to_numpy()
    latitudes = data["lat_centre"].to_numpy()

    cv = SpatialLatitudeFoldCV()
    folds: list[dict] = []
    residual_rows: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, groups=latitudes)):
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        if model_type == "elasticnet":
            model = ElasticNet(alpha=alpha, l1_ratio=0.2, random_state=42, max_iter=10000)
        elif model_type == "ridge":
            model = Ridge(alpha=alpha)
        else:
            raise ValueError("model_type must be 'ridge' or 'elasticnet'.")

        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        metrics = _regression_metrics(y[test_idx], pred)
        metrics.update(
            {
                "fold": int(fold_idx),
                "lat_band": f"{cv.folds[fold_idx][0]:+.0f} to {cv.folds[fold_idx][1]:+.0f}",
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
        folds.append(metrics)

        fold_df = data.iloc[test_idx].copy()
        for _, row in fold_df.assign(pred_log_power=pred).iterrows():
            residual_rows.append(
                {
                    "fold": int(fold_idx),
                    "cell_id": int(row["cell_id"]),
                    "longitude": float(row["lon_centre"]),
                    "latitude": float(row["lat_centre"]),
                    "observed_log_power": float(row["log_primary_power"]),
                    "predicted_log_power": float(row["pred_log_power"]),
                    "residual_log_power": float(row["log_primary_power"] - row["pred_log_power"]),
                    "primary_power_gw": float(row["primary_power_gw"]),
                    "power_count": int(row["power_count"]),
                    "power_names": row.get("power_names", ""),
                    "geology_unit": row.get("geology_unit", ""),
                }
            )

    residuals = pd.DataFrame(residual_rows)
    if residuals.empty:
        raise ValueError("Spatial CV produced no residual rows.")

    overall = _regression_metrics(
        residuals["observed_log_power"].to_numpy(),
        residuals["predicted_log_power"].to_numpy(),
    )
    summary = {}
    for key in ["r2", "rmse", "mae", "spearman"]:
        vals = np.array([f[key] for f in folds], dtype=float)
        vals = vals[np.isfinite(vals)]
        summary[key] = {
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "std": float(vals.std(ddof=0)) if vals.size else float("nan"),
        }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": model_type,
        "alpha": alpha,
        "target": "log1p(primary_power_gw)",
        "power_definition": (
            "estimated thermal-emission proxy derived from Davies/JIRAM 4.8 micron "
            "spectral radiance; not direct bolometric radiant power"
        ),
        "features": feature_cols,
        "excluded_leakage_features": sorted(LEAKAGE_FEATURES),
        "n_observed_power_cells": int(len(data)),
        "folds": folds,
        "summary": summary,
        "overall_oof": overall,
        "residuals": residuals,
    }
    logger.info(
        "Power regression complete: n=%d, OOF R2=%.3f, RMSE=%.3f, Spearman=%.3f.",
        len(data),
        overall["r2"],
        overall["rmse"],
        overall["spearman"],
    )
    return result


def save_power_regression(results: dict) -> tuple[Path, Path]:
    """Persist regression metrics JSON and residual CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    residuals = results.get("residuals", pd.DataFrame())
    if isinstance(residuals, pd.DataFrame):
        residuals.to_csv(POWER_RESIDUALS_CSV, index=False)

    payload = {k: v for k, v in results.items() if k != "residuals"}
    with POWER_REGRESSION_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved power regression results to %s and %s.", POWER_REGRESSION_JSON, POWER_RESIDUALS_CSV)
    return POWER_REGRESSION_JSON, POWER_RESIDUALS_CSV


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = train_power_regression()
    save_power_regression(res)
    print(json.dumps({k: res[k] for k in ["summary", "overall_oof"]}, indent=2))
