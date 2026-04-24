"""
models/spatial_cv.py
--------------------
Spatial cross-validation with latitude-band folds.

⚠️  CRITICAL: Random train/test splits are FORBIDDEN for this dataset.
    Spatial autocorrelation means nearby cells share feature values.
    Random splits leak spatial information from train to test, inflating
    all metrics. See CLAUDE.md for the full rationale.

This module implements 4-fold spatial CV using latitude bands as fold
boundaries. Each fold withholds one 45°-wide latitude band and trains
on the remaining three.

Usage
-----
    from models.spatial_cv import SpatialLatitudeFoldCV

    cv = SpatialLatitudeFoldCV()
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, groups=latitudes)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from config import SPATIAL_CV_FOLDS

logger = logging.getLogger(__name__)


class SpatialLatitudeFoldCV:
    """Spatial cross-validator using latitude-band folds.

    Splits data into 4 folds based on latitude bands:
        Fold 0: test on -90° to -45°
        Fold 1: test on -45° to   0°
        Fold 2: test on   0° to +45°
        Fold 3: test on +45° to +90°

    Compatible with sklearn's cross_validate() when used with groups.

    Attributes:
        folds: List of (lat_min, lat_max) tuples defining fold bands.
    """

    def __init__(
        self, folds: list[tuple[float, float]] | None = None
    ) -> None:
        self.folds = folds if folds is not None else SPATIAL_CV_FOLDS

    @property
    def n_splits(self) -> int:
        """Number of CV folds."""
        return len(self.folds)

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | None = None,
        groups: np.ndarray | pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate (train_indices, test_indices) for each fold.

        Args:
            X: Feature matrix (unused except for length).
            y: Labels (unused).
            groups: Latitude values for each sample. Required.

        Yields:
            Tuples of (train_indices, test_indices) as numpy arrays.

        Raises:
            ValueError: If groups (latitudes) are not provided.
        """
        if groups is None:
            raise ValueError(
                "groups (latitude values) are required for SpatialLatitudeFoldCV. "
                "Pass lat_centre values as the groups argument."
            )

        latitudes = np.asarray(groups)
        n_samples = len(latitudes)
        all_indices = np.arange(n_samples)

        for fold_idx, (lat_min, lat_max) in enumerate(self.folds):
            test_mask = (latitudes >= lat_min) & (latitudes < lat_max)
            test_indices = all_indices[test_mask]
            train_indices = all_indices[~test_mask]

            if len(test_indices) == 0:
                logger.warning(
                    "Fold %d (%.0f° to %.0f°) has no test samples.",
                    fold_idx, lat_min, lat_max,
                )

            logger.debug(
                "Fold %d: %d train, %d test (%.0f° to %.0f°).",
                fold_idx, len(train_indices), len(test_indices), lat_min, lat_max,
            )

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return number of splits (required by sklearn CV interface)."""
        return self.n_splits

    def describe_folds(self, latitudes: np.ndarray) -> pd.DataFrame:
        """Summarise fold composition for a given latitude array.

        Args:
            latitudes: Array of latitude values.

        Returns:
            DataFrame with fold statistics.
        """
        rows = []
        for fold_idx, (lat_min, lat_max) in enumerate(self.folds):
            mask = (latitudes >= lat_min) & (latitudes < lat_max)
            rows.append({
                "fold": fold_idx,
                "lat_band": f"{lat_min:+.0f}° to {lat_max:+.0f}°",
                "n_cells": int(mask.sum()),
                "pct_of_total": 100 * mask.mean(),
            })
        return pd.DataFrame(rows)
