"""
tests/test_spatial_cv.py
------------------------
Unit tests for models/spatial_cv.py

These tests verify that spatial CV does what it claims:
  - Exactly 4 folds
  - Test folds are mutually exclusive and exhaustive
  - No sample appears in both train and test for a given fold
  - Random splits are NOT used (no sklearn train_test_split in models/)
"""

import numpy as np
import pandas as pd
import pytest

from models.spatial_cv import SpatialLatitudeFoldCV
from config import SPATIAL_CV_FOLDS


class TestSpatialLatitudeFoldCV:
    def _make_latitudes(self, n: int = 1000) -> np.ndarray:
        """Generate uniform latitude samples."""
        return np.random.uniform(-90, 90, n)

    def test_n_splits(self):
        cv = SpatialLatitudeFoldCV()
        assert cv.n_splits == 4

    def test_requires_groups(self):
        cv = SpatialLatitudeFoldCV()
        X = np.zeros((100, 2))
        with pytest.raises(ValueError, match="groups"):
            list(cv.split(X, groups=None))

    def test_no_overlap_between_train_and_test(self):
        cv = SpatialLatitudeFoldCV()
        X = np.zeros((500, 2))
        latitudes = self._make_latitudes(500)

        for train_idx, test_idx in cv.split(X, groups=latitudes):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Train and test sets must not overlap"

    def test_test_folds_are_exhaustive(self):
        """All samples should appear in exactly one test fold."""
        cv = SpatialLatitudeFoldCV()
        X = np.zeros((500, 2))
        latitudes = self._make_latitudes(500)

        all_test_indices = []
        for _, test_idx in cv.split(X, groups=latitudes):
            all_test_indices.extend(test_idx.tolist())

        assert sorted(all_test_indices) == sorted(range(len(X))), (
            "Every sample must appear in exactly one test fold"
        )

    def test_latitude_bands_respected(self):
        """Test samples must fall within their fold's latitude band."""
        cv = SpatialLatitudeFoldCV()
        X = np.zeros((1000, 2))
        latitudes = np.linspace(-90, 90, 1000)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, groups=latitudes)):
            lat_min, lat_max = SPATIAL_CV_FOLDS[fold_idx]
            test_lats = latitudes[test_idx]
            assert np.all(test_lats >= lat_min), f"Fold {fold_idx}: test lats below lat_min"
            assert np.all(test_lats < lat_max), f"Fold {fold_idx}: test lats above lat_max"

    def test_describe_folds_returns_dataframe(self):
        cv = SpatialLatitudeFoldCV()
        latitudes = self._make_latitudes(100)
        df = cv.describe_folds(latitudes)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "fold" in df.columns
        assert "n_cells" in df.columns


class TestNoRandomSplitsInModelCode:
    """Verify that train_test_split is not used in the models/ directory."""

    def test_no_train_test_split_import(self):
        """Models should never import train_test_split."""
        import ast
        import os

        models_dir = "models"
        forbidden = "train_test_split"
        violations = []

        for fname in os.listdir(models_dir):
            if fname.endswith(".py"):
                with open(os.path.join(models_dir, fname), encoding="utf-8") as f:
                    source = f.read()
                if forbidden in source:
                    violations.append(fname)

        assert not violations, (
            f"CRITICAL: train_test_split found in models/: {violations}. "
            "Random splits are forbidden — use SpatialLatitudeFoldCV. "
            "See CLAUDE.md."
        )
