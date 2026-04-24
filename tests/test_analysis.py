"""
tests/test_analysis.py
-----------------------
Contract tests for the scientific analysis layer.

All tests use synthetic fixtures — no real data files are required.
Tests verify output shape, column presence, valid value ranges, and
basic statistical sanity checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_feature_matrix() -> pd.DataFrame:
    """Minimal feature matrix with 100 cells: 5 hotspot, 95 non-hotspot."""
    rng = np.random.default_rng(0)
    n = 100
    lons = rng.uniform(-180, 180, n)
    lats = rng.uniform(-90, 90, n)
    has_hotspot = np.zeros(n, dtype=int)
    has_hotspot[:5] = 1  # 5 positives
    units = rng.choice(["Fb", "Fd", "Fu", "Prb", "UNKNOWN"], size=n)
    return pd.DataFrame(
        {
            "cell_id": np.arange(n),
            "lon_centre": lons,
            "lat_centre": lats,
            "has_hotspot": has_hotspot,
            "geology_unit": units,
            "geology_encoded": rng.integers(0, 5, n),
            "tidal_heating_flux": rng.uniform(0, 1, n),
            "dist_nearest_hotspot_km": rng.uniform(0, 5000, n),
            "dist_tidal_stress_max_km": rng.uniform(0, 5000, n),
        }
    )


@pytest.fixture
def tiny_catalog() -> pd.DataFrame:
    """Minimal hotspot catalog with 20 hotspots."""
    rng = np.random.default_rng(1)
    n = 20
    return pd.DataFrame(
        {
            "name": [f"hs_{i}" for i in range(n)],
            "longitude": rng.uniform(-180, 180, n),
            "latitude": rng.uniform(-90, 90, n),
        }
    )


@pytest.fixture
def equal_split_catalog() -> pd.DataFrame:
    """Catalog split exactly 10/10 between sub-Jovian and anti-Jovian halves."""
    lons = np.array(
        [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 89.0,   # sub-Jov |lon|<90
         95.0, 100.0, 120.0, 140.0, 150.0, 160.0, 170.0, -170.0, -150.0, -100.0]  # anti-Jov
    )
    lats = np.zeros(len(lons))
    return pd.DataFrame({"name": [f"h{i}" for i in range(len(lons))],
                         "longitude": lons, "latitude": lats})


# ---------------------------------------------------------------------------
# 1. Leakage audit
# ---------------------------------------------------------------------------

def test_leakage_audit_columns(tiny_feature_matrix):
    from analysis.leakage_audit import audit_feature_leakage
    result = audit_feature_leakage(tiny_feature_matrix)

    expected_cols = {
        "feature", "derivation_source", "pearson_r", "spearman_r",
        "lr_coef", "abs_coef", "suspected_leakage", "flag_reason",
    }
    assert expected_cols.issubset(result.columns), (
        f"Missing columns: {expected_cols - set(result.columns)}"
    )
    assert len(result) == 4  # 4 feature columns in fixture
    assert result["suspected_leakage"].dtype == bool


def test_leakage_audit_dist_flagged(tiny_feature_matrix):
    """dist_nearest_hotspot_km must always be flagged as target-derived."""
    from analysis.leakage_audit import audit_feature_leakage
    result = audit_feature_leakage(tiny_feature_matrix)
    row = result[result["feature"] == "dist_nearest_hotspot_km"]
    assert len(row) == 1
    assert row["suspected_leakage"].iloc[0], (
        "dist_nearest_hotspot_km must be flagged as target-derived"
    )


# ---------------------------------------------------------------------------
# 2. Geology enrichment — p-values must be valid probabilities
# ---------------------------------------------------------------------------

def test_enrichment_p_values_valid(tiny_feature_matrix):
    from analysis.geology_enrichment import compute_geology_enrichment
    df = compute_geology_enrichment(tiny_feature_matrix)

    assert "p_bonf" in df.columns
    assert "enrichment_ratio" in df.columns
    assert "significant" in df.columns
    assert "wilson_lo" in df.columns
    assert "wilson_hi" in df.columns

    valid_p = df["p_bonf"].dropna()
    assert (valid_p >= 0.0).all(), "p_bonf must be >= 0"
    assert (valid_p <= 1.0).all(), "p_bonf must be <= 1"


def test_enrichment_returns_one_row_per_unit(tiny_feature_matrix):
    from analysis.geology_enrichment import compute_geology_enrichment
    df = compute_geology_enrichment(tiny_feature_matrix)
    n_unique = tiny_feature_matrix["geology_unit"].nunique()
    assert len(df) == n_unique, (
        f"Expected {n_unique} rows, got {len(df)}"
    )


# ---------------------------------------------------------------------------
# 3. Spatial statistics — uniform-on-sphere K must be inside CSR envelope
# ---------------------------------------------------------------------------

def test_spatial_stats_uniform_inside_csr():
    """K for uniform points should fall inside the 95% CSR envelope."""
    from analysis.spatial_stats import (
        compute_spatial_stats,
        DEFAULT_RADII_KM,
    )
    # Generate uniform-on-sphere sample of 30 points
    rng = np.random.default_rng(99)
    n = 30
    u = rng.uniform(0, 1, n)
    v = rng.uniform(0, 1, n)
    lats = np.degrees(np.arcsin(2 * u - 1))
    lons = 360.0 * v - 180.0
    cat = pd.DataFrame({"latitude": lats, "longitude": lons})

    # Use small radii grid and few sims to keep test fast
    radii = DEFAULT_RADII_KM[:10]
    results = compute_spatial_stats(cat, radii_km=radii, n_sims=49, seed=42)

    k_obs = results["k_observed"]
    k_lo = results["k_env_lo"]
    k_hi = results["k_env_hi"]

    # Expect K to be inside envelope for most radii (allow a few exceedances)
    inside = ((k_obs >= k_lo) & (k_obs <= k_hi)).sum()
    assert inside >= len(radii) * 0.6, (
        f"Uniform sample K should mostly be inside CSR envelope. "
        f"Inside: {inside}/{len(radii)}"
    )


# ---------------------------------------------------------------------------
# 4. Asymmetry — equal-split catalog should not be significant
# ---------------------------------------------------------------------------

def test_asymmetry_binomial_equal_split_not_significant(equal_split_catalog):
    from analysis.asymmetry import compute_asymmetry
    results = compute_asymmetry(equal_split_catalog, n_boot=50, seed=42)
    tests = results["binomial_tests"]

    # Sub-Jovian vs anti-Jovian: 10 vs 10 → p should be 1.0
    subj_test = next(t for t in tests if "Sub-Jovian" in t["comparison"])
    assert subj_test["p_binom"] > 0.05, (
        f"Equal split should not be significant: p={subj_test['p_binom']:.4f}"
    )


def test_asymmetry_total_counts_match_catalog(tiny_catalog):
    """Total counts in each binomial test must sum to catalog size."""
    from analysis.asymmetry import compute_asymmetry
    results = compute_asymmetry(tiny_catalog, n_boot=50, seed=42)
    n = len(tiny_catalog)
    for t in results["binomial_tests"]:
        assert t["n_total"] == n, (
            f"Test '{t['comparison']}': n_total={t['n_total']}, expected {n}"
        )


def test_asymmetry_histogram_bin_counts_sum_to_catalog(tiny_catalog):
    """Histogram bin counts must sum to catalog size."""
    from analysis.asymmetry import compute_asymmetry
    results = compute_asymmetry(tiny_catalog, n_boot=50, seed=42)
    n = len(tiny_catalog)

    lon_total = int(results["lon_histogram"]["counts"].sum())
    lat_total = int(results["lat_histogram"]["counts"].sum())
    assert lon_total == n, f"Lon histogram total {lon_total} != {n}"
    assert lat_total == n, f"Lat histogram total {lat_total} != {n}"


# ---------------------------------------------------------------------------
# 5. Coverage bias — all rates must be non-negative
# ---------------------------------------------------------------------------

def test_coverage_bias_rates_nonneg(tiny_feature_matrix):
    from analysis.coverage_bias import compute_coverage_bias
    df = compute_coverage_bias(tiny_feature_matrix)

    assert not df.empty
    assert "lat_band" in df.columns
    assert len(df) == 4, f"Expected 4 latitude bands, got {len(df)}"

    for col in ["raw_rate", "adjusted_rate", "unobserved_pct"]:
        valid = df[col].dropna()
        assert (valid >= 0.0).all(), f"Column '{col}' has negative values"


def test_coverage_bias_observed_le_total(tiny_feature_matrix):
    from analysis.coverage_bias import compute_coverage_bias
    df = compute_coverage_bias(tiny_feature_matrix)
    assert (df["observed_cells"] <= df["total_cells"]).all(), (
        "observed_cells must be <= total_cells"
    )


# ---------------------------------------------------------------------------
# 6. Uncertainty module — fold_variability_summary
# ---------------------------------------------------------------------------

def test_fold_variability_summary_structure():
    from analysis.uncertainty import fold_variability_summary

    # Minimal mock ablation dict
    mock_ablation = {
        "feature_sets": [
            {
                "name": "no_leakage",
                "is_leaky_baseline": False,
                "is_null_sanity_check": False,
                "summary": {
                    "auc_roc": {"mean": 0.75, "std": 0.05},
                    "pr_auc": {"mean": 0.12, "std": 0.03},
                    "f1": {"mean": 0.20, "std": 0.04},
                    "precision": {"mean": 0.15, "std": 0.02},
                    "recall": {"mean": 0.30, "std": 0.06},
                },
            },
            {
                "name": "null",
                "is_leaky_baseline": False,
                "is_null_sanity_check": True,
                "summary": {
                    "auc_roc": {"mean": 0.50, "std": 0.02},
                    "pr_auc": {"mean": 0.003, "std": 0.001},
                    "f1": {"mean": 0.005, "std": 0.001},
                    "precision": {"mean": 0.003, "std": 0.001},
                    "recall": {"mean": 0.01, "std": 0.005},
                },
            },
        ]
    }

    df = fold_variability_summary(mock_ablation)
    assert len(df) == 2
    assert "auc_mean" in df.columns
    assert "auc_std" in df.columns
    null_row = df[df["name"] == "null"].iloc[0]
    assert abs(null_row["auc_mean"] - 0.50) < 1e-6


# ---------------------------------------------------------------------------
# 7. Tidal templates — shape invariants
# ---------------------------------------------------------------------------

def test_tidal_templates_nonneg(tiny_feature_matrix):
    """All template values must be non-negative at any grid location."""
    from analysis.tidal_templates import evaluate_all_templates
    lats = tiny_feature_matrix["lat_centre"].to_numpy()
    lons = tiny_feature_matrix["lon_centre"].to_numpy()
    df = evaluate_all_templates(lats, lons)
    assert (df >= 0).all().all(), "All template values must be >= 0"


def test_tidal_templates_unit_mean(tiny_feature_matrix):
    """All templates must have area-weighted mean ≈ 1.0 after normalisation."""
    from analysis.tidal_templates import evaluate_all_templates, TEMPLATE_NAMES
    lats = tiny_feature_matrix["lat_centre"].to_numpy()
    lons = tiny_feature_matrix["lon_centre"].to_numpy()
    df = evaluate_all_templates(lats, lons)
    weights = np.cos(np.radians(lats))
    for name in TEMPLATE_NAMES:
        mean = np.sum(df[name].to_numpy() * weights) / np.sum(weights)
        assert abs(mean - 1.0) < 0.05, f"'{name}' area-weighted mean = {mean:.4f}, expected ≈ 1.0"


def test_asthenosphere_peaks_equatorial():
    """Asthenosphere template must be higher near equator than near poles."""
    from analysis.tidal_templates import evaluate_template
    lats = np.array([-80.0, -60.0, 0.0, 0.0, 60.0, 80.0])
    lons = np.array([0.0,   0.0,  0.0, 180.0, 0.0,  0.0])
    h = evaluate_template("asthenosphere", lats, lons)
    equatorial_mean = (h[2] + h[3]) / 2.0
    polar_mean = (h[0] + h[1] + h[4] + h[5]) / 4.0
    assert equatorial_mean > polar_mean, (
        f"Asthenosphere equatorial mean {equatorial_mean:.3f} should exceed "
        f"polar mean {polar_mean:.3f}"
    )


def test_deep_mantle_peaks_polar():
    """Deep-mantle template must be higher near poles than near equator."""
    from analysis.tidal_templates import evaluate_template
    lats = np.array([-80.0, 0.0, 80.0])
    lons = np.array([0.0,   0.0,  0.0])
    h = evaluate_template("deep_mantle", lats, lons)
    # Both poles must beat the equator
    assert h[0] > h[1] and h[2] > h[1], (
        f"Deep-mantle pole values ({h[0]:.3f}, {h[2]:.3f}) must exceed "
        f"equatorial value ({h[1]:.3f})"
    )


# ---------------------------------------------------------------------------
# 8. Hypothesis comparison — output structure
# ---------------------------------------------------------------------------

def test_hypothesis_comparison_columns(tiny_feature_matrix):
    """compare_hypotheses must return required columns and ΔAIC min = 0."""
    from analysis.hypothesis_comparison import compare_hypotheses
    result = compare_hypotheses(tiny_feature_matrix, include_geology=True)
    required = {"template", "log_lik", "aic", "delta_aic", "pseudo_r2",
                "lr_chi2", "lr_p_value"}
    assert required.issubset(result.columns), (
        f"Missing columns: {required - set(result.columns)}"
    )
    assert result["delta_aic"].min() == pytest.approx(0.0, abs=1e-9), (
        "Best template must have delta_aic = 0"
    )


def test_hypothesis_comparison_geology_best(tiny_feature_matrix):
    """In-sample geology template is fit to the labels so must have lowest AIC."""
    from analysis.hypothesis_comparison import compare_hypotheses
    result = compare_hypotheses(tiny_feature_matrix, include_geology=True)
    best_template = result.iloc[0]["template"]
    assert best_template == "geology_insample", (
        f"In-sample geology template should rank first; got '{best_template}'"
    )


# ---------------------------------------------------------------------------
# 9. Catalogue stability — output structure and range
# ---------------------------------------------------------------------------

def test_catalog_stability_output_columns(tiny_feature_matrix, tiny_catalog):
    """catalog_jackknife must return required columns with valid survival rates."""
    from analysis.catalog_stability import catalog_jackknife
    df = catalog_jackknife(
        tiny_feature_matrix, tiny_catalog,
        retention_fractions=(0.9,), n_replicates=5, seed=0,
    )
    for col in ("fraction_retained", "metric", "survival_rate", "summary"):
        assert col in df.columns, f"Missing column: '{col}'"
    assert (df["survival_rate"] >= 0.0).all(), "survival_rate must be >= 0"
    assert (df["survival_rate"] <= 1.0).all(), "survival_rate must be <= 1"


def test_catalog_stability_survival_rate_bounds(tiny_feature_matrix, tiny_catalog):
    """All survival_rate values must lie in [0, 1] at multiple retention levels."""
    from analysis.catalog_stability import catalog_jackknife
    df = catalog_jackknife(
        tiny_feature_matrix, tiny_catalog,
        retention_fractions=(0.9, 0.75), n_replicates=10, seed=1,
    )
    assert df["survival_rate"].between(0.0, 1.0).all(), (
        "survival_rate out of [0, 1] range"
    )
