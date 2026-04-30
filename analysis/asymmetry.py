"""
analysis/asymmetry.py
---------------------
Hemispheric and longitudinal asymmetry tests on the Io hotspot catalogue.

Three two-sided exact binomial tests (H0: equal split between two regions):
  1. Sub-Jovian (|lon| < 90°) vs anti-Jovian (|lon| >= 90°)
  2. Leading (0° < lon <= 180°) vs trailing (-180° < lon <= 0°)
  3. Northern (lat > 0°) vs southern (lat <= 0°)

Longitudinal (36 × 10° bins) and latitudinal (18 × 10° bins) density histograms
with bootstrap 95% CIs (n_boot=1000 by default).

Reference model predictions (noted for context; not directly tested without
published dissipation grids):
  - Segatz M3 (asthenosphere): maximum dissipation at sub-Jovian and anti-Jovian
    equatorial points → expect enrichment near lon ≈ 0° and lon ≈ ±180°
  - Segatz M4 (deep mantle): maximum dissipation at poles → expect poleward enrichment
  - Tyler et al. 2015 (magma ocean): sub-Jovian equatorial enhancement

We do NOT claim this catalogue analysis proves any interior model. We note which
pattern the observed data most resembles, with explicit coverage-bias caveats.

See docs/scientific_methods.md §Hemispheric and longitudinal asymmetry for
full methodology.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from config import IO_RADIUS_KM, PROCESSED_DIR

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
ASYMMETRY_PNG: Path = RESULTS_DIR / "asymmetry.png"
ASYMMETRY_CSV: Path = RESULTS_DIR / "asymmetry_tests.csv"

# Tidal-heating model reference patterns (qualitative markers for plots)
# lon_markers: list of (lon_deg, label) for vertical lines on longitudinal histogram
TIDAL_MODEL_LON_MARKERS: dict[str, list[tuple[float, str]]] = {
    "M3 / asthenosphere (Segatz)": [
        (0.0, "M3 max (sub-Jov)"),
        (180.0, "M3 max (anti-Jov)"),
        (-180.0, "M3 max (anti-Jov)"),
    ],
    "M4 / deep mantle (Segatz)": [],  # polar → no lon preference
}
# lat_markers: list of (lat_deg, label) for vertical lines on latitudinal histogram
TIDAL_MODEL_LAT_MARKERS: dict[str, list[tuple[float, str]]] = {
    "M3 / asthenosphere (Segatz)": [(0.0, "M3 equatorial max")],
    "M4 / deep mantle (Segatz)": [(90.0, "M4 polar max"), (-90.0, "M4 polar max")],
}

LON_BIN_EDGES: np.ndarray = np.linspace(-180.0, 180.0, 37)   # 36 × 10° bins
LAT_BIN_EDGES: np.ndarray = np.linspace(-90.0, 90.0, 19)     # 18 × 10° bins


def _binomial_test_result(n_a: int, n_b: int, label_a: str, label_b: str) -> dict:
    """Two-sided binomial test: H0 = P(A) = 0.5.

    Args:
        n_a: Count in region A.
        n_b: Count in region B.
        label_a: Name of region A.
        label_b: Name of region B.

    Returns:
        Dict with comparison, n_a, n_b, n_total, fraction_a, p_binom,
        ci_lo, ci_hi (95% Clopper–Pearson on fraction_a), interpretation.
    """
    n_total = n_a + n_b
    if n_total == 0:
        return {
            "comparison": f"{label_a} vs {label_b}",
            "n_a": 0, "n_b": 0, "n_total": 0, "fraction_a": float("nan"),
            "p_binom": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
            "interpretation": "no data",
        }

    result = binomtest(n_a, n_total, p=0.5, alternative="two-sided")
    p = result.pvalue

    # 95% Clopper–Pearson CI on fraction_a
    ci = result.proportion_ci(confidence_level=0.95, method="exact")

    if p < 0.05:
        dominant = label_a if n_a > n_b else label_b
        interp = f"Significant asymmetry (p={p:.3f}): {dominant} dominant"
    else:
        interp = f"No significant asymmetry (p={p:.3f})"

    return {
        "comparison": f"{label_a} vs {label_b}",
        "n_a": n_a,
        "n_b": n_b,
        "n_total": n_total,
        "fraction_a": n_a / n_total,
        "p_binom": p,
        "ci_lo": float(ci.low),
        "ci_hi": float(ci.high),
        "interpretation": interp,
    }


def _bootstrap_histogram(
    values: np.ndarray,
    bin_edges: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap 95% CI for a histogram of values.

    Returns:
        (bin_centres, observed_counts, ci_lo, ci_hi)
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    counts_obs, _ = np.histogram(values, bins=bin_edges)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if n_boot <= 0:
        counts = counts_obs.astype(float)
        return bin_centres, counts, counts.copy(), counts.copy()

    boot_counts = np.empty((n_boot, len(bin_centres)), dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_counts[i], _ = np.histogram(sample, bins=bin_edges)

    ci_lo = np.quantile(boot_counts, 0.025, axis=0)
    ci_hi = np.quantile(boot_counts, 0.975, axis=0)
    return bin_centres, counts_obs.astype(float), ci_lo, ci_hi


def compute_asymmetry(
    catalog: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute all hemispheric and longitudinal asymmetry results.

    Args:
        catalog: Hotspot catalog DataFrame with columns longitude, latitude.
        n_boot: Number of bootstrap samples for histogram CIs.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
          - binomial_tests: list of test-result dicts (one per comparison)
          - lon_histogram: dict with bin_centres, counts, ci_lo, ci_hi
          - lat_histogram: dict with bin_centres, counts, ci_lo, ci_hi
          - n_hotspots: total catalog size
          - coverage_caveat: string warning
    """
    lons = catalog["longitude"].astype(float).values
    lats = catalog["latitude"].astype(float).values
    n = len(lons)

    # --- Binomial tests ---
    # 1. Sub-Jovian (|lon| < 90) vs anti-Jovian (|lon| >= 90)
    n_subjov = int((np.abs(lons) < 90.0).sum())
    n_antijov = int((np.abs(lons) >= 90.0).sum())

    # 2. Leading (0 < lon <= 180) vs trailing (-180 < lon <= 0)
    n_leading = int(((lons > 0.0) & (lons <= 180.0)).sum())
    n_trailing = int(((lons <= 0.0) & (lons > -180.0)).sum())

    # 3. Northern (lat > 0) vs southern (lat <= 0)
    n_north = int((lats > 0.0).sum())
    n_south = int((lats <= 0.0).sum())

    tests = [
        _binomial_test_result(n_subjov, n_antijov, "Sub-Jovian (|lon|<90°)", "Anti-Jovian (|lon|≥90°)"),
        _binomial_test_result(n_leading, n_trailing, "Leading (0°<lon≤180°)", "Trailing (-180°<lon≤0°)"),
        _binomial_test_result(n_north, n_south, "Northern (lat>0°)", "Southern (lat≤0°)"),
    ]

    # --- Histograms with bootstrap CIs ---
    lon_centres, lon_counts, lon_lo, lon_hi = _bootstrap_histogram(
        lons, LON_BIN_EDGES, n_boot=n_boot, seed=seed
    )
    lat_centres, lat_counts, lat_lo, lat_hi = _bootstrap_histogram(
        lats, LAT_BIN_EDGES, n_boot=n_boot, seed=seed
    )

    logger.info(
        "Asymmetry: n=%d | Sub-Jovian=%d Anti-Jovian=%d | North=%d South=%d",
        n, n_subjov, n_antijov, n_north, n_south,
    )

    return {
        "binomial_tests": tests,
        "lon_histogram": {
            "bin_centres": lon_centres,
            "counts": lon_counts,
            "ci_lo": lon_lo,
            "ci_hi": lon_hi,
            "bin_edges": LON_BIN_EDGES,
        },
        "lat_histogram": {
            "bin_centres": lat_centres,
            "counts": lat_counts,
            "ci_lo": lat_lo,
            "ci_hi": lat_hi,
            "bin_edges": LAT_BIN_EDGES,
        },
        "n_hotspots": n,
        "coverage_caveat": (
            "Distribution reflects spacecraft observational coverage (primarily Galileo "
            "and Voyager), not the true planetary hotspot distribution. The southern "
            "hemisphere (lat < −45°) is systematically undersampled. Asymmetries must "
            "be interpreted as properties of the catalogue, not necessarily of Io."
        ),
    }


def plot_asymmetry(results: dict, out_path: Path | None = None) -> Path:
    """Three-panel figure: longitudinal histogram; latitudinal histogram; test summary.

    Args:
        results: Output from compute_asymmetry().
        out_path: Output path for PNG. Defaults to ASYMMETRY_PNG.

    Returns:
        Path to saved figure.
    """
    out_path = out_path or ASYMMETRY_PNG
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lon_h = results["lon_histogram"]
    lat_h = results["lat_histogram"]
    tests = results["binomial_tests"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Io hotspot asymmetry analysis (n = {results['n_hotspots']})\n"
        "Caution: reflects observational coverage, not true planetary distribution",
        fontsize=10,
    )

    # --- Panel 1: Longitudinal histogram ---
    ax = axes[0]
    ax.bar(
        lon_h["bin_centres"], lon_h["counts"],
        width=10.0, color="C0", alpha=0.7, label="Observed",
    )
    ax.fill_between(
        lon_h["bin_centres"], lon_h["ci_lo"], lon_h["ci_hi"],
        alpha=0.3, color="C0", label="95% bootstrap CI",
    )
    for model, markers in TIDAL_MODEL_LON_MARKERS.items():
        for lon_m, lbl in markers:
            ax.axvline(lon_m, linestyle="--", lw=1.2, label=lbl if lbl not in [
                x for _, x in TIDAL_MODEL_LON_MARKERS.get(model, [])[:markers.index((lon_m, lbl))]
            ] else "")
    ax.axvline(-90, linestyle=":", color="grey", lw=0.8, label="±90° boundary")
    ax.axvline(90, linestyle=":", color="grey", lw=0.8)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Hotspot count per 10° bin")
    ax.set_title("Longitudinal distribution")
    ax.set_xlim(-180, 180)
    ax.legend(fontsize=7, loc="upper right")

    # --- Panel 2: Latitudinal histogram ---
    ax = axes[1]
    ax.bar(
        lat_h["bin_centres"], lat_h["counts"],
        width=10.0, color="C1", alpha=0.7, label="Observed",
    )
    ax.fill_between(
        lat_h["bin_centres"], lat_h["ci_lo"], lat_h["ci_hi"],
        alpha=0.3, color="C1", label="95% bootstrap CI",
    )
    for model, markers in TIDAL_MODEL_LAT_MARKERS.items():
        for lat_m, lbl in markers:
            ax.axvline(lat_m, linestyle="--", lw=1.2, label=lbl)
    ax.axvline(0, linestyle=":", color="grey", lw=0.8, label="Equator")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Hotspot count per 10° bin")
    ax.set_title("Latitudinal distribution")
    ax.set_xlim(-90, 90)
    ax.legend(fontsize=7, loc="upper right")

    # --- Panel 3: Binomial test summary table ---
    ax = axes[2]
    ax.axis("off")
    rows = []
    for t in tests:
        p = t["p_binom"]
        rows.append([
            t["comparison"].replace(" vs ", "\nvs "),
            f"{t['n_a']} / {t['n_b']}",
            f"{t['fraction_a']:.2f}",
            f"{t['p_binom']:.3f}{'*' if p < 0.05 else ''}",
        ])
    col_labels = ["Comparison", "n_A / n_B", "Fraction A", "p (binomial)"]
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 2.0)
    # Colour rows with significant p-value
    for row_i, t in enumerate(tests, start=1):
        if t["p_binom"] < 0.05:
            for col_i in range(len(col_labels)):
                tbl[row_i, col_i].set_facecolor("#ffcccc")
    ax.set_title("Binomial tests (* p < 0.05)", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved asymmetry figure to %s", out_path)
    return out_path


def save_asymmetry_tests(results: dict, path: Path | None = None) -> Path:
    """Persist binomial test results to CSV."""
    path = path or ASYMMETRY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results["binomial_tests"])
    df.to_csv(path, index=False)
    logger.info("Saved asymmetry tests to %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ingest.hotspot_catalog import load_hotspot_catalog

    cat = load_hotspot_catalog()
    res = compute_asymmetry(cat)
    save_asymmetry_tests(res)
    plot_asymmetry(res)
    for t in res["binomial_tests"]:
        print(f"{t['comparison']:50s}  p={t['p_binom']:.4f}  {t['interpretation']}")
