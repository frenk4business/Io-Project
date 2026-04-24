"""
scripts/run_scientific_analysis.py
-----------------------------------
Run the full scientific analysis pipeline and save all artifacts to data/results/.

Usage:
    python scripts/run_scientific_analysis.py
    python scripts/run_scientific_analysis.py --fast    # fewer MC sims / bootstrap

Output artifacts
----------------
    data/results/ablation.json
    data/results/geology_enrichment.csv
    data/results/geology_enrichment.png
    data/results/spatial_stats.npz
    data/results/spatial_stats.png
    data/results/asymmetry_tests.csv
    data/results/asymmetry.png
    data/results/coverage_bias.csv
    data/results/coverage_bias.png
    data/results/validation/summary.json
    data/results/validation/calibration_fold{0..3}.png
    data/results/validation/pr_fold{0..3}.png
    data/results/power_regression.json
    data/results/power_residuals.csv
    data/results/power_by_latitude.csv
    data/results/power_by_geology.csv
    data/results/power_outlier_sensitivity.csv

This script writes only to data/results/. It does not modify:
    data/raw/, data/processed/, data/models/,
    the existing notebooks, or any source modules.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_scientific_analysis")


def _separator(title: str) -> None:
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def main(fast: bool = False) -> None:
    n_sims = 99 if fast else 199
    n_boot = 200 if fast else 500

    t0 = time.time()
    _separator("Loading data")

    from features.build import load_feature_matrix
    from ingest.hotspot_catalog import load_hotspot_catalog

    fm = load_feature_matrix()
    catalog = load_hotspot_catalog()
    logger.info(
        "Feature matrix: %d cells, %d positive. Catalog: %d hotspots.",
        len(fm), int(fm["has_hotspot"].sum()), len(catalog),
    )

    # ── 1. Leakage audit ─────────────────────────────────────────────────
    _separator("1 / 8 — Leakage audit")
    from analysis.leakage_audit import audit_feature_leakage
    leakage = audit_feature_leakage(fm)
    n_flagged = int(leakage["suspected_leakage"].sum())
    print(leakage.to_string(index=False))
    logger.info("%d feature(s) flagged as suspected leakage.", n_flagged)

    # ── 2. Ablation ───────────────────────────────────────────────────────
    _separator("2 / 8 — Ablation study")
    from models.ablation import run_ablation, save_ablation
    ablation = run_ablation(fm)
    save_ablation(ablation)
    for s in ablation["feature_sets"]:
        summ = s["summary"]
        tag = " [LEAKY]" if s.get("is_leaky_baseline") else (
            " [NULL]" if s.get("is_null_sanity_check") else ""
        )
        logger.info(
            "  %-14s AUC=%s±%s  PR-AUC=%s±%s  F1=%s±%s%s",
            s["name"],
            f"{summ['auc_roc']['mean']:.3f}", f"{summ['auc_roc']['std']:.3f}",
            f"{summ['pr_auc']['mean']:.3f}", f"{summ['pr_auc']['std']:.3f}",
            f"{summ['f1']['mean']:.3f}", f"{summ['f1']['std']:.3f}",
            tag,
        )

    # ── 3. Geology enrichment ─────────────────────────────────────────────
    _separator("3 / 8 — Geology enrichment")
    from analysis.geology_enrichment import (
        compute_geology_enrichment, plot_enrichment_bar, save_enrichment,
    )
    enr = compute_geology_enrichment(fm)
    save_enrichment(enr)
    plot_enrichment_bar(enr)
    n_sig = int(enr["significant"].sum())
    logger.info(
        "%d of %d units are Bonferroni-significant (p_bonf < 0.05).",
        n_sig, len(enr),
    )
    if n_sig > 0:
        logger.info(
            "  Significant units: %s",
            ", ".join(enr.loc[enr["significant"], "unit"].tolist()),
        )

    # ── 4. Spatial statistics ─────────────────────────────────────────────
    _separator(f"4 / 8 — Spatial statistics ({n_sims} MC sims)")
    from analysis.spatial_stats import compute_spatial_stats, plot_spatial_stats, save_spatial_stats
    sp = compute_spatial_stats(catalog, n_sims=n_sims)
    save_spatial_stats(sp)
    plot_spatial_stats(sp)
    logger.info(
        "Clustering at small scales: %s. n_points=%d, n_sims=%d.",
        sp["clustered_at_small_radii"], sp["n_points"], sp["n_sims"],
    )

    # ── 5. Hemispheric asymmetry ──────────────────────────────────────────
    _separator("5 / 8 — Hemispheric asymmetry")
    from analysis.asymmetry import compute_asymmetry, plot_asymmetry, save_asymmetry_tests
    asym = compute_asymmetry(catalog, n_boot=n_boot)
    save_asymmetry_tests(asym)
    plot_asymmetry(asym)
    for t in asym["binomial_tests"]:
        logger.info("  %-55s p=%.4f  %s", t["comparison"], t["p_binom"], t["interpretation"])

    # ── 6. Coverage bias ──────────────────────────────────────────────────
    _separator("6 / 8 — Coverage bias")
    from analysis.coverage_bias import compute_coverage_bias, plot_coverage_bias, save_coverage_bias
    cov = compute_coverage_bias(fm)
    save_coverage_bias(cov)
    plot_coverage_bias(cov)
    if not cov.empty:
        max_ratio = cov["rate_ratio"].max()
        max_band = cov.loc[cov["rate_ratio"].idxmax(), "lat_band"]
        logger.info(
            "Largest coverage-adjusted rate ratio: %.2f in %s",
            max_ratio, max_band,
        )

    # ── 7. Tidal hypothesis comparison ────────────────────────────────────
    _separator("7 / 8 — Tidal hypothesis comparison (Poisson log-likelihood)")
    from analysis.hypothesis_comparison import compare_hypotheses, save_hypothesis_comparison
    hyp = compare_hypotheses(fm, include_geology=True)
    save_hypothesis_comparison(hyp)
    best = hyp.iloc[0]
    logger.info(
        "Best template: %-20s  pseudo-R²=%.4f  LR-p=%.4f",
        best["template"], best["pseudo_r2"], best["lr_p_value"],
    )
    for _, row in hyp.iterrows():
        logger.info(
            "  %-22s  log_lik=%10.1f  ΔAIC=%7.2f  pseudo-R²=%.4f",
            row["template"], row["log_lik"], row["delta_aic"], row["pseudo_r2"],
        )

    # ── 8. Catalogue jackknife stability ──────────────────────────────────
    _separator("8 / 8 — Catalogue jackknife stability")
    from analysis.catalog_stability import catalog_jackknife, save_stability
    n_rep = 50 if fast else 100
    stab = catalog_jackknife(
        fm, catalog,
        retention_fractions=(0.9, 0.75, 0.5),
        n_replicates=n_rep,
        seed=42,
    )
    save_stability(stab)
    fragile = stab[stab["survival_rate"] < 0.5]
    if not fragile.empty:
        logger.warning("FRAGILE CLAIMS (survival < 50%% at some retention level):")
        for _, row in fragile.iterrows():
            logger.warning(
                "  retention=%.0f%%  %-35s  survival=%.0f%%",
                row["fraction_retained"] * 100, row["metric"],
                row["survival_rate"] * 100,
            )
    else:
        logger.info(
            "All tested claims survive ≥ 50%% of jackknife replicates at all retention levels."
        )

    # ── Thermal intensity analysis ────────────────────────────────────────
    _separator("Bonus — Thermal intensity analysis (Davies/JIRAM proxy)")
    try:
        from ingest.power_catalog import load_power_catalog
        from preprocess.grid import load_base_grid
        from preprocess.power_grid import assign_power_to_grid, save_power_grid
        from models.regression import save_power_regression, train_power_regression
        from analysis.power_intensity import (
            compute_power_intensity_suite,
            save_power_intensity_suite,
        )

        power_catalog = load_power_catalog()
        power_grid = assign_power_to_grid(load_base_grid(), power_catalog)
        save_power_grid(power_grid)

        power_suite = compute_power_intensity_suite(fm, power_grid)
        save_power_intensity_suite(power_suite)

        power_reg = train_power_regression(fm, power_grid)
        save_power_regression(power_reg)
        logger.info(
            "Estimated thermal-emission proxy regression: OOF R2=%.3f, "
            "RMSE=%.3f, Spearman=%.3f over %d cells.",
            power_reg["overall_oof"]["r2"],
            power_reg["overall_oof"]["rmse"],
            power_reg["overall_oof"]["spearman"],
            power_reg["n_observed_power_cells"],
        )
    except Exception as exc:
        logger.warning("Thermal intensity analysis failed: %s", exc)

    # ── Validation metrics ────────────────────────────────────────────────
    _separator("Bonus — Validation metrics (calibration, PR, Brier, MCC)")
    try:
        from models.validation_metrics import compute_validation_metrics
        compute_validation_metrics(fm)
        logger.info("Validation figures saved to data/results/validation/")
    except Exception as exc:
        logger.warning("Validation metrics failed: %s", exc)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _separator(f"Done in {elapsed:.1f}s")
    logger.info(
        "All artifacts written to data/results/. "
        "Review leakage audit and no_leakage ablation results before drawing "
        "scientific conclusions."
    )

    # Print honest scientific status
    print()
    print("=" * 70)
    print("SCIENTIFIC STATUS SUMMARY")
    print("=" * 70)
    no_leak = next(
        (s for s in ablation["feature_sets"] if s["name"] == "no_leakage"), None
    )
    if no_leak:
        auc = no_leak["summary"]["auc_roc"]["mean"]
        auc_std = no_leak["summary"]["auc_roc"]["std"]
        print(f"Honest baseline AUC-ROC (no leakage): {auc:.3f} ± {auc_std:.3f}")
        if auc < 0.7:
            print("  → Low: geology + synthetic tidal carry minimal signal.")
        elif auc < 0.85:
            print("  → Moderate: some spatial structure beyond CSR. "
                  "Caution: synthetic tidal proxy inflates this.")
        else:
            print("  → High: strong signal even without leakage. "
                  "Investigate which features drive this.")

    print(f"Geology units with Bonferroni-significant association: {n_sig}")
    print(f"Spatial clustering at small scales: {sp['clustered_at_small_radii']}")
    for t in asym["binomial_tests"]:
        print(f"  {t['comparison'][:55]:55s} p={t['p_binom']:.4f}")
    print()
    print("WHAT THIS PROJECT CANNOT YET CLAIM:")
    print("  - Tidal heating drives distribution (synthetic proxy)")
    print("  - M3 or M4 model is supported (no real dissipation grids)")
    print("  - Southern hemisphere is intrinsically less active (coverage bias)")
    print("  - Any hotspot will erupt (no temporal model)")
    print("  - True bolometric radiant power follows any pattern")
    print("    (current power_gw is an estimated Davies/JIRAM 4.8 micron proxy)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Io hotspot scientific analysis pipeline."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer MC sims (99) and bootstrap iterations (200) for speed.",
    )
    args = parser.parse_args()
    main(fast=args.fast)
