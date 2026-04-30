"""
analysis/catalog_stability.py
-----------------------------
Jackknife / random-deletion sensitivity analysis for catalogue-level
statistical claims.

Purpose
-------
The USGS hotspot catalogue has finite size (~172 entries) and uneven
detection history. Any published claim derived from it — "unit Pfd is
Bonferroni-significantly enriched", "sub-Jovian contains X% of hotspots",
"small-scale clustering is detected at radius r km" — should be robust to
realistic catalogue perturbations: a few entries missing, a few merged,
a few mis-localised.

This module tests that robustness by repeated random deletion of a fraction
``f`` of the catalogue, re-running the key catalogue-level tests, and
reporting the fraction of replicates in which each claim survives.

It is *not* a substitute for independent replication on a distinct catalogue
(JIRAM, Keck AO) — that requires external data. It is the minimum
within-catalogue stability check a reviewer will expect.

What is re-run per replicate
----------------------------
1. Geology enrichment — for each unit, record enrichment ratio and whether
   ``p_bonf < 0.05`` survives.
2. Hemispheric binomial tests — sub-Jovian/anti-Jovian, leading/trailing,
   northern/southern p-values.

What is NOT re-run
------------------
Ripley's K is expensive (Monte Carlo envelopes) and is deferred to the CLI
script when a full analysis is requested.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
STABILITY_CSV: Path = RESULTS_DIR / "catalog_stability.csv"


def _recompute_geology_enrichment(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Local wrapper that imports the compute function lazily."""
    from analysis.geology_enrichment import compute_geology_enrichment
    return compute_geology_enrichment(feature_matrix)


def _recompute_asymmetry(catalog: pd.DataFrame) -> dict:
    from analysis.asymmetry import compute_asymmetry
    return compute_asymmetry(catalog, n_boot=0, seed=0)


def catalog_jackknife(
    feature_matrix: pd.DataFrame,
    catalog: pd.DataFrame,
    retention_fractions: tuple[float, ...] = (0.9, 0.75, 0.5),
    n_replicates: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Random-deletion stability of enrichment significance and binomial p.

    For each retention fraction f and each replicate, drop a random (1−f)
    share of hotspots, re-label the grid (approximate — we drop rows from
    the *catalogue* and recount per cell), then recompute:
        - per-unit significance (p_bonf < 0.05)
        - hemispheric binomial p-values.

    Args:
        feature_matrix: Full grid feature matrix with ``cell_id``,
            ``has_hotspot``, ``hotspot_count`` (optional), ``geology_unit``,
            ``lat_centre``, ``lon_centre``.
        catalog: USGS hotspot catalogue with at least ``longitude``, ``latitude``.
        retention_fractions: Fractions of the catalogue to retain.
        n_replicates: Number of random deletions per fraction.
        seed: RNG seed.

    Returns:
        DataFrame with one row per (fraction, metric) summarising the
        survival probability of each claim.
    """
    rng = np.random.default_rng(seed)
    n = len(catalog)
    records: list[dict] = []
    _SCHEMA = ["fraction_retained", "metric", "survival_rate", "summary"]
    if n == 0:
        logger.warning("Catalogue jackknife skipped: hotspot catalogue is empty.")
        return pd.DataFrame(columns=_SCHEMA)

    # Need per-cell hotspot positions; rebuild from catalog each replicate.
    # Cheap approach: for each replicate, re-label has_hotspot in-place using
    # a nearest-cell lookup by rounding longitude/latitude to the 1° grid.
    lon_round = np.floor(catalog["longitude"].to_numpy()).astype(int)
    lat_round = np.floor(catalog["latitude"].to_numpy()).astype(int)

    # Map from (lon_floor, lat_floor) to cell_id
    fm = feature_matrix.copy()
    if "lon_floor" not in fm.columns:
        fm["lon_floor"] = np.floor(fm["lon_centre"]).astype(int)
        fm["lat_floor"] = np.floor(fm["lat_centre"]).astype(int)
    cell_lookup = fm.set_index(["lon_floor", "lat_floor"])["cell_id"].to_dict()

    for f in retention_fractions:
        keep_n = max(1, int(round(f * n)))
        enrichment_hits: dict[str, int] = {}
        p_subj: list[float] = []
        p_lead: list[float] = []
        p_north: list[float] = []
        enrichment_failures = 0
        asymmetry_failures = 0
        first_enrichment_error: str | None = None
        first_asymmetry_error: str | None = None

        for rep in range(n_replicates):
            idx = rng.choice(n, size=keep_n, replace=False)
            sub_cat = catalog.iloc[idx]
            # Re-label the grid.
            fm_rep = fm.copy()
            fm_rep["has_hotspot"] = 0
            fm_rep["hotspot_count"] = 0
            for ln, lt in zip(
                np.floor(sub_cat["longitude"].to_numpy()).astype(int),
                np.floor(sub_cat["latitude"].to_numpy()).astype(int),
            ):
                cid = cell_lookup.get((ln, lt))
                if cid is None:
                    continue
                mask = fm_rep["cell_id"] == cid
                fm_rep.loc[mask, "has_hotspot"] = 1
                fm_rep.loc[mask, "hotspot_count"] += 1

            try:
                enr = _recompute_geology_enrichment(fm_rep)
                for _, row in enr.iterrows():
                    unit = row["unit"]
                    is_sig = bool(row.get("significant", False))
                    enrichment_hits.setdefault(unit, 0)
                    if is_sig:
                        enrichment_hits[unit] += 1
            except Exception as exc:
                enrichment_failures += 1
                if first_enrichment_error is None:
                    first_enrichment_error = str(exc)

            try:
                asym = _recompute_asymmetry(sub_cat.reset_index(drop=True))
                for t in asym["binomial_tests"]:
                    c = t["comparison"]
                    if "Sub-Jovian" in c:
                        p_subj.append(float(t["p_binom"]))
                    elif "Leading" in c:
                        p_lead.append(float(t["p_binom"]))
                    elif "Northern" in c:
                        p_north.append(float(t["p_binom"]))
            except Exception as exc:
                asymmetry_failures += 1
                if first_asymmetry_error is None:
                    first_asymmetry_error = str(exc)

        if enrichment_failures:
            logger.warning(
                "Enrichment unavailable for %d/%d replicates at fraction_retained=%.2f. First error: %s",
                enrichment_failures,
                n_replicates,
                f,
                first_enrichment_error,
            )
        if asymmetry_failures:
            logger.warning(
                "Asymmetry unavailable for %d/%d replicates at fraction_retained=%.2f. First error: %s",
                asymmetry_failures,
                n_replicates,
                f,
                first_asymmetry_error,
            )

        # Summarise enrichment: fraction of replicates each unit was significant
        for unit, hits in enrichment_hits.items():
            records.append(
                {
                    "fraction_retained": f,
                    "metric": f"enrichment_{unit}",
                    "survival_rate": hits / max(1, n_replicates),
                    "summary": f"{hits}/{n_replicates} replicates Bonferroni-sig.",
                }
            )

        # Summarise binomial p-values
        for label, arr in [
            ("sub_jovian_vs_anti", p_subj),
            ("leading_vs_trailing", p_lead),
            ("northern_vs_southern", p_north),
        ]:
            if not arr:
                continue
            arr_np = np.asarray(arr)
            sig_rate = float(np.mean(arr_np < 0.05))
            records.append(
                {
                    "fraction_retained": f,
                    "metric": f"binomial_{label}",
                    "survival_rate": sig_rate,
                    "summary": (
                        f"p < 0.05 in {int(sig_rate * len(arr_np))}/"
                        f"{len(arr_np)} replicates. "
                        f"Median p = {np.median(arr_np):.3f}."
                    ),
                }
            )

    if not records:
        return pd.DataFrame(columns=_SCHEMA)
    out = pd.DataFrame.from_records(records)[_SCHEMA]
    return out


def save_stability(df: pd.DataFrame, path: Path | None = None) -> Path:
    out_path = path or STABILITY_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote stability results → %s", out_path)
    return out_path
