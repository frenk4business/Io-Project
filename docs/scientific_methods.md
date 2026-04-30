# Scientific Methods

[Nederlandse versie](scientific_methods.nl.md)

Canonical methodology reference for the Io Hotspot Prediction project's
**Scientific Analysis layer**. If a claim surfaces anywhere in the dashboard,
README, or a generated figure, it is defined here.

The existing **Exploration / Teaching layer** (the 3D globe, interactive maps,
FAQ) remains as a learning aid. This document governs the analysis layer that
sits alongside it.

---

## Research Question

> *Given the USGS Io hotspot catalogue, which geological units and physical-proxy
> covariates show statistically significant association with observed hotspot
> locations, after adjusting for imaging-coverage bias — and how do those
> associations hold under leakage-free spatial cross-validation?*

This is a descriptive / inferential question, not a predictive one. We do not
claim to predict undiscovered hotspots. We characterise the observed catalogue
and report which covariates plausibly co-vary with it, with explicit uncertainty.

---

## Exploratory Multi-Metric Volcanism Question

> **To what extent do different volcanic activity metrics, including named
> hotspot occurrence, estimated thermal intensity, and metadata-normalized
> observation activity, produce different spatial interpretations of Io's
> volcanism on a common 1 degree grid?**

This project also includes a reproducible, exploratory comparison framework for
asking whether Io looks volcanically different depending on the metric used. A
map of known volcanic centers is not necessarily the same as a map of thermal
power, and both may differ from activity normalized by observation metadata.

The comparison separates three metric families:

- **Named hotspot occurrence**: `has_hotspot` and `hotspot_count`, derived from
  catalogued volcanic centers and useful as a geological/reference layer.
- **Estimated thermal intensity**: Davies/JIRAM estimated proxy GW is kept
  separate from JIRAM radiance, NIMS radiance, AO brightness, and unitless
  normalized percentile proxies.
- **Metadata-normalized observation activity**: activity proxies divided by
  metadata such as observation counts or coverage weights. This is a
  metadata-based normalization, not a true footprint/sensitivity correction.

The current generated outputs include Spearman correlation, top-10 percent rank
overlap, Jensen-Shannon divergence, latitude-band contribution tables, and top-N
concentration curves. The analysis is designed to show how metric choice changes
spatial interpretation, not to claim a new physical model of Io.

Strict limitation: metadata-normalized activity must not be described as true
coverage-corrected volcanism. Real footprints, systematic non-detections,
instrument sensitivity, and observing-geometry masks are required before making
that stronger claim. Persistence and episodicity are provisional for the same
reason: event rows can define activity and metadata coverage at the same time.

---

## Known limitations we do not hide

1. **Target leakage.** The feature `dist_nearest_hotspot_km`
   ([features/synthetic.py:80-115](../features/synthetic.py)) is computed from
   the same hotspot catalogue that defines the target label. When included, a
   logistic regression trivially learns "hotspots are near hotspots" (coefficient
   ≈ −30 in standardised units). Any reported performance that includes this
   feature is inflated and **not a valid measure of generalisation**. The
   Scientific Analysis layer always reports a non-leaky feature set alongside
   the leaky baseline for direct comparison.

2. **Synthetic tidal heating.** The `tidal_heating_flux` column produced by
   [ingest/tidal_heating.py:99-142](../ingest/tidal_heating.py) is an analytical
   proxy `cos²(lon)·cos²(lat) + 0.3·sin²(lat)`. This is **not** the Segatz M3
   asthenosphere model, the Segatz M4 deep-mantle model, Hamilton et al. (2013),
   or Tyler et al. (2015). It is kept because the pipeline depends on *some*
   tidal column, but every use site is labelled as a proxy. Real dissipation
   grids are scaffolded by `ingest/tidal_models.py` and, when ingested, swapped
   in via `TIDAL_MODEL_REGISTRY` in `config.py`.

3. **Observational coverage bias.** The USGS catalogue reflects where Galileo
   and Voyager observed. Absence of a hotspot in a grid cell does not imply
   absence of hotspots in reality. The `analysis/coverage_bias.py` module
   quantifies and (partially) adjusts for this.

4. **Catalogue vintage.** The hotspot list is dominated by pre-2003 detections.
   Post-2003 additions (New Horizons, JWST, ground-based) are not fully merged.
   Temporal variability is not modelled.

5. **Estimated thermal-emission proxy, not direct radiant power.** The project
   now includes Davies et al. (2024) JIRAM 4.8 micron spectral radiance for
   266 hot spots. `data/raw/io_hotspot_power.csv` converts that table to an
   estimated `power_gw` proxy using the article-level radiance-to-thermal-
   emission scaling. This supports intensity analysis, but it is not directly
   measured bolometric radiant power.

These limitations are enumerated in the README "Known Limitations" section and
surfaced in the dashboard's "Scientific Analysis → Research Question & Methods"
tab. Removing them is an explicit non-goal.

---

## Methodology summary

### Spatial cross-validation

All model performance metrics reported in the Scientific Analysis layer use the
existing 4-latitude-band spatial CV from [models/spatial_cv.py](../models/spatial_cv.py).
Random splits are forbidden per [CLAUDE.md](../CLAUDE.md) and would invalidate
any claim of generalisation on a spatially autocorrelated dataset.

### Leakage audit — `analysis/leakage_audit.py`

For each feature we report:
- derivation source (target-derived / analytical proxy / observed)
- Pearson and Spearman correlation with the label
- standardised logistic-regression coefficient (diagnostic, full-data fit)
- boolean `suspected_leakage` flag with `flag_reason`

### Ablation — `models/ablation.py`

Multiple feature sets are trained through spatial CV:
- `all` — the current four features (leaky baseline). Results labelled as such.
- `no_leakage` — drops `dist_nearest_hotspot_km`. This is the project's honest headline.
- `geology_only`, `tidal_only` — single-feature-family reports.
- `null` — labels permuted; AUC must be ≈ 0.5 (sanity check for the CV implementation).

All results are persisted to `data/results/ablation.json`.

### Thermal intensity analysis — `analysis/power_intensity.py`

Davies/JIRAM thermal activity is analysed separately from binary hotspot
presence. The pipeline aggregates estimated `power_gw` proxy records onto the
1 degree grid, then reports latitude-band summaries, geology-unit summaries,
polar-cap threshold sensitivity, and top-outlier sensitivity. `models/regression.py`
trains a leakage-aware Ridge regression on `log1p(primary_power_gw)` using the
same latitude-band spatial CV, excluding `dist_nearest_hotspot_km` by default.

### Validation metrics — `models/validation_metrics.py`

Per fold: reliability curve, PR curve, Brier score, Matthews correlation.
Figures to `data/results/validation/`.

### Geological association — `analysis/geology_enrichment.py`

For each of the 16 USGS map units
(see [visualization/globe_3d.py](../visualization/globe_3d.py) `IO_GEOLOGY_LEGEND`):
- observed hotspot count
- expected count under a complete spatial randomness (CSR) null, proportional
  to the unit's cell area
- enrichment ratio (obs / exp) with a 95% Wilson confidence interval
- chi-square contribution and Bonferroni-corrected p-value

### Spatial point-pattern — `analysis/spatial_stats.py`

Ripley's K on a sphere, its pair-correlation derivative g(r), and the
nearest-neighbour distance CDF. CSR envelopes are computed with ≥ 199 Monte Carlo
simulations of uniform-on-sphere point sets.

CSR expectation on a sphere of radius R:

    K_csr(r) = 2π R² (1 − cos(r/R))

Values above the upper envelope indicate clustering; values below indicate
regularity.

### Hemispheric and longitudinal asymmetry — `analysis/asymmetry.py`

- Sub-Jovian (|lon| < 90°) vs anti-Jovian (|lon| ≥ 90°) hotspot counts with
  binomial CIs.
- Leading (lon in (0°, 180°)) vs trailing (lon in (−180°, 0°)) counts.
- Northern vs southern hemisphere counts.
- Longitudinal and latitudinal density bins with bootstrap CIs.

We note which of the **Segatz M3** (asthenosphere → sub-Jovian / anti-Jovian
equatorial enhancement) vs **Tyler et al. 2015** (magma-ocean → polar enhancement)
patterns the observed catalogue most resembles. We do **not** claim this proves
either interior model.

### Coverage-bias awareness — `analysis/coverage_bias.py`

Cells whose geology unit is `NoData` or `UNKNOWN` are treated as unobserved.
We compare raw hotspot density to a simple inhomogeneous-Poisson adjustment
λ̂ = N_hotspots / Area_observed per latitude band, producing a rate-ratio plot.

### Uncertainty — `analysis/uncertainty.py`

- Bootstrap percentile CIs for each headline CV metric.
- Sensitivity of AUC / PR-AUC to the choice of tidal column
  (synthetic proxy vs uniform vs M3 stub vs M4 stub when ingested).
  A flag is raised whenever the direction of a conclusion changes with the tidal
  choice.

---

## Language rules

In any text produced by the Scientific Analysis layer — code comments, plot
captions, markdown, dashboard copy — we use neutral scientific phrasing:

- ✅ "associated with," "consistent with," "cannot rule out," "observed catalogue"
- ❌ "predicts," "AI identifies," "our model discovers," "accurately detects"

Headline numbers always cite their feature set, so the leaky AUC never appears
without the non-leaky AUC beside it.

---

## Reproducibility

`scripts/run_scientific_analysis.py` runs the full pipeline and regenerates
every artifact in `data/results/`. `tests/test_analysis.py` contains shape and
contract checks for each analysis module; it also verifies that Ripley's K on a
uniform-on-sphere sample falls inside the CSR envelope.

---

## Nothing is mutated

The Scientific Analysis layer writes only to `data/results/`. It does not
rewrite `data/raw/`, `data/processed/`, `data/models/`, or the existing
`FEATURE_COLUMNS`. The Exploration / Teaching pages, the persisted
`logistic_regression.pkl` / `scaler.pkl`, and the existing notebooks are all
preserved as-is.
