# Io Hotspot Prediction: A Research-Oriented Project Summary

**Audience:** Io researchers, planetary geologists, planetary volcanology researchers, and data-oriented planetary scientists.

---

## Page 1 - What I Built

This project is a reproducible data-analysis and visualization pipeline for studying volcanic hotspots on Io. The central idea is to bring several public Io datasets into one common spatial framework, then use that framework to ask careful questions about hotspot distribution, surface geology, thermal-emission intensity, and observational bias.

The project starts from a simple but scientifically important problem: Io is extremely active, but our view of Io is incomplete. The known hotspot catalogues are shaped by spacecraft coverage, observing geometry, instrument sensitivity, wavelength, and catalogue vintage. A blank region on a map is therefore not automatically a quiet region on Io. It may simply be a region that has not been observed with the right sensitivity at the right time. This project treats that limitation as a core part of the analysis rather than as an afterthought.

The pipeline aligns multiple sources onto a shared 1 deg x 1 deg longitude-latitude grid. That grid acts as the common coordinate system for hotspot presence, geological unit labels, synthetic or imported tidal-heating fields, distance-based covariates, and thermal-emission proxy summaries. By using one grid throughout the project, the analysis can compare spatial patterns consistently across data sources that were originally collected for different purposes.

The project includes several main layers:

- A data-ingestion layer for hotspot catalogues, geological maps, tidal-heating proxy data, and Juno/JIRAM-derived thermal-emission proxy records.
- A preprocessing layer that builds and aligns the global Io grid.
- A feature-engineering layer that creates covariates such as geology, tidal-heating proxy values, and distances to known hotspots or stress-related reference points.
- A modelling layer that trains baseline logistic-regression models and evaluates them using spatial cross-validation.
- A scientific-analysis layer that reports leakage audits, ablation studies, geology enrichment, spatial clustering, hemispheric asymmetry, coverage-bias diagnostics, and thermal-intensity summaries.
- A dashboard layer that presents both public-facing visual exploration and researcher-facing scientific interpretation.

The most visible part of the project is the Streamlit dashboard. It includes an "Io Experience" page with a 3D globe, 2D maps, a 3D globe view, a scientific-analysis page, and bilingual support in English and Dutch. The 3D experience is not meant to replace scientific analysis; it is a way to make the spatial structure of the data immediately understandable. The scientific-analysis page is where the more cautious interpretation lives.

A recent technical improvement was the repair of the 3D visualization dependency chain. The 3D pages require Plotly. The repository now records that dependency in the pip requirements, the Conda environment, and the Python project metadata. The dashboard also gives clear English and Dutch error messages if Plotly is missing from the active Streamlit environment. This matters because scientific tools are only useful if another researcher can actually run them without guessing which environment was used.

The project deliberately separates "visual impression" from "scientific claim." It does not present the model as an AI discovery engine for unknown volcanoes. Instead, it asks what structure exists in the observed catalogues, which associations are robust or fragile, and how much interpretation changes when leakage-prone or proxy-based variables are removed.

<div style="page-break-after: always;"></div>

## Page 2 - What the Analysis Does Scientifically

The project is built around a descriptive and inferential research question: given the observed Io hotspot catalogues, the USGS geology map, and several proxy covariates, what spatial structure and associations are visible, and how sensitive are those conclusions to data leakage, observation coverage, and proxy assumptions?

One of the most important methodological choices is the explicit treatment of target leakage. A feature such as distance to the nearest known hotspot can be very informative in a statistical model, but it is also derived from the same catalogue used to define the target label. If that feature is included without caution, the model can appear more predictive than it really is. In practical terms, it learns that hotspots are near hotspots. That may produce strong metrics, but it does not prove that the model has learned a transferable physical relationship.

To address this, the project includes a leakage audit and ablation workflow. The analysis compares feature sets that include all variables with more honest feature sets that remove leakage-prone variables. This allows the dashboard to show the difference between a diagnostic baseline and a more defensible generalization estimate. For Io researchers, that distinction is important because spatial autocorrelation and catalogue-derived predictors can easily inflate apparent performance.

The project also uses spatial cross-validation rather than random train-test splits. Random splits are often misleading for spatial planetary data because neighbouring cells are not independent. A random split can place nearby cells from the same spatial structure in both training and test sets. This project instead uses latitude-band spatial cross-validation, which is a more conservative way to ask whether model behaviour transfers across broad regions of Io.

Beyond modelling, the project includes several analyses that are directly relevant to Io science:

- **Geological association:** The project compares observed hotspot counts across geological units against expected counts under a spatial null model. It reports enrichment ratios and uncertainty rather than simply ranking units visually.
- **Spatial point-pattern analysis:** The project uses spherical spatial statistics, including Ripley's K on a sphere, to test whether hotspot locations are more clustered than expected under complete spatial randomness.
- **Hemispheric and longitudinal asymmetry:** The project compares sub-Jovian and anti-Jovian hemispheres, leading and trailing hemispheres, and northern and southern regions. These comparisons connect directly to long-standing questions about tidal forcing, interior structure, and observation geometry.
- **Coverage-bias diagnostics:** The project treats unknown or unmapped regions as a warning signal. It asks how observed hotspot rates change when only mapped or better-characterized regions are considered.
- **Thermal-emission proxy analysis:** The project uses Davies/JIRAM-derived 4.8 micron spectral-radiance information as an estimated thermal-emission proxy. It summarizes intensity by latitude band and geology and tests sensitivity to the strongest sources.

The thermal-intensity part is especially useful because Io's volcanic activity is not only a question of where hotspots occur. A small number of very strong sources can dominate the thermal-emission signal. Therefore, a binary hotspot map and an intensity-weighted map can tell different stories. This project keeps those two questions separate: hotspot presence is treated differently from estimated thermal-emission strength.

The project also makes the limits of its tidal-heating input explicit. The current tidal-heating field is a proxy unless replaced by a published physical dissipation grid. This is not hidden. The code and documentation explain that the proxy is useful for pipeline development and qualitative comparison, but it should not be interpreted as a definitive Segatz, Beuthe, Hamilton, or Tyler model. For researchers, that transparency is valuable because it shows exactly where a future physical model could be inserted.

In short, the project is not trying to overclaim. It is trying to make the existing Io hotspot data easier to interrogate, easier to visualize, and harder to misinterpret.

<div style="page-break-after: always;"></div>

## Page 3 - Why This Could Be Interesting for Io Researchers

This project may be useful to Io researchers because it provides an open, inspectable framework for connecting catalogues, geology, thermal activity, and spatial statistics in one place. Many Io studies focus deeply on one data source or one physical question. This project is different: it is a structured integration layer that helps compare several kinds of evidence while keeping uncertainty visible.

For researchers working on volcanic distribution, the project offers a reproducible way to examine whether known hotspots are associated with particular mapped surface units or hemispheric regions. It does not replace expert geological interpretation, but it can help identify which associations are strong enough to deserve closer inspection and which may be artefacts of catalogue coverage.

For researchers interested in tidal-heating hypotheses, the project provides a ready-made testing scaffold. The existing tidal field can be replaced with real gridded outputs from interior models. Once those grids are available, the same pipeline can compare hotspot distributions, intensity summaries, and spatial likelihoods against physically motivated heating patterns. That makes the project a useful bridge between catalogue analysis and interior-model evaluation.

For researchers working with Juno/JIRAM or future thermal datasets, the project already separates thermal-emission proxy intensity from binary hotspot occurrence. This is important because Io's thermal output is highly uneven. A few major sources can shape global summaries, while weaker or intermittent sources may still matter for geological interpretation. The project's outlier-sensitivity and latitude-band summaries are designed to make that imbalance visible.

For researchers concerned with observational bias, the project is intentionally conservative. It repeatedly warns that non-detection is not the same as absence. It treats coverage and catalogue vintage as scientific constraints rather than small technical details. That makes it a useful starting point for more rigorous coverage correction, such as incorporating Galileo, Voyager, New Horizons, Juno/JIRAM, JWST, or ground-based observing footprints.

The bilingual dashboard is also useful beyond outreach. English remains the primary scientific language, but the Dutch interface shows that the same scientific content can be localized without changing the underlying analysis. That matters for teaching, public communication, and interdisciplinary presentation. The public-facing 3D globe can help non-specialists understand Io's spatial structure, while the scientific-analysis layer preserves the caveats that researchers need.

The strongest contribution of the project is not a single headline result. Its value is the workflow:

1. Put the data on a common grid.
2. Make each feature's origin explicit.
3. Separate observed catalogue structure from predictive claims.
4. Test leakage-prone and leakage-aware versions side by side.
5. Report spatial, geological, hemispheric, and intensity-based summaries together.
6. Keep limitations visible in the same interface as the results.

That workflow can be extended. Future versions could ingest updated hotspot catalogues, include coordinate uncertainty, add real tidal-dissipation grids, add observation-footprint rasters, compare SIM3168 with JIRAM detections more formally, or test whether intensity-weighted volcanic activity follows different spatial patterns from named volcanic centres.

For an Io researcher, the project is therefore best understood as a reproducible research scaffold. It does not claim to solve Io's volcanism problem. It makes the problem easier to explore with traceable assumptions, explicit uncertainty, and a dashboard that connects visual intuition with statistical caution.

The project is interesting because Io research often lives at the intersection of geology, thermal observations, orbital forcing, and incomplete coverage. This tool sits exactly at that intersection. It gives researchers a way to ask: what do the current catalogues actually support, where are the results fragile, and what new data would most improve the interpretation?

