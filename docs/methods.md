# Methods Reference

This document describes the scientific and technical methods used in
`io-hotspot-prediction`. It is intended as a companion to the notebooks,
not a replacement for reading the code.

---

## 1. The Problem

Io, Jupiter's innermost Galilean moon, is the most volcanically active body
in the solar system. Its volcanism is driven by tidal heating — periodic
flexing of Io's interior as it orbits Jupiter in a slightly elliptical orbit
maintained by resonance with Europa and Ganymede (the Laplace resonance).

The USGS has compiled a catalog of ~150–200 known volcanic hotspots from
Galileo, Voyager, and ground-based near-infrared observations. But this
catalog reflects observational coverage, not the true spatial distribution
of volcanic activity. Spacecraft coverage of Io was uneven — the southern
hemisphere in particular is undersampled.

**The question:** Given what we know about the physical drivers of volcanism
(tidal stress, surface geology) and the spatial pattern of known hotspots,
can we estimate where undiscovered hotspots are most likely to be?

---

## 2. Grid

All analysis is performed on a 1°×1° longitude-latitude grid covering the
full Io surface:

- **Cells:** 360 × 180 = 64,800
- **Cell size:** 1° × 1° ≈ 32 km × 32 km at the equator (Io radius = 1821.6 km)
- **Convention:** Cell centres at (lon + 0.5°, lat + 0.5°) within each 1° bin

This resolution matches the spatial precision of the tidal heating model and
the geology map at mid-latitudes. It is coarser than the hotspot catalog
itself, which is appropriate — we are modeling at the scale of the physics,
not the scale of individual vents.

---

## 3. Features

### 3.1 Tidal Heating Flux (`tidal_heating_flux`)

Tidal heating arises from viscous dissipation of tidal energy in Io's
mantle and crust. The spatial pattern depends on Io's interior rheology and
the tidal forcing geometry.

For a homogeneous body in synchronous rotation, tidal heating is maximum
near the sub-Jupiter point (longitude 0°) and the anti-Jupiter point
(longitude 180°), and suppressed near the poles.

We use a published time-averaged model (see `data/external/SOURCES.md`).

**Assumption:** The model is time-averaged and assumes a specific interior
structure. Real tidal dissipation varies with orbital phase and is not fully
constrained by observation.

**Feature engineering:** Nearest-neighbour interpolation from model grid
to 1°×1° cell centres (`scipy.spatial.cKDTree`).

### 3.2 Geology Unit (`geology_encoded`)

The USGS geologic map (Williams et al. 2011) classifies Io's surface into
geologic units (plains, mountains, volcanic deposits, etc.). Some unit types
may be preferentially associated with active or past volcanic vents.

**Feature engineering:** Point-in-polygon spatial join from cell centres to
geology polygons (`geopandas.sjoin`). Label-encoded to integer. Cells outside
all polygons receive `UNKNOWN`.

**Limitation:** The geology map has limited resolution in some polar regions.
Cells with `UNKNOWN` geology introduce noise into the feature.

### 3.3 Distance to Nearest Known Hotspot (`dist_nearest_hotspot_km`)

Great-circle distance from each grid cell centre to the nearest hotspot in
the catalog, computed using the haversine formula with Io's radius.

**Rationale:** Volcanic systems tend to cluster. Cells near known hotspots
may be part of the same volcanic field.

**Bias warning:** This feature encodes the observational coverage pattern of
the catalog. Cells far from known hotspots may be far because they are
genuinely inactive, or because they were not well observed. The model cannot
distinguish these cases. This is acknowledged in the Limitations section.

### 3.4 Distance to Tidal Stress Maximum (`dist_tidal_stress_max_km`)

Great-circle distance to the nearest of four theoretical tidal stress maxima:
sub-Jupiter (0°N, 0°E), anti-Jupiter (0°N, 180°E), and the two leading/trailing
hemisphere apexes.

**Rationale:** Tidal flexing is greatest near these points, providing the
energy for volcanic activity.

**Limitation:** This uses a simplified point-source approximation. A full
tidal stress tensor calculation would account for body shape, interior
rheology, and orbital phase. This is a first-order proxy.

---

## 4. Model

### Logistic Regression

We use `sklearn.linear_model.LogisticRegression` with:
- `class_weight='balanced'` — reweights each sample by inverse class frequency
- `solver='lbfgs'`
- `max_iter=1000`
- Features standardized (zero mean, unit variance) within each CV fold

**Why logistic regression?**
- Interpretable: coefficients directly quantify feature-hotspot association
- Appropriate for a binary classification problem with continuous features
- No hyperparameter search required at baseline
- Cannot silently overfit in complex ways that would be hard to detect

**When to move beyond logistic regression:**
Only if you can demonstrate (a) residual structure in the predictions that
a more complex model could capture, and (b) sufficient data to prevent
overfitting. At ~150 positive examples on a 64,800-cell grid, this bar is
high.

### Class Imbalance

The positive class (hotspot cells) is approximately 0.2% of all cells.
`class_weight='balanced'` adjusts the loss function so that each positive
example receives ~500× the weight of a negative example.

This does not solve the underlying problem of having few positive examples.
It prevents the model from achieving low loss by predicting all-negative.

---

## 5. Validation

### Why Spatial Cross-Validation

Standard k-fold CV randomly assigns samples to folds without regard to
spatial location. For geospatial data, nearby cells share feature values
(spatial autocorrelation). If a cell is in the training set, its neighbors
in the test set will have similar features — the model appears to generalize
but is actually interpolating.

This is called **spatial data leakage** and is one of the most common errors
in geospatial machine learning.

### Our Approach: 4 Latitude-Band Folds

We split the globe into 4 latitude bands:

| Fold | Test band | Train bands |
|------|-----------|-------------|
| 0 | −90° to −45° | −45° to +90° |
| 1 | −45° to 0° | −90° to −45° and 0° to +90° |
| 2 | 0° to +45° | −90° to 0° and +45° to +90° |
| 3 | +45° to +90° | −90° to +45° |

Each fold trains on ~75% of cells and tests on ~25%. The train and test
regions are spatially separated by the fold boundary.

**Limitation:** Latitude bands are a simple spatial split. More sophisticated
approaches (buffered leave-one-out, custom geographic blocks) could reduce
leakage further, but are harder to implement and explain. For a baseline
model, latitude bands are defensible and transparent.

---

## 6. Visualization

### KDE Heatmap

Kernel density estimation using `scipy.stats.gaussian_kde` with Scott's
bandwidth selection. Evaluated at 0.25° resolution for display.

**This is not a model prediction.** It shows the density of *known hotspots*,
not the model's probability surface. Every KDE output is labelled:
> "Kernel density estimate — spatially interpolated. Not a model prediction."

### Prediction Surface

The logistic regression's `predict_proba()` output on the full 1°×1° grid,
displayed as a scatter plot. This is a model output and is labelled as such.

---

## 7. Assumptions Summary

| Assumption | Impact | Mitigation |
|------------|--------|------------|
| Catalog = ground truth | Model trains on observational bias | Acknowledge in all outputs |
| Tidal model is time-averaged | Feature may miss orbital phase variation | Use best available model; note limitation |
| Geology map is complete | UNKNOWN cells introduce noise | Track UNKNOWN fraction; report |
| 1°×1° resolution is appropriate | May miss sub-degree clustering | Defensible given source data resolution |
| Interior is homogeneous | Tidal heating proxy is approximate | Use published model when available |
