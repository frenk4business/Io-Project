# io-hotspot-prediction

[Nederlandse versie](README.nl.md)

**Mapping and predicting volcanic hotspots on Io, Jupiter's most volcanically
active moon — using open planetary datasets, spatial machine learning, and
honest uncertainty quantification.**

---

## Why Io?

Io is the most volcanically active body in the solar system, driven by tidal
heating from Jupiter's immense gravity. Its surface is covered by hundreds of
volcanic hotspots — but observational coverage is uneven and incomplete. This
project asks a simple, honest question:

> *Given what we know about Io's tidal environment and surface geology,
> can we predict where undiscovered hotspots are most likely to be?*

The answer is "probably, a little" — and making that uncertainty explicit is
the point.

---

## Research Question

> *Given the USGS Io hotspot catalogue, which geological units and
> physical-proxy covariates show statistically significant association with
> observed hotspot locations, after adjusting for imaging-coverage bias — and
> how do those associations hold under leakage-free spatial cross-validation?*

This is a descriptive / inferential question, not a predictive one. The
**Scientific Analysis layer** (see `docs/scientific_methods.md` and the
"Scientific Analysis" dashboard page) reports spatial statistics, geological
enrichment, hemispheric asymmetry, coverage-bias adjustment, and leakage-free
model performance. The **Exploration / Teaching layer** (overview, 3D globe,
hotspot map, FAQ) remains as an educational aid alongside it.

---

## What This Project Does

1. **Ingests** the USGS Io Volcanic Hotspot catalog and supporting spatial layers
2. **Preprocesses** all inputs onto a common 1°×1° planetary grid
3. **Engineers four features** per grid cell: tidal heating flux, geology type,
   distance to nearest known hotspot, and distance to tidal stress maxima
4. **Trains a logistic regression classifier** using spatial cross-validation
   (4 latitude-band folds) to avoid spatial data leakage
5. **Generates a KDE density heatmap** at high resolution, explicitly labelled
   as interpolated
6. **Exposes everything** through a Streamlit dashboard
7. **Analyses thermal activity intensity** using the Davies/JIRAM 4.8 µm
   spectral-radiance catalogue as an estimated thermal-emission proxy

---

## Repository Structure

```
io-hotspot-prediction/
├── data/
│   ├── raw/              # Untouched source files (read-only after ingest)
│   ├── processed/        # Cleaned, gridded intermediates
│   └── external/         # Manually downloaded files + SOURCES.md manifest
├── ingest/               # Data download and loading
├── preprocess/           # Cleaning, reprojection, grid alignment
├── features/             # Feature engineering on the 1°×1° grid
├── models/               # Logistic regression + spatial CV
├── visualization/        # Matplotlib/Cartopy figures
├── dashboard/            # Streamlit app
├── tests/                # Pytest stubs and unit tests
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_spatial_alignment.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_baseline_modeling.ipynb
├── docs/
├── environment.yml       # Conda environment (primary)
├── requirements.txt      # Pip fallback
├── CLAUDE.md             # AI assistant instructions
└── README.md
```

---

## Data Sources

| Dataset | Source | Format | Provenance |
|---------|--------|--------|------------|
| Io Volcanic Hotspot Catalog | USGS Astrogeology / Lopes & Williams (2005) | CSV | [USGS Io hotspot list](https://astrogeology.usgs.gov/) |
| Davies/JIRAM Hotspot Radiance | Davies et al. (2024), Nature Astronomy | CSV | Supplementary Table S1, 266 hot spots |
| Io Geologic Map | Williams et al. (2011), USGS | Shapefile | [USGS Planetary GIS](https://astrogeology.usgs.gov/search/map/Io/Geology/Io_Geology_Geologic_Map) |
| Tidal Heating Flux Grid | Tackley et al. / Hamilton et al. (published model) | GeoTIFF / CSV | See `data/external/SOURCES.md` |

See `data/external/SOURCES.md` for full citation details and download instructions.

---

## Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git

### Installation (conda, recommended)

```bash
git clone https://github.com/yourname/io-hotspot-prediction.git
cd io-hotspot-prediction
conda env create -f environment.yml
conda activate io-hotspot
```

> **Why conda?** `cartopy` and `rasterio` have C dependencies that install
> reliably via conda but can be painful via pip alone.

### Installation (pip fallback)

```bash
pip install -r requirements.txt
# Note: cartopy may require manual GEOS/PROJ installation on some systems
```

### Data Setup

Raw data is not committed to this repository. After installing dependencies:

```bash
python -m ingest.hotspot_catalog    # Downloads/places USGS hotspot CSV
python -m ingest.geology_map        # Places geology shapefile in data/raw/
python -m ingest.tidal_heating      # Places tidal heating grid in data/raw/
```

See `data/external/SOURCES.md` for manual download links and expected filenames.

---

## Usage

### Run the full pipeline

```bash
python -m preprocess.grid           # Build 1°×1° base grid
python -m preprocess.align_layers   # Align all inputs to base grid
python -m features.build            # Compute all four features
python -m models.train              # Train logistic regression with spatial CV
python -m visualization.hotspot_map # Generate figures
```

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

If the 3D Io Experience or 3D Globe reports that Plotly is missing, update the
same environment that runs Streamlit:

```bash
conda env update -f environment.yml
# or
python -m pip install "plotly>=5.18"
```

### Run tests

```bash
pytest tests/ -v
```

### Run notebooks (in order)

```bash
jupyter lab notebooks/
```

Start with `00_data_audit.ipynb` and proceed sequentially.

---

## Methods

### Grid

All spatial features and predictions are computed on a 1°×1° longitude-latitude
grid covering the full Io surface (360 × 180 = 64,800 cells).

### Features

| Feature | Type | Source |
|---------|------|--------|
| `tidal_heating_flux` | Continuous | Published tidal model |
| `geology_class` | Categorical (encoded) | USGS geology shapefile |
| `dist_nearest_hotspot` | Continuous (km) | Computed from catalog |
| `dist_tidal_stress_max` | Continuous (km) | Analytically derived |

### Model

Logistic regression (`sklearn.linear_model.LogisticRegression`) with
`class_weight='balanced'` to address the ~0.2% positive class rate.

### Validation

Spatial cross-validation with 4 latitude-band folds (−90°→−45°, −45°→0°,
0°→+45°, +45°→+90°). Each fold trains on 3 bands, tests on 1. This prevents
spatial autocorrelation from inflating metrics.

**Metrics reported:** Precision, Recall, F1, AUC-ROC per fold and mean ± std.

### Visualization

KDE heatmaps are rendered at higher resolution than the modeling grid and are
explicitly labelled as spatially interpolated. They show *density of known
hotspots*, not model predictions.

---

## Known Limitations (read before interpreting any result)

The Scientific Analysis layer exists because the original baseline had two
credibility issues that must remain visible:

1. **Target leakage.** `dist_nearest_hotspot_km` is computed from the hotspot
   catalogue, which is also the label source. Any logistic-regression AUC that
   includes this feature (≈ 0.95) is an artifact of a feature that encodes the
   answer. The **Scientific Analysis → Ablation Results** page reports a
   leakage-free feature set alongside the leaky baseline. Treat the non-leaky
   number as the honest headline.
2. **Synthetic tidal heating proxy.** `tidal_heating_flux` in the feature
   matrix is currently an analytical expression
   `cos²(lon)·cos²(lat) + 0.3·sin²(lat)`, not a physical tidal-dissipation
   model. Real published grids (Segatz M3, Segatz M4, Hamilton 2013, Tyler 2015)
   are scaffolded via `ingest/tidal_models.py` and swap in automatically when
   the corresponding files are placed in `data/external/`. Until then, all
   tidal-related inferences are conditional on the proxy and **the sensitivity
   analysis** in the Scientific Analysis layer tells you whether the conclusion
   survives other tidal definitions.
3. **Observational coverage bias**, **catalogue vintage**, and **estimated
   thermal-emission proxy limitations** are discussed in
   `docs/scientific_methods.md` and surfaced per-page in the dashboard.

4. **Thermal intensity is now proxy-based, not direct radiant power.**
   `data/raw/io_hotspot_power.csv` is derived from Davies/JIRAM 4.8 µm spectral
   radiance. It enables regression and power-density summaries, but every
   output must label `power_gw` as an estimated thermal-emission proxy rather
   than a directly measured bolometric radiant-power product.

Full methodology: [`docs/scientific_methods.md`](docs/scientific_methods.md).

---

## Limitations & Assumptions

This section is not a disclaimer — it is part of the science.

1. **Observational bias:** The USGS hotspot catalog reflects where Galileo and
   Voyager had good coverage. The southern hemisphere is undersampled. The model
   will inherit this bias.

2. **Static tidal model:** We use a published time-averaged tidal heating flux.
   Io's actual tidal dissipation varies with orbital phase and is not fully
   constrained.

3. **Geology map vintage:** The USGS geology map is Galileo-era (pre-2003 primary
   data). Some regions have low-resolution or gap-filled coverage.

4. **What "prediction" means here:** The classifier predicts relative likelihood
   given the four features. It does not predict hotspot activity, temperature,
   or timing. It should not be extrapolated beyond its training distribution.

5. **Class imbalance:** ~0.2% positive rate. All results should be interpreted
   with this in mind. A model predicting all-negative achieves 99.8% accuracy.
   We do not report accuracy.

---

## Development Roadmap

Steps in recommended order:

- [ ] 1. Acquire raw data (see `data/external/SOURCES.md`)
- [ ] 2. Run `00_data_audit.ipynb` — verify catalog completeness and CRS
- [ ] 3. Implement `preprocess/grid.py` — build base 1°×1° grid
- [ ] 4. Run `01_spatial_alignment.ipynb` — verify all layers register correctly
- [ ] 5. Implement `features/tidal_heating.py` and `features/geology.py`
- [ ] 6. Implement `features/synthetic.py` (distance features)
- [ ] 7. Run `02_eda.ipynb` — check feature distributions and correlations
- [ ] 8. Run `03_feature_engineering.ipynb` — finalize feature matrix
- [ ] 9. Implement `models/spatial_cv.py`
- [ ] 10. Run `04_baseline_modeling.ipynb` — train and evaluate
- [ ] 11. Implement `visualization/hotspot_map.py`
- [ ] 12. Wire up `dashboard/app.py`
- [ ] 13. Write tests in `tests/`

---

## Citation

If you use this project or its methods, please cite the underlying data sources
listed in `data/external/SOURCES.md`. This project itself is not peer-reviewed.

---

## License

MIT License. See `LICENSE`.

---

## Acknowledgements

- USGS Astrogeology Science Center for the Io hotspot catalog and geology map
- The Galileo and Voyager mission teams whose observations underpin all of this
