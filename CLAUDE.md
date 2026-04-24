# CLAUDE.md — io-hotspot-prediction

Project-specific instructions for AI assistants working on this codebase.
Read this before making any changes.

---

## Project Purpose

A modular, portfolio-quality Python pipeline that harmonizes open planetary datasets
about Io (Jupiter's volcanic moon) to map and predict volcanic hotspot locations at
a baseline level. The goal is scientific honesty + clean engineering, not impressive
numbers.

---

## Architecture Principles

### Module Responsibilities (do not blur these)

| Folder | Responsibility |
|--------|---------------|
| `ingest/` | Download or load raw data. Write to `data/raw/` only. No transforms. |
| `preprocess/` | Clean, reproject, grid-align. Write to `data/processed/`. Never modify `data/raw/`. |
| `features/` | Compute features on the 1°×1° grid. Inputs from `data/processed/`. |
| `models/` | Train, evaluate, persist models. No data loading — accept DataFrames. |
| `visualization/` | Produce figures. No modeling. No data loading from disk. |
| `dashboard/` | Streamlit app only. Calls other modules, does not reimplement logic. |
| `tests/` | Pytest stubs and unit tests. One file per module. |
| `notebooks/` | Exploration only. Numbered, sequential. No production logic lives here. |

---

## Critical Scientific Constraints

### ⚠️ RANDOM SPLITS ARE FORBIDDEN

This dataset has severe spatial autocorrelation. Random train/test splits will
inflate all metrics and produce scientifically invalid results.

**Always use spatial cross-validation with 4 latitude-band folds.**

The spatial CV utility is in `models/spatial_cv.py`. Use it. Never use
`sklearn.model_selection.train_test_split` on the hotspot grid data.

### Grid Resolution

- **Modeling and features:** 1°×1° grid (360 × 180 cells)
- **Visualization only:** KDE heatmaps may be rendered at higher resolution
- Always label interpolated/upsampled outputs explicitly in figures and notebooks

### Class Imbalance

The positive class (hotspot-containing cells) is ~0.2% of the grid. Always:
- Report the imbalance ratio in any notebook that trains a model
- Use `class_weight='balanced'` or explicit resampling with justification
- Report precision, recall, F1, and AUC-ROC — never accuracy alone

### Data Provenance

Every processed dataset must carry a `provenance` field or header comment
linking back to its raw source. If you can't trace it, don't use it.

---

## Code Standards

- **Python 3.11**
- Type hints on all public functions
- Docstrings on all public functions (one-line summary + Args/Returns)
- No magic numbers — use named constants or config
- Prefer explicit over clever
- No `import *`

---

## Data Rules

- `data/raw/` is **read-only** after ingest. Never write to it from preprocess or later.
- `data/processed/` contains intermediate outputs with clear filenames (e.g. `hotspots_1deg_grid.parquet`)
- `data/external/` contains manually downloaded files with a `SOURCES.md` manifest

---

## Uncertainty & Assumptions

This project makes several strong assumptions that must remain visible:

1. **Hotspot catalog completeness:** The USGS catalog reflects observational coverage,
   not true hotspot distribution. Absence of data ≠ absence of hotspots.
2. **Tidal heating model:** We use a published static model. Real tidal heating
   varies with Io's orbital position and internal structure.
3. **Geology map vintage:** The USGS Io geology map (Williams et al. 2011) is derived
   from Galileo-era imagery. Some regions have limited coverage.
4. **Prediction target:** We predict "likelihood given available features" not
   "ground truth hotspot presence."

These assumptions must appear in the README, the modeling notebook, and any
published outputs.

---

## What NOT to Do

- Do not add deep learning. Logistic regression is the right baseline here.
- Do not interpolate hotspot locations beyond the KDE visualization layer.
- Do not cite papers you haven't read — stub citations with `[VERIFY]` instead.
- Do not overfit to the known catalog — the model should generalize, not memorize.
- Do not remove the Limitations section from the README.

---

## First Development Steps (in order)

See README.md § Development Roadmap for the full ordered task list.

Quick orientation:
1. Run `notebooks/00_data_audit.ipynb` to confirm raw data integrity
2. Run `notebooks/01_spatial_alignment.ipynb` to verify grid registration
3. Only then proceed to feature engineering and modeling

---

## Contact / Attribution

Project by: [Your Name]
Data sources: See `data/external/SOURCES.md` and README § Data Sources
