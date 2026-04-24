"""
ingest/tidal_models.py
----------------------
Ingestion scaffold for published tidal heating dissipation grids.

Status: SCAFFOLD — only the uniform null and synthetic proxy are runnable.
M3 and M4 grids require external data files not currently available locally.

Model registry
--------------
TIDAL_MODEL_REGISTRY maps short model names to provenance and ingestion metadata.
When a grid file is placed at the expected path, load_tidal_model(name) reads it,
aligns it to the project's 1°×1° grid, and returns a DataFrame compatible with
the existing feature pipeline.

Registered models
-----------------
  "synthetic_proxy" — AVAILABLE. Analytical cosine² formula (see ingest/tidal_heating.py).
                      No physical validity. Already ingested into the pipeline as default.
  "uniform"         — AVAILABLE. Spatially uniform null (all cells equal flux = 1.0).
                      Useful as a model-free null hypothesis for enrichment comparison.
  "M3"              — NOT AVAILABLE. Segatz et al. (2013) asthenosphere-dominated heating.
                      Maximum dissipation at sub-Jovian and anti-Jovian equatorial points.
  "M4"              — NOT AVAILABLE. Segatz et al. (2013) deep-mantle-dominated heating.
                      Maximum dissipation at poles.

To obtain M3 / M4 grids
------------------------
1. Contact: Spohn, T. / Segatz, M. — original model authors.
   Or: Hamilton, C.W. et al. (2013) — EPSL, supplementary data.
   Or: Tyler, R.H. et al. (2015) — ApJS, supplementary data.
2. Expected file format:
     CSV with columns: longitude (-180 to 180), latitude (-90 to 90), flux_w_m2
     OR GeoTIFF raster (reprojection handled by load_tidal_model).
3. Place files at the paths defined in TIDAL_MODEL_REGISTRY[name]["path"].
4. Run load_tidal_model(name) — it will validate and align to 1°×1° grid.

See data/external/SOURCES.md for citation details.
See docs/scientific_methods.md §Replace fake physics for scientific context.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import EXTERNAL_DIR, GRID_RESOLUTION_DEG, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

TIDAL_MODEL_REGISTRY: dict[str, dict] = {
    "synthetic_proxy": {
        "description": (
            "Analytical cosine² proxy: cos²(lon)·cos²(lat) + 0.3·sin²(lat). "
            "No physical calibration. Not a published model. "
            "Used as a development placeholder only."
        ),
        "path": RAW_DIR / "io_tidal_heating_flux.csv",
        "reference": "None — synthetic formula (see ingest/tidal_heating.py)",
        "status": "available",
        "format": "csv",
    },
    "uniform": {
        "description": (
            "Spatially uniform null: all grid cells assigned flux = 1.0. "
            "Tests whether any spatially structured model outperforms a featureless baseline."
        ),
        "path": None,  # generated in memory — no file needed
        "reference": "None — generated null",
        "status": "available",
        "format": "generated",
    },
    "M3": {
        "description": (
            "Segatz et al. (2013) asthenosphere-dominated tidal heating. "
            "Maximum dissipation at sub-Jovian (0°, 0°) and anti-Jovian (0°, 180°) "
            "equatorial points. Predicts hotspot concentration at equatorial longitudes."
        ),
        "path": EXTERNAL_DIR / "tidal_models" / "segatz_M3.csv",
        "reference": (
            "Segatz, M. et al. (1988) Tidal dissipation, surface heat flow, and figure "
            "of viscoelastic models of Io. Icarus, 75(2), 187–206. "
            "Grid: [VERIFY — contact authors or see Hamilton et al. 2013 EPSL supplement]"
        ),
        "status": "not_available",
        "format": "csv",
    },
    "M4": {
        "description": (
            "Segatz et al. (2013) deep-mantle-dominated tidal heating. "
            "Maximum dissipation at the poles. Predicts poleward hotspot concentration."
        ),
        "path": EXTERNAL_DIR / "tidal_models" / "segatz_M4.csv",
        "reference": (
            "Segatz, M. et al. (1988) Tidal dissipation, surface heat flow, and figure "
            "of viscoelastic models of Io. Icarus, 75(2), 187–206. "
            "Grid: [VERIFY — contact authors or see Hamilton et al. 2013 EPSL supplement]"
        ),
        "status": "not_available",
        "format": "csv",
    },
}

# Expected columns after loading any model grid
_REQUIRED_OUTPUT_COLS = ["longitude", "latitude", "flux_w_m2", "model_name", "provenance"]


def list_available_models() -> list[str]:
    """Return names of tidal models whose data files are present or generated."""
    available = []
    for name, info in TIDAL_MODEL_REGISTRY.items():
        if info["status"] == "available":
            if info["path"] is None or Path(info["path"]).exists():
                available.append(name)
    return available


def list_all_models() -> pd.DataFrame:
    """Return a summary DataFrame of all registered models and their availability."""
    rows = []
    for name, info in TIDAL_MODEL_REGISTRY.items():
        path = info["path"]
        file_exists = path is not None and Path(path).exists()
        rows.append(
            {
                "name": name,
                "status": info["status"],
                "file_exists": file_exists,
                "path": str(path) if path else "(generated)",
                "description": info["description"][:80] + "...",
                "reference": info["reference"][:60] + "...",
            }
        )
    return pd.DataFrame(rows)


def _align_to_grid(
    df: pd.DataFrame,
    resolution_deg: float = GRID_RESOLUTION_DEG,
) -> pd.DataFrame:
    """Align a tidal heating DataFrame to the project 1°×1° grid via nearest-neighbour.

    Args:
        df: DataFrame with longitude, latitude, flux_w_m2 columns.
        resolution_deg: Grid resolution in degrees.

    Returns:
        DataFrame with 64,800 rows (one per grid cell centre), columns:
        longitude, latitude, flux_w_m2.
    """
    from preprocess.grid import build_base_grid

    grid = build_base_grid()
    grid_lons = grid["lon_centre"].values
    grid_lats = grid["lat_centre"].values

    src_lons = df["longitude"].values
    src_lats = df["latitude"].values
    src_flux = df["flux_w_m2"].values

    # KD-tree on (lon, lat) — simple planar approximation adequate at 1° resolution
    tree = cKDTree(np.column_stack([src_lons, src_lats]))
    _, idx = tree.query(np.column_stack([grid_lons, grid_lats]))

    return pd.DataFrame(
        {
            "longitude": grid_lons,
            "latitude": grid_lats,
            "flux_w_m2": src_flux[idx],
        }
    )


def load_uniform_null(resolution_deg: float = GRID_RESOLUTION_DEG) -> pd.DataFrame:
    """Return a spatially uniform tidal heating grid (flux = 1.0 everywhere).

    Args:
        resolution_deg: Grid resolution in degrees.

    Returns:
        DataFrame with columns: longitude, latitude, flux_w_m2, model_name, provenance.
    """
    lons = np.arange(-180 + resolution_deg / 2, 180, resolution_deg)
    lats = np.arange(-90 + resolution_deg / 2, 90, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    df = pd.DataFrame(
        {
            "longitude": lon_grid.ravel(),
            "latitude": lat_grid.ravel(),
            "flux_w_m2": np.ones(lon_grid.size),
            "model_name": "uniform",
            "provenance": "Spatially uniform null — generated, no physical model",
        }
    )
    logger.info("Generated uniform null tidal model: %d cells.", len(df))
    return df


def load_tidal_model(name: str) -> pd.DataFrame:
    """Load a registered tidal model, aligned to the 1°×1° grid.

    Args:
        name: Model name from TIDAL_MODEL_REGISTRY. Use list_available_models()
              to see what is currently loadable.

    Returns:
        DataFrame with columns: longitude, latitude, flux_w_m2, model_name, provenance.

    Raises:
        KeyError: If name is not in TIDAL_MODEL_REGISTRY.
        FileNotFoundError: If the model's data file is not present locally.
        NotImplementedError: If the model format is not yet implemented.
    """
    if name not in TIDAL_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown tidal model '{name}'. "
            f"Registered names: {list(TIDAL_MODEL_REGISTRY)}"
        )

    info = TIDAL_MODEL_REGISTRY[name]

    if name == "uniform":
        return load_uniform_null()

    if name == "synthetic_proxy":
        from ingest.tidal_heating import load_tidal_heating_csv
        df = load_tidal_heating_csv(info["path"])
        df["model_name"] = "synthetic_proxy"
        return df

    # M3 / M4 (and future models)
    path = info["path"]
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"Tidal model '{name}' data file not found at: {path}\n\n"
            f"Description: {info['description']}\n"
            f"Reference: {info['reference']}\n\n"
            "To install:\n"
            "  1. Obtain the grid from the reference above.\n"
            f"  2. Place the CSV file at: {path}\n"
            "  3. Ensure columns: longitude (-180..180), latitude (-90..90), flux_w_m2.\n"
            "  4. Re-run this function.\n\n"
            "See data/external/SOURCES.md for further guidance."
        )

    logger.info("Loading tidal model '%s' from %s", name, path)

    if info["format"] == "csv":
        raw = pd.read_csv(path)
        required = ["longitude", "latitude", "flux_w_m2"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise ValueError(
                f"Tidal model CSV for '{name}' is missing columns: {missing}. "
                f"Available: {list(raw.columns)}"
            )
        raw[["longitude", "latitude", "flux_w_m2"]] = raw[
            ["longitude", "latitude", "flux_w_m2"]
        ].apply(pd.to_numeric, errors="coerce")

    elif info["format"] == "geotiff":
        raise NotImplementedError(
            f"GeoTIFF loading for model '{name}' is not yet implemented. "
            "Convert the raster to CSV (lon, lat, flux_w_m2) first."
        )
    else:
        raise NotImplementedError(f"Unknown format '{info['format']}' for model '{name}'.")

    aligned = _align_to_grid(raw)
    aligned["model_name"] = name
    aligned["provenance"] = info["reference"]
    logger.info("Loaded and aligned tidal model '%s': %d cells.", name, len(aligned))
    return aligned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Available tidal models:", list_available_models())
    print()
    print(list_all_models().to_string(index=False))
    print()
    uniform = load_tidal_model("uniform")
    print("Uniform model (first 5 rows):")
    print(uniform.head())
