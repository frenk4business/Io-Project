"""
config.py
---------
Project-wide constants, paths, and configuration.

All magic numbers live here. Import from this module; never hardcode values
in pipeline modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
GRID_LON_MIN: float = -180.0
GRID_LON_MAX: float = 180.0
GRID_LAT_MIN: float = -90.0
GRID_LAT_MAX: float = 90.0
GRID_RESOLUTION_DEG: float = 1.0  # degrees per cell (modeling resolution)

# Derived grid dimensions
GRID_N_LON: int = int((GRID_LON_MAX - GRID_LON_MIN) / GRID_RESOLUTION_DEG)  # 360
GRID_N_LAT: int = int((GRID_LAT_MAX - GRID_LAT_MIN) / GRID_RESOLUTION_DEG)  # 180
GRID_N_CELLS: int = GRID_N_LON * GRID_N_LAT  # 64,800

# ---------------------------------------------------------------------------
# Io physical constants
# ---------------------------------------------------------------------------
IO_RADIUS_KM: float = 1821.6  # mean radius in km (IAU 2015)

# ---------------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------------
# Latitude band boundaries for 4-fold spatial CV
# Each fold is one band; model trains on the other three.
SPATIAL_CV_FOLDS: list[tuple[float, float]] = [
    (-90.0, -45.0),
    (-45.0, 0.0),
    (0.0, 45.0),
    (45.0, 90.0),
]

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS: dict = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
    "solver": "lbfgs",
}

# ---------------------------------------------------------------------------
# Raw data filenames (expected after ingest)
# ---------------------------------------------------------------------------
HOTSPOT_CATALOG_FILENAME: str = "io_volcanic_hotspots.csv"
POWER_CATALOG_FILENAME: str = "io_hotspot_power.csv"
GEOLOGY_MAP_FILENAME: str = "io_geology_map.shp"
TIDAL_HEATING_FILENAME: str = "io_tidal_heating_flux.csv"

# ---------------------------------------------------------------------------
# Processed data filenames
# ---------------------------------------------------------------------------
BASE_GRID_FILENAME: str = "base_grid_1deg.parquet"
FEATURE_MATRIX_FILENAME: str = "feature_matrix.parquet"
HOTSPOT_GRID_FILENAME: str = "hotspots_1deg_grid.parquet"
POWER_GRID_FILENAME: str = "power_grid_1deg.parquet"

# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------
# All spatial data is reprojected to this CRS before gridding.
# Simple geographic CRS — appropriate for a full-body planetary grid.
TARGET_CRS: str = "EPSG:4326"

# ---------------------------------------------------------------------------
# Tidal model registry (Scientific Analysis layer)
# ---------------------------------------------------------------------------
# Directory for externally obtained tidal heating grid files.
# Place M3 / M4 CSVs here; see ingest/tidal_models.py for format details.
TIDAL_MODEL_DIR: Path = EXTERNAL_DIR / "tidal_models"

# Canonical names used by ingest.tidal_models.TIDAL_MODEL_REGISTRY
TIDAL_MODEL_NAMES: list[str] = ["synthetic_proxy", "uniform", "M3", "M4"]

# Models whose data files are generated in memory (no external file required)
TIDAL_MODELS_GENERATED: list[str] = ["uniform"]

# Models that are available after a fresh repo clone (no external download needed)
TIDAL_MODELS_DEFAULT_AVAILABLE: list[str] = ["synthetic_proxy", "uniform"]
