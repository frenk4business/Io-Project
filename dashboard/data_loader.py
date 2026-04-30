"""
dashboard/data_loader.py
------------------------
Centralized dashboard data loading, caching, and production diagnostics.

The dashboard should serve restored, dashboard-ready artifacts in production.
This module keeps file expectations in one place so Streamlit pages, tests,
and deployment checks report the same paths and recovery guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import (
    BASE_GRID_FILENAME,
    FEATURE_MATRIX_FILENAME,
    HOTSPOT_CATALOG_FILENAME,
    HOTSPOT_GRID_FILENAME,
    POWER_CATALOG_FILENAME,
    POWER_GRID_FILENAME,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
)

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
NASA_3D_DIR = PROJECT_ROOT / "data" / "external" / "nasa_io_3d"


@dataclass(frozen=True)
class DashboardFileSpec:
    key: str
    path: Path
    purpose: str
    restore_policy: str
    regenerate_command: str | None = None
    required: bool = True

    def exists(self) -> bool:
        return self.path.exists()


FILE_SPECS: dict[str, DashboardFileSpec] = {
    "base_grid": DashboardFileSpec(
        "base_grid",
        PROCESSED_DIR / BASE_GRID_FILENAME,
        "Shared 1 degree Io longitude-latitude grid used by map, globe, and activity layers.",
        "Committed artifact; restore by Git checkout or from the production data archive.",
        "python -m preprocess.grid",
    ),
    "feature_matrix": DashboardFileSpec(
        "feature_matrix",
        PROCESSED_DIR / FEATURE_MATRIX_FILENAME,
        "Main dashboard feature matrix on the common 1 degree grid.",
        "Committed artifact; restore by Git checkout or from the production data archive.",
        "python -m features.build",
    ),
    "hotspot_catalog": DashboardFileSpec(
        "hotspot_catalog",
        RAW_DIR / HOTSPOT_CATALOG_FILENAME,
        "Curated named Io hotspot catalog used by Explore Io maps and 3D globe markers.",
        "Committed small runtime input; restore by Git checkout.",
        "python -m ingest.download",
    ),
    "hotspot_grid": DashboardFileSpec(
        "hotspot_grid",
        PROCESSED_DIR / HOTSPOT_GRID_FILENAME,
        "Named hotspot occurrences aligned to the common 1 degree grid.",
        "Committed artifact; restore by Git checkout or from the production data archive.",
        "python -m preprocess.align_layers",
    ),
    "thermal_proxy_csv": DashboardFileSpec(
        "thermal_proxy_csv",
        RAW_DIR / POWER_CATALOG_FILENAME,
        "Curated Davies/JIRAM estimated thermal-emission proxy input.",
        "Committed small runtime input; restore by Git checkout.",
        None,
    ),
    "power_grid": DashboardFileSpec(
        "power_grid",
        PROCESSED_DIR / POWER_GRID_FILENAME,
        "Estimated thermal-emission proxy aggregated to the common 1 degree grid.",
        "Committed artifact; restore by Git checkout or from the production data archive.",
        "python -m preprocess.power_grid",
    ),
    "coverage_cell_maps": DashboardFileSpec(
        "coverage_cell_maps",
        PROCESSED_DIR / "io_coverage_corrected_cell_maps.parquet",
        "Dashboard-ready metadata-normalized activity cell maps.",
        "Restore externally from the v2 production data artifact if not committed.",
        "python -m analysis.coverage_corrected_volcanism",
    ),
    "coverage_cube": DashboardFileSpec(
        "coverage_cube",
        PROCESSED_DIR / "io_multi_instrument_coverage_cube.parquet",
        "Metadata observation cube for instrument/product/time-bin activity summaries.",
        "Restore externally from the v2 production data artifact if not committed.",
        "python -m analysis.coverage_corrected_volcanism",
    ),
    "activity_events": DashboardFileSpec(
        "activity_events",
        PROCESSED_DIR / "io_thermal_activity_events.parquet",
        "Normalized multi-instrument thermal activity event table.",
        "Restore externally from the v2 production data artifact if not committed.",
        "python -m ingest.thermal_activity_events",
    ),
    "jiram_coverage": DashboardFileSpec(
        "jiram_coverage",
        PROCESSED_DIR / "jiram_observation_coverage.parquet",
        "JIRAM product metadata coverage layer used by time-resolved activity views.",
        "Restore externally from the v2 production data artifact if not committed.",
        "python -m ingest.jiram_coverage",
    ),
    "research_question": DashboardFileSpec(
        "research_question",
        RESULTS_DIR / "io_research_question_evaluation.md",
        "Reviewer-style summary of the final multi-metric research question evaluation.",
        "Committed result file; restore by Git checkout.",
        None,
    ),
    "metric_interpretation": DashboardFileSpec(
        "metric_interpretation",
        RESULTS_DIR / "io_metric_interpretation_summary.csv",
        "Metric-pair interpretation table used by Scientific Analysis.",
        "Committed result file; restore by Git checkout.",
        "python -m analysis.coverage_corrected_volcanism",
    ),
    "metric_correlation": DashboardFileSpec(
        "metric_correlation",
        RESULTS_DIR / "io_metric_correlation_matrix.csv",
        "Spearman correlation matrix between key grid metrics.",
        "Committed result file; restore by Git checkout.",
        "python -m analysis.coverage_corrected_volcanism",
    ),
    "rank_overlap": DashboardFileSpec(
        "rank_overlap",
        RESULTS_DIR / "io_rank_overlap.csv",
        "Top-rank overlap table for metric comparisons.",
        "Committed result file; restore by Git checkout.",
        "python -m analysis.coverage_corrected_volcanism",
    ),
    "js_divergence": DashboardFileSpec(
        "js_divergence",
        RESULTS_DIR / "io_js_divergence.csv",
        "Jensen-Shannon divergence table for metric distributions.",
        "Committed result file; restore by Git checkout.",
        "python -m analysis.coverage_corrected_volcanism",
    ),
    "latitude_contributions": DashboardFileSpec(
        "latitude_contributions",
        RESULTS_DIR / "io_latitude_band_contributions.csv",
        "Latitude-band contribution table for spatial metric comparison.",
        "Committed result file; restore by Git checkout.",
        "python -m analysis.coverage_corrected_volcanism",
    ),
    "top_n_intensity": DashboardFileSpec(
        "top_n_intensity",
        RESULTS_DIR / "io_top_n_cumulative_intensity.csv",
        "Top-N cumulative concentration curve for intensity layers.",
        "Committed result file; restore by Git checkout.",
        "python -m analysis.coverage_corrected_volcanism",
    ),
    "power_concentration": DashboardFileSpec(
        "power_concentration",
        RESULTS_DIR / "io_power_concentration_summary.csv",
        "Top-N concentration summary for the Davies/JIRAM estimated proxy power layer.",
        "Committed result file; restore by Git checkout.",
        "python -m analysis.coverage_corrected_volcanism",
    ),
    "nasa_glb": DashboardFileSpec(
        "nasa_glb",
        NASA_3D_DIR / "io_nasa.glb",
        "NASA Io 3D model asset used by the Io Experience viewer.",
        "Committed asset; restore by Git checkout or external artifact if repository policy changes.",
        None,
    ),
    "nasa_texture": DashboardFileSpec(
        "nasa_texture",
        NASA_3D_DIR / "io_nasa_texture.png",
        "NASA Io texture used by the Io Experience Three.js viewer.",
        "Committed asset; restore by Git checkout or external artifact if repository policy changes.",
        None,
    ),
    "model": DashboardFileSpec(
        "model",
        MODELS_DIR / "logistic_regression.pkl",
        "Legacy logistic-regression model artifact for exploration diagnostics.",
        "Committed model artifact; restore by Git checkout.",
        "python -m models.train",
    ),
    "scaler": DashboardFileSpec(
        "scaler",
        MODELS_DIR / "scaler.pkl",
        "Feature scaler paired with the legacy logistic-regression model.",
        "Committed model artifact; restore by Git checkout.",
        "python -m models.train",
    ),
}

REQUIRED_DASHBOARD_FILE_KEYS: tuple[str, ...] = tuple(FILE_SPECS)


def get_file_spec(key: str) -> DashboardFileSpec:
    return FILE_SPECS[key]


def missing_file_diagnostic(key: str) -> str:
    spec = get_file_spec(key)
    lines = [
        f"Expected path: `{spec.path}`",
        f"Purpose: {spec.purpose}",
        f"Restore policy: {spec.restore_policy}",
    ]
    if spec.regenerate_command:
        lines.append(f"Regenerate command: `{spec.regenerate_command}`")
    else:
        lines.append("Regenerate command: not available; restore the file from Git or the production data artifact.")
    return "\n".join(f"- {line}" for line in lines)


def combined_missing_file_diagnostic(keys: list[str] | tuple[str, ...]) -> str:
    missing = [get_file_spec(key) for key in keys if not get_file_spec(key).exists()]
    if not missing:
        return "All required files are present."
    sections = []
    for spec in missing:
        sections.append(f"**{spec.key}**\n{missing_file_diagnostic(spec.key)}")
    return "\n\n".join(sections)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading hotspot catalog...")
def load_hotspot_catalog_cached(path: str | Path | None = None) -> pd.DataFrame:
    from ingest.hotspot_catalog import load_hotspot_catalog

    return load_hotspot_catalog(Path(path) if path is not None else None)


@st.cache_data(show_spinner="Loading feature matrix...")
def load_feature_matrix_cached(path: str | Path | None = None) -> pd.DataFrame:
    from features.build import load_feature_matrix

    return load_feature_matrix(Path(path) if path is not None else None)


@st.cache_data(show_spinner="Loading base grid...")
def load_base_grid_cached(path: str | Path | None = None) -> pd.DataFrame:
    from preprocess.grid import load_base_grid

    return load_base_grid(Path(path) if path is not None else None)


@st.cache_data(show_spinner="Loading power grid...")
def load_power_grid_cached(path: str | Path | None = None) -> pd.DataFrame:
    from preprocess.power_grid import load_power_grid

    return load_power_grid(Path(path) if path is not None else None)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model_cached(
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
) -> tuple[Any, Any]:
    from models.train import load_model

    return load_model(
        Path(model_path) if model_path is not None else None,
        Path(scaler_path) if scaler_path is not None else None,
    )


@st.cache_data(show_spinner="Loading JIRAM observation coverage...")
def load_jiram_observation_coverage_cached(path: str | Path | None = None) -> pd.DataFrame:
    target = Path(path) if path is not None else FILE_SPECS["jiram_coverage"].path
    if not target.exists():
        raise FileNotFoundError(f"JIRAM observation coverage not found at {target}.")
    return _read_parquet(target)


@st.cache_data(show_spinner="Loading thermal activity events...")
def load_saved_activity_events_cached(path: str | Path | None = None) -> pd.DataFrame:
    from ingest.thermal_activity_events import load_saved_activity_events

    return load_saved_activity_events(Path(path) if path is not None else None)


@st.cache_data(show_spinner=False)
def load_result_csv_cached(path: str | Path, columns: tuple[str, ...] = ()) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=list(columns))
    try:
        return pd.read_csv(target)
    except Exception:
        return pd.DataFrame(columns=list(columns))


@st.cache_data(show_spinner=False)
def load_research_question_text_cached(path: str | Path | None = None) -> str:
    target = Path(path) if path is not None else FILE_SPECS["research_question"].path
    if not target.exists():
        return ""
    try:
        return target.read_text(encoding="utf-8")
    except Exception:
        return ""


def _time_activity_from_saved(
    feature_matrix: pd.DataFrame,
    activity_events: pd.DataFrame,
    cell_maps: pd.DataFrame,
    coverage_cube: pd.DataFrame,
    min_observations: int,
    optional_status: dict[str, str] | None = None,
) -> dict:
    from analysis.coverage_corrected_volcanism import (
        _comparison_metrics,
        _scientific_summary,
        _time_maps,
        prepare_activity_events,
    )

    events = prepare_activity_events(activity_events)
    time_maps = _time_maps(events, coverage_cube, min_observations=min_observations)
    comparison = _comparison_metrics(cell_maps)
    corrected = (
        pd.to_numeric(cell_maps["coverage_corrected_intensity"], errors="coerce").fillna(0)
        if "coverage_corrected_intensity" in cell_maps
        else pd.Series(0, index=cell_maps.index)
    )
    has_hotspot = (
        pd.to_numeric(feature_matrix["has_hotspot"], errors="coerce").fillna(0)
        if "has_hotspot" in feature_matrix
        else pd.Series(0, index=feature_matrix.index)
    )
    occurrence = (
        pd.to_numeric(cell_maps["occurrence_event_count"], errors="coerce").fillna(0)
        if "occurrence_event_count" in cell_maps
        else pd.Series(0, index=cell_maps.index)
    )
    data_quality = {
        "activity_event_rows": int(len(events)),
        "coverage_cube_rows": int(len(coverage_cube)),
        "coverage_cells": int(coverage_cube["cell_id"].nunique()) if "cell_id" in coverage_cube else 0,
        "coverage_time_bins": int(coverage_cube["time_bin"].nunique()) if "time_bin" in coverage_cube else 0,
        "coverage_instruments": sorted(coverage_cube["instrument"].dropna().astype(str).unique().tolist())
        if "instrument" in coverage_cube
        else [],
        "nonzero_coverage_corrected_cells": int((corrected > 0).sum()),
        "min_observations": int(min_observations),
        "coverage_method": "metadata-based event/product cell observation approximation",
        "unit_policy": "raw units are kept in separate layers; combined map uses within-family percentiles only",
        "optional_dataset_status_detail": optional_status or {"saved_artifacts": "loaded"},
        "named_hotspot_cells": int((has_hotspot > 0).sum()),
        "thermal_cells": int((occurrence > 0).sum()),
    }
    return {
        "cell_maps": cell_maps,
        "cell_activity": cell_maps,
        "time_maps": time_maps,
        "coverage_cube": coverage_cube,
        "comparison_metrics": comparison,
        "regional_summary": comparison["latitude_band_contributions"],
        "comparison_summary": comparison["spearman_correlation"],
        "data_quality": data_quality,
        "scientific_summary": _scientific_summary(cell_maps, comparison),
        "available_instruments": data_quality["coverage_instruments"],
        "available_time_bins": sorted(coverage_cube["time_bin"].dropna().astype(str).unique().tolist())
        if "time_bin" in coverage_cube
        else [],
    }


@st.cache_data(show_spinner="Loading time-resolved activity...")
def load_saved_or_compute_time_activity_cached(
    _feature_matrix: pd.DataFrame,
    _power_grid: pd.DataFrame,
    _coverage: pd.DataFrame | None,
    instrument: str = "combined",
    time_bin: str = "all",
    allow_compute: bool = False,
) -> dict:
    required = ["activity_events", "coverage_cell_maps", "coverage_cube"]
    if all(FILE_SPECS[key].exists() for key in required):
        activity_events = _read_parquet(FILE_SPECS["activity_events"].path)
        cell_maps = _read_parquet(FILE_SPECS["coverage_cell_maps"].path)
        coverage_cube = _read_parquet(FILE_SPECS["coverage_cube"].path)
        return _time_activity_from_saved(
            _feature_matrix,
            activity_events,
            cell_maps,
            coverage_cube,
            min_observations=1,
        )

    if not allow_compute:
        missing = [key for key in required if not FILE_SPECS[key].exists()]
        raise FileNotFoundError(
            "Time-resolved activity artifacts are missing:\n"
            + combined_missing_file_diagnostic(tuple(missing))
        )

    from analysis.coverage_corrected_volcanism import compute_coverage_corrected_volcanism
    from ingest.thermal_activity_events import load_activity_events

    activity_events, optional_status = load_activity_events(include_optional=True)
    result = compute_coverage_corrected_volcanism(
        _feature_matrix,
        activity_events,
        jiram_coverage=_coverage,
        min_observations=1,
        persist_outputs=False,
    )
    comparison = result["comparison_metrics"]
    result["cell_activity"] = result["cell_maps"]
    result["regional_summary"] = comparison["latitude_band_contributions"]
    result["comparison_summary"] = comparison["spearman_correlation"]
    result["data_quality"]["optional_dataset_status_detail"] = optional_status
    has_hotspot = (
        pd.to_numeric(_feature_matrix["has_hotspot"], errors="coerce").fillna(0)
        if "has_hotspot" in _feature_matrix
        else pd.Series(0, index=_feature_matrix.index)
    )
    result["data_quality"]["named_hotspot_cells"] = int((has_hotspot > 0).sum())
    result["data_quality"]["thermal_cells"] = int((result["cell_maps"]["occurrence_event_count"] > 0).sum())
    result["available_instruments"] = result["data_quality"].get("coverage_instruments", [])
    result["available_time_bins"] = sorted(result["coverage_cube"]["time_bin"].dropna().astype(str).unique().tolist())
    return result
