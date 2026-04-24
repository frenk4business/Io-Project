"""
analysis/tidal_templates.py
---------------------------
Analytical tidal-heating *templates* for hypothesis comparison.

Scientific status
-----------------
These are NOT the full Segatz / Beuthe / Tyler dissipation integrals. They are
closed-form analytical patterns whose spatial morphology reproduces, at the
qualitative level, the dominant surface-heat-flux patterns predicted by the
three families of published tidal-heating hypotheses for Io:

    1. ``asthenosphere`` (Segatz M3-like)
         Near-surface dissipation in a thin low-viscosity asthenosphere.
         Equatorial maxima near the sub-Jovian (0°, 0°) and anti-Jovian
         (180°, 0°) points; polar minima.
         Qualitative reference: Segatz et al. (1988), Icarus 75, 187–206
         (their Fig. 4a); Beuthe (2013), Icarus 223, 308–329 (eq. 92).

    2. ``deep_mantle`` (Segatz M4-like)
         Dissipation concentrated in a high-viscosity deep mantle.
         Polar maxima, equatorial minima.
         Qualitative reference: Segatz et al. (1988), Fig. 4b;
         Beuthe (2013), eq. 93.

    3. ``magma_ocean`` (Tyler-like / Hamilton 2013-like)
         Dissipation dominated by a global subsurface magma ocean
         (Obliquity tide + eccentricity tide on a thin fluid layer).
         Equatorial band, with modest longitudinal modulation.
         Qualitative reference: Tyler et al. (2015), ApJS 218, 22;
         Hamilton et al. (2013), EPSL 361, 272–286.

Each template is NORMALISED to unit global mean (∫ h cos φ dφ dλ / 4π = 1),
which makes them directly comparable as candidate *intensity fields* for a
Poisson point process (see ``analysis.hypothesis_comparison``).

What this module is for
-----------------------
The templates let us ask the question:

    "Given the observed USGS catalogue, which published hypothesis about the
     spatial distribution of tidal heating would produce the catalogue with
     highest likelihood?"

The answer is not a claim that the catalogue was *caused* by any one
hypothesis — coverage bias, catalogue vintage, and catalogue completeness
all confound this. But the relative log-likelihoods quantify which pattern
is *most consistent* with the observations, and that is a testable,
publishable question.

What this module is NOT
-----------------------
- It is not a geodynamic simulation.
- It is not a reproduction of Segatz or Beuthe's numerical integrals.
- The coefficients are stylised; exact amplitudes should not be compared
  to published heat-flux values (W m⁻²). Only the *spatial pattern* is used.
- Replace these with real gridded outputs as soon as they are available;
  the ``ingest/tidal_models.py`` registry is the drop-in point.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TemplateName = Literal[
    "asthenosphere", "deep_mantle", "magma_ocean", "uniform", "synthetic_proxy"
]

TEMPLATE_NAMES: tuple[TemplateName, ...] = (
    "asthenosphere",
    "deep_mantle",
    "magma_ocean",
    "uniform",
    "synthetic_proxy",
)

TEMPLATE_REFERENCES: dict[str, str] = {
    "asthenosphere": "Segatz+1988 Fig.4a; Beuthe 2013 eq.92 (stylised)",
    "deep_mantle": "Segatz+1988 Fig.4b; Beuthe 2013 eq.93 (stylised)",
    "magma_ocean": "Tyler+2015; Hamilton+2013 (stylised)",
    "uniform": "Null hypothesis",
    "synthetic_proxy": "Project's original cos²·cos² placeholder",
}


def _asthenosphere(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Segatz M3-like pattern.

    Dominant term: cos²φ · (1 + 2 cos²λ) / 3. Peaks at (0°, 0°) and (180°, 0°),
    falls to small values at poles. The 2·cos²λ term captures the longitudinal
    tidal-bulge modulation; the cos²φ envelope confines heating to the equator.
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    h = np.cos(phi) ** 2 * (1.0 + 2.0 * np.cos(lam) ** 2) / 3.0
    return h


def _deep_mantle(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Segatz M4-like pattern.

    Dominant term: sin²φ + small equatorial baseline. Peaks at ±90° latitude,
    minima at equator. Weak longitudinal modulation reflects the residual
    tidal-bulge contribution at depth.
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    h = np.sin(phi) ** 2 + 0.15 * np.cos(phi) ** 2 * (1.0 + 0.3 * np.cos(2 * lam))
    return h


def _magma_ocean(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Tyler / Hamilton magma-ocean-like pattern.

    Dominated by the eccentricity tide on a global thin fluid layer:
    broad equatorial band, weak longitudinal modulation (the pattern is
    more uniform in longitude than the asthenosphere case).
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    h = np.cos(phi) ** 2 * (1.0 + 0.3 * np.cos(2 * lam))
    return h


def _synthetic_proxy(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Project's original cos²λ·cos²φ + 0.3·sin²φ placeholder (for reference)."""
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    return np.cos(lam) ** 2 * np.cos(phi) ** 2 + 0.3 * np.sin(phi) ** 2


def _uniform(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    return np.ones_like(lat_deg, dtype=float)


_TEMPLATE_FUNCS = {
    "asthenosphere": _asthenosphere,
    "deep_mantle": _deep_mantle,
    "magma_ocean": _magma_ocean,
    "uniform": _uniform,
    "synthetic_proxy": _synthetic_proxy,
}


def _normalise_to_unit_mean(values: np.ndarray, lats_deg: np.ndarray) -> np.ndarray:
    """Rescale so the area-weighted mean (on a sphere) equals 1."""
    weights = np.cos(np.radians(lats_deg))
    mean = np.sum(values * weights) / np.sum(weights)
    if mean <= 0:
        raise ValueError("Template has non-positive area-weighted mean.")
    return values / mean


def evaluate_template(
    name: TemplateName,
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
) -> np.ndarray:
    """Evaluate a named template at the given grid-cell centres.

    Returns values normalised so the area-weighted mean on the sphere is 1.
    This lets templates be compared as Poisson-process intensity fields.
    """
    if name not in _TEMPLATE_FUNCS:
        raise KeyError(
            f"Unknown template '{name}'. Available: {list(_TEMPLATE_FUNCS)}"
        )
    raw = _TEMPLATE_FUNCS[name](lats_deg, lons_deg)
    # Guard against negative values from stylised coefficients.
    raw = np.clip(raw, a_min=1e-6, a_max=None)
    return _normalise_to_unit_mean(raw, lats_deg)


def evaluate_all_templates(
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
) -> pd.DataFrame:
    """Return a DataFrame with one column per template, indexed by input order."""
    out = {name: evaluate_template(name, lats_deg, lons_deg) for name in TEMPLATE_NAMES}
    return pd.DataFrame(out)


def attach_templates_to_grid(grid: pd.DataFrame) -> pd.DataFrame:
    """Attach tidal_template_<name> columns to a feature matrix.

    Args:
        grid: DataFrame with `lat_centre`, `lon_centre` columns (1°×1° grid).

    Returns:
        A copy with new columns ``tidal_template_asthenosphere``,
        ``tidal_template_deep_mantle``, ``tidal_template_magma_ocean``,
        ``tidal_template_uniform``, ``tidal_template_synthetic_proxy``.
    """
    required = {"lat_centre", "lon_centre"}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"Grid is missing required columns: {missing}")

    lats = grid["lat_centre"].to_numpy()
    lons = grid["lon_centre"].to_numpy()

    out = grid.copy()
    for name in TEMPLATE_NAMES:
        out[f"tidal_template_{name}"] = evaluate_template(name, lats, lons)
    return out
