"""
analysis/spatial_stats.py
-------------------------
Spatial point-pattern analysis of the Io hotspot catalogue.

Implements:
  - Ripley's K-function on a sphere, compared to the CSR expectation
        K_csr(r) = 2π R² (1 − cos(r/R))
  - Pair-correlation function g(r) = (1 / (2π R sin(r/R))) · dK/dr
  - Nearest-neighbour (NN) distance CDF compared to the Poisson expectation
        F_csr(r) = 1 − exp(−λ · 2π R² (1 − cos(r/R)))

CSR envelopes are obtained by Monte Carlo simulation of N uniform-on-sphere
points; the default permutation count (199) gives a 95% two-sided envelope.

Great-circle distances use the project's existing haversine utility in
`features/synthetic._haversine_km`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import IO_RADIUS_KM, PROCESSED_DIR
from features.synthetic import _haversine_km

logger = logging.getLogger(__name__)

RESULTS_DIR: Path = PROCESSED_DIR.parent / "results"
SPATIAL_STATS_NPZ: Path = RESULTS_DIR / "spatial_stats.npz"
SPATIAL_STATS_PNG: Path = RESULTS_DIR / "spatial_stats.png"

DEFAULT_N_SIMULATIONS: int = 199
DEFAULT_RADII_KM: np.ndarray = np.linspace(50.0, 2500.0, 50)
SPHERE_SURFACE_FACTOR: float = 4.0 * np.pi  # * R² = total surface area


def _pairwise_great_circle(
    lats: np.ndarray, lons: np.ndarray, radius_km: float
) -> np.ndarray:
    """All pairwise great-circle distances (km). Diagonal set to +inf."""
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, :] = _haversine_km(lats, lons, float(lats[i]), float(lons[i]), radius_km)
    np.fill_diagonal(D, np.inf)
    return D


def _uniform_on_sphere(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw n uniform points on a unit sphere; return (lat_deg, lon_deg)."""
    u = rng.uniform(0.0, 1.0, size=n)
    v = rng.uniform(0.0, 1.0, size=n)
    lat = np.degrees(np.arcsin(2 * u - 1))
    lon = 360.0 * v - 180.0
    return lat, lon


def ripley_k(
    lats: np.ndarray,
    lons: np.ndarray,
    radii_km: np.ndarray = DEFAULT_RADII_KM,
    sphere_radius_km: float = IO_RADIUS_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Ripley's K and the CSR expectation.

    Returns (K_observed, K_csr) on the supplied radii grid.
    """
    n = len(lats)
    if n < 2:
        raise ValueError("Need at least 2 points for Ripley's K.")
    area = SPHERE_SURFACE_FACTOR * sphere_radius_km * sphere_radius_km
    lam = n / area
    D = _pairwise_great_circle(np.asarray(lats), np.asarray(lons), sphere_radius_km)
    k_obs = np.empty_like(radii_km, dtype=float)
    for i, r in enumerate(radii_km):
        # K(r) = (1/λ) · (1/n) · Σ_i Σ_{j≠i} 1[d_ij ≤ r]
        counts = (D <= r).sum(axis=1)  # neighbours within r for each point
        k_obs[i] = counts.mean() / lam
    k_csr = 2 * np.pi * sphere_radius_km * sphere_radius_km * (
        1 - np.cos(radii_km / sphere_radius_km)
    )
    return k_obs, k_csr


def pair_correlation_g(
    k_values: np.ndarray,
    radii_km: np.ndarray,
    sphere_radius_km: float = IO_RADIUS_KM,
) -> np.ndarray:
    """Pair-correlation function derived from K values."""
    dk_dr = np.gradient(k_values, radii_km)
    denom = 2 * np.pi * sphere_radius_km * np.sin(radii_km / sphere_radius_km)
    # Avoid division by zero at r=0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(denom > 0, dk_dr / denom, np.nan)
    return g


def nn_distance_cdf(
    lats: np.ndarray,
    lons: np.ndarray,
    radii_km: np.ndarray = DEFAULT_RADII_KM,
    sphere_radius_km: float = IO_RADIUS_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical nearest-neighbour CDF and Poisson expectation."""
    n = len(lats)
    D = _pairwise_great_circle(np.asarray(lats), np.asarray(lons), sphere_radius_km)
    nn = D.min(axis=1)  # nearest-neighbour distances
    cdf = np.array([(nn <= r).mean() for r in radii_km])
    area = SPHERE_SURFACE_FACTOR * sphere_radius_km * sphere_radius_km
    lam = n / area
    csr_cdf = 1 - np.exp(
        -lam * 2 * np.pi * sphere_radius_km * sphere_radius_km
        * (1 - np.cos(radii_km / sphere_radius_km))
    )
    return cdf, csr_cdf


def csr_envelopes_k(
    n_points: int,
    radii_km: np.ndarray = DEFAULT_RADII_KM,
    sphere_radius_km: float = IO_RADIUS_KM,
    n_sims: int = DEFAULT_N_SIMULATIONS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo 95% envelopes of K under CSR for n_points on a sphere."""
    rng = np.random.default_rng(seed)
    area = SPHERE_SURFACE_FACTOR * sphere_radius_km * sphere_radius_km
    lam = n_points / area
    all_k = np.empty((n_sims, len(radii_km)))
    for s in range(n_sims):
        lat_s, lon_s = _uniform_on_sphere(n_points, rng)
        D = _pairwise_great_circle(lat_s, lon_s, sphere_radius_km)
        for i, r in enumerate(radii_km):
            all_k[s, i] = (D <= r).sum(axis=1).mean() / lam
    lo = np.quantile(all_k, 0.025, axis=0)
    hi = np.quantile(all_k, 0.975, axis=0)
    return lo, hi


def compute_spatial_stats(
    catalog: pd.DataFrame,
    radii_km: np.ndarray = DEFAULT_RADII_KM,
    sphere_radius_km: float = IO_RADIUS_KM,
    n_sims: int = DEFAULT_N_SIMULATIONS,
    seed: int = 42,
) -> dict:
    """Full spatial-stats bundle: K, g, NN CDF, and CSR envelopes for K."""
    lats = catalog["latitude"].astype(float).values
    lons = catalog["longitude"].astype(float).values
    k_obs, k_csr = ripley_k(lats, lons, radii_km, sphere_radius_km)
    g_obs = pair_correlation_g(k_obs, radii_km, sphere_radius_km)
    nn_obs, nn_csr = nn_distance_cdf(lats, lons, radii_km, sphere_radius_km)
    lo, hi = csr_envelopes_k(len(lats), radii_km, sphere_radius_km, n_sims, seed)
    clustered = bool((k_obs > hi).any())
    return {
        "radii_km": radii_km,
        "k_observed": k_obs,
        "k_csr": k_csr,
        "k_env_lo": lo,
        "k_env_hi": hi,
        "g_observed": g_obs,
        "nn_cdf_observed": nn_obs,
        "nn_cdf_csr": nn_csr,
        "n_points": len(lats),
        "n_sims": n_sims,
        "sphere_radius_km": sphere_radius_km,
        "clustered_at_small_radii": clustered,
    }


def plot_spatial_stats(results: dict, out_path: Path | None = None) -> Path:
    """Three-panel figure: K vs CSR envelope; g(r); NN CDF vs Poisson."""
    out_path = out_path or SPATIAL_STATS_PNG
    out_path.parent.mkdir(parents=True, exist_ok=True)

    r = results["radii_km"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].fill_between(r, results["k_env_lo"], results["k_env_hi"],
                         alpha=0.25, color="grey", label="95% CSR envelope")
    axes[0].plot(r, results["k_csr"], "--", color="grey", label="K_CSR (expectation)")
    axes[0].plot(r, results["k_observed"], color="C3", label="K_observed")
    axes[0].set_xlabel("r (km)")
    axes[0].set_ylabel("K(r)  (km²)")
    axes[0].set_title("Ripley's K on a sphere")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].axhline(1.0, linestyle="--", color="grey", lw=1, label="CSR (g=1)")
    axes[1].plot(r, results["g_observed"], color="C0")
    axes[1].set_xlabel("r (km)")
    axes[1].set_ylabel("g(r)")
    axes[1].set_title("Pair-correlation function")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(r, results["nn_cdf_csr"], "--", color="grey", label="Poisson CDF")
    axes[2].plot(r, results["nn_cdf_observed"], color="C2", label="Observed CDF")
    axes[2].set_xlabel("r (km)")
    axes[2].set_ylabel("P(NN ≤ r)")
    axes[2].set_title("Nearest-neighbour CDF")
    axes[2].legend(loc="lower right", fontsize=8)

    caption = (
        f"n={results['n_points']} hotspots, {results['n_sims']} Monte Carlo CSR simulations. "
        "K above the envelope ⇒ clustering; below ⇒ regularity."
    )
    fig.suptitle(caption, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_spatial_stats(results: dict, path: Path | None = None) -> Path:
    path = path or SPATIAL_STATS_NPZ
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        radii_km=results["radii_km"],
        k_observed=results["k_observed"],
        k_csr=results["k_csr"],
        k_env_lo=results["k_env_lo"],
        k_env_hi=results["k_env_hi"],
        g_observed=results["g_observed"],
        nn_cdf_observed=results["nn_cdf_observed"],
        nn_cdf_csr=results["nn_cdf_csr"],
    )
    logger.info("Saved spatial stats arrays to %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ingest.hotspot_catalog import load_hotspot_catalog

    cat = load_hotspot_catalog()
    res = compute_spatial_stats(cat, n_sims=99)  # lighter for CLI
    save_spatial_stats(res)
    plot_spatial_stats(res)
    print("Clustering detected at small radii:", res["clustered_at_small_radii"])
