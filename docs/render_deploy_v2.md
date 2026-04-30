# Render Deployment Notes For Io Dashboard v2

These notes prepare the v2 dashboard branch for a future Render deployment while keeping the existing v1 service safe.

## Recommended Deployment Strategy

- Keep the current Render production service on version 1 until you intentionally switch it.
- Use branch `v2-metric-dashboard` for a separate preview service, or point Render to this branch only after review.
- Do not commit Render secrets, API tokens, private URLs, or environment variable values.

## Runtime

- Python: 3.11
- App entry point: `dashboard/app.py`
- Required data checklist: `docs/data_manifest_v2.md`

## Render Web Service Settings

Use these settings for a Python web service unless you later add a Dockerfile or Render blueprint.

| setting | value |
|---|---|
| Runtime | Python |
| Branch | `v2-metric-dashboard` for preview, `main` only after merge |
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0` |
| Python version | 3.11 |

If the existing Render v1 service uses Docker, keep that configuration unchanged until a v2 Dockerfile is intentionally added or updated.

## Required Data Restore Step

The repository intentionally does not commit ignored raw files, most processed parquet files, or bulk source extraction caches. The small runtime-critical feature matrix, Explore Io power-grid files, and Explore Io hotspot catalog files are committed so Render can load the same feature, thermal-emission proxy, and hotspot coordinate data as local without a rebuild. Before launching the full v2 dashboard, restore any remaining files listed in `docs/data_manifest_v2.md` that are still marked `committed_to_github = No`.

Minimum runtime restore for full dashboard behavior:

- `data/processed/base_grid_1deg.parquet` (committed; restored by Git checkout)
- `data/processed/feature_matrix.parquet` (committed; restored by Git checkout)
- `data/raw/io_hotspot_power.csv` (committed; restored by Git checkout)
- `data/raw/io_volcanic_hotspots.csv` (committed; restored by Git checkout)
- `data/processed/hotspots_1deg_grid.parquet` (committed; restored by Git checkout)
- `data/processed/power_grid_1deg.parquet` (committed; restored by Git checkout)
- `data/processed/io_coverage_corrected_cell_maps.parquet`
- `data/processed/io_multi_instrument_coverage_cube.parquet`
- `data/processed/io_thermal_activity_events.parquet`
- `data/processed/jiram_observation_coverage.parquet`
- committed `data/results/io_*.csv` and `data/results/io_*.md`
- committed NASA 3D assets under `data/external/nasa_io_3d/`

Recommended storage options:

- Render persistent disk mounted at the repository root or copied into the repository data paths before app start.
- GitHub Release asset, for example `io-project-v2-data-artifacts.zip`, downloaded during build.
- Cloud storage bucket or manually managed artifact archive.

If you choose to regenerate artifacts during build, restore the raw/source inputs first. Build-time regeneration is slower and may be less reliable on free/limited Render instances.

## Environment Variables

Only commit environment variable names in documentation, never values. Useful non-secret names to document later:

- `PYTHON_VERSION` if Render does not infer Python 3.11 from project settings.
- Storage location variables for the external data artifact, if used.
- Any Streamlit configuration variables needed by Render.

## Render Details To Collect Later

For a clean v2 deployment handoff, record these non-secret details from the current Render setup:

- Current service name.
- Runtime type: Docker, Python web service, or Blueprint.
- Current build command.
- Current start command.
- Current branch Render follows.
- Python version.
- Environment variable names only, without values.
- Whether v2 should replace v1 or run as a separate preview service.

## Safety Notes

- Version 1 remains safe as long as Render continues following the current production branch or service configuration.
- A pull request from `v2-metric-dashboard` into `main` is reviewable without changing the live Render service.
- Do not commit `.env` files, private storage credentials, or Render deploy hooks.
