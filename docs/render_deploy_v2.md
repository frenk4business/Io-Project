# Render Deployment Notes For Io Dashboard v2

These notes prepare the v2 dashboard branch for a future Render deployment while keeping the existing v1 service safe.

## Recommended Deployment Strategy

- Keep the current Render production service on version 1 until you intentionally switch it.
- Use branch `v2-metric-dashboard` for a separate preview service, or point Render to this branch only after review.
- Do not commit Render secrets, API tokens, private URLs, or environment variable values.

## Runtime

- Python: 3.11 pinned by repository-root `runtime.txt` (`python-3.11.9`)
- App entry point: `dashboard/app.py`
- Required data checklist: `docs/data_manifest_v2.md`
- Production data check: `python scripts/check_dashboard_data.py`

Do not rely on Python 3.14 for this app. Render should detect `runtime.txt`; if the service has an explicit Python version override, set it to Python 3.11.

## Render Web Service Settings

Use these settings for a Python web service unless you later add a Dockerfile or Render blueprint.

| setting | value |
|---|---|
| Runtime | Python |
| Branch | `v2-metric-dashboard` for preview, `main` only after merge |
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false` |
| Python version | 3.11 (`runtime.txt`: `python-3.11.9`) |
| Recommended instance size | Paid instance with enough RAM for full Plotly/Three.js/science pages |

Free or very small Render instances may still restart under concurrent users, first-load cache warming, large dataframe serialization, Plotly/WebGL figure creation, or the embedded NASA texture data URL. The production refactor reduces repeated loads and rerun recomputation; it does not create a light/demo version.

If the existing Render v1 service uses Docker, keep that configuration unchanged until a v2 Dockerfile is intentionally added or updated.

## Required Data Restore Step

The repository intentionally does not commit ignored raw files, most processed parquet files, or bulk source extraction caches. The small runtime-critical feature matrix, Explore Io power-grid files, and Explore Io hotspot catalog files are committed so Render can load the same feature, thermal-emission proxy, and hotspot coordinate data as local without a rebuild. Before launching the full v2 dashboard, restore any remaining files listed in `docs/data_manifest_v2.md` that are still marked `committed_to_github = No`.

Production should serve dashboard-ready artifacts. Do not run heavy ingest, preprocessing, training, or coverage-analysis pipelines during Streamlit request handling on Render.

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

After restore, run:

```bash
python scripts/check_dashboard_data.py
```

Deploy only after this check prints `PASS`.

Recommended storage options:

- Render persistent disk mounted at the repository root or copied into the repository data paths before app start.
- GitHub Release asset, for example `io-project-v2-data-artifacts.zip`, downloaded during build.
- Cloud storage bucket or manually managed artifact archive.

If you choose to regenerate artifacts during build, restore the raw/source inputs first. Build-time regeneration is slower and may be less reliable on free/limited Render instances.

## Streamlit Compatibility Notes

- The dashboard uses `width="stretch"` / `width="content"` instead of deprecated `use_container_width`.
- `visualization/nasa_io_model_viewer.py` intentionally still uses `components.html(...)`. That viewer embeds generated inline Three.js HTML and a local texture data URL, not a remote/local URL suitable for `st.iframe`. Replacing it with `st.iframe` would require serving a separate HTML asset and is outside this production-stability refactor.

## Redeploy Steps

1. Push the branch to GitHub.
2. In Render, open the Io dashboard web service.
3. Use **Manual Deploy**.
4. Choose **Clear build cache & deploy** so Render picks up `runtime.txt`, dependency changes, and the Streamlit API update.

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
