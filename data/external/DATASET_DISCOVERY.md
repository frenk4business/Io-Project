# Dataset Discovery for Coverage-Corrected Io Volcanism

This manifest records the datasets that can support a quantitative comparison
between hotspot occurrence, thermal intensity, persistence, and observation
coverage. Large calibrated image bundles should stay outside Git; this project
commits source manifests, schemas, small metadata tables, and reproducible
ingest code.

## Core Sources

| Dataset | Source URL | Access method | Expected local files | Size risk | Roles | Integration status |
|---|---|---|---|---|---|---|
| USGS SIM3168 Io geologic map and named volcanic centers | https://pubs.usgs.gov/publication/sim3168 | Download GIS/table products from USGS | `data/raw/io_volcanic_hotspots.csv`, `data/raw/io_geology_map.*` | Moderate GIS package | Occurrence, supporting geology | Integrated into feature matrix |
| Davies et al. 2024 Juno/JIRAM hotspot table | https://doi.org/10.1038/s41550-023-02123-5 | Article supplement | `data/external/davies_2024/41550_2023_2123_MOESM2_ESM.csv`, `data/raw/io_hotspot_power.csv` | Small | Intensity, occurrence | Integrated as estimated thermal-emission proxy |
| Juno/JIRAM calibrated PDS metadata | https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/JUNO/jiram/ | PDS indexes/product logs first; calibrated images only by explicit cache | `data/raw/jiram_io_calibrated_orbit57_products.csv`, `data/raw/jiram_io_calibrated_orbit58_products.csv` | Very large if full images are downloaded | Coverage, selected intensity after future radiance processing | Metadata coverage cube integrated |
| Mura et al. 2024 hotspot variability | https://doi.org/10.3389/fspas.2024.1369472 | Article HTML first; PDF Table 2 layout-text fallback | `data/raw/mura_2024_hotspot_timeseries.csv` | Small after extraction | Intensity, persistence | Integrated: 288 orbit-band peak-radiance rows |
| Galileo NIMS Io hotspot spectral-radiance products | NASA PDS / PDS Atmospheres Galileo NIMS archive | PDS inventory + night converted Excel log + selected converted CSV products | `data/raw/galileo_nims_io_hotspot_spectral_radiance.csv` | Large if all products are downloaded | Intensity, temporal context | Integrated: 236 usable product-level max spectral-radiance rows from first 250 log rows |
| Ground-based adaptive-optics hotspot catalogues | de Kleer et al. 2019 AAS machine-readable Table 5 | MRT table download | `data/raw/ground_based_ao_io_hotspots.csv` | Small | Intensity, persistence, temporal recurrence | Integrated: 1596 filter brightness rows |

## Manual Structured Schemas

When machine-readable extraction is unreliable, create curated CSVs with these
columns and keep provenance in a `source` column.

### `data/raw/mura_2024_hotspot_timeseries.csv`

`source_id,name,longitude,latitude,orbit,observation_time,power_gw,instrument,source`

### `data/raw/galileo_nims_io_hotspot_spectral_radiance.csv`

`source_id,name,longitude,latitude,observation_time,wavelength_um,spectral_radiance,spectral_radiance_unit,source`

### `data/raw/ground_based_ao_io_hotspots.csv`

`source_id,name,longitude,latitude,observation_time,brightness_value,brightness_unit,instrument,source`

## Processed Outputs

The v2 ingest path produces these reproducible processed tables when the
dashboard or tests call the loader functions:

- `data/processed/io_observation_coverage_cube.parquet`
- `data/processed/io_thermal_activity_events.parquet`

Both files are ignored by Git because they are generated artifacts.

## Scientific Limitation

The current coverage correction is metadata-based: product counts are weighted
by available spatial resolution and emission-angle metadata. It is not a true
pixel-footprint or radiometric-sensitivity correction. Non-detections should
therefore be treated as coverage-limited unless footprint rasters and calibrated
per-pixel sensitivity are integrated.

## Current External Extraction Results

- Mura 2024: Frontiers HTML did not expose static tables reliably, so the
  parser uses the downloaded PDF with `pdftotext -layout` and extracts Table 2
  peak radiances. These are stored as source-unit intensity values, not
  bolometric radiant power.
- Galileo NIMS: the PDS collection inventory contains 2063 products. The
  current ingest uses the PDS Atmospheres night converted log and downloads the
  first 250 converted CSV spectra, producing 236 rows with matched hotspot
  coordinates and maximum spectral radiance.
- AO: de Kleer et al. 2019 Table 5 is available as an AAS MRT table and is
  parsed directly. Values remain filter brightness / spectral-radiance-style
  quantities in `GW/um/sr`; no power conversion is applied.
