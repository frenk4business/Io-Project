# Io Dashboard v2 Data Manifest

This manifest is the data checklist for restoring the full v2 dashboard experience. It separates files that are intended to live in GitHub from local or bulky artifacts that should be restored from external storage before deployment.

`committed_to_github` describes the intended v2 branch policy, not whether a local file happens to exist on a development machine.

Recommended external storage for uncommitted data:

- Render persistent disk mounted into the app working directory, if the production service uses one.
- GitHub Release asset such as `io-project-v2-data-artifacts.zip`.
- Cloud storage or a manual artifact archive controlled by the project owner.
- Original public source URLs where the file can be re-fetched reproducibly.

## Runtime-Critical Derived Data

These files make the dashboard fast and complete. They are derived from raw/source data and should be restored for Render unless the build step regenerates them.

| file_path | purpose | approx_size | required_for_full_dashboard | committed_to_github | external_storage_if_not_committed | restore_or_regenerate_notes |
|---|---|---:|---|---|---|---|
| `data/processed/base_grid_1deg.parquet` | Shared 1 deg Io longitude-latitude grid; rebuild input for the Explore Io thermal-emission proxy grid. | ~406 KB | Yes | Yes | Not needed | Restored automatically by Git checkout on Render. Regenerate with `python -m preprocess.grid` if missing locally. |
| `data/processed/feature_matrix.parquet` | Main merged feature table for dashboard maps and older model diagnostics; this is the feature matrix loaded by `dashboard/app.py`. | ~1.3 MB | Yes | Yes | Not needed | Restored automatically by Git checkout on Render. If it is missing locally, regenerate with `python -m features.build` after restoring raw/source inputs, then commit only this processed parquet artifact. |
| `data/processed/hotspots_1deg_grid.parquet` | Named hotspot occurrence mapped to the common 1 deg grid. | ~413 KB | Yes | No | Render persistent disk or v2 GitHub Release data asset | Regenerate from `data/raw/io_volcanic_hotspots.csv`. |
| `data/processed/power_grid_1deg.parquet` | Estimated thermal-emission proxy mapped to the common 1 deg grid; primary Explore Io power layer. | ~433 KB | Yes | Yes | Not needed | Restored automatically by Git checkout on Render. Regenerate with `python -m preprocess.power_grid` after `data/raw/io_hotspot_power.csv` and `data/processed/base_grid_1deg.parquet` are present. |
| `data/processed/io_coverage_corrected_cell_maps.parquet` | Main v2 metric comparison table with occurrence, intensity, and metadata-normalized activity fields. | ~517 KB | Yes | No | Render persistent disk or v2 GitHub Release data asset | Regenerate with `analysis.coverage_corrected_volcanism`. Despite the legacy filename, dashboard wording should treat it as metadata-normalized. |
| `data/processed/io_multi_instrument_coverage_cube.parquet` | Metadata observation cube for instrument/product/time-bin activity summaries. | ~50 KB | Yes | No | Render persistent disk or v2 GitHub Release data asset | Regenerate with `ingest.observation_coverage_cube`. |
| `data/processed/io_thermal_activity_events.parquet` | Multi-instrument thermal event table used by time-resolved and metric activity views. | ~99 KB | Yes | No | Render persistent disk or v2 GitHub Release data asset | Regenerate with `ingest.thermal_activity_events`. |
| `data/processed/jiram_observation_coverage.parquet` | JIRAM product metadata coverage layer. | ~10 KB | Yes | No | Render persistent disk or v2 GitHub Release data asset | Regenerate from JIRAM PDS inventory and raw orbit product CSVs. |

## Result Summaries And Dashboard Evidence

These outputs are small enough to commit and are used by the Science and Metric Evidence views.

| file_path | purpose | approx_size | required_for_full_dashboard | committed_to_github | external_storage_if_not_committed | restore_or_regenerate_notes |
|---|---|---:|---|---|---|---|
| `data/results/io_research_question_evaluation.md` | Reviewer-style evaluation of whether v2 answers the multi-metric research question. | ~4 KB | Yes | Yes | Not needed | Regenerate manually or from the review workflow if scientific framing changes. |
| `data/results/io_metric_interpretation_summary.csv` | Compact interpretation table for key metric-pair comparisons. | ~1 KB | Yes | Yes | Not needed | Regenerate from coverage/metric comparison analysis. |
| `data/results/io_metric_correlation_matrix.csv` | Spearman correlation matrix between key grid metrics. | <1 KB | Yes | Yes | Not needed | Regenerate from `analysis.coverage_corrected_volcanism`. |
| `data/results/io_rank_overlap.csv` | Top-rank overlap and Jaccard-style comparison outputs. | ~2 KB | Yes | Yes | Not needed | Regenerate from `analysis.coverage_corrected_volcanism`. |
| `data/results/io_js_divergence.csv` | Jensen-Shannon divergence between metric distributions. | ~2 KB | Yes | Yes | Not needed | Regenerate from `analysis.coverage_corrected_volcanism`. |
| `data/results/io_latitude_band_contributions.csv` | Latitude-band contribution table for comparing spatial emphasis. | ~1 KB | Yes | Yes | Not needed | Regenerate from `analysis.coverage_corrected_volcanism`. |
| `data/results/io_top_n_cumulative_intensity.csv` | Top-N cumulative concentration curve for intensity layers. | ~228 KB | Yes | Yes | Not needed | Regenerate from `analysis.coverage_corrected_volcanism`. |
| `data/results/io_power_concentration_summary.csv` | Top 1/5/10/25/50 concentration summary for Davies/JIRAM proxy power. | ~1 KB | Yes | Yes | Not needed | Regenerate from `analysis.coverage_corrected_volcanism`. |
| `data/results/coverage_bias.png` | Supporting static figure for observation/coverage bias diagnostics. | ~57 KB | Yes | Yes | Not needed | Regenerate from analysis notebooks/scripts if source data changes. |
| `data/results/geology_enrichment.png` | Supporting static figure for geology enrichment context. | ~33 KB | Yes | Yes | Not needed | Regenerate from spatial/geology analysis if source data changes. |
| `data/results/spatial_stats.png` | Supporting static figure for spatial statistics context. | ~82 KB | Yes | Yes | Not needed | Regenerate from spatial statistics analysis if source data changes. |

## Raw Rebuild Inputs

Raw files are required only if the processed parquet artifacts are regenerated. They are intentionally not committed because `data/raw/` is ignored.

| file_path | purpose | approx_size | required_for_full_dashboard | committed_to_github | external_storage_if_not_committed | restore_or_regenerate_notes |
|---|---|---:|---|---|---|---|
| `data/raw/io_volcanic_hotspots.csv` | Named hotspot catalogue input for occurrence layers. | ~14 KB | Needed for rebuild | No | v2 data archive or source catalogue URL | Restore before running hotspot alignment preprocessing. |
| `data/raw/io_hotspot_power.csv` | Small curated Davies/JIRAM estimated thermal-emission proxy input used to rebuild `power_grid_1deg.parquet`; not a bulk raw/source archive. | ~76 KB | Needed for rebuild | Yes | Not needed | Restored automatically by Git checkout on Render. Keep larger raw/source downloads excluded. |
| `data/raw/io_hotspot_radiance_davies2024.csv` | Davies/JIRAM radiance-derived thermal proxy input. | ~59 KB | Needed for rebuild | No | v2 data archive or Davies/JIRAM source supplement | Restore before regenerating multi-instrument thermal events. |
| `data/raw/mura_2024_hotspot_timeseries.csv` | Mura 2024 JIRAM hotspot temporal/radiance extraction. | ~39 KB | Needed for rebuild | No | v2 data archive or Mura source extraction | Restore before regenerating thermal event rows. |
| `data/raw/galileo_nims_io_hotspot_spectral_radiance.csv` | Galileo NIMS spectral-radiance summary used as an intensity family. | ~82 KB | Needed for rebuild | No | v2 data archive or regenerated from NIMS products | Restore before regenerating NIMS thermal event rows. |
| `data/raw/ground_based_ao_io_hotspots.csv` | Ground-based AO hotspot brightness summary. | ~296 KB | Needed for rebuild | No | v2 data archive or de Kleer source table extraction | Restore before regenerating AO thermal event rows. |
| `data/raw/jiram_io_calibrated_orbit57_products.csv` | JIRAM orbit 57 product metadata for observation activity. | ~6 KB | Needed for rebuild | No | v2 data archive or JIRAM PDS metadata extraction | Restore before rebuilding JIRAM coverage metadata. |
| `data/raw/jiram_io_calibrated_orbit58_products.csv` | JIRAM orbit 58 product metadata for observation activity. | ~6 KB | Needed for rebuild | No | v2 data archive or JIRAM PDS metadata extraction | Restore before rebuilding JIRAM coverage metadata. |
| `data/raw/io_tidal_heating_flux.csv` | Supporting tidal-heating proxy grid used by older context/model diagnostics. | ~6.2 MB | Needed for full supporting diagnostics | No | v2 data archive or source model URL | Restore if retaining tidal-comparison diagnostics. |
| `data/raw/io_geology_map.shp` | Io geology shapefile geometry. | ~2.7 MB | Needed for full supporting diagnostics | No | v2 data archive or USGS/PDS source | Restore with matching `.dbf`, `.shx`, and `.prj`. |
| `data/raw/io_geology_map.dbf` | Io geology shapefile attributes. | ~175 KB | Needed for full supporting diagnostics | No | v2 data archive or USGS/PDS source | Must match `io_geology_map.shp`. |
| `data/raw/io_geology_map.shx` | Io geology shapefile index. | ~12 KB | Needed for full supporting diagnostics | No | v2 data archive or USGS/PDS source | Must match `io_geology_map.shp`. |
| `data/raw/io_geology_map.prj` | Io geology shapefile projection metadata. | <1 KB | Needed for full supporting diagnostics | No | v2 data archive or USGS/PDS source | Must match `io_geology_map.shp`. |
| `data/raw/FETCH_LOG.txt` | Local provenance log for data fetch/extraction commands. | ~3 KB | Useful for rebuild | No | v2 data archive | Keep with raw archive for reproducibility. |

## External Source And Provenance Files

These files document or preserve external source products. Large extraction caches should remain external unless already tracked.

| file_path | purpose | approx_size | required_for_full_dashboard | committed_to_github | external_storage_if_not_committed | restore_or_regenerate_notes |
|---|---|---:|---|---|---|---|
| `data/external/SOURCES.md` | Human-readable source provenance for external datasets. | small | Yes | Yes | Not needed | Update whenever source products change. |
| `data/external/DATASET_DISCOVERY.md` | Discovery notes for newly added source products. | small | Yes | Yes | Not needed | Update with future source discovery decisions. |
| `data/external/davies_2024/41550_2023_2123_MOESM1_ESM.pdf` | Davies/JIRAM supplementary source document. | ~594 KB | Source provenance | Yes | Not needed | Already tracked; keep if repository size remains acceptable. |
| `data/external/davies_2024/41550_2023_2123_MOESM2_ESM.csv` | Davies/JIRAM supplementary table used for thermal proxy extraction. | ~29 KB | Source provenance / rebuild | Yes | Not needed | Already tracked and small. |
| `data/external/jiram_pds/bundle_juno_jiram.xml` | JIRAM PDS bundle metadata. | small | Source provenance / rebuild | Yes | Not needed | Already tracked. |
| `data/external/jiram_pds/collection_data_calibrated_inventory.csv` | JIRAM calibrated-data inventory used for product metadata coverage. | ~25.6 MB | Source provenance / rebuild | Yes | Not needed if kept tracked | Already tracked but relatively large; consider release asset if repo size becomes an issue. |
| `data/external/jiram_pds/Juno_JIRAM_Io_Calibrated_Orbit-57.htm` | JIRAM orbit 57 source page snapshot. | ~18 KB | Source provenance | Yes | Not needed | Already tracked. |
| `data/external/jiram_pds/Juno_JIRAM_Io_Calibrated_Orbit-58.htm` | JIRAM orbit 58 source page snapshot. | ~22 KB | Source provenance | Yes | Not needed | Already tracked. |
| `data/external/nasa_io_3d/io_nasa.glb` | 3D Io globe asset for Explore Io. | ~16.7 MB | Yes | Yes | Not needed if kept tracked | Already tracked; required for full Explore Io visual experience. |
| `data/external/nasa_io_3d/io_nasa_texture.png` | Io texture asset for Explore Io. | ~16.2 MB | Yes | Yes | Not needed if kept tracked | Already tracked; required for full Explore Io visual experience. |
| `data/external/nasa_io_3d/SOURCE.md` | NASA 3D model provenance. | small | Yes | Yes | Not needed | Already tracked. |
| `data/external/mura_2024/mura_2024_temporal_variability_io_hotspots.pdf` | Mura 2024 source article PDF. | ~21.6 MB | Source provenance | Yes | Not needed if kept tracked | Already tracked; consider release asset if repo size becomes an issue. |
| `data/external/mura_2024/EXTRACTION_REPORT.txt` | Notes on Mura extraction status. | small | Source provenance / rebuild | Yes | Not needed | Commit as lightweight provenance. |
| `data/external/mura_2024/mura_2024_frontiers_article.html` | HTML extraction cache for Mura source. | ~1.5 MB | No | No | v2 data archive if needed | Ignored cache; can be re-fetched or regenerated. |
| `data/external/mura_2024/mura_2024_pdf_layout.txt` | PDF layout extraction cache. | ~113 KB | No | No | v2 data archive if needed | Ignored cache; regenerate from PDF if needed. |
| `data/external/ao_dekleer/AO_SOURCE_MANIFEST.md` | Ground-based AO source provenance. | small | Source provenance / rebuild | Yes | Not needed | Commit as lightweight provenance. |
| `data/external/ao_dekleer/ajab2380t5_mrt.txt` | de Kleer AO machine-readable source table. | ~143 KB | Source provenance / rebuild | Yes | Not needed | Commit as source table if acceptable. |
| `data/external/ao_dekleer/de_Kleer_2019_AJ_158_29.pdf` | AO paper PDF. | ~13.4 MB | No | No | v2 data archive or journal/source URL | Ignored; keep outside GitHub unless intentionally archived as a release asset. |
| `data/external/ao_dekleer/de_Kleer_2019_AJ_158_29_layout.txt` | AO PDF layout extraction cache. | ~84 KB | No | No | v2 data archive if needed | Ignored cache; regenerate from PDF/source table. |
| `data/external/galileo_nims/bundle_go_nims_io_rad.xml` | Galileo NIMS bundle metadata. | small | Source provenance / rebuild | Yes | Not needed | Commit as lightweight provenance. |
| `data/external/galileo_nims/collection_go_nims_io_rad_data_derived_inventory.csv` | Galileo NIMS derived collection inventory. | ~225 KB | Source provenance / rebuild | Yes | Not needed | Commit as lightweight inventory. |
| `data/external/galileo_nims/EXTRACTION_REPORT.txt` | Notes on NIMS extraction status. | small | Source provenance / rebuild | Yes | Not needed | Commit as lightweight provenance. |
| `data/external/galileo_nims/products/*.csv` | Bulk converted Galileo NIMS hotspot spectra products. | ~1.0 MB total locally | Needed only for regenerating NIMS summaries | No | v2 data archive, PDS source, or GitHub Release data asset | Ignored by policy; regenerate or restore before rebuilding raw NIMS summaries. |
| `data/external/galileo_nims/galileo_night_converted_log.xlsx` | Local NIMS conversion log workbook. | ~30 KB | Useful for rebuild audit | No | v2 data archive | Ignored by policy; store with data archive if needed. |
| `data/external/galileo_nims/sample_*.csv` and `sample_*.xml` | Local sample products used during extraction development. | small | No | No | v2 data archive if desired | Ignored; not needed for runtime dashboard. |

## Model Artifacts

These artifacts support older model/diagnostic pages. They are not the central v2 research claim, but keeping them preserves the full dashboard experience.

| file_path | purpose | approx_size | required_for_full_dashboard | committed_to_github | external_storage_if_not_committed | restore_or_regenerate_notes |
|---|---|---:|---|---|---|---|
| `data/models/logistic_regression.pkl` | Legacy logistic-regression model artifact for older diagnostic views. | small | Yes, for legacy diagnostics | Yes | Not needed | Retrain only if the feature matrix or old diagnostic view changes. |
| `data/models/scaler.pkl` | Legacy feature scaler used with `logistic_regression.pkl`. | small | Yes, for legacy diagnostics | Yes | Not needed | Must stay paired with the model artifact. |

## Deployment Checklist

Before deploying v2 on Render:

1. Restore `data/processed/*.parquet` from a v2 data artifact or regenerate them from the raw/source files.
2. Confirm the committed `data/results/io_*.csv` and `.md` files are present.
3. Confirm `data/external/nasa_io_3d/io_nasa.glb` and `data/external/nasa_io_3d/io_nasa_texture.png` are present for the Explore Io page.
4. If rebuilding data, restore the ignored `data/raw/` and necessary `data/external/` source products first.
5. Do not commit secrets, Render credentials, or private storage tokens.
