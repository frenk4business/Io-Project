# Data Sources

This file documents all external datasets used in this project, their provenance,
download instructions, and expected filenames.

**Keep this file up to date.** Every file in `data/raw/` must have an entry here.

---

## 1. USGS Io Volcanic Hotspot Catalog

**Canonical name:** `io_volcanic_hotspots.csv`  
**Placed in:** `data/raw/io_volcanic_hotspots.csv`

**Description:**  
Catalog of volcanic hotspots identified on Io from Galileo, Voyager, and
ground-based near-infrared observations.

**Primary reference:**  
Lopes, R.M.C. & Williams, D.A. (2005). Io after Galileo.
*Reports on Progress in Physics*, 68(2), 303.

**Additional sources to consult:**  
- Davies, A.G. (2007). *Volcanism on Io: A Comparison with Earth*. Cambridge.  
- Veeder, G.J. et al. (1994). Io's heat flow from infrared radiometry. *JGR Planets*, 99(E8).

**Download / access:**  
- USGS Astrogeology: https://astrogeology.usgs.gov/
- Check for compiled hotspot tables in published papers (Table 1 format)
- A community-maintained list is sometimes available via the Io geology working group

**Expected CSV columns (update ingest/hotspot_catalog.py COLUMN_MAP if different):**

| Column | Type | Description |
|--------|------|-------------|
| Name | str | Hotspot name |
| Longitude | float | Degrees, 0–360 or -180–180 |
| Latitude | float | Degrees, -90 to 90 |
| Temperature_K | float | Brightness temperature (may be NaN) |
| Source | str | Originating observation/paper |

**Status:** [ ] Downloaded  [ ] Placed in data/raw/  [ ] Verified in 00_data_audit.ipynb

---

## 2. USGS Io Geologic Map

**Canonical name:** `io_geology_map.shp` (+ .dbf, .prj, .shx, .cpg)  
**Placed in:** `data/raw/`

**Description:**  
Geologic map of Io at 1:15,000,000 scale. Derived from Galileo SSI and
Voyager imaging data. The primary mapping unit is the geologic unit polygon.

**Reference:**  
Williams, D.A., Keszthelyi, L.P., Crown, D.A., Yff, J.A., Jaeger, W.L.,
Schenk, P.M., Geissler, P.E., & Becker, T.L. (2011).
Geologic Map of Io. USGS Scientific Investigations Map 3168.

**Download:**  
https://pubs.usgs.gov/sim/3168/

Direct GIS data package download available on that page.

**Expected attributes (update ingest/geology_map.py ATTRIBUTE_MAP if different):**

| Attribute | Type | Description |
|-----------|------|-------------|
| UNIT_NAME | str | Full geologic unit name |
| UNIT_CODE | str | Short alphanumeric code |

**Coverage note:**  
Some polar and high-latitude regions have limited coverage. Cells in these
areas will receive `geology_unit = 'UNKNOWN'` — see `features/geology.py`.

**Status:** [ ] Downloaded  [ ] Placed in data/raw/  [ ] Verified in 00_data_audit.ipynb

---

## 3. Tidal Heating Flux Grid

**Canonical name:** `io_tidal_heating_flux.csv`  
**Placed in:** `data/raw/io_tidal_heating_flux.csv`

**Description:**  
Time-averaged tidal heating flux model on a regular lon/lat grid.

⚠️  **Important:** This is a modelled quantity, not a direct observation.
The model assumes a specific interior structure and time-averaged orbital
parameters. Real tidal dissipation varies with orbital phase and is not
fully constrained.

**Reference candidates (check which model/grid is available):**  
- Tackley, P.J., Schubert, G., Glatzmaier, G.A., Schenk, P., Ratcliff, J.T.,
  & Matas, J.P. (2001). Three-dimensional simulations of mantle convection
  in Io. *Icarus*, 149(1), 79–93.
- Hamilton, C.W., Beggan, C.D., Still, S., Beuthe, M., Lopes, R.M.C., et al.
  (2013). Spatial distribution of volcanoes on Io: Implications for tidal
  heating models. *Earth and Planetary Science Letters*, 361, 272–286.
- Tyler, R.H., Henning, W.G., & Hamilton, C.W. (2015). Tidal heating in a
  magma ocean within Jupiter's moon Io. *ApJS*, 218(2), 22.

**If no grid is publicly available:**  
Use `ingest.tidal_heating.load_synthetic_tidal_proxy()` for development.
Mark all outputs as provisional and replace before final analysis.

**Expected CSV columns:**

| Column | Type | Description |
|--------|------|-------------|
| longitude | float | Degrees, -180 to 180 |
| latitude | float | Degrees, -90 to 90 |
| flux_w_m2 | float | Tidal heating flux in W/m² |

**Status:** [ ] Obtained  [ ] Placed in data/raw/  [ ] Verified in 00_data_audit.ipynb

---

## Checklist

Before proceeding to modeling, confirm all three sources are in place:

- [ ] `data/raw/io_volcanic_hotspots.csv`
- [ ] `data/raw/io_geology_map.shp` (and associated files)
- [ ] `data/raw/io_tidal_heating_flux.csv` (or synthetic proxy documented)

If using synthetic or placeholder data, all downstream outputs **must** be
labelled as provisional in notebooks and figures.

---

## 4. Davies et al. (2024) Juno/JIRAM Hotspot Radiance Catalogue

**Canonical raw supplementary file:** `data/external/davies_2024/41550_2023_2123_MOESM2_ESM.csv`  
**Normalized radiance table:** `data/raw/io_hotspot_radiance_davies2024.csv`  
**Regression proxy table:** `data/raw/io_hotspot_power.csv`

**Reference:**  
Davies, A.G., Perry, J.E., Williams, D.A., & Nelson, D.M. (2024).
Io's polar volcanic thermal emission indicative of magma ocean and shallow
tidal heating models. *Nature Astronomy*, 8, 94-100.
https://doi.org/10.1038/s41550-023-02123-5

**Downloaded files:**
- `41550_2023_2123_MOESM2_ESM.csv` - Supplementary Table S1, 266 JIRAM hot spots.
- `41550_2023_2123_MOESM1_ESM.pdf` - Supplementary information.

**Important science note:**  
Supplementary Table S1 reports 4.8 micron spectral radiance in GW/um, not a
direct bolometric radiant power table. The project file `io_hotspot_power.csv`
therefore contains an explicit derived proxy, `power_gw`, calculated from the
article-level relation that 1 TW/um maximum unsaturated spectral radiance
corresponds to about 18 TW total thermal emission. Treat this as an estimated
thermal-emission proxy for model experiments, not as directly measured power.

**Normalized columns:**

| Column | Type | Description |
|--------|------|-------------|
| source_id | str | Davies/JIRAM hotspot ID |
| name | str | Hotspot name or source ID if unnamed |
| longitude | float | Degrees east, normalized to -180 to 180 |
| latitude | float | Degrees north |
| max_unsaturated_spectral_radiance_gw_um | float | 4.8 micron spectral radiance proxy |
| power_gw | float | Estimated total thermal emission proxy in GW |

**Status:** [x] Downloaded  [x] Placed in data/raw/  [ ] Verified in notebooks

---

## 5. Juno/JIRAM PDS Io Product Logs

**PDS bundle metadata:** `data/external/jiram_pds/bundle_juno_jiram.xml`  
**PDS calibrated inventory:** `data/external/jiram_pds/collection_data_calibrated_inventory.csv`  
**Normalized product logs:**
- `data/raw/jiram_io_calibrated_orbit57_products.csv`
- `data/raw/jiram_io_calibrated_orbit58_products.csv`

**Source:**  
NASA Planetary Data System Atmospheres Node, Juno JIRAM bundle:
https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/JUNO/jiram/

**Description:**  
Calibrated JIRAM product metadata for close Io observations on orbit 57
(2023-12-30) and orbit 58 (2024-02-03). These rows describe image/spectral
products, geometry, resolution, and observing times. They are not yet a
published hotspot catalogue; deriving hotspot locations/power from these files
requires image/radiance processing.

**Status:** [x] Downloaded  [x] Converted to CSV  [ ] Hotspot extraction performed

---

## 6. Mura et al. (2024) JIRAM Hotspot Variability Paper

**Downloaded file:** `data/external/mura_2024/mura_2024_temporal_variability_io_hotspots.pdf`

**Reference:**  
Mura, A. et al. (2024). The temporal variability of Io's hotspots.
*Frontiers in Astronomy and Space Sciences*, 11, 1369472.
https://doi.org/10.3389/fspas.2024.1369472

**Description:**  
Open-access paper containing JIRAM hotspot location and power-output tables for
Juno orbits 41, 43, 47, and 49. The article notes that orbits 53, 55, 57, 58,
and 60 were expected to improve southern/polar coverage, but a fully processed
public hotspot catalogue for those later close flybys was not found during this
data pass.

**Status:** [x] PDF downloaded  [ ] Tables extracted  [ ] Integrated with pipeline
