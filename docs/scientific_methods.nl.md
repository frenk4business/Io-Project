# Scientific Methods (NL)

[English version](scientific_methods.md)

Dit document is de Nederlandstalige tegenhanger van de canonieke methodereferentie voor de **Scientific Analysis**-laag van dit project.

---

## Onderzoeksvraag

> *Gegeven de USGS Io-hotspotcatalogus, de USGS-geologiekaart en proxy-covariaten (waaronder een expliciet synthetische getijdenverwarmingsproxy en een Davies/JIRAM thermische-emissieproxy), welke ruimtelijke structuur en associaties zijn aanwezig in de waargenomen catalogus, en hoe gevoelig zijn deze conclusies voor leakage-gevoelige features en observationele dekkingsproxies?*

Dit is een **beschrijvende / inferentiele** vraag. We claimen niet dat het project onbekende hotspots ontdekt. De analyse karakteriseert de waargenomen catalogus en rapporteert associaties, gevoeligheden en caveats.

---

## Beperkingen die niet verborgen worden

1. **Target leakage**: `dist_nearest_hotspot_km` is geconstrueerd uit dezelfde hotspotcatalogus als de targetlabels.
2. **Synthetische getijdenverwarming**: `tidal_heating_flux` is momenteel een analytische placeholder, geen gepubliceerd fysisch dissipatiemodel.
3. **Observationele coverage bias**: de catalogus weerspiegelt waar missies hebben gekeken, niet noodzakelijk waar alle hotspots bestaan.
4. **Catalogusvintage**: de hotspotcatalogus wordt gedomineerd door oudere detecties.
5. **`power_gw` is een geschatte thermische-emissieproxy**: afgeleid van Davies/JIRAM 4.8 micron spectrale radiantie, niet direct gemeten bolometrisch radiant vermogen.

---

## Methodesamenvatting

- **Spatial CV**: 4 latitude-band folds; random splits zijn niet toegestaan.
- **Leakage audit**: controle op target-derived features, verdachte correlaties en extreme coefficienten.
- **Ablation**: vergelijking van feature sets met en zonder leakage-feature.
- **Geological enrichment**: verrijkingsratio's per USGS-eenheid.
- **Spatial statistics**: Ripley's K op een bol, `g(r)` en nearest-neighbour CDF.
- **Asymmetry**: hemisferische en longitudinale asymmetrie via binomiale toetsen en bootstrap-CIs.
- **Coverage bias**: eerste-orde correctie op basis van observatieproxy's.
- **Thermal intensity**: samenvattingen en regressie op `power_gw` als geschatte thermische-emissieproxy.

---

## Reproduceerbaarheid

- `scripts/run_scientific_analysis.py` regenereert de analyse-artifacten in `data/results/`.
- De dashboardlaag is display-only.
- De Engelse versie blijft de primaire canonieke referentie voor contributors en tests.
