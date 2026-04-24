# io-hotspot-prediction

[English version](README.md)

**Het in kaart brengen en analyseren van vulkanische hotspots op Io, de meest vulkanisch actieve maan van Jupiter, met open planetaire datasets, ruimtelijke machine learning en expliciete onzekerheidsanalyse.**

---

## Waarom Io?

Io is het meest vulkanisch actieve object in het zonnestelsel. De maan wordt verhit door getijdenwerking vanuit Jupiter. Het oppervlak bevat honderden vulkanische hotspots, maar de observationele dekking is ongelijk en onvolledig.

Dit project stelt daarom geen grote discovery-claim, maar een eerlijke vraag: welke patronen zien we in de **waargenomen catalogus**, welke geologische en ruimtelijke associaties zijn zichtbaar, en hoe gevoelig zijn die conclusies voor bias, leakage en proxy-aannames?

---

## Onderzoeksvraag

> *Gegeven de USGS Io-hotspotcatalogus, de USGS-geologiekaart en proxy-covariaten (waaronder een expliciet synthetische getijdenverwarmingsproxy en een Davies/JIRAM thermische-emissieproxy), welke ruimtelijke structuur en associaties zijn aanwezig in de waargenomen catalogus, en hoe gevoelig zijn deze conclusies voor leakage-gevoelige features en observationele dekkingsproxies?*

Dit is een **beschrijvende / inferentiele** vraag, geen discovery-claim. De pagina **Scientific Analysis** rapporteert ruimtelijke statistiek, geologische verrijking, hemisferische asymmetrie, coverage-bias, hypothesis checks en thermische-intensiteitsanalyse. De laag **Explore Io** blijft daarnaast bestaan als publieksvriendelijke visualisatielaag.

---

## Wat dit project doet

1. Leest de USGS Io-hotspotcatalogus en ondersteunende ruimtelijke lagen in.
2. Lijnt die data uit op een gemeenschappelijk `1 deg x 1 deg` grid.
3. Bouwt featurelagen voor geologie, getijdenproxy en afstandsvariabelen.
4. Traineert een logistieke-regressie-baseline met latitude-band spatial CV.
5. Voert een leakage-audit, ablaties en bias-analyses uit.
6. Integreert Davies/JIRAM `power_gw` als **geschatte thermische-emissieproxy** voor intensiteitsanalyse.
7. Toont alles in een Streamlit-dashboard met `Io Experience`, `2D Maps`, `3D Globe` en `Scientific Analysis`.

---

## Belangrijke datasets

- USGS SIM3168 hotspotcatalogus
- USGS SIM3168 geologiekaart
- Davies et al. (2024) JIRAM 4.8 micron spectrale-radiantietabel
- NASA Io 3D-model/textuur voor de publieksviewer
- Een synthetische getijdenverwarmingsproxy, totdat een gepubliceerd fysisch grid wordt ingelezen

Zie `data/external/SOURCES.md` voor provenance en downloadinstructies.

---

## Belangrijke beperkingen

1. **Target leakage**: `dist_nearest_hotspot_km` is afgeleid van dezelfde catalogus als de labels.
2. **Synthetische getijdenproxy**: de huidige `tidal_heating_flux` is geen gepubliceerd dissipatiemodel.
3. **Observationele dekkingsbias**: geen hotspotrecord betekent niet automatisch geen vulkanisme.
4. **Catalogusvintage**: de USGS-catalogus is vooral gebaseerd op oudere waarnemingen.
5. **Thermische intensiteit is proxy-gebaseerd**: `power_gw` is geen direct gemeten bolometrisch radiant vermogen.

---

## Gebruik

```bash
streamlit run dashboard/app.py
```

Als de 3D Io Beleving of 3D Globe meldt dat Plotly ontbreekt, werk dan dezelfde
omgeving bij waarin Streamlit draait:

```bash
conda env update -f environment.yml
# of
python -m pip install "plotly>=5.18"
```

Voor de wetenschappelijke methode en beperkingen:

- `docs/scientific_methods.md`
- `docs/scientific_methods.nl.md`
