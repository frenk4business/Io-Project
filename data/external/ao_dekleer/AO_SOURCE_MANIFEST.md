# Ground-Based AO Io Hotspot Sources

Automated ingest uses the AAS machine-readable Table 5 MRT file from de Kleer et al. 2019.
The normalized CSV is written to `data/raw/ground_based_ao_io_hotspots.csv` with columns:

`source_id,name,longitude,latitude,observation_time,brightness_value,brightness_unit,instrument,source`

Sources:
- AAS machine-readable Table 5: https://content.cld.iop.org/journals/1538-3881/158/1/29/revision1/ajab2380t5_mrt.txt
- de Kleer et al. 2019 AJ / Caltech: https://authors.library.caltech.edu/records/cwxa2-29g80/latest
- de Kleer and de Pater 2016 Icarus time variability: https://doi.org/10.1016/j.icarus.2016.06.019
- de Kleer and de Pater 2016 spatial companion: https://doi.org/10.1016/j.icarus.2016.06.018

Scientific note: preserve AO values as brightness/radiance unless a table explicitly reports power.
