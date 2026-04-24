"""
ingest
------
Raw data loading modules. Write to data/raw/ only.
"""

from ingest.hotspot_catalog import load_hotspot_catalog
from ingest.tidal_heating import load_tidal_heating_csv, load_synthetic_tidal_proxy

__all__ = [
    "load_hotspot_catalog",
    "load_tidal_heating_csv",
    "load_synthetic_tidal_proxy",
]
