"""
ingest/external_activity_sources.py
-----------------------------------
Download and normalize external Io thermal-activity sources.

The functions in this module are intentionally conservative: they keep large
source products in ``data/external/`` and only write small normalized CSVs into
``data/raw/`` when extracted tables pass basic schema and row-count checks.
"""

from __future__ import annotations

import csv
import logging
import re
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile
from typing import Iterable

import numpy as np
import pandas as pd

from config import EXTERNAL_DIR, RAW_DIR
from ingest.thermal_activity_events import AO_SCHEMA, MURA_MANUAL_SCHEMA, NIMS_SCHEMA

logger = logging.getLogger(__name__)

MURA_ARTICLE_URL = (
    "https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/"
    "10.3389/fspas.2024.1369472/full"
)
NIMS_BASE_URL = "https://pds.nasa.gov/data/pds4/releases/atmos/go_nims_io_rad-20240308"
NIMS_ATMOS_BASE_URL = "https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/go_nims_io_rad"
NIMS_LOG_URL = (
    "https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/Galileo/"
    "logs/galileo_night_converted_log.xlsx"
)
NIMS_INVENTORY_URL = f"{NIMS_BASE_URL}/collection_go_nims_io_rad_data_derived_inventory.csv"
NIMS_BUNDLE_URL = f"{NIMS_BASE_URL}/bundle_go_nims_io_rad.xml"
AO_CALTECH_2018_URL = "https://authors.library.caltech.edu/records/cwxa2-29g80/latest"
AO_TABLE5_URL = "https://content.cld.iop.org/journals/1538-3881/158/1/29/revision1/ajab2380t5_mrt.txt"
AO_ICARUS_2015_URL = "https://doi.org/10.1016/j.icarus.2016.06.019"

MURA_DIR = EXTERNAL_DIR / "mura_2024"
NIMS_DIR = EXTERNAL_DIR / "galileo_nims"
AO_DIR = EXTERNAL_DIR / "ao_dekleer"

ORBIT_DATE_MAP = {
    37: "2021-10-16",
    41: "2022-04-09",
    43: "2022-07-05",
    47: "2022-12-15",
    49: "2023-03-01",
}

WAVELENGTH_BY_BAND = {"M": 4.78, "L": 3.45}
AO_FILTER_WAVELENGTHS = {
    "Kc": 2.27,
    "H2O": 3.06,
    "PAH": 3.29,
    "Lp": 3.78,
    "BrAlphaCont": 3.99,
    "BrAlpha": 4.05,
    "Ms": 4.67,
}
MONTHS = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_cell = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif self._in_table and tag == "tr":
            self._current_row = []
        elif self._in_table and tag in {"td", "th"}:
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_table and tag in {"td", "th"}:
            self._in_cell = False
            self._current_row.append(" ".join(self._current_cell).strip())
        elif self._in_table and tag == "tr":
            if any(cell != "" for cell in self._current_row):
                self._current_table.append(self._current_row)
        elif tag == "table" and self._in_table:
            self._in_table = False
            if self._current_table:
                self.tables.append(self._current_table)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                self._current_cell.append(text)


def _download(url: str, path: Path, *, binary: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if binary:
        path.write_bytes(data)
    else:
        path.write_text(data.decode("utf-8", errors="replace"), encoding="utf-8")
    return path


def _normalize_number(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip().replace(",", ".")
    text = text.replace("\u2212", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^0-9eE+\-.]", "", text)
    if text in {"", "-", ".", "-."}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _west_to_project_longitude(lon_west: float) -> float:
    east = (-float(lon_west)) % 360.0
    return ((east + 180.0) % 360.0) - 180.0


def _flatten_columns(columns: Iterable[object]) -> list[str]:
    flat = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(part) for part in col if not str(part).startswith("Unnamed")]
            text = " ".join(parts)
        else:
            text = str(col)
        text = re.sub(r"\s+", " ", text).strip()
        flat.append(text)
    return flat


def _read_html_tables(html: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(StringIO(html))
    except ImportError:
        parser = _TableHTMLParser()
        parser.feed(html)
        tables = []
        for raw_table in parser.tables:
            if not raw_table:
                continue
            width = max(len(row) for row in raw_table)
            rows = [row + [""] * (width - len(row)) for row in raw_table]
            header, body = rows[0], rows[1:]
            tables.append(pd.DataFrame(body, columns=header))
        return tables


def _find_mura_tables(tables: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    hotspot_table = None
    power_table = None
    for table in tables:
        df = table.copy()
        df.columns = _flatten_columns(df.columns)
        joined_cols = " ".join(df.columns).lower()
        joined_values = " ".join(df.head(3).astype(str).to_numpy().ravel()).lower()
        if "name" in joined_cols and "lat" in joined_cols and "lon" in joined_cols:
            hotspot_table = df
        if "temperature" in joined_cols or "estimated total output" in joined_cols:
            power_table = df
        if power_table is None and "single-band total output" in joined_values:
            power_table = df
        if (
            power_table is None
            and "41" in joined_cols
            and "43" in joined_cols
            and "lat" not in joined_cols
            and "lon" not in joined_cols
        ):
            power_table = df
    if hotspot_table is None:
        raise ValueError("Mura Table 2 hotspot-location table was not found in article HTML.")
    if power_table is None:
        raise ValueError("Mura Table 3 power-output table was not found in article HTML.")
    return hotspot_table, power_table


def _standardize_mura_hotspot_table(table: pd.DataFrame) -> pd.DataFrame:
    df = table.copy()
    df.columns = _flatten_columns(df.columns)
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in {"#", "number", "# name"} or lc.startswith("#"):
            rename[col] = "source_number"
        elif lc == "name" or lc.endswith(" name"):
            rename[col] = "name"
        elif lc in {"lat", "latitude", "location lat"}:
            rename[col] = "latitude"
        elif lc in {"lon", "longitude", "location lon"}:
            rename[col] = "longitude_west"
    df = df.rename(columns=rename)
    if "source_number" not in df.columns:
        df = df.rename(columns={df.columns[0]: "source_number"})
    if "name" not in df.columns and len(df.columns) > 1:
        df = df.rename(columns={df.columns[1]: "name"})
    for candidate in df.columns:
        if candidate.lower().endswith("lat") and "latitude" not in df.columns:
            df = df.rename(columns={candidate: "latitude"})
        if candidate.lower().endswith("lon") and "longitude_west" not in df.columns:
            df = df.rename(columns={candidate: "longitude_west"})
    required = {"source_number", "name", "latitude", "longitude_west"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mura hotspot table missing columns after parsing: {sorted(missing)}")
    out = df[list(required)].copy()
    out["source_number"] = out["source_number"].map(_normalize_number).astype("Int64")
    out["latitude"] = out["latitude"].map(_normalize_number)
    out["longitude_west"] = out["longitude_west"].map(_normalize_number)
    out["longitude"] = out["longitude_west"].map(_west_to_project_longitude)
    out = out.dropna(subset=["source_number", "latitude", "longitude"])
    return out.reset_index(drop=True)


def _standardize_mura_power_table(table: pd.DataFrame) -> pd.DataFrame:
    df = table.copy()
    df.columns = _flatten_columns(df.columns)
    if "source_number" not in df.columns:
        df = df.rename(columns={df.columns[0]: "source_number"})
    df["source_number"] = df["source_number"].map(_normalize_number).astype("Int64")
    return df.dropna(subset=["source_number"]).reset_index(drop=True)


def _mura_power_value(row: pd.Series, orbit: int, band: str) -> tuple[float, str]:
    patterns = [
        rf"\b{orbit}\s*{band}\b",
        rf"\b{orbit}{band}\b",
        rf"\b{orbit}\b.*\b{band}\b",
    ]
    candidates = []
    for col in row.index:
        col_text = str(col)
        if any(re.search(pattern, col_text, flags=re.IGNORECASE) for pattern in patterns):
            value = _normalize_number(row[col])
            if not np.isnan(value):
                candidates.append((value, col_text))
    if not candidates:
        return float("nan"), ""
    # The right-side estimated-total columns usually have larger values and
    # contain the physically useful event intensity. Pick the largest numeric
    # candidate for the orbit/band, while preserving the source column name.
    return max(candidates, key=lambda item: item[0])


def extract_mura_2024_events_from_pdf_layout_text(text: str) -> tuple[pd.DataFrame, str]:
    """Extract Mura Table 2 peak-radiance events from pdftotext -layout output."""
    orbit_bands = [(41, "M"), (41, "L"), (43, "M"), (43, "L"), (47, "M"), (47, "L"), (49, "M"), (49, "L")]
    rows = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line or "TABLE 2" in line or "Peak radiances" in line or "frontiersin.org" in line:
            continue
        match = re.match(r"^(?:\d{2}\s+)?(?P<num>\d{1,3})\s+(?P<rest>.+)$", line)
        if not match:
            continue
        parts = match.group("rest").split()
        if len(parts) < 18:
            continue
        numeric_start = None
        for idx, part in enumerate(parts):
            if part == "-" or re.match(r"^[<>]?\d+(?:[.,]\d+)?$", part):
                numeric_start = idx
                break
        if numeric_start is None or numeric_start < 1:
            continue
        name = " ".join(parts[:numeric_start])
        values = parts[numeric_start:]
        if len(values) < 16:
            continue
        radiance_values = values[:8]
        lat = _normalize_number(values[-4])
        lon_west = _normalize_number(values[-2])
        if np.isnan(lat) or np.isnan(lon_west):
            continue
        source_number = int(match.group("num"))
        for (orbit, band), raw_value in zip(orbit_bands, radiance_values):
            intensity = _normalize_number(raw_value)
            if np.isnan(intensity) or intensity <= 0:
                continue
            rows.append(
                {
                    "source_id": f"Mura2024-{source_number}-{orbit}{band}",
                    "name": name,
                    "longitude": _west_to_project_longitude(lon_west),
                    "latitude": lat,
                    "orbit": orbit,
                    "observation_time": ORBIT_DATE_MAP.get(orbit, f"{orbit}-01-01"),
                    "power_gw": np.nan,
                    "instrument": "JIRAM",
                    "source": "Mura et al. 2024 Table 2 peak radiance",
                    "wavelength_um": WAVELENGTH_BY_BAND[band],
                    "intensity_value": intensity,
                    "intensity_unit": "W m-2 sr-1 peak_radiance",
                }
            )
    out = pd.DataFrame(rows)
    if len(out) < 40:
        raise ValueError(f"Mura PDF Table 2 extraction produced too few event rows: {len(out)}")
    return out, f"Extracted {len(out)} orbit-band peak-radiance rows from Mura Table 2 PDF layout text."


def extract_mura_2024_events_from_html(html: str) -> tuple[pd.DataFrame, str]:
    tables = _read_html_tables(html)
    hotspot_table, power_table = _find_mura_tables(tables)
    hotspots = _standardize_mura_hotspot_table(hotspot_table)
    powers = _standardize_mura_power_table(power_table)
    merged = hotspots.merge(powers, on="source_number", how="inner", suffixes=("", "_power"))
    rows = []
    for _, row in merged.iterrows():
        for orbit in (41, 43, 47, 49):
            for band in ("M", "L"):
                value, source_column = _mura_power_value(row, orbit, band)
                if np.isnan(value) or value <= 0:
                    continue
                rows.append(
                    {
                        "source_id": f"Mura2024-{int(row['source_number'])}-{orbit}{band}",
                        "name": row["name"],
                        "longitude": row["longitude"],
                        "latitude": row["latitude"],
                        "orbit": orbit,
                        "observation_time": ORBIT_DATE_MAP.get(orbit, f"{orbit}-01-01"),
                        "power_gw": value,
                        "instrument": "JIRAM",
                        "source": f"Mura et al. 2024 Table 3 column '{source_column}', band {band}",
                    }
                )
    out = pd.DataFrame(rows, columns=MURA_MANUAL_SCHEMA)
    if len(out) < 20:
        raise ValueError(f"Mura extraction produced too few event rows: {len(out)}")
    return out, f"Parsed {len(tables)} HTML tables; extracted {len(out)} orbit-band event rows."


def fetch_and_normalize_mura_2024() -> tuple[Path, str]:
    html_path = _download(MURA_ARTICLE_URL, MURA_DIR / "mura_2024_frontiers_article.html")
    html = html_path.read_text(encoding="utf-8", errors="replace")
    try:
        events, report = extract_mura_2024_events_from_html(html)
    except ValueError as html_exc:
        pdf_path = MURA_DIR / "mura_2024_temporal_variability_io_hotspots.pdf"
        if not pdf_path.exists():
            _download(
                "https://public-pages-files-2025.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2024.1369472/pdf",
                pdf_path,
                binary=True,
            )
        if shutil.which("pdftotext") is None:
            raise ValueError(f"{html_exc}; pdftotext is not available for PDF fallback.") from html_exc
        text_path = MURA_DIR / "mura_2024_pdf_layout.txt"
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(text_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        events, pdf_report = extract_mura_2024_events_from_pdf_layout_text(
            text_path.read_text(encoding="utf-8", errors="replace")
        )
        report = f"HTML extraction fallback reason: {html_exc}. {pdf_report}"
    out_path = RAW_DIR / "mura_2024_hotspot_timeseries.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_path, index=False)
    (MURA_DIR / "EXTRACTION_REPORT.txt").write_text(report + "\n", encoding="utf-8")
    return out_path, report


def parse_nims_inventory(inventory_csv: str) -> pd.DataFrame:
    rows = []
    reader = csv.reader(StringIO(inventory_csv))
    for parts in reader:
        if not parts:
            continue
        text = ",".join(parts)
        if "xml" not in text.lower() and "urn:nasa:pds:" not in text:
            continue
        xml_path = next((part.strip() for part in parts if part.strip().lower().endswith(".xml")), "")
        lidvid = next((part.strip() for part in parts if part.strip().startswith("urn:nasa:pds:")), "")
        if not xml_path and len(parts) >= 2:
            xml_path = parts[-1].strip()
        product_id = Path(xml_path).stem if xml_path else (lidvid.rsplit(":", 1)[-1] if lidvid else "")
        product_id = product_id.split("::", 1)[0]
        if not xml_path and product_id:
            xml_path = f"data_derived/{product_id}.xml"
        rows.append(
            {
                "product_id": product_id,
                "label_path": xml_path,
                "label_url": f"{NIMS_BASE_URL}/{xml_path.lstrip('./')}" if xml_path else "",
                "lidvid": lidvid,
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["product_id"])
    if out.empty:
        raise ValueError("NIMS inventory did not contain any XML product labels.")
    return out.reset_index(drop=True)


def _xml_text(root: ET.Element, suffixes: Iterable[str]) -> str:
    suffixes = tuple(suffixes)
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag in suffixes and elem.text:
            return elem.text.strip()
    return ""


def parse_nims_label_xml(xml_text: str, product_id: str) -> dict[str, object]:
    root = ET.fromstring(xml_text)
    start_time = _xml_text(root, ("start_date_time", "observation_start_date_time"))
    stop_time = _xml_text(root, ("stop_date_time", "observation_stop_date_time"))
    longitude = _normalize_number(_xml_text(root, ("longitude", "center_longitude", "sub_spacecraft_longitude")))
    latitude = _normalize_number(_xml_text(root, ("latitude", "center_latitude", "sub_spacecraft_latitude")))
    wavelength = _normalize_number(_xml_text(root, ("wavelength", "central_wavelength", "wavelength_center")))
    return {
        "product_id": product_id,
        "observation_time": start_time or stop_time,
        "longitude": longitude,
        "latitude": latitude,
        "wavelength_um": wavelength,
    }


def _read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(ns + "si"):
                shared.append("".join(t.text or "" for t in si.iter(ns + "t")))
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.findall(".//" + ns + "row"):
            values = []
            for cell in row.findall(ns + "c"):
                value = cell.find(ns + "v")
                text = value.text if value is not None else ""
                if cell.attrib.get("t") == "s" and text:
                    text = shared[int(text)]
                values.append(text)
            rows.append(values)
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    return pd.DataFrame(rows[1:], columns=header)


def _clean_hotspot_name(name: object) -> str:
    text = str(name or "").lower()
    text = re.sub(r"\bgp\d+\b", "", text)
    text = re.sub(r"\b(region|patera|fluctus|and-or|or|close to|inc)\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _hotspot_coordinate_lookup() -> dict[str, tuple[float, float]]:
    path = RAW_DIR / "io_volcanic_hotspots.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        key = _clean_hotspot_name(row.get("Name"))
        lon = _normalize_number(row.get("Longitude"))
        lat = _normalize_number(row.get("Latitude"))
        if key and not np.isnan(lon) and not np.isnan(lat):
            lookup[key] = (((lon + 180.0) % 360.0) - 180.0, lat)
    return lookup


def _match_hotspot_coordinates(target_name: object, lookup: dict[str, tuple[float, float]]) -> tuple[float, float]:
    target = _clean_hotspot_name(target_name)
    if target in lookup:
        return lookup[target]
    for key, value in lookup.items():
        if key and (key in target or target in key):
            return value
    return float("nan"), float("nan")


def _read_nims_product_max_radiance(path: Path) -> tuple[float, float]:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 5:
        return float("nan"), float("nan")
    wavelength = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    radiance = pd.to_numeric(df.iloc[:, 4], errors="coerce")
    if radiance.dropna().empty:
        return float("nan"), float("nan")
    idx = radiance.idxmax()
    return float(wavelength.loc[idx]), float(radiance.loc[idx])


def fetch_and_normalize_nims(max_labels: int = 250) -> tuple[Path, str]:
    inventory_path = _download(NIMS_INVENTORY_URL, NIMS_DIR / "collection_go_nims_io_rad_data_derived_inventory.csv")
    _download(NIMS_BUNDLE_URL, NIMS_DIR / "bundle_go_nims_io_rad.xml")
    log_path = _download(NIMS_LOG_URL, NIMS_DIR / "galileo_night_converted_log.xlsx", binary=True)
    inventory = parse_nims_inventory(inventory_path.read_text(encoding="utf-8", errors="replace"))
    log_df = _read_xlsx_first_sheet(log_path)
    lookup = _hotspot_coordinate_lookup()
    records = []
    for _, row in log_df.head(max_labels).iterrows():
        file_name = str(row.get("File Name", "")).strip()
        if not file_name:
            continue
        product_id = Path(file_name).stem
        product_url = f"{NIMS_ATMOS_BASE_URL}/data_derived/unresolved_night/converted/{file_name}"
        label_url = product_url.replace(".csv", ".xml")
        lon, lat = _match_hotspot_coordinates(row.get("Target", ""), lookup)
        try:
            product_path = _download(product_url, NIMS_DIR / "products" / file_name)
            wavelength, radiance = _read_nims_product_max_radiance(product_path)
        except Exception as exc:
            logger.warning("Could not parse NIMS product %s: %s", product_url, exc)
            wavelength, radiance = np.nan, np.nan
        records.append(
            {
                "source_id": product_id,
                "name": str(row.get("Target", "")).strip() or product_id,
                "longitude": lon,
                "latitude": lat,
                "observation_time": row.get("START_TIME") or "1996-01-01",
                "wavelength_um": wavelength,
                "spectral_radiance": radiance,
                "spectral_radiance_unit": "PDS converted spectral_radiance",
                "source": product_url,
            }
        )
    out = pd.DataFrame(records, columns=NIMS_SCHEMA)
    usable = out.dropna(subset=["longitude", "latitude", "spectral_radiance"])
    if len(usable) < 10:
        raise ValueError(f"NIMS extraction produced too few usable rows: {len(usable)}")
    out_path = RAW_DIR / "galileo_nims_io_hotspot_spectral_radiance.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    usable.to_csv(out_path, index=False)
    report = (
        f"Downloaded NIMS inventory with {len(inventory)} product labels and night converted log; "
        f"normalized {len(usable)} usable product-level max spectral-radiance rows from first {min(max_labels, len(log_df))} log rows."
    )
    (NIMS_DIR / "EXTRACTION_REPORT.txt").write_text(report + "\n", encoding="utf-8")
    return out_path, report


def normalize_ao_curated_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(AO_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"AO curated table missing columns: {sorted(missing)}")
    return df[AO_SCHEMA].copy()


def parse_ao_mrt_table5(text: str) -> pd.DataFrame:
    filters = [
        ("Kc", 57, 62),
        ("H2O", 69, 74),
        ("PAH", 80, 85),
        ("Lp", 91, 97),
        ("BrAlphaCont", 105, 110),
        ("BrAlpha", 116, 122),
        ("Ms", 129, 135),
    ]
    rows = []
    in_data = False
    for line in text.splitlines():
        if line.startswith("Nusku Patera"):
            in_data = True
        if not in_data or len(line) < 55:
            continue
        site = line[0:23].strip()
        year = line[24:28].strip()
        month = line[29:32].strip()
        day = line[33:35].strip()
        lat = _normalize_number(line[36:41])
        lon_west = _normalize_number(line[46:51])
        if not site or not year.isdigit() or month not in MONTHS or not day.strip().isdigit():
            continue
        for filter_name, start, stop in filters:
            brightness = _normalize_number(line[start:stop])
            if np.isnan(brightness) or brightness <= 0:
                continue
            rows.append(
                {
                    "source_id": f"dekleer2019-{site}-{year}{MONTHS[month]}{int(day):02d}-{filter_name}",
                    "name": site,
                    "longitude": _west_to_project_longitude(lon_west),
                    "latitude": lat,
                    "observation_time": f"{year}-{MONTHS[month]}-{int(day):02d}",
                    "brightness_value": brightness,
                    "brightness_unit": "GW/um/sr filter_integrated_flux_density",
                    "instrument": "Keck/Gemini AO",
                    "source": f"de Kleer et al. 2019 AJ Table 5, filter {filter_name}",
                    "wavelength_um": AO_FILTER_WAVELENGTHS[filter_name],
                }
            )
    out = pd.DataFrame(rows)
    if len(out) < 100:
        raise ValueError(f"AO MRT Table 5 extraction produced too few rows: {len(out)}")
    return out


def fetch_ao_source_manifests() -> tuple[Path, str]:
    AO_DIR.mkdir(parents=True, exist_ok=True)
    mrt_path = _download(AO_TABLE5_URL, AO_DIR / "ajab2380t5_mrt.txt")
    ao = parse_ao_mrt_table5(mrt_path.read_text(encoding="utf-8", errors="replace"))
    out_path = RAW_DIR / "ground_based_ao_io_hotspots.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ao.to_csv(out_path, index=False)
    manifest = AO_DIR / "AO_SOURCE_MANIFEST.md"
    manifest.write_text(
        "# Ground-Based AO Io Hotspot Sources\n\n"
        "Automated ingest uses the AAS machine-readable Table 5 MRT file from de Kleer et al. 2019.\n"
        "The normalized CSV is written to `data/raw/ground_based_ao_io_hotspots.csv` with columns:\n\n"
        f"`{','.join(AO_SCHEMA)}`\n\n"
        "Sources:\n"
        f"- AAS machine-readable Table 5: {AO_TABLE5_URL}\n"
        f"- de Kleer et al. 2019 AJ / Caltech: {AO_CALTECH_2018_URL}\n"
        f"- de Kleer and de Pater 2016 Icarus time variability: {AO_ICARUS_2015_URL}\n"
        "- de Kleer and de Pater 2016 spatial companion: https://doi.org/10.1016/j.icarus.2016.06.018\n\n"
        "Scientific note: preserve AO values as brightness/radiance unless a table explicitly reports power.\n",
        encoding="utf-8",
    )
    return out_path, f"Parsed {len(ao)} AO brightness rows from de Kleer et al. 2019 Table 5 MRT."


def fetch_all_external_activity_sources() -> dict[str, str]:
    """Fetch and normalize all supported external activity sources."""
    status: dict[str, str] = {}
    for label, func in {
        "Mura 2024": fetch_and_normalize_mura_2024,
        "Galileo NIMS": fetch_and_normalize_nims,
        "Ground-based AO": fetch_ao_source_manifests,
    }.items():
        try:
            path, report = func()
            status[label] = f"{path}: {report}"
        except Exception as exc:
            status[label] = f"failed: {exc}"
            logger.exception("%s ingest failed", label)
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for source, message in fetch_all_external_activity_sources().items():
        print(f"{source}: {message}")
