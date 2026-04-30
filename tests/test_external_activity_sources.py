from __future__ import annotations

import pandas as pd
import pytest


def test_mura_html_extraction_maps_orbit_band_and_west_longitude():
    from ingest.external_activity_sources import extract_mura_2024_events_from_html

    html = """
    <table>
      <tr><th>#</th><th>Name</th><th>41M</th><th>41L</th><th>43M</th><th>43L</th><th>47M</th><th>47L</th><th>49M</th><th>49L</th><th>LAT</th><th>ERR</th><th>LON</th><th>ERR</th></tr>
      {rows}
    </table>
    <table>
      <tr><th>#</th><th>41M</th><th>41L</th><th>43M</th><th>43L</th><th>47M</th><th>47L</th><th>49M</th><th>49L</th><th>41 M/L</th><th>43 M/L</th><th>47 M/L</th><th>49 M/L</th><th>41</th><th>43</th><th>47</th><th>49</th></tr>
      {power_rows}
    </table>
    """.format(
        rows="\n".join(
            f"<tr><td>{i}</td><td>Hotspot {i}</td><td>0.1</td><td>0.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td>{i % 80}</td><td>1</td><td>154</td><td>1</td></tr>"
            for i in range(1, 25)
        ),
        power_rows="\n".join(
            f"<tr><td>{i}</td><td>1.0</td><td>0.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td>10.0</td><td></td><td></td><td></td><td>500</td><td></td><td></td><td></td></tr>"
            for i in range(1, 25)
        ),
    )

    events, report = extract_mura_2024_events_from_html(html)
    assert "extracted" in report
    assert len(events) >= 24
    first = events.iloc[0]
    assert first["orbit"] == 41
    assert first["instrument"] == "JIRAM"
    assert first["longitude"] == pytest.approx(-154.0)


def test_nims_inventory_parser_accepts_pds4_inventory_lines():
    from ingest.external_activity_sources import parse_nims_inventory

    inventory = (
        "P,urn:nasa:pds:go_nims_io_rad:data_derived:prod1::1.0,data_derived/prod1.xml\n"
        "P,urn:nasa:pds:go_nims_io_rad:data_derived:prod2::1.0,data_derived/prod2.xml\n"
    )
    out = parse_nims_inventory(inventory)
    assert list(out["product_id"]) == ["prod1", "prod2"]
    assert out.loc[0, "label_url"].endswith("/data_derived/prod1.xml")


def test_nims_label_parser_extracts_basic_metadata():
    from ingest.external_activity_sources import parse_nims_label_xml

    xml = """
    <Product_Observational>
      <Observation_Area>
        <Time_Coordinates><start_date_time>1999-10-11T00:00:00Z</start_date_time></Time_Coordinates>
      </Observation_Area>
      <longitude>120.0</longitude>
      <latitude>-15.0</latitude>
      <wavelength>4.7</wavelength>
    </Product_Observational>
    """
    meta = parse_nims_label_xml(xml, "prod1")
    assert meta["observation_time"] == "1999-10-11T00:00:00Z"
    assert meta["longitude"] == pytest.approx(120.0)
    assert meta["latitude"] == pytest.approx(-15.0)
    assert meta["wavelength_um"] == pytest.approx(4.7)


def test_auto_fetch_status_can_be_disabled(tmp_path, monkeypatch):
    from ingest.thermal_activity_events import load_activity_events

    power_path = tmp_path / "power.csv"
    pd.DataFrame(
        {
            "name": ["A"],
            "longitude": [1.0],
            "latitude": [2.0],
            "power_gw": [3.0],
            "source_id": ["A1"],
            "epoch": ["2024"],
            "instrument": ["JIRAM"],
        }
    ).to_csv(power_path, index=False)

    import ingest.thermal_activity_events as events_module

    monkeypatch.setattr(
        events_module,
        "load_power_catalog",
        lambda: pd.read_csv(power_path),
    )
    events, status = load_activity_events(include_optional=False, auto_fetch_external=False)
    assert len(events) == 1
    assert "external fetch" not in " ".join(status)


def test_ao_mrt_parser_preserves_filter_brightness_and_west_longitude():
    from ingest.external_activity_sources import parse_ao_mrt_table5

    line = (
        "Loki Patera            "
        "2017 Jan  3 "
        " 38.8 1.0 "
        "304.3  1.0 "
        "  1.0 "
        "  0.1 "
        " 2.00 "
        "0.20 "
        " 3.00 "
        "0.30 "
        "  4.00 "
        "  0.40 "
        "  5.0 "
        " 0.5 "
        "  6.00 "
        " 0.60 "
        "  7.00 "
        " 0.70 "
    )
    # Pad with a real downloaded-table style starter so the parser enters data mode.
    text = (line.replace("Loki Patera", "Nusku Patera", 1) + "\n") * 20
    events = parse_ao_mrt_table5(text)
    assert not events.empty
    assert {"Lp", "Ms"} & set(events["source"].str.extract(r"filter ([A-Za-z0-9]+)")[0].dropna())
    first = events.iloc[0]
    assert first["longitude"] == pytest.approx(55.7)
    assert first["brightness_unit"].startswith("GW/um/sr")
