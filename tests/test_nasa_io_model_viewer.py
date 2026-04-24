"""
tests/test_nasa_io_model_viewer.py
----------------------------------
Contract tests for the official NASA Io model viewer.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pandas as pd


def _power_grid(n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        power = float((i + 1) * 10)
        rows.append(
            {
                "cell_id": i,
                "lon_centre": -120.0 + i * 18.0,
                "lat_centre": -65.0 + i * 10.0,
                "power_count": 1,
                "primary_power_gw": power,
                "mean_power_gw": power,
                "sum_power_gw": power,
                "log_primary_power": np.log1p(power),
                "power_names": f"hotspot_{i}",
            }
        )
    return pd.DataFrame(rows)


def _write_tiny_glb(path, image_bytes: bytes = b"\x89PNG\r\n\x1a\nfake") -> None:
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(image_bytes)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(image_bytes)}],
        "images": [{"bufferView": 0, "mimeType": "image/png"}],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = image_bytes + b"\0" * ((4 - len(image_bytes) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total)
        + struct.pack("<I4s", len(json_bytes), b"JSON")
        + json_bytes
        + struct.pack("<I4s", len(bin_bytes), b"BIN\0")
        + bin_bytes
    )


def test_nasa_model_path_constant_points_to_glb():
    from ingest.nasa_io_model import NASA_IO_MODEL_PATH

    assert NASA_IO_MODEL_PATH.name == "io_nasa.glb"
    assert "nasa_io_3d" in str(NASA_IO_MODEL_PATH)


def test_nasa_model_missing_asset_detection(tmp_path):
    from visualization.nasa_io_model_viewer import nasa_model_available

    assert nasa_model_available(tmp_path / "missing.glb", tmp_path / "missing.png") is False


def test_nasa_texture_extraction_reads_embedded_png(tmp_path):
    from ingest.nasa_io_model import extract_nasa_io_texture, nasa_io_asset_status

    glb = tmp_path / "tiny.glb"
    texture = tmp_path / "texture.png"
    _write_tiny_glb(glb)

    result = extract_nasa_io_texture(model_path=glb, texture_path=texture)
    status = nasa_io_asset_status(model_path=glb, texture_path=texture)

    assert result == texture
    assert texture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert status["model_exists"] is True
    assert status["texture_exists"] is True
    assert status["texture_png_header"] is True


def test_nasa_model_viewer_html_uses_proxy_language_and_texture(tmp_path):
    from visualization.nasa_io_model_viewer import build_nasa_io_model_viewer_html

    texture = tmp_path / "texture.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    html, insights = build_nasa_io_model_viewer_html(
        _power_grid(3),
        show_all_heat_sources=True,
        texture_path=texture,
    )

    assert "three" in html
    assert "TextureLoader" in html
    assert "SphereGeometry(1, 128, 64)" in html
    assert "OrbitControls" in html
    assert "GLTFLoader" not in html
    assert "data:image/png;base64" in html
    assert "data:model/gltf-binary;base64" not in html
    assert "texture ready" in html
    assert "three loaded" in html
    assert "texture loaded" in html
    assert "sphere added" in html
    assert "controls.autoRotate = false" in html
    assert "planetGroup.rotation.y +=" not in html
    assert "estimated thermal-emission proxy" in html
    assert "estimated thermal-emission proxy" in insights["power_definition"]


def test_nasa_model_viewer_top_10_limits_hotspot_buttons(tmp_path):
    from visualization.nasa_io_model_viewer import build_nasa_io_model_viewer_html

    texture = tmp_path / "texture.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    html, insights = build_nasa_io_model_viewer_html(
        _power_grid(18),
        show_all_heat_sources=True,
        highlight_top_10=True,
        texture_path=texture,
    )

    assert insights["visible_hotspots"] == 10
    assert html.count('"name": "hotspot_') == 10
    assert "front > 0.04" in html
    assert "marker.visible = visible" in html


def test_nasa_model_viewer_optional_controls_are_wired(tmp_path):
    from visualization.nasa_io_model_viewer import build_nasa_io_model_viewer_html

    texture = tmp_path / "texture.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    html, _ = build_nasa_io_model_viewer_html(
        _power_grid(4),
        show_heat_glow=False,
        show_power_towers=True,
        show_polar_bands=True,
        show_coverage_uncertainty=True,
        texture_path=texture,
    )

    assert "const showHeatGlow = false" in html
    assert "if (!showHeatGlow) return" in html
    assert "const showPowerTowers = true" in html
    assert "function addPowerTowers()" in html
    assert "const showPolarBands = true" in html
    assert "function addPolarBands()" in html
    assert "const showCoverageUncertainty = true" in html
    assert "coverage uncertainty" in html
