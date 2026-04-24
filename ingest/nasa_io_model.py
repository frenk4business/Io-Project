"""
ingest/nasa_io_model.py
-----------------------
Download and document the official NASA Io 3D model asset.

The dashboard can render this local GLB in the public Io Experience page.
This helper intentionally uses NASA sources only; it does not scrape Google
viewer assets.
"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

from config import EXTERNAL_DIR

NASA_IO_MODEL_PAGE_URL = "https://science.nasa.gov/resource/io-3d-model/"
NASA_3D_RESOURCES_URL = "https://science.nasa.gov/3d-resources"
NASA_IO_BASEMAP_PAGE_URL = "https://science.nasa.gov/photojournal/io-in-motion/"
NASA_IO_GLB_URL = (
    "https://assets.science.nasa.gov/content/dam/science/psd/solar/2023/09/i/"
    "Io_1_3643.glb?emrc=68503a602d969"
)

NASA_IO_MODEL_DIR: Path = EXTERNAL_DIR / "nasa_io_3d"
NASA_IO_MODEL_PATH: Path = NASA_IO_MODEL_DIR / "io_nasa.glb"
NASA_IO_TEXTURE_PATH: Path = NASA_IO_MODEL_DIR / "io_nasa_texture.png"
NASA_IO_SOURCE_PATH: Path = NASA_IO_MODEL_DIR / "SOURCE.md"


def nasa_io_model_exists(path: Path = NASA_IO_MODEL_PATH) -> bool:
    """Return True when the local NASA Io GLB asset is present."""
    return path.exists() and path.stat().st_size > 0


def nasa_io_texture_exists(path: Path = NASA_IO_TEXTURE_PATH) -> bool:
    """Return True when the extracted NASA Io texture is present."""
    return path.exists() and path.stat().st_size > 0


def nasa_io_asset_status(
    model_path: Path = NASA_IO_MODEL_PATH,
    texture_path: Path = NASA_IO_TEXTURE_PATH,
) -> dict:
    """Return a compact status snapshot for the NASA Io visual assets."""
    model_exists = nasa_io_model_exists(model_path)
    texture_exists = nasa_io_texture_exists(texture_path)
    texture_png_header = False
    if texture_exists:
        with texture_path.open("rb") as fh:
            texture_png_header = fh.read(8) == b"\x89PNG\r\n\x1a\n"

    return {
        "model_exists": model_exists,
        "texture_exists": texture_exists,
        "texture_png_header": texture_png_header,
        "model_size_bytes": model_path.stat().st_size if model_exists else 0,
        "texture_size_bytes": texture_path.stat().st_size if texture_exists else 0,
        "model_path": str(model_path),
        "texture_path": str(texture_path),
    }


def _read_glb_chunks(model_path: Path = NASA_IO_MODEL_PATH) -> tuple[dict, bytes]:
    """Read a GLB file and return its JSON chunk and binary chunk."""
    data = model_path.read_bytes()
    if len(data) < 28 or data[:4] != b"glTF":
        raise ValueError(f"Not a valid GLB file: {model_path}")

    version, _total_length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"Unsupported GLB version {version}: {model_path}")

    json_length, json_type = struct.unpack_from("<I4s", data, 12)
    if json_type != b"JSON":
        raise ValueError(f"First GLB chunk is not JSON: {model_path}")

    json_start = 20
    json_end = json_start + json_length
    gltf_json = json.loads(data[json_start:json_end].decode("utf-8").rstrip(" \t\r\n\0"))

    bin_length, bin_type = struct.unpack_from("<I4s", data, json_end)
    if bin_type != b"BIN\x00":
        raise ValueError(f"Second GLB chunk is not BIN: {model_path}")

    bin_start = json_end + 8
    bin_end = bin_start + bin_length
    return gltf_json, data[bin_start:bin_end]


def extract_nasa_io_texture(
    model_path: Path = NASA_IO_MODEL_PATH,
    texture_path: Path = NASA_IO_TEXTURE_PATH,
) -> Path:
    """Extract the embedded NASA Io texture PNG from the official GLB."""
    if not nasa_io_model_exists(model_path):
        raise FileNotFoundError(
            f"NASA Io GLB not found at {model_path}. Run python -m ingest.nasa_io_model first."
        )

    gltf_json, bin_chunk = _read_glb_chunks(model_path)
    images = gltf_json.get("images") or []
    buffer_views = gltf_json.get("bufferViews") or []
    if not images or "bufferView" not in images[0]:
        raise ValueError("NASA Io GLB does not contain an embedded image bufferView.")

    image = images[0]
    if image.get("mimeType") != "image/png":
        raise ValueError(f"Expected embedded PNG texture, found {image.get('mimeType')!r}.")

    view = buffer_views[int(image["bufferView"])]
    offset = int(view.get("byteOffset", 0))
    length = int(view["byteLength"])
    texture_bytes = bin_chunk[offset : offset + length]
    if not texture_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Extracted NASA Io texture does not start with a PNG header.")

    texture_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path.write_bytes(texture_bytes)
    write_source_file(model_path=model_path, texture_path=texture_path)
    return texture_path


def ensure_nasa_io_texture(
    model_path: Path = NASA_IO_MODEL_PATH,
    texture_path: Path = NASA_IO_TEXTURE_PATH,
) -> Path:
    """Return the extracted texture path, extracting from the GLB if needed."""
    if nasa_io_texture_exists(texture_path):
        return texture_path
    return extract_nasa_io_texture(model_path=model_path, texture_path=texture_path)


def write_source_file(
    model_path: Path = NASA_IO_MODEL_PATH,
    texture_path: Path = NASA_IO_TEXTURE_PATH,
    source_path: Path = NASA_IO_SOURCE_PATH,
) -> None:
    """Write provenance for the local NASA Io model asset."""
    downloaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0.0
    texture_size_mb = texture_path.stat().st_size / (1024 * 1024) if texture_path.exists() else 0.0
    source_path.write_text(
        "\n".join(
            [
                "# NASA Io 3D Model",
                "",
                f"Downloaded at: {downloaded_at}",
                f"Local file: `{model_path.as_posix()}`",
                f"File size: {size_mb:.2f} MB",
                f"Extracted texture: `{texture_path.as_posix()}`",
                f"Texture size: {texture_size_mb:.2f} MB",
                "",
                "Official sources:",
                f"- NASA Io 3D Model: {NASA_IO_MODEL_PAGE_URL}",
                f"- NASA 3D Resources hub: {NASA_3D_RESOURCES_URL}",
                f"- NASA Io basemap / Io in Motion: {NASA_IO_BASEMAP_PAGE_URL}",
                "",
                "Direct GLB asset:",
                f"- {NASA_IO_GLB_URL}",
                "",
                "Usage note:",
                "- This asset is used as a realistic NASA Io model/basemap layer.",
                "- The dashboard renders the extracted NASA texture on a controlled sphere for reliability.",
                "- The hotspot overlay is project data and represents an estimated thermal-emission proxy.",
                "- The NASA texture itself is not a live or current thermal hotspot map.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def download_nasa_io_model(
    overwrite: bool = False,
    model_path: Path = NASA_IO_MODEL_PATH,
) -> Path:
    """Download the official NASA Io GLB asset and write provenance."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if nasa_io_model_exists(model_path) and not overwrite:
        write_source_file(model_path=model_path)
        ensure_nasa_io_texture(model_path=model_path)
        return model_path

    urlretrieve(NASA_IO_GLB_URL, model_path)
    write_source_file(model_path=model_path)
    ensure_nasa_io_texture(model_path=model_path)
    return model_path


if __name__ == "__main__":
    path = download_nasa_io_model()
    print(f"NASA Io model ready: {path}")
