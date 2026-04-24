"""
visualization/nasa_io_model_viewer.py
-------------------------------------
Streamlit HTML renderer for the official NASA Io visual layer.

The runtime view renders the extracted NASA Io texture on a controlled Three.js
sphere. This is more reliable in Streamlit than asking the browser to decode
the embedded GLB texture inside an iframe. All heat values are labelled as an
estimated thermal-emission proxy.
"""

from __future__ import annotations

import base64
import html
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from dashboard.i18n import translate as t

from ingest.nasa_io_model import (
    NASA_IO_MODEL_PATH,
    NASA_IO_TEXTURE_PATH,
    ensure_nasa_io_texture,
    nasa_io_asset_status,
    nasa_io_model_exists,
    nasa_io_texture_exists,
)


def nasa_model_available(
    model_path: Path = NASA_IO_MODEL_PATH,
    texture_path: Path = NASA_IO_TEXTURE_PATH,
) -> bool:
    """Return True when the NASA visual assets can be prepared for rendering."""
    return nasa_io_texture_exists(texture_path) or nasa_io_model_exists(model_path)


def nasa_visual_asset_status(
    model_path: Path = NASA_IO_MODEL_PATH,
    texture_path: Path = NASA_IO_TEXTURE_PATH,
) -> dict:
    """Return asset status for dashboard diagnostics."""
    return nasa_io_asset_status(model_path=model_path, texture_path=texture_path)


@lru_cache(maxsize=1)
def _texture_data_url(texture_path_text: str) -> str:
    texture_path = Path(texture_path_text)
    encoded = base64.b64encode(texture_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _power_points(
    power_grid: pd.DataFrame,
    min_power_gw: float,
    show_all_heat_sources: bool,
    highlight_top_10: bool,
) -> pd.DataFrame:
    points = power_grid[power_grid["power_count"] > 0].copy()
    points = points[points["primary_power_gw"].astype(float) >= min_power_gw]
    points = points.sort_values("primary_power_gw", ascending=False)
    if highlight_top_10:
        return points.head(10).reset_index(drop=True)
    if not show_all_heat_sources:
        return points.head(45).reset_index(drop=True)
    return points.reset_index(drop=True)


def _sphere_xyz(lon_deg: float, lat_deg: float, radius: float = 1.035) -> tuple[float, float, float]:
    lon = np.deg2rad(float(lon_deg))
    lat = np.deg2rad(float(lat_deg))
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.sin(lat)
    z = -radius * np.cos(lat) * np.sin(lon)
    return float(x), float(y), float(z)


def _marker_size(power_gw: float, max_power_gw: float, highlight_top_10: bool) -> float:
    scale = np.log1p(max(float(power_gw), 0.0)) / max(np.log1p(max(float(max_power_gw), 1.0)), 1.0)
    if highlight_top_10:
        return 0.014 + 0.026 * scale
    return 0.006 + 0.012 * scale


def _hotspot_payload(points: pd.DataFrame, highlight_top_10: bool) -> list[dict]:
    """Convert hotspot rows to a small JSON-safe payload for Three.js."""
    if points.empty:
        return []
    max_power = float(points["primary_power_gw"].max())
    payload: list[dict] = []
    for row in points.itertuples(index=False):
        x, y, z = _sphere_xyz(row.lon_centre, row.lat_centre)
        payload.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "size": _marker_size(row.primary_power_gw, max_power, highlight_top_10),
                "name": str(getattr(row, "power_names", "") or "Davies/JIRAM hot spot"),
                "lat": float(row.lat_centre),
                "lon": float(row.lon_centre),
                "power": float(row.primary_power_gw),
            }
        )
    return payload


def nasa_io_insights(points: pd.DataFrame, language: str = "en") -> dict:
    """Return the plain-English insight metrics used below the viewer."""
    polar = points[points["lat_centre"].abs() >= 60.0] if not points.empty else points
    total_power = float(points["sum_power_gw"].sum()) if not points.empty else 0.0
    polar_power = float(polar["sum_power_gw"].sum()) if not polar.empty else 0.0
    strongest = points.iloc[0] if not points.empty else None
    return {
        "visible_hotspots": int(len(points)),
        "visible_power_gw": total_power,
        "strongest_name": str(strongest["power_names"] or t("viewer.hotspot.default_name", language)) if strongest is not None else t("viewer.none", language),
        "strongest_power_gw": float(strongest["primary_power_gw"]) if strongest is not None else 0.0,
        "polar_fraction": polar_power / total_power if total_power > 0 else 0.0,
        "power_definition": t("viewer.power_definition", language),
    }


def build_nasa_io_model_viewer_html(
    power_grid: pd.DataFrame,
    min_power_gw: float = 0.0,
    scene: str = "Meet Io",
    show_heat_glow: bool = True,
    show_power_towers: bool = False,
    show_polar_bands: bool = False,
    show_coverage_uncertainty: bool = False,
    show_all_heat_sources: bool = False,
    highlight_top_10: bool = False,
    language: str = "en",
    model_path: Path = NASA_IO_MODEL_PATH,
    texture_path: Path = NASA_IO_TEXTURE_PATH,
) -> tuple[str, dict]:
    """Build a self-contained HTML viewer for Streamlit components."""
    texture_path = ensure_nasa_io_texture(model_path=model_path, texture_path=texture_path)
    points = _power_points(
        power_grid=power_grid,
        min_power_gw=min_power_gw,
        show_all_heat_sources=show_all_heat_sources,
        highlight_top_10=highlight_top_10,
    )
    texture_src = json.dumps(_texture_data_url(str(texture_path)))
    hotspots_json = json.dumps(_hotspot_payload(points, highlight_top_10=highlight_top_10))
    show_heat_glow_js = json.dumps(bool(show_heat_glow))
    show_power_towers_js = json.dumps(bool(show_power_towers))
    show_polar_bands_js = json.dumps(bool(show_polar_bands))
    show_coverage_uncertainty_js = json.dumps(bool(show_coverage_uncertainty))
    insights = nasa_io_insights(points, language=language)
    source_count = len(points)
    mode_label = t("viewer.top10", language) if highlight_top_10 else f"{t('viewer.visible_heat_sources', language)}: {source_count}"
    if show_coverage_uncertainty:
        mode_label += f" - {t('viewer.coverage_uncertainty', language)}"

    html_doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: #000;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .wrap {{
      position: relative;
      width: 100%;
      height: 760px;
      background:
        radial-gradient(circle at 52% 48%, rgba(255, 223, 126, 0.08), transparent 30%),
        #000;
      border: 1px solid rgba(255, 255, 255, 0.12);
    }}
    #viewer {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .badge {{
      position: absolute;
      top: 18px;
      left: 18px;
      z-index: 2;
      color: rgba(255,255,255,0.84);
      background: rgba(0,0,0,0.48);
      border: 1px solid rgba(255,255,255,0.16);
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 13px;
      line-height: 1.35;
      backdrop-filter: blur(4px);
    }}
    .badge strong {{
      display: block;
      color: #ffe58a;
      font-size: 14px;
      margin-bottom: 2px;
    }}
    .tooltip {{
      position: absolute;
      display: none;
      z-index: 3;
      max-width: 280px;
      color: white;
      background: rgba(0, 0, 0, 0.82);
      border: 1px solid rgba(255,255,255,0.22);
      border-radius: 7px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.35;
      text-align: left;
      pointer-events: none;
      transform: translate(12px, -12px);
    }}
    .hint {{
      position: absolute;
      right: 18px;
      bottom: 16px;
      z-index: 2;
      color: rgba(255,255,255,0.70);
      background: rgba(0,0,0,0.40);
      border: 1px solid rgba(255,255,255,0.12);
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
    }}
    .status {{
      position: absolute;
      left: 18px;
      bottom: 16px;
      z-index: 2;
      color: rgba(255,255,255,0.58);
      background: rgba(0,0,0,0.34);
      border: 1px solid rgba(255,255,255,0.10);
      padding: 7px 9px;
      border-radius: 8px;
      font-size: 11px;
      max-width: 52%;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="badge" id="badge">
      <strong>{html.escape(t("page.iox.surface.nasa", language))}</strong>
      {html.escape(mode_label)}<br>
      {html.escape(t("viewer.badge_proxy", language))}
    </div>
    <div id="viewer"></div>
    <div id="tooltip" class="tooltip"></div>
    <div id="status" class="status">{html.escape(t("viewer.status.texture_ready", language))}</div>
    <div class="hint">{html.escape(t("viewer.hint", language))}</div>
  </div>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
      }}
    }}
  </script>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

    const textureSrc = {texture_src};
    const hotspots = {hotspots_json};
    const showHeatGlow = {show_heat_glow_js};
    const showPowerTowers = {show_power_towers_js};
    const showPolarBands = {show_polar_bands_js};
    const showCoverageUncertainty = {show_coverage_uncertainty_js};
    const container = document.getElementById("viewer");
    const tooltip = document.getElementById("tooltip");
    const badge = document.getElementById("badge");
    const status = document.getElementById("status");

    function setStatus(message) {{
      status.textContent = message;
    }}

    setStatus({json.dumps(t("viewer.status.three_loaded", language))});

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    container.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(27, container.clientWidth / container.clientHeight, 0.01, 100);
    camera.position.set(0.02, 0.12, 2.65);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 0.0;
    controls.minDistance = 1.62;
    controls.maxDistance = 3.9;
    controls.target.set(0, 0, 0);

    const planetGroup = new THREE.Group();
    scene.add(planetGroup);
    const hotspotGroup = new THREE.Group();
    planetGroup.add(hotspotGroup);
    const towerGroup = new THREE.Group();
    planetGroup.add(towerGroup);
    const polarGroup = new THREE.Group();
    planetGroup.add(polarGroup);
    const clickable = [];
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2(-10, -10);

    function addHotspots() {{
      if (!showHeatGlow) return;
      hotspots.forEach((point) => {{
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(point.size, 16, 16),
          new THREE.MeshBasicMaterial({{ color: 0xff2a1a, transparent: true, opacity: 0.78, depthWrite: false }})
        );
        marker.position.set(point.x, point.y, point.z);
        marker.userData = point;
        hotspotGroup.add(marker);
        clickable.push(marker);

        const glow = new THREE.Mesh(
          new THREE.SphereGeometry(point.size * 2.15, 16, 16),
          new THREE.MeshBasicMaterial({{ color: 0xff1f14, transparent: true, opacity: 0.10, depthWrite: false }})
        );
        glow.position.copy(marker.position);
        glow.userData.isGlow = true;
        marker.userData.glow = glow;
        hotspotGroup.add(glow);
      }});
    }}

    function addPowerTowers() {{
      if (!showPowerTowers) return;
      const material = new THREE.LineBasicMaterial({{ color: 0xff2a1a, transparent: true, opacity: 0.64 }});
      hotspots.forEach((point) => {{
        const base = new THREE.Vector3(point.x, point.y, point.z).normalize().multiplyScalar(1.015);
        const height = Math.min(0.23, 0.055 + point.size * 3.25);
        const tip = base.clone().normalize().multiplyScalar(1.015 + height);
        const geometry = new THREE.BufferGeometry().setFromPoints([base, tip]);
        const line = new THREE.Line(geometry, material);
        towerGroup.add(line);
      }});
    }}

    function addPolarBands() {{
      if (!showPolarBands) return;
      const material = new THREE.LineBasicMaterial({{ color: 0x38c9ff, transparent: true, opacity: 0.86 }});
      [60, -60].forEach((latDeg) => {{
        const lat = THREE.MathUtils.degToRad(latDeg);
        const points = [];
        for (let lonDeg = -180; lonDeg <= 180; lonDeg += 3) {{
          const lon = THREE.MathUtils.degToRad(lonDeg);
          points.push(new THREE.Vector3(
            1.045 * Math.cos(lat) * Math.cos(lon),
            1.045 * Math.sin(lat),
            -1.045 * Math.cos(lat) * Math.sin(lon)
          ));
        }}
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        polarGroup.add(new THREE.Line(geometry, material));
      }});
    }}

    new THREE.TextureLoader().load(
      textureSrc,
      (texture) => {{
        setStatus({json.dumps(t("viewer.status.texture_loaded", language))});
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
        const material = new THREE.MeshBasicMaterial({{ map: texture }});
        const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 64), material);
        planetGroup.add(sphere);
        if (showCoverageUncertainty) {{
          const haze = new THREE.Mesh(
            new THREE.SphereGeometry(1.006, 128, 64),
            new THREE.MeshBasicMaterial({{ color: 0x111827, transparent: true, opacity: 0.16, depthWrite: false }})
          );
          planetGroup.add(haze);
        }}
        addHotspots();
        addPowerTowers();
        addPolarBands();
        setStatus({json.dumps(t("viewer.status.sphere_added", language))});
      }},
      undefined,
      (error) => {{
        console.error("NASA Io texture failed to load", error);
        setStatus({json.dumps(t("viewer.status.texture_failed", language))});
        badge.innerHTML = "<strong>{html.escape(t('page.iox.surface.nasa', language))}</strong>{html.escape(t('viewer.texture_failed_badge', language))}";
      }}
    );

    renderer.domElement.addEventListener("pointermove", (event) => {{
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      tooltip.style.left = `${{event.clientX - rect.left}}px`;
      tooltip.style.top = `${{event.clientY - rect.top}}px`;
    }});

    function updateHotspotVisibility() {{
      const cameraLocal = planetGroup.worldToLocal(camera.position.clone());
      clickable.forEach((marker) => {{
        const normal = marker.position.clone().normalize();
        const toCamera = cameraLocal.clone().sub(marker.position).normalize();
        const front = normal.dot(toCamera);
        const visible = front > 0.04;
        marker.visible = visible;
        if (marker.userData.glow) marker.userData.glow.visible = visible;
        if (visible) {{
          marker.material.opacity = Math.min(0.70, 0.24 + front * 0.46);
          if (marker.userData.glow) marker.userData.glow.material.opacity = Math.min(0.08, 0.025 + front * 0.055);
        }}
      }});
      towerGroup.children.forEach((line) => {{
        const end = line.geometry.attributes.position;
        const base = new THREE.Vector3(end.getX(0), end.getY(0), end.getZ(0));
        const normal = base.clone().normalize();
        const toCamera = cameraLocal.clone().sub(base).normalize();
        line.visible = normal.dot(toCamera) > 0.04;
      }});
    }}

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      updateHotspotVisibility();
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(clickable.filter((item) => item.visible), false)[0];
      if (hit) {{
        const p = hit.object.userData;
        tooltip.style.display = "block";
        tooltip.innerHTML = `<b>${{p.name}}</b><br>{html.escape(t("viewer.hover.lat", language))} ${{p.lat.toFixed(1)}} deg, {html.escape(t("viewer.hover.lon", language))} ${{p.lon.toFixed(1)}} deg<br>` +
          `{html.escape(t("viewer.hover.proxy", language))}: ${{p.power.toLocaleString(undefined, {{maximumFractionDigits: 1}})}} GW<br>` +
          "{html.escape(t('viewer.hover.derived', language))}";
      }} else {{
        tooltip.style.display = "none";
      }}
      renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener("resize", () => {{
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }});
  </script>
</body>
</html>
"""
    return html_doc, insights


def render_nasa_io_model_viewer(
    power_grid: pd.DataFrame,
    min_power_gw: float = 0.0,
    scene: str = "Meet Io",
    show_heat_glow: bool = True,
    show_power_towers: bool = False,
    show_polar_bands: bool = False,
    show_coverage_uncertainty: bool = False,
    show_all_heat_sources: bool = False,
    highlight_top_10: bool = False,
    language: str = "en",
    height: int = 780,
) -> dict:
    """Render the NASA texture viewer and return insight metrics."""
    html_doc, insights = build_nasa_io_model_viewer_html(
        power_grid=power_grid,
        min_power_gw=min_power_gw,
        scene=scene,
        show_heat_glow=show_heat_glow,
        show_power_towers=show_power_towers,
        show_polar_bands=show_polar_bands,
        show_coverage_uncertainty=show_coverage_uncertainty,
        show_all_heat_sources=show_all_heat_sources,
        highlight_top_10=highlight_top_10,
        language=language,
    )
    components.html(html_doc, height=height, scrolling=False)
    return insights
