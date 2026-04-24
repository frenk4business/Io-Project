"""
dashboard/app.py
----------------
Streamlit dashboard scaffold for io-hotspot-prediction.

Run with:
    streamlit run dashboard/app.py

This dashboard provides:
  - Interactive KDE heatmap of known hotspots
  - Model prediction surface
  - Feature importance (logistic regression coefficients)
  - Data provenance summary

Architecture note
-----------------
This file orchestrates display only. No data processing or modeling logic
lives here - all computation is delegated to pipeline modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when running via streamlit
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dashboard.i18n import (
    DEFAULT_LANGUAGE,
    GROUP_LABEL_KEYS,
    PAGE_LABEL_KEYS,
    SUPPORTED_LANGUAGES,
    language_label,
    option_labels,
    translate as t,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Io Hotspot Prediction",
    page_icon="ðŸŒ‹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Top Navigation
# ---------------------------------------------------------------------------
NAV_GROUPS: dict[str, list[str]] = {
    "Explore Io": ["Io Experience", "2D Maps", "3D Globe"],
    "Science": ["Scientific Analysis"],
    "Info": ["About", "FAQ"],
}

PAGE_TO_GROUP = {
    page_name: group_name
    for group_name, page_names in NAV_GROUPS.items()
    for page_name in page_names
}


def get_language() -> str:
    language = st.session_state.get("language", DEFAULT_LANGUAGE)
    return language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def _inject_navigation_css() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none;
        }
        .io-topbar {
            position: sticky;
            top: 0;
            z-index: 99;
            background: rgba(255, 255, 255, 0.96);
            border-bottom: 1px solid rgba(49, 51, 63, 0.12);
            padding: 0.65rem 0 0.35rem;
            margin-bottom: 0.6rem;
            backdrop-filter: blur(8px);
        }
        .io-brand {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.4rem;
        }
        .io-brand-title {
            font-weight: 800;
            font-size: 1.05rem;
            color: #172033;
        }
        .io-brand-meta {
            color: #6b7280;
            font-size: 0.84rem;
            white-space: nowrap;
        }
        div[data-testid="stSegmentedControl"] label[data-baseweb="radio"][aria-checked="true"] {
            border-color: #ff4b4b;
            color: #ff4b4b;
        }
        @media (max-width: 900px) {
            .io-brand {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.15rem;
            }
            .io-brand-meta {
                white-space: normal;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _segmented_control(label: str, options: list[str], default: str, key: str) -> str:
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options=options, default=default, key=key)
    return st.radio(label, options=options, index=options.index(default), horizontal=True, key=key)


def render_top_navigation() -> str:
    _inject_navigation_css()

    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE
    if "nav_group" not in st.session_state:
        st.session_state.nav_group = "Science"
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Scientific Analysis"
    if st.session_state.nav_group not in NAV_GROUPS:
        st.session_state.nav_group = "Science"
    if st.session_state.nav_page not in PAGE_TO_GROUP:
        st.session_state.nav_page = "Scientific Analysis"

    language = get_language()
    st.markdown(
        f"""
        <div class="io-topbar">
          <div class="io-brand">
            <div class="io-brand-title">{t("nav.brand.title", language)}</div>
            <div class="io-brand-meta">{t("nav.brand.meta", language)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_col, lang_col = st.columns([5.2, 1.3])
    with nav_col:
        group_options = list(NAV_GROUPS.keys())
        group_labels = option_labels(group_options, GROUP_LABEL_KEYS, language)
        default_group = t(GROUP_LABEL_KEYS[st.session_state.nav_group], language)
        group_label = _segmented_control(
            t("nav.section", language),
            options=group_labels,
            default=default_group,
            key="nav_group",
        )
        group = group_options[group_labels.index(group_label)]

    with lang_col:
        lang_options = [language_label("en"), language_label("nl")]
        selected_lang = _segmented_control(
            t("nav.language", language),
            options=lang_options,
            default=language_label(language),
            key="language_toggle",
        )
        st.session_state.language = "en" if selected_lang == language_label("en") else "nl"
        language = get_language()

    page_options = NAV_GROUPS[group]
    if PAGE_TO_GROUP.get(st.session_state.nav_page) != group:
        st.session_state.nav_page = page_options[0]

    page_labels = option_labels(page_options, PAGE_LABEL_KEYS, language)
    page_label = _segmented_control(
        t("nav.page", language),
        options=page_labels,
        default=t(PAGE_LABEL_KEYS[st.session_state.nav_page], language),
        key=f"nav_page_{group}",
    )
    page_name = page_options[page_labels.index(page_label)]
    st.session_state.nav_page = page_name
    return page_name


page = render_top_navigation()
# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading hotspot catalog...")
def get_catalog() -> pd.DataFrame | None:
    """Load hotspot catalog with graceful failure."""
    try:
        from ingest.hotspot_catalog import load_hotspot_catalog
        return load_hotspot_catalog()
    except FileNotFoundError:
        return None
@st.cache_data(show_spinner="Loading feature matrix...")
def get_feature_matrix() -> pd.DataFrame | None:
    try:
        from features.build import load_feature_matrix
        return load_feature_matrix()
    except FileNotFoundError:
        return None
@st.cache_data(show_spinner="Loading trained model...")
def get_model() -> tuple | None:
    try:
        from models.train import load_model
        return load_model()
    except FileNotFoundError:
        return None
@st.cache_data(show_spinner="Loading power grid...")
def get_power_grid() -> pd.DataFrame | None:
    try:
        from preprocess.power_grid import (
            assign_power_to_grid,
            load_power_grid,
            save_power_grid,
        )
        from ingest.power_catalog import load_power_catalog
        from preprocess.grid import load_base_grid
        return load_power_grid()
    except FileNotFoundError:
        try:
            grid = load_base_grid()
            power_catalog = load_power_catalog()
            power_grid = assign_power_to_grid(grid, power_catalog)
            save_power_grid(power_grid)
            return power_grid
        except Exception:
            return None
    except ImportError:
        return None
def page_overview_v2() -> None:
    language = get_language()
    catalog = get_catalog()
    feature_matrix = get_feature_matrix()
    power_grid = get_power_grid()
    hotspot_count = len(catalog) if catalog is not None else 0
    grid_count = len(feature_matrix) if feature_matrix is not None else 0
    positive_rate = (
        float(feature_matrix["has_hotspot"].mean()) if feature_matrix is not None and "has_hotspot" in feature_matrix else 0.0
    )
    power_count = len(power_grid) if power_grid is not None else 0
    title = "Overview" if language == "en" else "Overzicht"
    caption = (
        "This dashboard combines a public-facing Io exploration layer with a researcher-first analysis layer."
        if language == "en"
        else "Dit dashboard combineert een publieksgerichte Io-verkenningslaag met een onderzoekgerichte analyselaag."
    )
    explore_copy = (
        "Explore Io keeps the visual tools front and center: Io Experience, 2D maps, and the 3D globe."
        if language == "en"
        else "Verken Io zet de visuele tools centraal: Io Beleving, 2D-kaarten en de 3D Globe."
    )
    science_copy = (
        "Scientific Analysis focuses on leakage-aware interpretation, spatial evidence, bias, and thermal-intensity proxy results."
        if language == "en"
        else "Wetenschappelijke Analyse focust op leakage-bewuste interpretatie, ruimtelijk bewijs, bias en resultaten rond de thermische-intensiteitsproxy."
    )
    st.title(title)
    st.caption(caption)
    st.info(explore_copy)
    st.info(science_copy)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hotspots", f"{hotspot_count:,}")
    c2.metric("Grid cells", f"{grid_count:,}")
    c3.metric("Positive rate", f"{positive_rate:.2%}")
    c4.metric("Power rows", f"{power_count:,}")
def page_2d_maps() -> None:
    language = get_language()
    st.title(t("page.2d.title", language))
    st.caption(t("page.2d.caption", language))
    catalog = get_catalog()
    feature_matrix = get_feature_matrix()
    if catalog is None:
        st.error(t("common.error.catalog_missing", language))
        return
    if feature_matrix is None:
        st.error(t("page.iox.error.feature_missing", language))
        return
    from visualization.hotspot_map import plot_hotspot_catalog, plot_kde_heatmap, plot_prediction_surface
    tab_observed, tab_model = st.tabs([
        t("page.2d.tab.observed", language),
        t("page.2d.tab.model", language),
    ])
    with tab_observed:
        st.markdown(t("page.2d.observed.body", language))
        fig = plot_hotspot_catalog(catalog)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.expander(t("page.2d.kde.expander", language), expanded=False):
            fig_kde = plot_kde_heatmap(catalog)
            st.pyplot(fig_kde, use_container_width=True)
            plt.close(fig_kde)
            st.caption(t("page.2d.kde.note", language))
    with tab_model:
        st.markdown(t("page.2d.model.body", language))
        result = get_model()
        if result is None or result[0] is None:
            st.warning(t("page.globe.warning.model_missing", language))
        else:
            from features.build import FEATURE_COLUMNS
            model, scaler = result
            X = feature_matrix[FEATURE_COLUMNS].fillna(0).values
            probabilities = model.predict_proba(scaler.transform(X))[:, 1]
            fig_model = plot_prediction_surface(feature_matrix, probabilities)
            st.pyplot(fig_model, use_container_width=True)
            plt.close(fig_model)
            st.warning(t("page.2d.model.warning", language))
            st.caption(t("page.2d.model.caption", language))
def _render_io_appearance_legend(
    n_hotspots: int,
    selected_indices: frozenset[int],
) -> None:
    from visualization.globe_3d import IO_GEOLOGY_LEGEND
    language = get_language()
    with st.expander(t("page.globe.legend.surface", language), expanded=False):
        st.markdown(t("page.globe.legend.units", language))
        items_html = ""
        for i, (code, name, color) in enumerate(IO_GEOLOGY_LEGEND):
            border_color = "#aaa" if int(color[1:3], 16) > 0xC0 else "#555"
            is_selected = i in selected_indices
            ring = "box-shadow:0 0 0 2.5px #4af, 0 0 0 4px #0009;" if is_selected else ""
            label_weight = "font-weight:700;" if is_selected else ""
            check = " *" if is_selected else ""
            items_html += (
                f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:5px;">'
                f'<div style="width:16px;height:16px;flex-shrink:0;background:{color};border:1px solid {border_color};border-radius:3px;{ring}"></div>'
                f'<span style="font-size:0.85em;{label_weight}"><b>{code}</b> - {name}{check}</span></div>'
            )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 24px;">{items_html}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(t("page.globe.legend.markers", language))
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:7px;">'
            f'<div style="width:14px;height:14px;border-radius:50%;background:#ff3311;border:1.5px solid white;"></div>'
            f'<span style="font-size:0.85em;">{t("page.globe.legend.hotspot", language, count=n_hotspots)}</span></div>',
            unsafe_allow_html=True,
        )
_LAYER_LABEL_KEYS: dict[str, str] = {
    "io_appearance": "globe.layer.io_appearance",
    "tidal_heating_flux": "globe.layer.tidal_heating_flux",
    "geology_encoded": "globe.layer.geology_encoded",
    "model_probability": "globe.layer.model_probability",
}
def page_3d_globe() -> None:
    language = get_language()
    st.title(t("page.globe.title", language))
    catalog = get_catalog()
    feature_matrix = get_feature_matrix()
    if catalog is None:
        st.error(t("common.error.catalog_missing", language))
        return
    if feature_matrix is None:
        st.error(t("page.iox.error.feature_missing", language))
        return
    layer_labels = {key: t(label_key, language) for key, label_key in _LAYER_LABEL_KEYS.items()}
    with st.expander(t("page.globe.controls", language), expanded=True):
        layer_label = st.selectbox(
            t("page.globe.surface_layer", language),
            options=list(layer_labels.values()),
            index=0,
        )
    color_by = next(key for key, value in layer_labels.items() if value == layer_label)
    unit_filter: frozenset[int] = frozenset()
    if color_by == "io_appearance":
        from visualization.globe_3d import IO_GEOLOGY_LEGEND
        unit_options = [f"{code} - {name}" for code, name, _ in IO_GEOLOGY_LEGEND]
        selected_labels = st.multiselect(
            t("page.globe.filter_units", language),
            options=unit_options,
            default=[],
            help=t("page.globe.filter_help", language),
        )
        unit_filter = frozenset(i for i, label in enumerate(unit_options) if label in selected_labels)
    probabilities = None
    if color_by == "model_probability":
        result = get_model()
        if result is None or result[0] is None:
            st.warning(t("page.globe.warning.model_missing", language))
            color_by = "tidal_heating_flux"
        else:
            from features.build import FEATURE_COLUMNS
            model, scaler = result
            X = feature_matrix[FEATURE_COLUMNS].fillna(0).values
            probabilities = model.predict_proba(scaler.transform(X))[:, 1]
    if color_by == "io_appearance":
        if unit_filter:
            from visualization.globe_3d import IO_GEOLOGY_LEGEND as _leg
            selected_names = ", ".join(f"**{_leg[i][0]}**" for i in sorted(unit_filter))
            st.info(t("page.globe.info.filtered", language, selected=selected_names))
        else:
            st.info(t("page.globe.info.appearance", language, count=len(catalog)))
    try:
        from visualization.globe_3d import build_io_globe_3d
    except ImportError:
        st.error(t("page.globe.error.plotly", language))
        return
    fig = build_io_globe_3d(
        catalog=catalog,
        feature_matrix=feature_matrix,
        probabilities=probabilities,
        color_by=color_by,
        unit_filter=unit_filter if unit_filter else None,
    )
    st.plotly_chart(fig, use_container_width=True)
    if color_by == "io_appearance":
        st.caption(t("page.globe.caption.appearance", language))
        _render_io_appearance_legend(len(catalog), unit_filter)
    else:
        st.caption(
            t(
                "page.globe.caption.other",
                language,
                surface=t(_LAYER_LABEL_KEYS[color_by], language),
            )
        )
def _experience_probabilities(feature_matrix: pd.DataFrame) -> np.ndarray | None:
    result = get_model()
    if result is None or result[0] is None:
        return None
    from features.build import FEATURE_COLUMNS

    model, scaler = result
    X = feature_matrix[FEATURE_COLUMNS].fillna(0).values
    return model.predict_proba(scaler.transform(X))[:, 1]


def page_io_experience() -> None:
    language = get_language()
    st.title(t("page.iox.title", language))
    st.caption(t("page.iox.caption", language))

    feature_matrix = get_feature_matrix()
    power_grid = get_power_grid()
    if feature_matrix is None:
        st.error(t("page.iox.error.feature_missing", language))
        return
    if power_grid is None:
        st.error(t("page.iox.error.power_missing", language))
        return

    try:
        import importlib
        import visualization.io_experience_3d as io_experience_3d

        io_experience_3d = importlib.reload(io_experience_3d)
        SCENE_ORDER = io_experience_3d.SCENE_ORDER
        build_io_experience_3d = io_experience_3d.build_io_experience_3d
    except ImportError as exc:
        st.error(t("page.iox.error.module", language, exc=exc))
        return

    scene_label_keys = {
        "Meet Io": "scene.label.meet_io",
        "Not all volcanoes are equal": "scene.label.not_equal",
        "The giants dominate": "scene.label.giants",
        "Poles vs equator": "scene.label.poles",
        "What we still do not know": "scene.label.unknown",
    }
    scene_options = [t(scene_label_keys.get(scene_id, scene_id), language) for scene_id in SCENE_ORDER]
    scene_lookup = dict(zip(scene_options, SCENE_ORDER))
    default_scene_label = scene_options[0]

    if hasattr(st, "segmented_control"):
        selected_scene_label = st.segmented_control(
            t("page.iox.story_mode", language),
            options=scene_options,
            default=default_scene_label,
            help=t("page.iox.story_help", language),
        )
    else:
        selected_scene_label = st.radio(
            t("page.iox.story_mode", language),
            options=scene_options,
            horizontal=True,
            help=t("page.iox.story_help", language),
        )

    surface_labels = {
        t("page.iox.surface.nasa", language): "nasa_io_model",
        t("page.iox.surface.natural", language): "natural_io",
        t("page.iox.surface.thermal", language): "thermal_intensity",
        t("page.iox.surface.geology", language): "geology",
        t("page.iox.surface.model", language): "model_probability",
    }

    with st.expander(t("page.iox.explore", language), expanded=False):
        c1, c2, c3 = st.columns(3)
        show_heat_glow = c1.toggle(t("page.iox.show_heat_glow", language), value=True)
        show_power_towers = c1.toggle(t("page.iox.show_power_towers", language), value=False)
        highlight_top_10 = c2.toggle(t("page.iox.highlight_top_10", language), value=False)
        show_all_heat_sources = c2.toggle(t("page.iox.show_all_heat_sources", language), value=False)
        show_polar_bands = c2.toggle(t("page.iox.show_polar_bands", language), value=False)
        show_coverage_uncertainty = c3.toggle(t("page.iox.show_coverage_uncertainty", language), value=False)
        surface_label = c3.selectbox(
            t("page.iox.surface_view", language),
            options=list(surface_labels.keys()),
            index=0,
        )

        max_power = float(power_grid["primary_power_gw"].max())
        min_power_gw = st.slider(
            t("page.iox.min_power", language),
            min_value=0.0,
            max_value=max(max_power, 1.0),
            value=0.0,
            step=max(max_power / 100.0, 0.1),
        )
    scene = scene_lookup[selected_scene_label]

    probabilities = None
    surface_mode = surface_labels[surface_label]
    use_nasa_model = surface_mode == "nasa_io_model"
    nasa_asset_status = None
    if surface_mode == "model_probability":
        probabilities = _experience_probabilities(feature_matrix)
        if probabilities is None:
            st.warning(t("page.iox.warning.model_surface_missing", language))
            surface_mode = "natural_io"

    if use_nasa_model:
        try:
            import visualization.nasa_io_model_viewer as nasa_io_model_viewer

            nasa_io_model_viewer = importlib.reload(nasa_io_model_viewer)
            nasa_asset_status = nasa_io_model_viewer.nasa_visual_asset_status()
            if not nasa_io_model_viewer.nasa_model_available():
                st.warning(
                    t("page.iox.warning.nasa_missing", language)
                )
                use_nasa_model = False
                surface_mode = "natural_io"
            else:
                if scene == "The giants dominate":
                    highlight_top_10 = True
                insights = nasa_io_model_viewer.render_nasa_io_model_viewer(
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
                insights["scene_copy"] = io_experience_3d._scene_copy(scene, language)
        except Exception as exc:
            if nasa_asset_status is not None and (
                nasa_asset_status.get("texture_exists") or nasa_asset_status.get("model_exists")
            ):
                st.error(
                    t("page.iox.error.nasa_build", language, exc=exc)
                )
                st.json(nasa_asset_status)
                insights = {
                    "scene_copy": t("page.iox.info.nasa_fail", language),
                    "visible_hotspots": 0,
                    "visible_power_gw": 0.0,
                    "strongest_name": "Viewer failed",
                    "strongest_power_gw": 0.0,
                    "polar_fraction": 0.0,
                }
            else:
                st.warning(t("page.iox.warning.nasa_fallback", language, exc=exc))
                use_nasa_model = False
                surface_mode = "natural_io"

    if not use_nasa_model:
        fig, insights = build_io_experience_3d(
            feature_matrix=feature_matrix,
            power_grid=power_grid,
            probabilities=probabilities,
            scene=scene,
            surface_mode=surface_mode,
            show_heat_glow=show_heat_glow,
            show_power_towers=show_power_towers,
            show_only_top_10=highlight_top_10,
            show_polar_bands=show_polar_bands,
            show_coverage_uncertainty=show_coverage_uncertainty,
            min_power_gw=min_power_gw,
            language=language,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(insights["scene_copy"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("page.iox.metric.visible_hotspots", language), f"{insights['visible_hotspots']:,}")
    m2.metric(
        t("page.iox.metric.visible_proxy", language),
        f"{insights['visible_power_gw']:,.0f} GW",
        help=t("page.iox.metric.visible_proxy_help", language),
    )
    m3.metric(
        t("page.iox.metric.strongest", language),
        insights["strongest_name"][:28],
        f"{insights['strongest_power_gw']:,.0f} GW",
    )
    m4.metric(
        t("page.iox.metric.polar_share", language),
        f"{insights['polar_fraction']:.0%}",
        help=t("page.iox.metric.polar_share_help", language),
    )

    st.caption(t("page.iox.caption.footer", language))


def page_model_predictions() -> None:
    language = get_language()
    st.title("Model Prediction Surface")

    feature_matrix = get_feature_matrix()
    result = get_model()

    if result is None or result[0] is None:
        st.error("Model not loaded. Run the training pipeline first.")
        return

    model, scaler = result

    if feature_matrix is None:
        st.error(t("page.iox.error.feature_missing", language))
        return

    from features.build import FEATURE_COLUMNS
    from visualization.hotspot_map import plot_prediction_surface

    X = feature_matrix[FEATURE_COLUMNS].fillna(0).values
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    fig = plot_prediction_surface(feature_matrix, probabilities)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Predicted P(hotspot) per 1 deg x 1 deg grid cell. "
        "Logistic regression trained with spatial cross-validation."
    )


def page_feature_analysis() -> None:
    st.title("Feature Analysis")

    result = get_model()
    if result is None or result[0] is None:
        st.error("Model not loaded. Run the training pipeline first.")
        return

    model, _ = result
    from features.build import FEATURE_COLUMNS

    coef_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient", ascending=True)

    # ---- Leakage and proxy warnings ----
    dist_coef = coef_df.loc[
        coef_df["Feature"] == "dist_nearest_hotspot_km", "Coefficient"
    ]
    dist_coef_val = float(dist_coef.iloc[0]) if len(dist_coef) else float("nan")
    st.error(
        f"**Target leakage detected - `dist_nearest_hotspot_km` coefficient: "
        f"{dist_coef_val:.1f}** (standardised).\n\n"
        "`dist_nearest_hotspot_km` is computed from the same hotspot catalogue used as "
        "the training label. A cell containing a known hotspot has distance ~ 0 km to "
        "itself - the model trivially learns this and dominates all predictions. "
        "**AUC-ROC 0.998-0.999 and recall 1.000 in all folds are artefacts of this "
        "feature, not evidence of scientific prediction.** "
        "See **Scientific Analysis -> Baseline Model Credibility** for leakage-free results."
    )

    st.warning(
        "**`tidal_heating_flux` is a synthetic analytical placeholder**, not a published "
        "physical model. The formula `cos^2(lon)*cos^2(lat) + 0.3*sin^2(lat)` was generated "
        "in-house as a development proxy. Any coefficient for this feature reflects "
        "the assumed cosine^2 pattern, not observational physics. "
        "See **Scientific Analysis -> Research Question** for the path to real tidal grids."
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    IO_BACKGROUND_COLOR = "#1a1a2e"
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient (standardised)")
    ax.set_title("Feature coefficients - Exploration / Teaching layer\n"
                 "(includes leaky feature - see Scientific Analysis for honest baseline)")
    ax.set_facecolor(IO_BACKGROUND_COLOR)
    fig.patch.set_facecolor(IO_BACKGROUND_COLOR)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.info(
        "Positive coefficients increase predicted probability. "
        "Coefficients are on standardised features - magnitudes are comparable. "
        "The dominant negative coefficient for `dist_nearest_hotspot_km` is a "
        "leakage artefact, not scientific signal."
    )


# ---------------------------------------------------------------------------
# Scientific Analysis cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running leakage audit...")
def get_leakage_audit(_feature_matrix: pd.DataFrame) -> pd.DataFrame:
    from analysis.leakage_audit import audit_feature_leakage
    return audit_feature_leakage(_feature_matrix)


@st.cache_data(show_spinner="Running ablation study (4-fold spatial CV x 5 feature sets)...")
def get_ablation(_feature_matrix: pd.DataFrame) -> dict:
    from models.ablation import run_ablation
    return run_ablation(_feature_matrix)


@st.cache_data(show_spinner="Computing geology enrichment...")
def get_geology_enrichment(_feature_matrix: pd.DataFrame) -> pd.DataFrame:
    from analysis.geology_enrichment import compute_geology_enrichment
    return compute_geology_enrichment(_feature_matrix)


@st.cache_data(show_spinner="Computing spatial statistics (99 Monte Carlo sims)...")
def get_spatial_stats(_catalog: pd.DataFrame) -> dict:
    from analysis.spatial_stats import compute_spatial_stats
    return compute_spatial_stats(_catalog, n_sims=99)


@st.cache_data(show_spinner="Computing hemispheric asymmetry...")
def get_asymmetry(_catalog: pd.DataFrame) -> dict:
    from analysis.asymmetry import compute_asymmetry
    return compute_asymmetry(_catalog, n_boot=500)


@st.cache_data(show_spinner="Computing coverage bias...")
def get_coverage_bias(_feature_matrix: pd.DataFrame) -> pd.DataFrame:
    from analysis.coverage_bias import compute_coverage_bias
    return compute_coverage_bias(_feature_matrix)


@st.cache_data(show_spinner="Running Poisson hypothesis comparison...")
def get_hypothesis_comparison(_feature_matrix: pd.DataFrame) -> "pd.DataFrame":
    from analysis.hypothesis_comparison import compare_hypotheses
    return compare_hypotheses(_feature_matrix, include_geology=True)


@st.cache_data(show_spinner="Running catalogue jackknife (100 replicates x 3 retention levels)...")
def get_catalog_stability(_feature_matrix: "pd.DataFrame", _catalog: "pd.DataFrame") -> "pd.DataFrame":
    from analysis.catalog_stability import catalog_jackknife
    return catalog_jackknife(
        _feature_matrix, _catalog,
        retention_fractions=(0.9, 0.75, 0.5),
        n_replicates=100,
        seed=42,
    )


@st.cache_data(show_spinner="Computing thermal intensity summaries...")
def get_power_intensity_suite(
    _feature_matrix: "pd.DataFrame",
    _power_grid: "pd.DataFrame",
) -> dict[str, "pd.DataFrame"]:
    from analysis.power_intensity import compute_power_intensity_suite
    return compute_power_intensity_suite(_feature_matrix, _power_grid)


@st.cache_data(show_spinner="Running estimated thermal-emission proxy regression...")
def get_power_regression(
    _feature_matrix: "pd.DataFrame",
    _power_grid: "pd.DataFrame",
) -> dict:
    from models.regression import train_power_regression
    return train_power_regression(_feature_matrix, _power_grid)


# ---------------------------------------------------------------------------
# Scientific Analysis page helpers
# ---------------------------------------------------------------------------

def _style_leakage_table(df: pd.DataFrame) -> object:
    """Return a Styler that highlights suspected-leakage rows in red."""
    def _row_style(row: pd.Series):
        color = "background-color: #ffcccc" if row.get("suspected_leakage", False) else ""
        return [color] * len(row)
    return df.style.apply(_row_style, axis=1).format(
        {
            "pearson_r": "{:.3f}",
            "spearman_r": "{:.3f}",
            "lr_coef": "{:.2f}",
            "abs_coef": "{:.2f}",
        },
        na_rep="-",
    )


def _ablation_plotly_chart(ablation: dict):
    """Render ablation results as a grouped Plotly bar chart."""
    import plotly.graph_objects as go

    metric_keys = [
        ("auc_roc", "AUC-ROC"),
        ("pr_auc", "PR-AUC"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]

    fs_list = ablation.get("feature_sets", [])
    names = [fs["name"] for fs in fs_list]
    colors = []
    for fs in fs_list:
        if fs.get("is_null_sanity_check"):
            colors.append("rgba(180,180,180,0.7)")
        elif fs.get("is_leaky_baseline"):
            colors.append("rgba(220,50,50,0.75)")
        else:
            colors.append("rgba(50,130,220,0.85)")

    fig = go.Figure()
    for metric_key, metric_label in metric_keys:
        means = []
        errs = []
        for fs in fs_list:
            s = fs.get("summary", {}).get(metric_key, {})
            means.append(s.get("mean", float("nan")))
            errs.append(s.get("std", 0.0))
        fig.add_trace(go.Bar(
            name=metric_label,
            x=names,
            y=means,
            error_y=dict(type="data", array=errs, visible=True),
            marker_color=colors,
        ))

    fig.update_layout(
        barmode="group",
        title="Ablation study: 5 feature sets x spatial CV (mean +/- std across 4 folds)",
        xaxis_title="Feature set",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        legend_title="Metric",
        height=420,
        annotations=[
            dict(
                x=0, y=-0.22, xref="paper", yref="paper",
                text="Red = leaky baseline (dist_nearest_hotspot_km included). "
                     "Blue = honest. Grey = null sanity check.",
                showarrow=False, font=dict(size=10),
            )
        ],
    )
    return fig


def _style_enrichment_table(df: pd.DataFrame) -> object:
    def _row_style(row: pd.Series):
        color = "background-color: #ffcccc" if row.get("significant", False) else ""
        return [color] * len(row)
    return (
        df[["unit", "n_cells", "n_hotspots", "expected",
            "enrichment_ratio", "wilson_lo", "wilson_hi", "p_bonf", "significant"]]
        .style.apply(_row_style, axis=1)
        .format(
            {
                "expected": "{:.1f}",
                "enrichment_ratio": "{:.2f}",
                "wilson_lo": "{:.3f}",
                "wilson_hi": "{:.3f}",
                "p_bonf": "{:.4f}",
            },
            na_rep="-",
        )
    )


def _asymmetry_plotly_histograms(results: dict):
    """Render longitudinal and latitudinal histograms as two Plotly figures."""
    import plotly.graph_objects as go

    lon_h = results["lon_histogram"]
    lat_h = results["lat_histogram"]

    # Longitudinal
    fig_lon = go.Figure()
    fig_lon.add_trace(go.Bar(
        x=lon_h["bin_centres"].tolist(),
        y=lon_h["counts"].tolist(),
        name="Observed",
        marker_color="rgba(50,130,220,0.7)",
        width=10,
    ))
    fig_lon.add_trace(go.Scatter(
        x=lon_h["bin_centres"].tolist(),
        y=lon_h["ci_hi"].tolist(),
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig_lon.add_trace(go.Scatter(
        x=lon_h["bin_centres"].tolist(),
        y=lon_h["ci_lo"].tolist(),
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(50,130,220,0.2)", name="95% bootstrap CI",
    ))
    # M3 markers
    for lon_m, lbl in [(0, "M3 sub-Jov"), (180, "M3 anti-Jov"), (-180, "M3 anti-Jov")]:
        fig_lon.add_vline(x=lon_m, line_dash="dash", line_color="orange",
                          annotation_text=lbl, annotation_position="top")
    fig_lon.add_vline(x=-90, line_dash="dot", line_color="grey")
    fig_lon.add_vline(x=90, line_dash="dot", line_color="grey",
                      annotation_text="+/-90 deg boundary", annotation_position="top right")
    fig_lon.update_layout(
        title="Longitudinal distribution (10 deg bins)",
        xaxis_title="Longitude (deg)", yaxis_title="Hotspot count",
        xaxis=dict(range=[-180, 180]), height=360,
    )

    # Latitudinal
    fig_lat = go.Figure()
    fig_lat.add_trace(go.Bar(
        x=lat_h["bin_centres"].tolist(),
        y=lat_h["counts"].tolist(),
        name="Observed",
        marker_color="rgba(220,130,50,0.7)",
        width=10,
    ))
    fig_lat.add_trace(go.Scatter(
        x=lat_h["bin_centres"].tolist(),
        y=lat_h["ci_hi"].tolist(),
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig_lat.add_trace(go.Scatter(
        x=lat_h["bin_centres"].tolist(),
        y=lat_h["ci_lo"].tolist(),
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(220,130,50,0.2)", name="95% bootstrap CI",
    ))
    fig_lat.add_vline(x=0, line_dash="dot", line_color="grey",
                      annotation_text="Equator", annotation_position="top right")
    for lat_m, lbl in [(90, "M4 polar max"), (-90, "M4 polar max"),
                       (0, "M3 equatorial max")]:
        color = "purple" if "M4" in lbl else "orange"
        fig_lat.add_vline(x=lat_m, line_dash="dash", line_color=color,
                          annotation_text=lbl, annotation_position="top")
    fig_lat.update_layout(
        title="Latitudinal distribution (10 deg bins)",
        xaxis_title="Latitude (deg)", yaxis_title="Hotspot count",
        xaxis=dict(range=[-90, 90]), height=360,
    )

    return fig_lon, fig_lat


# ---------------------------------------------------------------------------
# Scientific Analysis: diagnostics
# ---------------------------------------------------------------------------

def _render_lr_coefficients(model) -> None:
    """Render logistic-regression coefficients as a diagnostic only.

    This uses the trained exploration baseline (which may include leaky features).
    It is embedded in Scientific Analysis so the caveats remain visible.
    """
    from features.build import FEATURE_COLUMNS

    coef_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Coefficient": model.coef_[0],
        }
    ).sort_values("Coefficient", ascending=True)

    dist_coef = coef_df.loc[
        coef_df["Feature"] == "dist_nearest_hotspot_km", "Coefficient"
    ]
    dist_coef_val = float(dist_coef.iloc[0]) if len(dist_coef) else float("nan")
    st.error(
        f"Target leakage detected: `dist_nearest_hotspot_km` coefficient = {dist_coef_val:.1f} "
        "(standardised). Interpret any performance that includes this feature as inflated."
    )
    st.warning(
        "`tidal_heating_flux` is a synthetic analytical placeholder unless replaced by a published grid."
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    io_background = "#1a1a2e"
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient (standardised)")
    ax.set_title("Diagnostic coefficients (includes leaky feature; interpret with caution)")
    ax.set_facecolor(io_background)
    fig.patch.set_facecolor(io_background)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scientific Analysis page
# ---------------------------------------------------------------------------

def page_scientific_analysis() -> None:
    language = get_language()
    st.title(t("page.science.title", language))
    st.caption(t("page.science.caption", language))

    catalog = get_catalog()
    feature_matrix = get_feature_matrix()

    if feature_matrix is None:
        st.error(
            "Feature matrix not loaded. Run the pipeline first: "
            "`python -m features.build`"
        )
        return

    tab_labels = [
        t("page.science.tab.methods", language),
        t("page.science.tab.spatial", language),
        t("page.science.tab.bias", language),
        t("page.science.tab.thermal", language),
    ]
    tabs = st.tabs(tab_labels)

    # â”€â”€ Tab 1: Research Question & Methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tabs[0]:
        st.subheader(t("page.science.research_question", language))
        st.info(t("page.science.research_question.body", language))

        st.subheader(t("page.science.limitations", language))
        st.error(
            "**1. Target leakage.** `dist_nearest_hotspot_km` is computed from "
            "the same catalogue used as the label. Any model performance that includes "
            "this feature is inflated and not a valid measure of generalisation. "
            "The Scientific Analysis layer always reports non-leaky results alongside "
            "the leaky baseline for direct comparison."
        )
        st.warning(
            "**2. Synthetic tidal heating.** `tidal_heating_flux` uses the formula "
            "`cos^2(lon)*cos^2(lat) + 0.3*sin^2(lat)`. This is **not** the Segatz M3 "
            "asthenosphere model, M4 deep-mantle model, or any published physical "
            "dissipation grid. Results from this feature are provisional placeholders. "
            "See Ingest / Tidal Models scaffold for the path to real grids."
        )
        st.warning(
            "**3. Observational coverage bias.** The catalogue reflects where Galileo "
            "and Voyager observed. Absence of a hotspot is not the absence of volcanic activity. "
            "See the Bias & Hypotheses tab."
        )
        st.warning(
            "**4. Catalogue vintage.** Dominated by pre-2003 detections. "
            "Post-2003 observations (New Horizons, JWST, Juno/JIRAM 2023-2025, "
            "Keck AO) are not fully incorporated."
        )
        st.warning(
            "**5. Estimated thermal-emission proxy.** Davies/JIRAM 4.8 micron "
            "spectral radiance is now integrated as `power_gw`, but this is an "
            "estimated proxy, not directly measured bolometric radiant power. "
            "Intensity results are reported in the Thermal Intensity tab."
        )

        st.caption(
            "Interpretation note: this layer reports associations/consistency with the observed catalogue "
            "(not discovery). Leakage and proxy caveats apply."
        )

        st.subheader(t("page.science.methods_summary", language))
        st.markdown(
            "| Analysis | Module | Method |\n"
            "|----------|--------|--------|\n"
            "| Leakage audit | `analysis/leakage_audit.py` | Point-biserial r, Spearman r, LR coefficient |\n"
            "| Ablation | `models/ablation.py` | 5 feature sets x 4-fold spatial CV |\n"
            "| Geology enrichment | `analysis/geology_enrichment.py` | Chi-square, Wilson 95% CI, Bonferroni |\n"
            "| Spatial statistics | `analysis/spatial_stats.py` | Ripley's K on sphere, g(r), NN CDF, 99 MC sims |\n"
            "| Hemispheric asymmetry | `analysis/asymmetry.py` | Exact binomial test, bootstrap histogram CIs |\n"
            "| Coverage bias | `analysis/coverage_bias.py` | Geology-proxy adjusted hotspot rates |\n"
            "| Thermal intensity | `analysis/power_intensity.py` / `models/regression.py` | Estimated JIRAM proxy summaries + spatial-CV Ridge regression |\n"
        )

        try:
            docs_path = (
                Path(__file__).parent.parent / "docs" / "scientific_methods.md"
            )
            if docs_path.exists():
                content = docs_path.read_text(encoding="utf-8")
                st.download_button(
                    t("page.science.download_methods", language),
                    data=content,
                    file_name="io_hotspot_scientific_methods.md",
                    mime="text/markdown",
                )
        except Exception:
            pass

    # â”€â”€ Tab 2: Baseline Model Credibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tabs[0]:
        st.divider()
        st.subheader(t("page.science.baseline", language))
        st.subheader(t("page.science.leakage_audit", language))
        st.markdown(
            "Each feature is audited for three types of concern:\n"
            "- **target-derived**: feature constructed from the hotspot catalogue itself\n"
            "- **analytical proxy**: hand-crafted formula, not an observed quantity\n"
            "- **high correlation or coefficient**: statistically suspicious magnitude"
        )

        with st.spinner("Running leakage audit..."):
            leakage_df = get_leakage_audit(feature_matrix)

        st.dataframe(_style_leakage_table(leakage_df), use_container_width=True)
        st.caption(
            "Red rows are flagged as `suspected_leakage`. "
            "Threshold: |Pearson r| >= 0.6 OR |LR coefficient| >= 5.0 OR target-derived by construction."
        )

        n_flagged = int(leakage_df["suspected_leakage"].sum())
        if n_flagged > 0:
            st.error(
                f"{n_flagged} feature(s) flagged as suspected leakage. "
                "The ablation study below shows how model performance changes when "
                "these features are excluded."
            )

        st.divider()
        st.subheader(t("page.science.ablation", language))
        st.markdown(
            "Five feature sets are trained through the same 4-fold spatial CV. "
            "Bars show mean +/- std across folds. "
            "**Red = leaky baseline** (includes `dist_nearest_hotspot_km`). "
            "**Blue = honest baseline** (no leakage-prone features). "
            "**Grey = null sanity check** (labels permuted - AUC should be ~ 0.5)."
        )

        with st.spinner("Running ablation (this may take ~30 seconds)..."):
            ablation = get_ablation(feature_matrix)

        st.plotly_chart(_ablation_plotly_chart(ablation), use_container_width=True)

        # Per-fold table for the no_leakage set
        no_leak = next(
            (fs for fs in ablation["feature_sets"] if fs["name"] == "no_leakage"),
            None,
        )
        if no_leak:
            st.subheader("Honest baseline (no_leakage) - per-fold detail")
            fold_df = pd.DataFrame(no_leak["folds"])[
                ["fold", "lat_band", "precision", "recall", "f1", "auc_roc", "pr_auc",
                 "n_test", "n_positive_test"]
            ]
            fold_df.columns = [
                "Fold", "Lat band", "Precision", "Recall", "F1", "AUC-ROC", "PR-AUC",
                "N test", "N positive",
            ]
            st.dataframe(
                fold_df.style.format(
                    {c: "{:.3f}" for c in ["Precision", "Recall", "F1", "AUC-ROC", "PR-AUC"]}
                ),
                use_container_width=True,
            )
            st.info(
                "**How to read this:** Without `dist_nearest_hotspot_km`, the model "
                "relies only on geology, tidal proxy, and tidal-stress distance. "
                "If AUC-ROC is substantially lower than the leaky baseline (0.998-0.999), "
                "that confirms the leakage diagnosis. "
                "If AUC is still above ~0.6, geology and/or tidal geometry carry some "
                "genuine predictive signal - but interpret with caution given the "
                "synthetic tidal proxy."
            )

    # â”€â”€ Tab 3: Geological Association â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tabs[0]:
        with st.expander(t("page.science.coefficients", language), expanded=False):
            result = get_model()
            if result is None or result[0] is None:
                st.warning("Trained model not loaded. Coefficient diagnostic is unavailable.")
            else:
                model, _ = result
                _render_lr_coefficients(model)

    with tabs[1]:
        st.subheader(t("page.science.geology", language))
        st.subheader(t("page.science.geology.enrichment", language))
        st.markdown(
            "For each USGS map unit, we compute the **enrichment ratio**: observed "
            "hotspot count divided by the expected count under a complete spatial "
            "randomness (CSR) null (proportional to cell area). "
            "Ratios > 1 indicate enrichment; < 1 indicate depletion. "
            "Significance is assessed via chi-square with **Bonferroni correction** for "
            f"multiple testing across all units. Red rows survive p_bonf < 0.05."
        )
        st.caption(
            "Warning: this analysis treats all cells as equally observable. "
            "Coverage bias (see Bias & Hypotheses tab) may distort enrichment ratios, "
            "particularly for units concentrated in undersampled regions."
        )

        with st.spinner("Computing geology enrichment..."):
            enr_df = get_geology_enrichment(feature_matrix)

        from analysis.geology_enrichment import plot_enrichment_bar
        enr_img_path = plot_enrichment_bar(enr_df)
        st.image(str(enr_img_path), width="stretch")

        st.dataframe(_style_enrichment_table(enr_df), use_container_width=True)

        st.download_button(
            "Download enrichment table (CSV)",
            data=enr_df.to_csv(index=False).encode("utf-8"),
            file_name="io_geology_enrichment.csv",
            mime="text/csv",
        )

        n_sig = int(enr_df["significant"].sum())
        if n_sig > 0:
            sig_units = enr_df.loc[enr_df["significant"], "unit"].tolist()
            st.success(
                f"{n_sig} unit(s) show Bonferroni-significant enrichment or depletion: "
                f"{', '.join(sig_units)}. "
                "These are exploratory associations - not causal claims."
            )
        else:
            st.info(
                "No unit reaches Bonferroni significance. "
                "This may reflect low catalogue size (172 hotspots) or "
                "genuine absence of strong geological control."
            )

    # â”€â”€ Tab 4: Spatial Point-Pattern Analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tabs[1]:
        st.divider()
        if catalog is None:
            st.error("Hotspot catalog not loaded.")
            return

        st.subheader(t("page.science.spatial_stats", language))
        st.markdown(
            "**Ripley's K on a sphere** counts pairs of hotspots within great-circle "
            "distance r. Values above the CSR (complete spatial randomness) envelope "
            "indicate clustering; below indicate regularity. "
            "The **pair-correlation function g(r)** is the derivative of K. "
            "g > 1 at small r means excess close pairs. "
            "The **nearest-neighbour CDF** compares empirical NN distances against "
            "the Poisson expectation."
        )
        st.info(
            "Dashboard uses 99 Monte Carlo simulations for speed. "
            "Run `python scripts/run_scientific_analysis.py` for the full 199-sim result."
        )
        st.caption(
            "Warning: spatial statistics reflect the observed catalogue distribution, "
            "which is shaped by observational coverage. Detected clustering may partly "
            "reflect where instruments looked rather than where hotspots concentrate."
        )

        with st.spinner("Computing spatial statistics (99 MC sims)..."):
            sp_stats = get_spatial_stats(catalog)

        clustered = sp_stats.get("clustered_at_small_radii", False)
        col1, col2 = st.columns(2)
        col1.metric(
            "Small-scale clustering vs CSR",
            "Detected" if clustered else "Not detected",
            help="True if observed K(r) exceeds the 97.5th percentile CSR envelope "
                 "at any tested radius.",
        )
        col2.metric("Hotspots analysed", sp_stats.get("n_points", "-"))

        from analysis.spatial_stats import plot_spatial_stats
        sp_img_path = plot_spatial_stats(sp_stats)
        st.image(str(sp_img_path), width="stretch")

        st.markdown(
            "**How to read the panels:**\n"
            "- Left: K_observed above the grey CSR envelope -> significant clustering.\n"
            "- Centre: g(r) > 1 at small radii -> hotspots are closer together than random.\n"
            "- Right: NN CDF above the dashed Poisson line -> hotspots have shorter "
            "nearest-neighbour distances than random."
        )

    # â”€â”€ Tab 5: Hemispheric Asymmetry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tabs[1]:
        st.divider()
        if catalog is None:
            st.error("Hotspot catalog not loaded.")
            return

        st.subheader(t("page.science.asymmetry", language))
        st.markdown(
            "Three exact binomial tests (H0: equal split) test whether the hotspot "
            "catalogue is asymmetrically distributed between major hemispheres. "
            "Reference model predictions are shown as dashed lines:\n\n"
            "- **Segatz M3** (asthenosphere): hotspot maximum at sub-Jovian (0 deg) and "
            "anti-Jovian (+/-180 deg) -> concentration at |lon| < 90 deg and near equator\n"
            "- **Segatz M4** (deep mantle): hotspot maximum at poles -> "
            "poleward concentration\n\n"
            "We do **not** claim this tests the models directly - that requires the "
            "published dissipation grids, which are not yet available locally. "
            "This compares the catalogue pattern to model predictions qualitatively."
        )
        st.caption(
            "Warning: The USGS catalogue reflects spacecraft coverage, not intrinsic "
            "planetary distribution. Observed asymmetries must be interpreted as "
            "properties of the catalogue before being attributed to planetary physics."
        )

        with st.spinner("Computing hemispheric asymmetry..."):
            asym = get_asymmetry(catalog)

        # Binomial test table
        test_df = pd.DataFrame(asym["binomial_tests"])
        st.dataframe(
            test_df[["comparison", "n_a", "n_b", "n_total", "fraction_a",
                      "p_binom", "ci_lo", "ci_hi", "interpretation"]]
            .style.format(
                {"fraction_a": "{:.3f}", "p_binom": "{:.4f}",
                 "ci_lo": "{:.3f}", "ci_hi": "{:.3f}"}
            )
            .apply(
                lambda row: [
                    "background-color: #ffcccc" if row["p_binom"] < 0.05 else ""
                ] * len(row),
                axis=1,
            ),
            use_container_width=True,
        )
        st.caption("Red rows: p_binom < 0.05 (two-sided exact binomial test). "
                   "ci_lo/ci_hi: 95% Clopper-Pearson CI on fraction_A.")

        st.divider()
        fig_lon, fig_lat = _asymmetry_plotly_histograms(asym)
        st.plotly_chart(fig_lon, use_container_width=True)
        st.plotly_chart(fig_lat, use_container_width=True)
        st.caption(
            f"Bootstrap 95% CIs from {500} catalogue resamples. "
            "Orange dashed lines: M3 (asthenosphere) model prediction maxima. "
            "Purple dashed lines: M4 (deep mantle) prediction maxima."
        )

        st.download_button(
            "Download asymmetry test results (CSV)",
            data=test_df.to_csv(index=False).encode("utf-8"),
            file_name="io_asymmetry_tests.csv",
            mime="text/csv",
        )

    # â”€â”€ Tab 6: Coverage Bias â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tabs[2]:
        st.subheader(t("page.science.coverage_bias", language))
        st.markdown(
            "A fundamental limitation of this project is that the hotspot catalogue "
            "reflects where Galileo and Voyager *looked*, not where volcanic hotspots "
            "*exist*. A grid cell with no hotspot record may be genuinely hotspot-free, "
            "or it may simply be unobserved.\n\n"
            "**Proxy used here:** cells with geology unit `NoData` or `UNKNOWN` are "
            "treated as likely unobserved (no geological mapping was possible - itself "
            "a coverage artefact). This is a conservative lower bound."
        )
        st.warning(
            "This adjustment is approximate. Exact bias correction requires "
            "spacecraft imaging coverage rasters (Galileo SSI/NIMS footprint maps or "
            "Juno/JIRAM coverage grids), which are not yet integrated into this project. "
            "See `analysis/coverage_bias.py` for the ingestion path."
        )

        with st.spinner("Computing coverage bias..."):
            cov_df = get_coverage_bias(feature_matrix)

        if cov_df.empty:
            st.error("Coverage bias computation failed - geology_unit column may be missing.")
            return

        # Summary table
        display_cols = ["lat_band", "total_cells", "observed_cells", "n_hotspots",
                        "unobserved_pct", "raw_rate", "adjusted_rate", "rate_ratio"]
        st.dataframe(
            cov_df[display_cols].style.format(
                {
                    "unobserved_pct": "{:.1f}%",
                    "raw_rate": "{:.5f}",
                    "adjusted_rate": "{:.5f}",
                    "rate_ratio": "{:.3f}",
                }
            ),
            use_container_width=True,
        )
        st.caption(
            "rate_ratio > 1: the raw hotspot density in this band is depressed by "
            "unobserved cells. The adjusted rate = hotspots / geologically-mapped cells only."
        )

        # Plot
        from analysis.coverage_bias import plot_coverage_bias
        cov_img_path = plot_coverage_bias(cov_df)
        st.image(str(cov_img_path), width="stretch")

        st.download_button(
            "Download coverage bias table (CSV)",
            data=cov_df.to_csv(index=False).encode("utf-8"),
            file_name="io_coverage_bias.csv",
            mime="text/csv",
        )

        st.subheader("What data is needed for a rigorous correction")
        st.markdown(
            "1. **Galileo SSI and NIMS imaging footprints** - rasterised to the "
            "1 deg x 1 deg grid to identify cells with at least one thermal observation.\n"
            "2. **Juno/JIRAM coverage maps** - for post-2016 flybys.\n"
            "3. **Detection sensitivity maps** - not all cells with imaging had "
            "sufficient sensitivity to detect weak hotspots (< 1 GW).\n\n"
            "Source: USGS Planetary Data System (PDS), Galileo mission archive. "
            "Contact: USGS Astrogeology Science Center."
        )

    # â”€â”€ Tab 7: Tidal Hypothesis Comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tabs[2]:
        st.divider()
        st.subheader(t("page.science.hypothesis", language))
        st.markdown(
            "Each tidal-heating hypothesis predicts a different *spatial distribution* "
            "of volcanic activity on Io. We test which analytical pattern is most "
            "consistent with the observed USGS catalogue by fitting an inhomogeneous "
            "Poisson point process and comparing log-likelihoods via AIC.\n\n"
            "**Templates compared:**\n"
            "- **asthenosphere** (Segatz M3-like): equatorial maxima at sub-Jovian and anti-Jovian points\n"
            "- **deep_mantle** (Segatz M4-like): polar maxima, equatorial minima\n"
            "- **magma_ocean** (Tyler/Hamilton-like): broad equatorial band\n"
            "- **uniform**: spatially flat null hypothesis\n"
            "- **synthetic_proxy**: the project's original cos^2(lambda)*cos^2(phi) placeholder\n"
            "- **geology_insample**: per-unit hotspot rates (in-sample upper bound - not a fair competitor)\n\n"
            "These are **stylised analytical templates** that reproduce the qualitative spatial morphology "
            "of published tidal-heating hypotheses. They are not the full Segatz (1988) or Beuthe (2013) "
            "dissipation integrals. Replace with real gridded outputs via `ingest/tidal_models.py` "
            "once digitised grid files are available."
        )
        st.warning(
            "**Causality caveat:** A high-likelihood template does not prove the underlying physics. "
            "Observational coverage bias, catalogue vintage, and detection thresholds are not modelled. "
            "The result quantifies *consistency with the spatial pattern*, not physical causation."
        )

        hyp_df = get_hypothesis_comparison(feature_matrix)

        # Î”AIC bar chart
        import plotly.graph_objects as go

        def _aic_colour(delta: float) -> str:
            if delta < 2.0:
                return "#2ecc71"   # green - substantial support
            if delta < 10.0:
                return "#f39c12"   # orange - weak support
            return "#e74c3c"       # red - no support

        hyp_sorted = hyp_df.sort_values("delta_aic")
        colours = [_aic_colour(d) for d in hyp_sorted["delta_aic"]]

        fig_hyp = go.Figure(go.Bar(
            x=hyp_sorted["delta_aic"],
            y=hyp_sorted["template"],
            orientation="h",
            marker_color=colours,
            text=[f"Delta AIC = {d:.1f}" for d in hyp_sorted["delta_aic"]],
            textposition="outside",
        ))
        fig_hyp.add_vline(x=2.0,  line_dash="dash", line_color="#f39c12",
                          annotation_text="Delta AIC = 2", annotation_position="top right")
        fig_hyp.add_vline(x=10.0, line_dash="dash", line_color="#e74c3c",
                          annotation_text="Delta AIC = 10", annotation_position="top right")
        fig_hyp.update_layout(
            title="Delta AIC relative to best template (lower = more consistent with catalogue)",
            xaxis_title="Delta AIC",
            yaxis_title="Template",
            height=380,
            margin=dict(l=180, r=80, t=60, b=40),
        )
        st.plotly_chart(fig_hyp, use_container_width=True)

        st.caption(
            "Burnham & Anderson (2002) thresholds: Delta AIC < 2 = substantial support (green); "
            "2-10 = weak support (orange); > 10 = no support (red). "
            "geology_insample is in-sample and not a fair competitor - it is shown as an upper bound."
        )

        # Winner interpretation
        best_row = hyp_df.iloc[0]
        if best_row["delta_aic"] == 0.0:
            runner = hyp_df.iloc[1]
            st.info(
                f"**Most consistent template:** `{best_row['template']}` "
                f"(pseudo-R^2 = {best_row['pseudo_r2']:.3f}, "
                f"LR p = {best_row['lr_p_value']:.4f} vs uniform null). "
                f"Second-ranked: `{runner['template']}` (Delta AIC = {runner['delta_aic']:.1f}). "
                f"Reference: {best_row['reference']}"
            )

        # Full results table
        with st.expander("Full results table"):
            display_cols = ["template", "log_lik", "aic", "delta_aic",
                            "pseudo_r2", "lr_chi2", "lr_p_value", "reference"]
            st.dataframe(
                hyp_df[display_cols].style.format({
                    "log_lik": "{:.1f}",
                    "aic": "{:.1f}",
                    "delta_aic": "{:.2f}",
                    "pseudo_r2": "{:.4f}",
                    "lr_chi2": "{:.2f}",
                    "lr_p_value": "{:.4f}",
                }),
                use_container_width=True,
            )
            st.download_button(
                "Download hypothesis comparison (CSV)",
                data=hyp_df.to_csv(index=False).encode("utf-8"),
                file_name="io_hypothesis_comparison.csv",
                mime="text/csv",
            )

        st.divider()

        # Catalogue jackknife stability
        st.subheader(t("page.science.stability", language))
        st.markdown(
            "How robust are the scientific claims to catalogue perturbation? "
            "We randomly delete 10%, 25%, or 50% of hotspots (100 replicates each), "
            "re-run geology enrichment and hemispheric binomial tests, and report the "
            "fraction of replicates in which each claim survives (p_bonf < 0.05 or p_binom < 0.05). "
            "A claim with survival rate < 0.5 at 90% retention should be treated as fragile."
        )

        stab_df = get_catalog_stability(feature_matrix, catalog)

        if not stab_df.empty:
            # Pivot: metrics as rows, retention fractions as columns
            pivot = stab_df.pivot_table(
                index="metric",
                columns="fraction_retained",
                values="survival_rate",
            ).reset_index()
            pivot.columns.name = None

            def _colour_survival(val):
                if pd.isna(val) or not isinstance(val, (int, float)):
                    return ""
                if val >= 0.8:
                    return "background-color: #d5f5e3"   # green
                if val >= 0.5:
                    return "background-color: #fef9e7"   # yellow
                return "background-color: #fadbd8"       # red

            format_cols = {c: "{:.0%}" for c in pivot.columns if isinstance(c, float)}
            survival_cols = [c for c in pivot.columns if isinstance(c, float)]
            st.dataframe(
                pivot.style.map(_colour_survival, subset=survival_cols)
                .format(format_cols),
                use_container_width=True,
            )

            # Fragile-claim alert
            fragile = stab_df[
                (stab_df["fraction_retained"] == 0.9) & (stab_df["survival_rate"] < 0.5)
            ]
            if not fragile.empty:
                st.warning(
                    "**Fragile claims** (survival < 50% even at 90% catalogue retention - "
                    "interpret with caution):\n"
                    + "\n".join(f"- `{row['metric']}`: {row['summary']}"
                                for _, row in fragile.iterrows())
                )
            else:
                st.success(
                    "All tested claims survive in >= 50% of replicates at 90% catalogue retention. "
                    "Results are not strongly sensitive to individual hotspot entries."
                )

            st.download_button(
                "Download stability results (CSV)",
                data=stab_df.to_csv(index=False).encode("utf-8"),
                file_name="io_catalog_stability.csv",
                mime="text/csv",
            )
        else:
            st.info("No stability results available.")

    # â”€â”€ Tab 8: Thermal Intensity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tabs[3]:
        st.subheader(t("page.science.thermal", language))
        st.warning(t("page.science.thermal.banner", language))

        power_grid = get_power_grid()
        if power_grid is None:
            st.error(
                "Estimated thermal-emission proxy grid is not available. "
                "Expected `data/raw/io_hotspot_power.csv` and a built base grid."
            )
            return

        obs = power_grid[power_grid["power_count"] > 0].copy()
        col1, col2, col3 = st.columns(3)
        col1.metric(t("page.science.thermal.power_cells", language), f"{len(obs):,}")
        col2.metric(
            t("page.science.thermal.total", language),
            f"{obs['sum_power_gw'].sum():,.0f} GW",
            help=t("page.science.thermal.total_help", language),
        )
        col3.metric(
            t("page.science.thermal.max", language),
            f"{obs['primary_power_gw'].max():,.0f} GW",
            help=t("page.science.thermal.max_help", language),
        )

        suite = get_power_intensity_suite(feature_matrix, power_grid)

        st.divider()
        st.subheader(t("page.science.thermal.latitude", language))
        st.dataframe(
            suite["by_latitude"].style.format(
                {
                    "sum_power_gw": "{:,.1f}",
                    "mean_primary_power_gw": "{:,.2f}",
                    "max_primary_power_gw": "{:,.1f}",
                    "fraction_total_power": "{:.1%}",
                }
            ),
            use_container_width=True,
        )
        st.dataframe(
            suite["polar_sensitivity"].style.format(
                {
                    "polar_sum_power_gw": "{:,.1f}",
                    "nonpolar_sum_power_gw": "{:,.1f}",
                    "polar_fraction_total_power": "{:.1%}",
                    "polar_mean_primary_power_gw": "{:,.2f}",
                    "nonpolar_mean_primary_power_gw": "{:,.2f}",
                }
            ),
            use_container_width=True,
        )

        st.subheader(t("page.science.thermal.geology", language))
        st.dataframe(
            suite["by_geology"].head(20).style.format(
                {
                    "sum_power_gw": "{:,.1f}",
                    "mean_primary_power_gw": "{:,.2f}",
                    "max_primary_power_gw": "{:,.1f}",
                    "fraction_total_power": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

        st.subheader(t("page.science.thermal.outlier", language))
        st.markdown(t("page.science.thermal.outlier.body", language))
        st.dataframe(
            suite["outlier_sensitivity"].style.format(
                {
                    "sum_power_gw": "{:,.1f}",
                    "polar_sum_power_gw": "{:,.1f}",
                    "nonpolar_sum_power_gw": "{:,.1f}",
                    "polar_fraction_total_power": "{:.1%}",
                    "max_remaining_primary_power_gw": "{:,.1f}",
                }
            ),
            use_container_width=True,
        )

        st.divider()
        st.subheader(t("page.science.thermal.regression", language))
        st.markdown(t("page.science.thermal.regression.body", language))
        try:
            reg = get_power_regression(feature_matrix, power_grid)
            overall = reg["overall_oof"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("OOF R^2", f"{overall['r2']:.3f}")
            c2.metric("OOF RMSE", f"{overall['rmse']:.3f}")
            c3.metric("OOF MAE", f"{overall['mae']:.3f}")
            c4.metric("OOF Spearman", f"{overall['spearman']:.3f}")

            folds_df = pd.DataFrame(reg["folds"])
            st.dataframe(
                folds_df[
                    ["fold", "lat_band", "n_train", "n_test", "r2", "rmse", "mae", "spearman"]
                ].style.format(
                    {"r2": "{:.3f}", "rmse": "{:.3f}", "mae": "{:.3f}", "spearman": "{:.3f}"}
                ),
                use_container_width=True,
            )

            residuals = reg["residuals"]
            st.download_button(
                t("page.science.thermal.download_residuals", language),
                data=residuals.to_csv(index=False).encode("utf-8"),
                file_name="io_power_residuals.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Power regression failed: {exc}")

        st.download_button(
            t("page.science.thermal.download_latitude", language),
            data=suite["by_latitude"].to_csv(index=False).encode("utf-8"),
            file_name="io_power_by_latitude.csv",
            mime="text/csv",
        )


def page_about() -> None:
    st.title("About This Project")
    st.markdown(
        """
        ### Data Sources
        | Dataset | Source |
        |---------|--------|
        | Io Volcanic Hotspot Catalog | USGS Astrogeology / Lopes & Williams (2005) |
        | Estimated thermal-emission proxy | Davies et al. (2024) Juno/JIRAM 4.8 micron spectral radiance |
        | Io Geologic Map | Williams et al. (2011), USGS SIM 3168 |
        | Tidal Heating Flux | Published model - see SOURCES.md |

        ### Assumptions & Limitations
        1. **Observational bias:** The catalog reflects where spacecraft had coverage,
           not where hotspots truly exist. The southern hemisphere is undersampled.
        2. **Static tidal model:** We use a time-averaged flux. Real tidal dissipation
           varies with orbital phase.
        3. **Geology map vintage:** Galileo-era data with limited resolution in some regions.
        4. **What "prediction" means:** Relative likelihood given features - not activity,
           temperature, or timing.
        5. **Class imbalance:** ~0.2% positive rate. Accuracy is not reported.

        ### Methods
        - **Grid:** 1 deg x 1 deg (64,800 cells)
        - **Model:** Logistic Regression, `class_weight='balanced'`
        - **Validation:** Spatial CV, 4 latitude-band folds (no random splits)
        - **Metrics:** Precision, Recall, F1, AUC-ROC per fold

        ### Repository
        See `README.md` and `CLAUDE.md` for full documentation.
        """
    )


def page_about_v2() -> None:
    language = get_language()
    st.title(t("page.about.title", language))
    st.markdown(t("page.about.body", language))


def page_faq() -> None:
    language = get_language()
    st.title(t("page.faq.title", language))
    st.caption(t("page.faq.caption", language))
    st.caption(t("page.faq.caption_split", language))
    st.subheader(t("page.faq.public", language))
    public_faq: list[tuple[str, str]] = [
        (t("faq.public.q1", language), t("faq.public.a1", language)),
        (t("faq.public.q2", language), t("faq.public.a2", language)),
        (t("faq.public.q3", language), t("faq.public.a3", language)),
        (t("faq.public.q4", language), t("faq.public.a4", language)),
        (t("faq.public.q5", language), t("faq.public.a5", language)),
    ]
    for q, a in public_faq:
        with st.expander(f"**{q}**", expanded=False):
            st.markdown(a)
    st.divider()
    st.subheader(t("page.faq.research", language))
    researcher_entries = [
        ("thermal",   "faq.research.q1",  "faq.research.a1"),
        ("viewer",    "faq.research.q2",  "faq.research.a2"),
        ("methods",   "faq.research.q3",  "faq.research.a3"),
        ("repro",     "faq.research.q4",  "faq.research.a4"),
        ("methods",   "faq.research.q5",  "faq.research.a5"),
        ("data",      "faq.research.q6",  "faq.research.a6"),
        ("catalogue", "faq.research.q7",  "faq.research.a7"),
        ("catalogue", "faq.research.q8",  "faq.research.a8"),
        ("tidal",     "faq.research.q9",  "faq.research.a9"),
        ("tidal",     "faq.research.q10", "faq.research.a10"),
        ("tidal",     "faq.research.q11", "faq.research.a11"),
        ("spatial",   "faq.research.q12", "faq.research.a12"),
        ("spatial",   "faq.research.q13", "faq.research.a13"),
        ("spatial",   "faq.research.q14", "faq.research.a14"),
        ("methods",   "faq.research.q15", "faq.research.a15"),
        ("methods",   "faq.research.q16", "faq.research.a16"),
        ("publish",   "faq.research.q17", "faq.research.a17"),
        ("methods",   "faq.research.q18", "faq.research.a18"),
    ]
    category_ids = list(dict.fromkeys(category for category, _, _ in researcher_entries))
    category_labels = {
        category_id: t(f"faq.category.{category_id}", language) for category_id in category_ids
    }
    all_topics_label = t("page.faq.filter.all", language)
    selected_label = st.selectbox(
        t("page.faq.filter", language),
        [all_topics_label] + [category_labels[category_id] for category_id in category_ids],
    )
    for category_id, question_key, answer_key in researcher_entries:
        if selected_label != all_topics_label and category_labels[category_id] != selected_label:
            continue
        with st.expander(f"**{t(question_key, language)}**", expanded=False):
            st.markdown(
                f"*{t('page.faq.topic', language, topic=category_labels[category_id])}*"
            )
            st.markdown(t(answer_key, language))
    st.divider()
    st.caption(t("page.faq.footer", language))
# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Overview":
    page_overview_v2()
elif page == "2D Maps":
    page_2d_maps()
elif page == "3D Globe":
    page_3d_globe()
elif page == "Io Experience":
    page_io_experience()
elif page == "Scientific Analysis":
    page_scientific_analysis()
elif page == "About":
    page_about_v2()
elif page == "FAQ":
    page_faq()



