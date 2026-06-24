"""
Streamlit web application for the Flood Risk Zonation System.

Run with: streamlit run app.py
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from flood_risk_zonation.config import BoundingBox, PipelineConfig, validate_bbox_size
from flood_risk_zonation.exceptions import FloodRiskError
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
from flood_risk_zonation.ingest.sample_data import DEMO_REGIONS, DemoRegion
from flood_risk_zonation.pipeline import FloodRiskPipeline, _load_land_mask
from flood_risk_zonation.scoring.susceptibility import (
    WeightedSusceptibilityModel,
    RandomForestSusceptibilityModel,  # noqa: F401 — ensure module is loaded
)
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Flood Risk Zonation System",
    page_icon="🌊",
    layout="wide",
)

# ── Cached API wrappers ───────────────────────────────────────────────────────
# Bbox coords are rounded to 4 dp before use so that tiny float jitter from
# the number_input widgets doesn't bust the cache on every rerender.

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_fetch_water_bodies(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    _cache_version: int = 3,  # increment to bust stale cache
) -> gpd.GeoDataFrame:
    """Cached thin wrapper around load_water_bodies."""
    from flood_risk_zonation.ingest.water_bodies import load_water_bodies

    bbox = BoundingBox(
        min_lon=round(min_lon, 4),
        min_lat=round(min_lat, 4),
        max_lon=round(max_lon, 4),
        max_lat=round(max_lat, 4),
    )
    return load_water_bodies(
        bbox,
        data_dir="data/water_bodies",
        allow_network=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🌊 Flood Risk Zonation")
st.sidebar.markdown("---")

# ── Region Selection ─────────────────────────────────────────────────────────
# All 4 preset regions + a "Custom" option for manual bbox entry.
# Offline sample data is enabled automatically for the 3 bundled demo regions.

PRESET_REGIONS = {
    "Gottigere, Bangalore": {
        "min_lon": 77.55, "min_lat": 12.84, "max_lon": 77.62, "max_lat": 12.91,
        "area_name": "Gottigere, Bangalore",
        "offline_key": "Bangalore (Gottigere)",   # matches DEMO_REGIONS key
    },
    "Chennai Marina (Coastal)": {
        "min_lon": 80.24, "min_lat": 12.98, "max_lon": 80.31, "max_lat": 13.05,
        "area_name": "Chennai Marina, Chennai",
        "offline_key": "Chennai Marina (Coastal)",
    },
    "Dal Lake, Srinagar": {
        "min_lon": 74.83, "min_lat": 34.07, "max_lon": 74.90, "max_lat": 34.14,
        "area_name": "Dal Lake, Srinagar",
        "offline_key": "Dal Lake, Srinagar",
    },
    "Puri, Odisha (Cyclone Coast)": {
        "min_lon": 85.80, "min_lat": 19.77, "max_lon": 85.87, "max_lat": 19.84,
        "area_name": "Puri, Odisha",
        "offline_key": None,                       # no bundled offline data — uses live API
    },
    "✏️ Custom Region": {
        "min_lon": 77.55, "min_lat": 12.84, "max_lon": 77.62, "max_lat": 12.91,
        "area_name": "",
        "offline_key": None,
    },
}

st.sidebar.subheader("Region Selection")
selected_preset = st.sidebar.selectbox(
    "Select Region",
    list(PRESET_REGIONS.keys()),
    index=0,
    help="Choose a preset region or select '✏️ Custom Region' to enter coordinates manually.",
)

preset = PRESET_REGIONS[selected_preset]
is_custom = selected_preset == "✏️ Custom Region"

# Determine offline mode: auto-enable for regions that have bundled demo data
offline_key = preset["offline_key"]
has_offline = offline_key is not None and offline_key in DEMO_REGIONS

if has_offline:
    use_offline = st.sidebar.checkbox(
        "📦 Use offline sample data",
        value=False,
        help="Use pre-bundled synthetic data — no network required. Good for demos.",
    )
else:
    use_offline = False

offline_region: DemoRegion | None = None
if use_offline and has_offline:
    offline_region = DEMO_REGIONS[offline_key]
    st.sidebar.info(
        f"📦 Offline mode active — using pre-bundled data for **{selected_preset}**."
    )

st.sidebar.markdown("---")

# ── Bbox inputs — pre-filled from preset, editable only in Custom mode ────────
area_name_input = st.sidebar.text_input(
    "Area Name (for PDF report)",
    value=preset["area_name"] if not is_custom else "",
    disabled=not is_custom,
)

col1, col2 = st.sidebar.columns(2)
min_lon = col1.number_input(
    "Min Lon", value=float(preset["min_lon"]),
    min_value=-180.0, max_value=180.0, step=0.01,
    disabled=not is_custom or use_offline,
)
min_lat = col2.number_input(
    "Min Lat", value=float(preset["min_lat"]),
    min_value=-90.0, max_value=90.0, step=0.01,
    disabled=not is_custom or use_offline,
)
max_lon = col1.number_input(
    "Max Lon", value=float(preset["max_lon"]),
    min_value=-180.0, max_value=180.0, step=0.01,
    disabled=not is_custom or use_offline,
)
max_lat = col2.number_input(
    "Max Lat", value=float(preset["max_lat"]),
    min_value=-90.0, max_value=90.0, step=0.01,
    disabled=not is_custom or use_offline,
)

if not is_custom and not use_offline:
    st.sidebar.caption(
        f"📍 Preset coordinates for **{selected_preset}**. "
        "Select '✏️ Custom Region' to enter your own."
    )

st.sidebar.subheader("Grid Resolution")
resolution_map = {"250m": 250, "500m": 500, "1000m": 1000}
resolution_label = st.sidebar.selectbox(
    "Cell Size", list(resolution_map.keys()), index=1
)
cell_size = resolution_map[resolution_label]

st.sidebar.subheader("Method")
method_choice = st.sidebar.radio(
    "Susceptibility Model",
    ["Ensemble (WSI + RF)", "Weighted Index (WSI)", "Random Forest (ML)"],
    index=0,
    help=(
        "Ensemble: blends WSI + Random Forest for best results — shows full CV metrics.\n"
        "WSI: transparent weighted index, instant results.\n"
        "RF: Random Forest only — shows AUC & F1 after each run."
    ),
)
_model_type_map = {
    "Ensemble (WSI + RF)": "ensemble",
    "Weighted Index (WSI)": "weighted_susceptibility",
    "Random Forest (ML)": "random_forest",
}
selected_model_type = _model_type_map[method_choice]

st.sidebar.subheader("Classification Thresholds")
low_threshold = st.sidebar.slider("Low / Medium boundary", 10.0, 49.0, 33.0, 1.0)
medium_threshold = st.sidebar.slider("Medium / High boundary", 51.0, 90.0, 66.0, 1.0)

run_button = st.sidebar.button(
    "🚀 Run Analysis", type="primary", use_container_width=True
)

# --- Land Mask Status ---
try:
    _load_land_mask()
    st.sidebar.markdown('<p style="color:#2ecc71;font-size:12px;margin-top:5px;margin-bottom:15px;text-align:center">🟢 <b>Land/Sea Mask</b>: Loaded & Active</p>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.markdown(f'<p style="color:#e74c3c;font-size:12px;margin-top:5px;margin-bottom:15px;text-align:center">🔴 <b>Land/Sea Mask</b>: Error ({e})</p>', unsafe_allow_html=True)

# ── Live bbox size preview ────────────────────────────────────────────────────
if not use_offline:
    from math import cos, radians as _rad
    _center_lat = (float(min_lat) + float(max_lat)) / 2.0
    _h = (float(max_lat) - float(min_lat)) * 111.32
    _w = (float(max_lon) - float(min_lon)) * 111.32 * cos(_rad(_center_lat))
    if _w > 0 and _h > 0:
        from flood_risk_zonation.config import BBOX_MIN_SIDE_KM, BBOX_MAX_SIDE_KM
        _ok = (
            _w >= BBOX_MIN_SIDE_KM and _h >= BBOX_MIN_SIDE_KM
            and _w <= BBOX_MAX_SIDE_KM and _h <= BBOX_MAX_SIDE_KM
        )
        _color = "✅" if _ok else "⚠️"
        st.sidebar.caption(
            f"{_color} Area: {_w:.1f} km × {_h:.1f} km "
            f"(limit: {BBOX_MIN_SIDE_KM:.0f}–{BBOX_MAX_SIDE_KM:.0f} km per side)"
        )

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("Flood Risk Zonation System")

if "result" not in st.session_state:
    st.session_state.result = None
if "fallback_warning" not in st.session_state:
    st.session_state.fallback_warning = False

if run_button:
    st.session_state.fallback_warning = False
    try:
        # Determine bbox
        if offline_region is not None:
            bbox = offline_region.bbox
        else:
            bbox = BoundingBox(
                min_lon=float(min_lon),
                min_lat=float(min_lat),
                max_lon=float(max_lon),
                max_lat=float(max_lat),
            )

        config = PipelineConfig(
            cell_size_meters=float(cell_size),
            model_type=selected_model_type,
            rf_n_estimators=100,
            low_threshold=float(low_threshold),
            medium_threshold=float(medium_threshold),
            use_cache=False,
            allow_network=not use_offline,
        )

        # ── Bbox size validation (skip for offline — bbox is pre-defined) ──
        if not use_offline:
            size_error = validate_bbox_size(bbox)
            if size_error:
                st.error(f"📐 **Invalid area size** — {size_error}")
                st.stop()

        # ── Phase 4: st.status() progress indicator ───────────────────────
        with st.status("Running analysis…", expanded=True) as status:

            if use_offline and offline_region is not None:
                # ── Offline path: serve bundled sample data ───────────────
                st.write("📦 Loading offline sample data…")
                from flood_risk_zonation.ingest.sample_data import (
                    get_demo_elevation,
                    get_demo_rainfall,
                    get_demo_water_bodies,
                )
                from flood_risk_zonation.ingest.drainage import (
                    generate_synthetic_drainage,
                )
                from flood_risk_zonation.ingest.population import load_population
                from flood_risk_zonation.grid.generator import generate_grid
                from flood_risk_zonation.features.extractor import extract_features
                from flood_risk_zonation.scoring.scorer import FloodRiskScorer
                from flood_risk_zonation.scoring.susceptibility import (
                    WeightedSusceptibilityModel,
                )
                from flood_risk_zonation.models import AnalysisResult, FloodRiskResult
                import time as _time

                t0 = _time.time()

                st.write("⚙️ Generating grid…")
                grid = generate_grid(
                    bbox,
                    config.cell_size_meters,
                    max_cells=config.max_grid_cells,
                )

                st.write("⛰️ Loading elevation…")
                elevation = get_demo_elevation(offline_region, resolution_m=30.0)

                st.write("🌧️ Loading rainfall…")
                rainfall = get_demo_rainfall(offline_region)

                st.write("💧 Loading water bodies…")
                water_bodies = get_demo_water_bodies(offline_region)

                st.write("👥 Loading population…")
                population = load_population(bbox, data_dir=str(config.cache_dir))
                drainage = generate_synthetic_drainage(grid, seed=config.random_seed)

                st.write("🔬 Computing features…")
                featured_grid = extract_features(
                    grid, elevation, rainfall, water_bodies, population, drainage
                )

                st.write("🤖 Running susceptibility model…")
                X = featured_grid[FEATURE_COLUMNS].copy()
                from flood_risk_zonation.scoring.susceptibility import (
                    WeightedSusceptibilityModel,
                    RandomForestSusceptibilityModel,
                    EnsembleSusceptibilityModel,
                )
                if selected_model_type == "ensemble":
                    model = EnsembleSusceptibilityModel(
                        n_estimators=100, cv_folds=5, random_state=config.random_seed,
                    ).fit(X)
                    analysis_result = AnalysisResult(
                        model=model, feature_names=list(model.feature_names),
                        feature_importances=model.feature_importances, method="ensemble",
                        validation_note=(
                            f"Ensemble (WSI + RF). 5-fold CV — "
                            f"AUC: {model.mean_cv_auc:.3f}, F1: {model.mean_cv_f1:.3f}, "
                            f"Accuracy: {model.mean_cv_accuracy:.3f}."
                        ),
                        mean_cv_auc=model.mean_cv_auc, mean_cv_f1=model.mean_cv_f1,
                        mean_cv_accuracy=model.mean_cv_accuracy,
                        mean_cv_precision=model.mean_cv_precision,
                        mean_cv_recall=model.mean_cv_recall,
                        cv_auc_scores=model.cv_auc_scores, cv_f1_scores=model.cv_f1_scores,
                        cv_accuracy_scores=model.cv_accuracy_scores,
                        cv_precision_scores=model.cv_precision_scores,
                        cv_recall_scores=model.cv_recall_scores,
                    )
                elif selected_model_type == "random_forest":
                    model = RandomForestSusceptibilityModel(
                        n_estimators=100, cv_folds=5, random_state=config.random_seed,
                    ).fit(X)
                    analysis_result = AnalysisResult(
                        model=model, feature_names=list(model.feature_names),
                        feature_importances=model.feature_importances, method="random_forest",
                        validation_note=(
                            f"Random Forest. 5-fold CV — "
                            f"AUC: {model.mean_cv_auc:.3f}, F1: {model.mean_cv_f1:.3f}."
                        ),
                        mean_cv_auc=model.mean_cv_auc, mean_cv_f1=model.mean_cv_f1,
                        cv_auc_scores=model.cv_auc_scores, cv_f1_scores=model.cv_f1_scores,
                    )
                else:
                    model = WeightedSusceptibilityModel().fit(X)
                    analysis_result = AnalysisResult(
                        model=model, feature_names=list(model.feature_names),
                        feature_importances=model.feature_importances,
                        method="weighted_susceptibility_index",
                        validation_note="Relative susceptibility index; not calibrated against observed flood events.",
                    )

                st.write("🗺️ Scoring and rendering map…")
                scorer = FloodRiskScorer()
                scorer.p_min = 0.0
                scorer.p_max = 1.0
                thresholds = {
                    "low_max": config.low_threshold,
                    "medium_max": config.medium_threshold,
                }
                scored_grid = scorer.score_grid(
                    featured_grid, model, FEATURE_COLUMNS, thresholds
                )

                # Apply water mask inline (reuse pipeline method)
                pipeline = FloodRiskPipeline(config)
                scored_grid = pipeline._apply_water_mask_and_proximity_boost(
                    scored_grid, water_bodies, config,
                    elevation_source="offline_sample",
                )

                duration = _time.time() - t0
                provenance = {
                    "elevation": "offline_sample",
                    "rainfall": "offline_sample",
                    "water_bodies": "offline_sample",
                    "population": population.source,
                    "drainage": "synthetic",
                }

                result = FloodRiskResult(
                    scored_grid=scored_grid,
                    analysis_result=analysis_result,
                    bounding_box=bbox,
                    config=config,
                    pipeline_duration_seconds=duration,
                    cell_count=len(scored_grid),
                    data_provenance=provenance,
                    data_tier=3,
                )

            else:
                # ── Live path with progress stages ────────────────────────
                from flood_risk_zonation.ingest.water_bodies import load_water_bodies

                # Stage 1: Elevation
                st.write("⛰️ Fetching elevation…")
                from flood_risk_zonation.ingest.elevation import (
                    load_elevation,
                    generate_synthetic_elevation,
                    fetch_elevation_api,
                )
                from pathlib import Path as _Path

                elev_dir = _Path("data/elevation")
                elevation = None
                if elev_dir.exists():
                    try:
                        elevation = load_elevation(bbox, elev_dir)
                    except FloodRiskError:
                        pass
                # Fallback 1: OpenTopoData SRTM API (ocean = 0m, land > 0)
                if elevation is None:
                    elevation = fetch_elevation_api(bbox, resolution_m=500)
                # Fallback 2: synthetic
                if elevation is None:
                    elevation = generate_synthetic_elevation(
                        bbox, resolution_m=500, seed=config.random_seed
                    )

                # Stage 2: Water bodies (cached + retried)
                st.write("💧 Fetching water bodies…")
                water_bodies = _cached_fetch_water_bodies(
                    round(bbox.min_lon, 4),
                    round(bbox.min_lat, 4),
                    round(bbox.max_lon, 4),
                    round(bbox.max_lat, 4),
                    3,
                )
                # Check if fallback was triggered
                if water_bodies.attrs.get("source") == "fallback":
                    st.session_state.fallback_warning = True

                # Stage 3–5: Delegate remaining stages to the pipeline
                # (rainfall, population, drainage, features, model, scoring)
                # We pass pre-fetched elevation and water_bodies through
                # by running the full pipeline — it will skip its own
                # elevation/water fetch since we pass allow_network=True
                # but the cache file will be warm from _cached_fetch_water_bodies.
                st.write("🌧️ Fetching rainfall…")
                from flood_risk_zonation.ingest.rainfall import (
                    load_rainfall,
                    generate_synthetic_rainfall,
                )

                rain_dir = _Path("data/rainfall")
                if list(rain_dir.glob("*.tif")):
                    try:
                        rainfall = load_rainfall(bbox, rain_dir)
                    except Exception:
                        rainfall = generate_synthetic_rainfall(
                            bbox, resolution_m=1000, seed=config.random_seed
                        )
                else:
                    rainfall = generate_synthetic_rainfall(
                        bbox, resolution_m=1000, seed=config.random_seed
                    )

                st.write("🔬 Computing features…")
                from flood_risk_zonation.ingest.population import load_population
                from flood_risk_zonation.ingest.drainage import (
                    generate_synthetic_drainage,
                )
                from flood_risk_zonation.grid.generator import generate_grid
                from flood_risk_zonation.features.extractor import extract_features
                from flood_risk_zonation.scoring.scorer import FloodRiskScorer
                from flood_risk_zonation.scoring.susceptibility import (
                    WeightedSusceptibilityModel,
                )
                from flood_risk_zonation.models import AnalysisResult, FloodRiskResult
                import time as _time
                from flood_risk_zonation.utils.validation import (
                    validate_bounding_box,
                    validate_config,
                )

                validate_bounding_box(bbox)
                validate_config(config)

                t0 = _time.time()
                grid = generate_grid(
                    bbox,
                    config.cell_size_meters,
                    max_cells=config.max_grid_cells,
                )
                population = load_population(bbox, data_dir=str(config.cache_dir))
                drainage = generate_synthetic_drainage(grid, seed=config.random_seed)
                featured_grid = extract_features(
                    grid, elevation, rainfall, water_bodies, population, drainage
                )

                st.write("🤖 Running susceptibility model…")
                X = featured_grid[FEATURE_COLUMNS].copy()
                from flood_risk_zonation.scoring.susceptibility import (
                    WeightedSusceptibilityModel as _WSI,
                    RandomForestSusceptibilityModel as _RF,
                    EnsembleSusceptibilityModel as _ENS,
                )
                if selected_model_type == "ensemble":
                    model = _ENS(
                        n_estimators=100, cv_folds=5, random_state=config.random_seed,
                    ).fit(X)
                    analysis_result = AnalysisResult(
                        model=model, feature_names=list(model.feature_names),
                        feature_importances=model.feature_importances, method="ensemble",
                        validation_note=(
                            f"Ensemble (WSI + RF). 5-fold CV — "
                            f"AUC: {model.mean_cv_auc:.3f}, F1: {model.mean_cv_f1:.3f}, "
                            f"Accuracy: {model.mean_cv_accuracy:.3f}."
                        ),
                        mean_cv_auc=model.mean_cv_auc, mean_cv_f1=model.mean_cv_f1,
                        mean_cv_accuracy=model.mean_cv_accuracy,
                        mean_cv_precision=model.mean_cv_precision,
                        mean_cv_recall=model.mean_cv_recall,
                        cv_auc_scores=model.cv_auc_scores, cv_f1_scores=model.cv_f1_scores,
                        cv_accuracy_scores=model.cv_accuracy_scores,
                        cv_precision_scores=model.cv_precision_scores,
                        cv_recall_scores=model.cv_recall_scores,
                    )
                elif selected_model_type == "random_forest":
                    model = _RF(
                        n_estimators=100, cv_folds=5, random_state=config.random_seed,
                    ).fit(X)
                    analysis_result = AnalysisResult(
                        model=model, feature_names=list(model.feature_names),
                        feature_importances=model.feature_importances, method="random_forest",
                        validation_note=(
                            f"Random Forest. 5-fold CV — "
                            f"AUC: {model.mean_cv_auc:.3f}, F1: {model.mean_cv_f1:.3f}."
                        ),
                        mean_cv_auc=model.mean_cv_auc, mean_cv_f1=model.mean_cv_f1,
                        cv_auc_scores=model.cv_auc_scores, cv_f1_scores=model.cv_f1_scores,
                    )
                else:
                    model = _WSI().fit(X)
                    analysis_result = AnalysisResult(
                        model=model, feature_names=list(model.feature_names),
                        feature_importances=model.feature_importances,
                        method="weighted_susceptibility_index",
                        validation_note="Relative susceptibility index; not calibrated against observed flood events.",
                    )

                st.write("🗺️ Scoring and rendering map…")
                scorer = FloodRiskScorer()
                scorer.p_min = 0.0
                scorer.p_max = 1.0
                thresholds = {
                    "low_max": config.low_threshold,
                    "medium_max": config.medium_threshold,
                }
                scored_grid = scorer.score_grid(
                    featured_grid, model, FEATURE_COLUMNS, thresholds
                )
                pipeline = FloodRiskPipeline(config)
                scored_grid = pipeline._apply_water_mask_and_proximity_boost(
                    scored_grid, water_bodies, config,
                    elevation_source=elevation.source,
                )

                duration = _time.time() - t0

                wb_source = water_bodies.attrs.get("source", "unavailable")
                provenance = {
                    "elevation": elevation.source,
                    "rainfall": rainfall.source,
                    "water_bodies": wb_source,
                    "population": population.source,
                    "drainage": "synthetic",
                }
                core_real = [
                    provenance["elevation"] != "synthetic",
                    provenance["rainfall"] != "synthetic",
                    wb_source in {"osm_overpass", "osm_cache"},
                ]
                data_tier = (
                    1 if all(core_real) else (2 if any(core_real) else 3)
                )

                result = FloodRiskResult(
                    scored_grid=scored_grid,
                    analysis_result=analysis_result,
                    bounding_box=bbox,
                    config=config,
                    pipeline_duration_seconds=duration,
                    cell_count=len(scored_grid),
                    data_provenance=provenance,
                    data_tier=data_tier,
                )

            status.update(label="✅ Analysis complete", state="complete", expanded=False)

        st.session_state.result = result
        st.success(
            f"✅ Analysis complete — {result.cell_count} cells in "
            f"{result.pipeline_duration_seconds:.1f}s  "
            f"(Tier {result.data_tier} data)"
        )

    except FloodRiskError as exc:
        st.error(f"Pipeline error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        logger.exception("Unhandled exception in pipeline")

# ── Fallback warning banner ───────────────────────────────────────────────────
if st.session_state.get("fallback_warning"):
    st.warning(
        "⚠️ **API fallback active** — the OSM Overpass API was unreachable after "
        "3 retries. Water body data is unavailable for this run. Risk scores are "
        "based on elevation, terrain, and rainfall only. "
        "Try again later, or enable **Use offline sample data** for a reliable demo.",
        icon="⚠️",
    )

result = st.session_state.result

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🗺️ Interactive Map",
        "📊 Risk Statistics",
        "📈 Factor Weights",
        "📋 Data Table",
        "📖 Methodology",
    ]
)

with tab1:
    if result is not None:
        center = result.bounding_box.center
        builder = FloodRiskMapBuilder()
        # Pass model normalisation bounds so factor bars scale to this dataset
        _model = result.analysis_result.model
        _model_bounds = None
        if hasattr(_model, "lower_") and hasattr(_model, "upper_"):
            _model_bounds = {
                f: (_model.lower_[f], _model.upper_[f])
                for f in _model.lower_
                if f in _model.upper_
            }
        m = builder.build_choropleth_map(
            result.scored_grid, center=center, zoom_start=11,
            model_bounds=_model_bounds,
        )
        map_html = m._repr_html_()
        st.components.v1.html(map_html, height=600, scrolling=False)
    else:
        st.info(
            "Configure parameters in the sidebar and click **Run Analysis** to generate a map."
        )

with tab2:
    if result is not None:
        dist = result.risk_distribution
        color_map = {
            "Low": "#2ecc71",
            "Medium": "#f39c12",
            "High": "#e74c3c",
            "Water": "#3498db",
        }
        dist_no_water = {k: v for k, v in dist.items() if k != "Water"}
        labels = list(dist_no_water.keys())
        counts = list(dist_no_water.values())
        bar_colors = [color_map.get(l, "#999") for l in labels]

        if counts:
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots()
                ax.bar(labels, counts, color=bar_colors)
                ax.set_ylabel("Cell Count")
                ax.set_title("Risk Class Distribution (excl. Water)")
                st.pyplot(fig)
                plt.close(fig)
            with col_b:
                fig2, ax2 = plt.subplots()
                ax2.pie(
                    counts,
                    labels=labels,
                    colors=bar_colors,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax2.set_title("Risk Class Share")
                st.pyplot(fig2)
                plt.close(fig2)
        else:
            st.info("The selected area contains only permanent-water cells.")

        n_water = dist.get("Water", 0)
        if n_water:
            st.info(
                f"ℹ️ {n_water} cells identified as permanent water bodies "
                "(lakes/tanks) — shown in blue on the map, excluded from flood "
                "risk statistics."
            )
    else:
        st.info("Run the analysis first.")

with tab3:
    if result is not None:
        fi = result.analysis_result.feature_importances
        features = list(fi.keys())
        importances = list(fi.values())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(features[::-1], importances[::-1], color="#3498db")
        ax.set_xlabel("Importance")
        ax.set_title("Susceptibility Factor Weights")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(result.analysis_result.validation_note)

        # Show CV metrics if RF or Ensemble was used
        ar = result.analysis_result
        if ar.method in ("random_forest", "ensemble") and ar.mean_cv_auc is not None:
            st.markdown("**Cross-Validation Results (5-fold stratified)**")
            method_label = "Ensemble (WSI + RF)" if ar.method == "ensemble" else "Random Forest"
            st.caption(f"Model: {method_label} · Labels: top-33% WSI scores = high risk")

            # Metric cards
            cols = st.columns(5)
            cols[0].metric("AUC-ROC", f"{ar.mean_cv_auc:.3f}")
            cols[1].metric("F1 Score", f"{ar.mean_cv_f1:.3f}")
            cols[2].metric("Accuracy", f"{ar.mean_cv_accuracy:.3f}" if ar.mean_cv_accuracy else "—")
            cols[3].metric("Precision", f"{ar.mean_cv_precision:.3f}" if ar.mean_cv_precision else "—")
            cols[4].metric("Recall", f"{ar.mean_cv_recall:.3f}" if ar.mean_cv_recall else "—")

            # Per-fold table
            if ar.cv_auc_scores:
                n_folds = len(ar.cv_auc_scores)
                fold_df = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(n_folds)],
                    "AUC-ROC":   [f"{v:.3f}" for v in ar.cv_auc_scores],
                    "F1":        [f"{v:.3f}" for v in (ar.cv_f1_scores or [])],
                    "Accuracy":  [f"{v:.3f}" for v in (ar.cv_accuracy_scores or ["—"]*n_folds)],
                    "Precision": [f"{v:.3f}" for v in (ar.cv_precision_scores or ["—"]*n_folds)],
                    "Recall":    [f"{v:.3f}" for v in (ar.cv_recall_scores or ["—"]*n_folds)],
                })
                st.dataframe(fold_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run the analysis first.")

with tab4:
    if result is not None:
        display_cols = (
            ["cell_id", "centroid_lat", "centroid_lon"]
            + FEATURE_COLUMNS
            + ["risk_score", "risk_class"]
        )
        available = [c for c in display_cols if c in result.scored_grid.columns]
        df = result.scored_grid[available].copy()

        risk_filter = st.multiselect(
            "Filter by Risk Class",
            ["Low", "Medium", "High", "Water"],
            default=["Low", "Medium", "High", "Water"],
        )
        df_filtered = df[df["risk_class"].isin(risk_filter)]
        st.dataframe(df_filtered, use_container_width=True)

        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            csv_buf = io.StringIO()
            df_filtered.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv_buf.getvalue(),
                file_name="flood_risk.csv",
                mime="text/csv",
            )
        with col_dl2:
            geojson_str = result.scored_grid.to_json()
            st.download_button(
                "⬇️ Download GeoJSON",
                geojson_str,
                file_name="flood_risk.geojson",
                mime="application/json",
            )
        with col_dl3:
            if st.button("📄 Generate PDF Report"):
                with st.spinner("Generating PDF report..."):
                    try:
                        import tempfile
                        from flood_risk_zonation.visualization.pdf_report import (
                            export_pdf_report,
                        )

                        area_name = (
                            area_name_input.strip()
                            if area_name_input.strip()
                            else preset["area_name"]
                            if preset["area_name"]
                            else (
                                f"Lat {result.bounding_box.min_lat:.3f}–"
                                f"{result.bounding_box.max_lat:.3f}, "
                                f"Lon {result.bounding_box.min_lon:.3f}–"
                                f"{result.bounding_box.max_lon:.3f}"
                            )
                        )
                        with tempfile.TemporaryDirectory() as tmpdir:
                            pdf_path = export_pdf_report(
                                result,
                                Path(tmpdir) / "flood_risk_report.pdf",
                                area_name=area_name,
                                data_tier=result.data_tier,
                            )
                            pdf_bytes = Path(pdf_path).read_bytes()
                        st.download_button(
                            "⬇️ Download PDF Report",
                            pdf_bytes,
                            file_name="flood_risk_report.pdf",
                            mime="application/pdf",
                        )
                        st.success("PDF report generated!")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
    else:
        st.info("Run the analysis first.")

with tab5:
    st.markdown(
        """
## Methodology

### Overview
This system generates a relative flood-susceptibility index from multiple
geospatial conditioning factors. It is a planning aid, not an event forecast.

### Data Sources
| Dataset | Source | Resolution |
|---|---|---|
| Elevation (DEM) | NASA SRTM (local GeoTIFF; synthetic fallback) | ~30m |
| Rainfall | NASA GPM IMERG / IMD (synthetic fallback) | 0.1° |
| Water Bodies | OpenStreetMap Overpass API (with local cache) | Vector |
| Drainage Capacity | Synthetic (population-density-correlated) | per-cell |
| Population Density | Local raster or synthetic fallback | ~1km |

### Features
Ten conditioning factors are computed per grid cell:
- **Elevation** — mean SRTM elevation (m)
- **Slope** — terrain slope in degrees (Horn 1981)
- **TWI** — Topographic Wetness Index: `ln(A / tan(β))`
- **Rainfall** — mean annual and max 24-hour rainfall (mm)
- **Distance to Water** — nearest OSM water body (m)
- **Drainage Capacity** — synthetic score [0, 1]
- **Population Density** — persons/km²
- **Aspect** — terrain aspect (degrees from north)
- **Curvature** — plan curvature

### Method
Each factor is robustly normalized using its 5th–95th percentile range and
combined with a declared weight and risk direction. Aspect is displayed but
not weighted because it has no globally consistent flood relationship.

### Risk Score
The weighted index is mapped directly to [0, 100]. Scores are relative to the
selected study area and are not calibrated against observed flood events.

### Reliability
- **Caching**: OSM water body results are cached locally (by bbox) and in
  Streamlit's in-memory cache for 1 hour, so repeated runs for the same
  region are instant.
- **Retry logic**: Overpass API calls are retried up to 3 times with
  exponential back-off (2 s → 4 s → 8 s) before falling back.
- **Offline mode**: The sidebar "Use offline sample data" checkbox serves
  pre-configured synthetic data for Bangalore, Chennai, and Srinagar —
  no network required.

### Data Tiers
- **Tier 1**: Real SRTM + GPM + OSM data
- **Tier 2**: Partial real data with synthetic gap-filling
- **Tier 3**: Fully synthetic data (demo/offline mode)
"""
    )
