"""
Streamlit web application for the Flood Risk Zonation System.

Run with: streamlit run app.py
"""
from __future__ import annotations

import io
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.exceptions import FloodRiskError
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
from flood_risk_zonation.pipeline import FloodRiskPipeline
from flood_risk_zonation.visualization.export import export_csv, export_geojson, export_html
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Flood Risk Zonation System",
    page_icon="🌊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🌊 Flood Risk Zonation")
st.sidebar.markdown("---")

st.sidebar.subheader("Region Selection")
area_name_input = st.sidebar.text_input("Area Name (for PDF report)", value="Gottigere, Bangalore")
col1, col2 = st.sidebar.columns(2)
min_lon = col1.number_input("Min Lon", value=77.0, min_value=-180.0, max_value=180.0, step=0.1)
min_lat = col2.number_input("Min Lat", value=28.0, min_value=-90.0, max_value=90.0, step=0.1)
max_lon = col1.number_input("Max Lon", value=77.5, min_value=-180.0, max_value=180.0, step=0.1)
max_lat = col2.number_input("Max Lat", value=28.5, min_value=-90.0, max_value=90.0, step=0.1)

st.sidebar.subheader("Grid Resolution")
resolution_map = {"250m": 250, "500m": 500, "1000m": 1000}
resolution_label = st.sidebar.selectbox("Cell Size", list(resolution_map.keys()), index=1)
cell_size = resolution_map[resolution_label]

st.sidebar.subheader("Model")
model_type = st.sidebar.selectbox("Algorithm", ["random_forest", "lightgbm"], index=0)

st.sidebar.subheader("Classification Thresholds")
low_threshold = st.sidebar.slider("Low / Medium boundary", 10.0, 49.0, 33.0, 1.0)
medium_threshold = st.sidebar.slider("Medium / High boundary", 51.0, 90.0, 66.0, 1.0)

run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("Flood Risk Zonation System")
if "area_name_input" not in dir():
    area_name_input = "Study Area"

if "result" not in st.session_state:
    st.session_state.result = None

if run_button:
    try:
        bbox = BoundingBox(
            min_lon=float(min_lon),
            min_lat=float(min_lat),
            max_lon=float(max_lon),
            max_lat=float(max_lat),
        )
        config = PipelineConfig(
            cell_size_meters=float(cell_size),
            model_type=model_type,
            rf_n_estimators=100,
            low_threshold=float(low_threshold),
            medium_threshold=float(medium_threshold),
            use_cache=False,
        )
        with st.spinner("Running pipeline…"):
            pipeline = FloodRiskPipeline(config)
            result = pipeline.run(bbox)
        st.session_state.result = result
        st.session_state.data_tier = pipeline._data_tier
        st.success(
            f"✅ Analysis complete — {result.cell_count} cells in "
            f"{result.pipeline_duration_seconds:.1f}s (Tier {pipeline._data_tier} data)"
        )
    except FloodRiskError as exc:
        st.error(f"Pipeline error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        logger.exception("Unhandled exception in pipeline")

result = st.session_state.result

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Interactive Map",
    "📊 Risk Statistics",
    "📈 Feature Importance",
    "📋 Data Table",
    "📖 Methodology",
])

with tab1:
    if result is not None:
        center = result.bounding_box.center
        builder = FloodRiskMapBuilder()
        m = builder.build_choropleth_map(result.scored_grid, center=center, zoom_start=11)
        map_html = m._repr_html_()
        st.components.v1.html(map_html, height=600, scrolling=False)
    else:
        st.info("Configure parameters in the sidebar and click **Run Analysis** to generate a map.")

with tab2:
    if result is not None:
        dist = result.risk_distribution
        labels = list(dist.keys())
        counts = list(dist.values())
        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#3498db"]
        color_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c", "Water": "#3498db"}
        # Exclude Water cells from risk statistics charts
        dist_no_water = {k: v for k, v in dist.items() if k != "Water"}
        labels = list(dist_no_water.keys())
        counts = list(dist_no_water.values())
        bar_colors = [color_map.get(l, "#999") for l in labels]

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
            ax2.pie(counts, labels=labels, colors=bar_colors, autopct="%1.1f%%", startangle=90)
            ax2.set_title("Risk Class Share")
            st.pyplot(fig2)
            plt.close(fig2)
        n_water = dist.get("Water", 0)
        if n_water:
            st.info(f"ℹ️ {n_water} cells identified as permanent water bodies (lakes/tanks) — shown in blue on the map, excluded from flood risk statistics.")
    else:
        st.info("Run the analysis first.")

with tab3:
    if result is not None:
        fi = result.training_result.feature_importances
        features = list(fi.keys())
        importances = list(fi.values())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(features[::-1], importances[::-1], color="#3498db")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importances")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Mean CV AUC: {result.training_result.mean_cv_auc:.3f} | "
                   f"Mean CV F1: {result.training_result.mean_cv_f1:.3f}")
    else:
        st.info("Run the analysis first.")

with tab4:
    if result is not None:
        display_cols = ["cell_id", "centroid_lat", "centroid_lon"] + FEATURE_COLUMNS + ["risk_score", "risk_class"]
        available = [c for c in display_cols if c in result.scored_grid.columns]
        df = result.scored_grid[available].copy()

        risk_filter = st.multiselect("Filter by Risk Class", ["Low", "Medium", "High", "Water"],
                                     default=["Low", "Medium", "High", "Water"])
        df_filtered = df[df["risk_class"].isin(risk_filter)]
        st.dataframe(df_filtered, use_container_width=True)

        # Download buttons
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            csv_buf = io.StringIO()
            df_filtered.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download CSV", csv_buf.getvalue(),
                               file_name="flood_risk.csv", mime="text/csv")
        with col_dl2:
            geojson_str = result.scored_grid.to_json()
            st.download_button("⬇️ Download GeoJSON", geojson_str,
                               file_name="flood_risk.geojson", mime="application/json")
        with col_dl3:
            if st.button("📄 Generate PDF Report"):
                with st.spinner("Generating PDF report..."):
                    try:
                        import tempfile
                        from flood_risk_zonation.visualization.pdf_report import export_pdf_report
                        area_name = area_name_input if area_name_input.strip() else f"Lat {result.bounding_box.min_lat:.3f}–{result.bounding_box.max_lat:.3f}, Lon {result.bounding_box.min_lon:.3f}–{result.bounding_box.max_lon:.3f}"
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            pdf_path = export_pdf_report(
                                result, tmp.name,
                                area_name=area_name,
                                data_tier=getattr(st.session_state, "data_tier", 3),
                            )
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        st.download_button(
                            "⬇️ Download PDF Report", pdf_bytes,
                            file_name="flood_risk_report.pdf", mime="application/pdf",
                        )
                        st.success("PDF report generated!")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
    else:
        st.info("Run the analysis first.")

with tab5:
    st.markdown("""
## Methodology

### Overview
This system generates micro-level flood risk zone maps using machine learning
applied to multi-source geospatial data.

### Data Sources
| Dataset | Source | Resolution |
|---|---|---|
| Elevation (DEM) | NASA SRTM | ~30m |
| Rainfall | NASA GPM IMERG / IMD | 0.1° |
| Water Bodies | OpenStreetMap | Vector |
| Population Density | WorldPop | ~100m |

### Features
Ten conditioning factors are computed per grid cell:
- **Elevation** — mean SRTM elevation (m)
- **Slope** — terrain slope in degrees (Horn 1981)
- **TWI** — Topographic Wetness Index: `ln(A / tan(β))`
- **Rainfall** — mean annual and max 24-hour rainfall (mm)
- **Distance to Water** — nearest OSM water body (m, log-scaled)
- **Drainage Capacity** — synthetic score [0, 1]
- **Population Density** — persons/km² (log-scaled)
- **Aspect** — terrain aspect (degrees from north)
- **Curvature** — plan curvature

### Model
Random Forest classifier (200 trees, 5-fold stratified CV) with AUC-ROC
and F1 reporting. LightGBM available as an alternative.

### Risk Score
Raw predicted probabilities are normalized to [0, 100] using min-max
scaling calibrated from the 1st–99th percentile of training probabilities.

### Data Tiers
- **Tier 1**: Real SRTM + GPM + OSM data
- **Tier 2**: Partial real data with synthetic gap-filling
- **Tier 3**: Fully synthetic data (demo/offline mode)
""")
