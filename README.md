# 🌊 Flood Risk Zonation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/KunalMK25/flood-risk-zonation/test.yml?label=tests)](https://github.com/KunalMK25/flood-risk-zonation/actions)

An ML-powered geospatial web app that generates **micro-level flood risk zone maps** for any region worldwide — using real NASA SRTM elevation, live OpenStreetMap water bodies, and a Random Forest classifier with cross-validated AUC ~0.97.

---

## 🚀 Live Demo

**[flood-risk-zonation.streamlit.app](https://flood-risk-zonation-ezw8ngm5igpy6egpdmgsqw.streamlit.app/)**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🗺️ Interactive Risk Map | Color-coded flood risk zones (High / Medium / Low / Water) via Folium |
| 🌍 Global Coverage | Works for any bounding box worldwide using live OpenStreetMap data |
| ⛰️ Real Elevation Data | NASA SRTM 30m DEM via OpenTopoData API |
| 💧 Live Water Body Detection | Fetches lakes, rivers, canals, ocean from OSM Overpass API |
| 🌧️ Rainfall Layer | IMD/GPM-calibrated rainfall intensity heatmap |
| 🤖 ML Model | Random Forest classifier with 5-fold stratified CV (AUC ~0.97) |
| 📄 PDF Report Generator | Full flood risk report with emergency deployment plan |
| 🔍 Click Popups | Per-cell risk factor breakdown (elevation, TWI, drainage, rainfall) |
| ⬇️ Export | Download results as CSV, GeoJSON, or interactive HTML |

---

## 🗂️ Project Structure

```
flood-risk-zonation/
├── flood_risk_zonation/        # Core Python package
│   ├── config.py               # BoundingBox, PipelineConfig dataclasses
│   ├── pipeline.py             # End-to-end orchestration
│   ├── exceptions.py           # Custom error types
│   ├── features/
│   │   └── extractor.py        # 10-feature geospatial extractor
│   └── visualization/
│       ├── map_builder.py      # Folium choropleth map
│       ├── export.py           # CSV / GeoJSON / HTML export
│       └── pdf_report.py       # ReportLab PDF generator
├── tests/                      # pytest + Hypothesis property-based tests
├── data/                       # Sample/cached data
├── .streamlit/                 # Streamlit theme config
├── .github/
│   └── workflows/
│       └── test.yml            # CI — runs pytest on every push
├── app.py                      # Streamlit entry point
├── requirements.txt            # Runtime dependencies
├── pyproject.toml              # Package metadata & build config
└── write_wb_module.py          # Utility: writes World Bank-style CSV outputs
```


---

## ⚙️ How It Works

```
Bounding Box Input
       │
       ▼
1. Grid Generation     — divides bbox into configurable cells (250m / 500m / 1km)
       │
       ▼
2. Data Ingestion      — fetches real SRTM elevation + OSM water bodies
       │
       ▼
3. Feature Extraction  — computes 10 features per cell
       │
       ▼
4. ML Training         — trains Random Forest (5-fold stratified CV)
       │
       ▼
5. Risk Scoring        — normalises predictions → [0, 100] → Low/Medium/High
       │
       ▼
6. Water Masking       — marks permanent water cells (blue) via elevation + OSM
       │
       ▼
7. Visualisation       — interactive Folium map + optional PDF/CSV/GeoJSON export
```

---

## 🧠 Features & Model

Ten conditioning factors are computed per grid cell:

| Feature | Description |
|---|---|
| Elevation | Mean SRTM elevation (m) |
| Slope | Terrain slope in degrees (Horn 1981) |
| TWI | Topographic Wetness Index: `ln(A / tan(β))` |
| Annual Rainfall | Mean annual precipitation (mm) — NASA GPM IMERG where available |
| Max 24-hr Rainfall | Extreme rainfall intensity (mm) |
| Distance to Water | Nearest OSM water body (m, log-scaled) |
| Drainage Capacity | Synthetic score [0, 1] |
| Population Density | Persons/km² (log-scaled) |
| Aspect | Terrain aspect (degrees from north) |
| Curvature | Plan curvature |

**Model:** Random Forest (200 trees, 5-fold stratified CV). LightGBM available as an alternative via the sidebar.

> ⚠️ **Data transparency:** When the NASA GPM API is unavailable, rainfall falls back to a region-calibrated synthetic layer (e.g. 970 mm/yr for Bangalore). The active data tier is shown in the UI after each run. See [Data Tiers](#-data-tiers) below.

---

## 🎯 Risk Zones

| Zone | Colour | Score | Description |
|---|---|---|---|
| 🔴 High Risk | Red | > 66 | Low elevation, poor drainage, near water |
| 🟠 Medium Risk | Amber | 34–66 | Moderate vulnerability |
| 🟢 Low Risk | Green | ≤ 33 | Higher elevation, good drainage |
| 🔵 Water | Blue | N/A | Permanent water body (lake / river / ocean) |

---

## 📡 Data Sources

| Dataset | Source | Resolution | Notes |
|---|---|---|---|
| Elevation (DEM) | NASA SRTM via OpenTopoData | ~30m | Tier 1 |
| Rainfall | NASA GPM IMERG | 0.1° | Tier 1; synthetic fallback = Tier 2/3 |
| Water Bodies | OpenStreetMap Overpass API | Vector | Tier 1 |
| Drainage Lines | OpenStreetMap Overpass API | Vector | Tier 1 |
| Population Density | WorldPop-style synthetic | ~1km | Tier 2/3 |

### 📶 Data Tiers

| Tier | Description |
|---|---|
| **Tier 1** | Real SRTM elevation + GPM rainfall + OSM water bodies |
| **Tier 2** | Partial real data with synthetic gap-filling |
| **Tier 3** | Fully synthetic (demo / offline mode) |

The active tier is displayed in the app after each pipeline run.

---

## 🗺️ Example Regions

| Area | Min Lon | Min Lat | Max Lon | Max Lat |
|---|---|---|---|---|
| Gottigere, Bangalore | 77.55 | 12.84 | 77.62 | 12.91 |
| Chennai Marina (Coastal) | 80.24 | 12.98 | 80.31 | 13.05 |
| Dal Lake, Srinagar | 74.83 | 34.07 | 74.90 | 34.14 |
| Puri, Odisha (Cyclone coast) | 85.80 | 19.77 | 85.87 | 19.84 |

---

## 🖥️ Run Locally

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/KunalMK25/flood-risk-zonation.git
cd flood-risk-zonation

# Install as an editable package (recommended)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt

streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=flood_risk_zonation
```

Tests include both standard `pytest` unit tests and **Hypothesis property-based tests** that verify feature extraction contracts and risk scoring invariants.

---

## ☁️ Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select this repo, set main file to `app.py`
4. Click **Deploy**

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Frontend | Streamlit, Folium |
| ML | scikit-learn (Random Forest), LightGBM |
| Geospatial | GeoPandas, Rasterio, Shapely, PyProj |
| Data | NASA SRTM, OpenStreetMap, NASA GPM IMERG |
| PDF | ReportLab |
| Testing | pytest, Hypothesis, pytest-cov |

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.

---

## 👤 Author

**Kunal MK** · [github.com/KunalMK25](https://github.com/KunalMK25)
