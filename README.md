# 🌊 Flood Risk Zonation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/KunalMK25/flood-risk-zonation/test.yml?label=tests)](https://github.com/KunalMK25/flood-risk-zonation/actions)

A geospatial web app that generates **micro-level flood risk zone maps** for any region worldwide — using real NASA SRTM elevation, live OpenStreetMap water bodies, and a transparent **Weighted Susceptibility Index** model with inspectable factor weights.

---

## 🚀 Live Demo

**[flood-risk-zonation.streamlit.app](https://flood-risk-zonation-ezw8ngm5igpy6egpdmgsqw.streamlit.app/)**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🗺️ Interactive Risk Map | Color-coded flood risk zones (High / Medium / Low / Water) via Folium |
| 🌍 Global Coverage | Works for any bounding box worldwide using live OpenStreetMap data |
| ⛰️ Real Elevation Data | NASA SRTM 30m DEM via local GeoTIFF; synthetic fallback |
| 💧 Live Water Body Detection | Fetches lakes, rivers, canals, ocean from OSM Overpass API (with retry + local cache) |
| 🌧️ Rainfall Layer | IMD/GPM-calibrated rainfall intensity heatmap |
| 🔍 Per-cell Explainability | Click any cell for a data-driven factor breakdown with risk contribution bars |
| ⚠️ Coastal Tsunami Flag | Cells within 1.5× cell-size of ocean/sea geometry are flagged in the popup |
| 🤖 Transparent Model | Weighted Susceptibility Index — declared weights and risk directions, no black-box ML |
| 📄 PDF Report Generator | Full flood risk report with emergency deployment plan |
| ⬇️ Export | Download results as CSV, GeoJSON, or interactive HTML |
| 📦 Offline Demo Mode | Pre-configured sample data for three regions — no network required |

---

## 🗂️ Project Structure

```
flood-risk-zonation/
├── flood_risk_zonation/        # Core Python package
│   ├── config.py               # BoundingBox, PipelineConfig dataclasses
│   ├── pipeline.py             # End-to-end orchestration
│   ├── exceptions.py           # Custom error types
│   ├── models.py               # RasterDataset, FloodRiskResult, AnalysisResult, etc.
│   ├── features/
│   │   └── extractor.py        # 10-feature geospatial extractor
│   ├── grid/
│   │   └── generator.py        # BoundingBox → cell GeoDataFrame
│   ├── ingest/
│   │   ├── elevation.py        # SRTM loader + synthetic fallback
│   │   ├── rainfall.py         # GPM/IMD loader + synthetic fallback
│   │   ├── water_bodies.py     # OSM Overpass fetcher with retry + cache
│   │   ├── drainage.py         # Synthetic drainage capacity generator
│   │   ├── population.py       # Population density loader
│   │   └── sample_data.py      # Bundled offline demo data (3 regions)
│   ├── scoring/
│   │   ├── scorer.py           # Grid scorer (maps model output → risk class)
│   │   └── susceptibility.py   # WeightedSusceptibilityModel
│   ├── utils/
│   │   ├── cache.py            # Grid/data cache helpers
│   │   └── validation.py       # Input validation + NaN imputation
│   └── visualization/
│       ├── map_builder.py      # Folium choropleth map orchestrator
│       ├── layers.py           # Individual Folium layer builders
│       ├── explainability.py   # Per-cell tooltip/popup HTML generator
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
└── pyproject.toml              # Package metadata & build config
```

---

## ⚙️ How It Works

```
Bounding Box Input
       │
       ▼
1. Grid Generation        — divides bbox into configurable cells (250m / 500m / 1km)
       │
       ▼
2. Data Ingestion         — fetches real SRTM elevation + OSM water bodies
       │                    (retried up to 3×; falls back to synthetic if unavailable)
       ▼
3. Feature Extraction     — computes 10 features per cell
       │
       ▼
4. Susceptibility Model   — transparent weighted index (declared weights, no ML training)
       │
       ▼
5. Risk Scoring           — normalises index → [0, 100] → Low / Medium / High
       │
       ▼
6. Post-processing        — water masking (elevation + OSM polygon centroid test)
       │                    proximity boost for cells near water
       │                    coastal tsunami flag for cells near ocean/sea
       ▼
7. Visualisation          — interactive Folium map + per-cell explainability popups
                            + optional PDF / CSV / GeoJSON export
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
| Distance to Water | Nearest OSM water body (m) |
| Drainage Capacity | Synthetic score [0, 1] |
| Population Density | Persons/km² |
| Aspect | Terrain aspect (degrees from north) — computed but not weighted |
| Curvature | Plan curvature |

**Model:** `WeightedSusceptibilityModel` — each factor is robustly normalised using its 5th–95th percentile range, then combined with a declared weight and risk direction (see table below). No ML training; output is a relative index whose weights are fully inspectable.

| Factor | Weight | Direction |
|---|---|---|
| Elevation | 0.15 | Low elevation → higher risk |
| TWI | 0.15 | High TWI → higher risk |
| Max 24-hr Rainfall | 0.15 | Higher → higher risk |
| Distance to Water | 0.15 | Closer → higher risk |
| Drainage Capacity | 0.15 | Poor drainage → higher risk |
| Annual Rainfall | 0.10 | Higher → higher risk |
| Slope | 0.05 | Flat terrain → higher risk |
| Population Density | 0.05 | Higher → higher risk |
| Curvature | 0.05 | Concave → higher risk |
| Aspect | — | Not weighted (no global relationship) |

> ⚠️ **Data transparency:** When the NASA GPM API is unavailable, rainfall falls back to a region-calibrated synthetic layer (e.g. 970 mm/yr for Bangalore). The active data tier is shown in the UI after each run. See [Data Tiers](#-data-tiers) below.

---

## 🎯 Risk Zones

| Zone | Colour | Score | Description |
|---|---|---|---|
| 🔴 High Risk | Red | > 66 | Low elevation, poor drainage, near water |
| 🟠 Medium Risk | Amber | 34–66 | Moderate vulnerability |
| 🟢 Low Risk | Green | ≤ 33 | Higher elevation, good drainage |
| 🔵 Water | Blue | N/A | Permanent water body (lake / river / ocean) |

Classification thresholds are adjustable via the sidebar sliders.

---

## 🗺️ Water Masking & Post-Processing

The pipeline applies a four-step post-scoring pass:

| Step | Description |
|---|---|
| Elevation mask | Cells with SRTM elevation ≤ 1 m → Water (skipped for synthetic elevation) |
| OSM polygon mask | Cells whose centroid lies inside an OSM area water body (lake, reservoir, bay, ocean) → Water |
| Proximity boost | Land cells within 0.6 × cell_size of any water geometry → boosted to at least Medium |
| Coastal flag | Land cells within 1.5 × cell_size of ocean/sea geometry → `is_coastal_tsunami_risk = True` flag shown in popup |

Linear water features (drains, streams, rivers, canals) contribute to the proximity boost but do **not** mask cells as Water, preventing false positives from thin sliver polygons.

---

## 📡 Data Sources

| Dataset | Source | Resolution | Notes |
|---|---|---|---|
| Elevation (DEM) | NASA SRTM via local GeoTIFF | ~30m | Tier 1; synthetic fallback = Tier 2/3 |
| Rainfall | NASA GPM IMERG / IMD | 0.1° | Tier 1; synthetic fallback = Tier 2/3 |
| Water Bodies | OpenStreetMap Overpass API | Vector | Tier 1; cached locally by bbox |
| Drainage Lines | OpenStreetMap (local GeoJSON) | Vector | Visualisation only |
| Population Density | Local raster or synthetic fallback | ~1km | Tier 2/3 |

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

These three regions are also available as **offline demo presets** via the sidebar — no network required.

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

Bounding boxes must be between **2 km and 50 km** per side. A live size preview is shown in the sidebar as you adjust the coordinates.

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
| Model | `WeightedSusceptibilityModel` (custom deterministic index) |
| Geospatial | GeoPandas, Rasterio, Shapely, PyProj |
| Data | NASA SRTM, OpenStreetMap Overpass API, NASA GPM IMERG |
| PDF | ReportLab |
| Testing | pytest, Hypothesis, pytest-cov |

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.

---

## 👤 Author

**Kunal MK** · [github.com/KunalMK25](https://github.com/KunalMK25)
