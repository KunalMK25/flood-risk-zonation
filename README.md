# Flood Risk Zonation System

An ML-powered geospatial web app that generates micro-level flood risk zone maps for any region worldwide.

## Live Demo
Deployed on Streamlit Cloud: [https://share.streamlit.io](https://flood-risk-zonation-ezw8ngm5igpy6egpdmgsqw.streamlit.app/)

## Features
- Interactive Risk Map with color-coded flood risk zones (High/Medium/Low/Water)
- Global Coverage - works for any bounding box worldwide using live OpenStreetMap data
- Real Elevation Data - NASA SRTM 30m DEM via OpenTopoData API
- Live Water Body Detection - fetches lakes, rivers, canals, ocean from OSM Overpass API
- Drainage Lines Overlay - real OSM drainage channels shown on map
- Rainfall Heatmap - IMD-calibrated rainfall intensity layer
- ML Model - Random Forest classifier with 5-fold cross-validation (AUC ~0.97)
- PDF Report Generator - full flood risk report with emergency deployment plan
- Click Popups - per-cell risk factor breakdown (elevation, TWI, drainage, rainfall)

## Risk Zones

| Zone | Color | Score | Description |
|------|-------|-------|-------------|
| High Risk | Red | > 66 | Low elevation, poor drainage, near water |
| Medium Risk | Amber | 34-66 | Moderate vulnerability factors |
| Low Risk | Green | <= 33 | Higher elevation, good drainage |
| Water | Blue | N/A | Permanent water body (lake/river/ocean) |

## Tech Stack

- Frontend: Streamlit + Folium
- ML: scikit-learn Random Forest
- Geospatial: GeoPandas, Rasterio, Shapely
- Data: NASA SRTM, OpenStreetMap, IMD/GPM rainfall
- PDF: ReportLab
- Testing: Hypothesis (property-based), pytest

## Run Locally

`ash
git clone https://github.com/KunalMK25/flood-risk-zonation.git
cd flood-risk-zonation
pip install -r requirements.txt
streamlit run app.py
`

Open http://localhost:8501

## Example Regions

| Area | Min Lon | Min Lat | Max Lon | Max Lat |
|------|---------|---------|---------|---------|
| Gottigere, Bangalore | 77.55 | 12.84 | 77.62 | 12.91 |
| Chennai Marina (Coastal) | 80.24 | 12.98 | 80.31 | 13.05 |
| Dal Lake, Srinagar | 74.83 | 34.07 | 74.90 | 34.14 |
| Puri, Odisha (Cyclone coast) | 85.80 | 19.77 | 85.87 | 19.84 |

## Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [https://share.streamlit.io](https://flood-risk-zonation-ezw8ngm5igpy6egpdmgsqw.streamlit.app/)
3. Click New app, select this repo, set main file to app.py
4. Click Deploy

## How It Works

1. Grid Generation - divides the bounding box into 500m x 500m cells
2. Data Ingestion - fetches real SRTM elevation + OSM water bodies for the bbox
3. Feature Extraction - computes 10 features per cell (elevation, slope, TWI, rainfall, etc.)
4. ML Training - trains Random Forest on feature-derived flood susceptibility labels
5. Risk Scoring - normalizes predictions to [0, 100] and classifies into Low/Medium/High
6. Water Masking - marks ocean/lake/river cells as Water (blue) using elevation + OSM
7. Visualization - renders interactive Folium map with multiple overlay layers

## Data Sources

| Dataset | Source | Resolution |
|---------|--------|------------|
| Elevation (DEM) | NASA SRTM via OpenTopoData | 30m |
| Water Bodies | OpenStreetMap Overpass API | Vector |
| Drainage Lines | OpenStreetMap Overpass API | Vector |
| Rainfall | IMD-calibrated synthetic (970mm/yr for Bangalore) | 1km |
| Population | Synthetic (WorldPop-style) | 1km |

## License
MIT
