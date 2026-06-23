"""
Individual layer builder functions for the Flood Risk Zonation map.
Uses bulk GeoJson rendering for performance with large grids.
"""
from __future__ import annotations
import json
from pathlib import Path
import folium
import folium.plugins
import geopandas as gpd
import numpy as np

RISK_COLOR_MAP = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
    "Water": "#3498db",
}


def add_risk_choropleth_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Render risk zones using bulk GeoJson per class — fast for any grid size.

    Rendered as non-interactive (no tooltip/popup) so that mouse events
    pass through to the per-cell explainability layer added on top.
    """
    fg = folium.FeatureGroup(name="Risk Classification", show=True)
    for risk_class, color in RISK_COLOR_MAP.items():
        subset = scored_grid[scored_grid["risk_class"] == risk_class]
        if len(subset) == 0:
            continue
        folium.GeoJson(
            subset.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 0.3,
                "fillOpacity": 0.6,
            },
        ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_rainfall_heatmap_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Add a rainfall intensity heatmap layer using cell centroids."""
    if "rainfall_mean_mm" not in scored_grid.columns:
        return folium_map
    rain_vals = scored_grid["rainfall_mean_mm"].values.astype(float)
    rain_max = float(rain_vals.max()) if rain_vals.max() > 0 else 1.0
    heat_data = [
        [row["centroid_lat"], row["centroid_lon"], float(row.get("rainfall_mean_mm", 0)) / rain_max]
        for _, row in scored_grid.iterrows()
    ]
    fg = folium.FeatureGroup(name="Rainfall Intensity", show=False)
    folium.plugins.HeatMap(
        heat_data, min_opacity=0.3, max_zoom=18, radius=25, blur=20,
        gradient={0.2: "#ffffb2", 0.5: "#fd8d3c", 0.8: "#e31a1c", 1.0: "#800026"},
    ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_drainage_lines_layer(folium_map: folium.Map, drainage_path: str | None = None) -> folium.Map:
    """Add OSM drainage lines as a layer. Searches data/drainage_lines/ for any geojson."""
    import logging
    logger = logging.getLogger(__name__)

    # Find any drainage geojson file
    drain_dir = Path("data/drainage_lines")
    if drainage_path is None:
        files = list(drain_dir.glob("*.geojson")) if drain_dir.exists() else []
        if not files:
            return folium_map
        drainage_path = str(files[0])

    path = Path(drainage_path)
    if not path.exists():
        return folium_map

    try:
        with open(path) as f:
            geojson_data = json.load(f)
        if not geojson_data.get("features"):
            return folium_map
        type_colors = {
            "drain": "#1a6faf", "canal": "#2980b9", "stream": "#5dade2",
            "river": "#1b4f72", "ditch": "#7fb3d3",
        }
        fg = folium.FeatureGroup(name="Drainage Lines", show=True)
        for feature in geojson_data["features"]:
            wtype = feature.get("properties", {}).get("waterway", "drain")
            color = type_colors.get(wtype, "#2980b9")
            weight = 3 if wtype in ("river", "canal") else 2
            folium.GeoJson(
                feature,
                style_function=lambda _, c=color, w=weight: {"color": c, "weight": w, "opacity": 0.85},
            ).add_to(fg)
        fg.add_to(folium_map)
    except Exception as e:
        logger.warning("Failed to load drainage lines: %s", e)
    return folium_map


def add_population_density_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Add population density overlay."""
    if "population_density" not in scored_grid.columns:
        return folium_map
    fg = folium.FeatureGroup(name="Population Density", show=False)
    pop_max = float(scored_grid["population_density"].max()) or 1.0
    for _, row in scored_grid.iterrows():
        intensity = float(row.get("population_density", 0)) / pop_max
        r = int(255 * intensity)
        color = f"#{r:02x}4444"
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {"fillColor": c, "color": "none", "fillOpacity": 0.4},
        ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_water_bodies_layer(folium_map: folium.Map, water_bodies: gpd.GeoDataFrame) -> folium.Map:
    """Add water bodies overlay."""
    if water_bodies is None or len(water_bodies) == 0:
        return folium_map
    fg = folium.FeatureGroup(name="Water Bodies", show=False)
    folium.GeoJson(
        water_bodies.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "#3498db", "color": "#2980b9", "weight": 1, "fillOpacity": 0.5,
        },
    ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map
