"""
Individual layer builder functions for the Flood Risk Zonation map.
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
    "Water": "#3498db",   # blue — permanent water body
}


def add_risk_choropleth_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Add color-coded risk classification polygons as a FeatureGroup layer."""
    fg = folium.FeatureGroup(name="Risk Classification", show=True)
    for _, row in scored_grid.iterrows():
        risk_class = str(row.get("risk_class", "Low"))
        color = RISK_COLOR_MAP.get(risk_class, "#2ecc71")
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 0.5,
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

    # Build heatmap points: [lat, lon, intensity]
    heat_data = []
    for _, row in scored_grid.iterrows():
        intensity = float(row.get("rainfall_mean_mm", 0)) / rain_max
        heat_data.append([row["centroid_lat"], row["centroid_lon"], intensity])

    fg = folium.FeatureGroup(name="Rainfall Intensity", show=False)
    folium.plugins.HeatMap(
        heat_data,
        name="Rainfall Intensity",
        min_opacity=0.3,
        max_zoom=18,
        radius=25,
        blur=20,
        gradient={0.2: "#ffffb2", 0.5: "#fd8d3c", 0.8: "#e31a1c", 1.0: "#800026"},
    ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_drainage_lines_layer(folium_map: folium.Map, drainage_path: str | None = None) -> folium.Map:
    """Add OSM drainage lines (drains, canals, streams) as a layer."""
    if drainage_path is None:
        drainage_path = "data/drainage_lines/gottigere_drains.geojson"

    path = Path(drainage_path)
    if not path.exists():
        return folium_map

    try:
        with open(path) as f:
            geojson_data = json.load(f)

        if not geojson_data.get("features"):
            return folium_map

        # Color by waterway type
        type_colors = {
            "drain": "#1a6faf",
            "canal": "#2980b9",
            "stream": "#5dade2",
            "river": "#1b4f72",
            "ditch": "#7fb3d3",
        }

        fg = folium.FeatureGroup(name="Drainage Lines", show=True)
        for feature in geojson_data["features"]:
            wtype = feature.get("properties", {}).get("waterway", "drain")
            color = type_colors.get(wtype, "#2980b9")
            weight = 3 if wtype in ("river", "canal") else 2
            folium.GeoJson(
                feature,
                style_function=lambda _, c=color, w=weight: {
                    "color": c,
                    "weight": w,
                    "opacity": 0.85,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["waterway", "name"] if "name" in (feature.get("properties") or {}) else ["waterway"],
                    aliases=["Type", "Name"] if "name" in (feature.get("properties") or {}) else ["Type"],
                    sticky=False,
                ),
            ).add_to(fg)
        fg.add_to(folium_map)

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to load drainage lines: %s", e)

    return folium_map


def add_population_density_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Add population density overlay as a FeatureGroup layer."""
    if "population_density" not in scored_grid.columns:
        return folium_map
    fg = folium.FeatureGroup(name="Population Density", show=False)
    pop_max = float(scored_grid["population_density"].max()) or 1.0
    for _, row in scored_grid.iterrows():
        pop = float(row.get("population_density", 0))
        intensity = pop / pop_max
        r = int(255 * intensity)
        color = f"#{r:02x}4444"
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": "none",
                "fillOpacity": 0.4,
            },
        ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_water_bodies_layer(folium_map: folium.Map, water_bodies: gpd.GeoDataFrame) -> folium.Map:
    """Add water bodies overlay as a FeatureGroup layer."""
    if water_bodies is None or len(water_bodies) == 0:
        return folium_map
    fg = folium.FeatureGroup(name="Water Bodies", show=False)
    for _, row in water_bodies.iterrows():
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#3498db",
                "color": "#2980b9",
                "weight": 1,
                "fillOpacity": 0.5,
            },
        ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_risk_choropleth_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Add color-coded risk classification polygons as a FeatureGroup layer."""
    fg = folium.FeatureGroup(name="Risk Classification", show=True)
    for _, row in scored_grid.iterrows():
        risk_class = str(row.get("risk_class", "Low"))
        color = RISK_COLOR_MAP.get(risk_class, "#2ecc71")
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 0.5,
                "fillOpacity": 0.6,
            },
        ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_population_density_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Add population density overlay as a FeatureGroup layer."""
    if "population_density" not in scored_grid.columns:
        return folium_map
    fg = folium.FeatureGroup(name="Population Density", show=False)
    pop_max = float(scored_grid["population_density"].max()) or 1.0
    for _, row in scored_grid.iterrows():
        pop = float(row.get("population_density", 0))
        intensity = pop / pop_max
        r = int(255 * intensity)
        color = f"#{r:02x}4444"
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": "none",
                "fillOpacity": 0.4,
            },
        ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_water_bodies_layer(folium_map: folium.Map, water_bodies: gpd.GeoDataFrame) -> folium.Map:
    """Add water bodies overlay as a FeatureGroup layer."""
    if water_bodies is None or len(water_bodies) == 0:
        return folium_map
    fg = folium.FeatureGroup(name="Water Bodies", show=False)
    for _, row in water_bodies.iterrows():
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#3498db",
                "color": "#2980b9",
                "weight": 1,
                "fillOpacity": 0.5,
            },
        ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map
