"""
Folium map construction for the Flood Risk Zonation System.
"""
from __future__ import annotations

import logging

import folium
import geopandas as gpd
import numpy as np

from flood_risk_zonation.visualization.layers import (
    RISK_COLOR_MAP,
    add_drainage_lines_layer,
    add_population_density_layer,
    add_rainfall_heatmap_layer,
    add_risk_choropleth_layer,
    add_water_bodies_layer,
)

logger = logging.getLogger(__name__)


class FloodRiskMapBuilder:
    """Builds interactive Folium maps from scored grid GeoDataFrames."""

    RISK_COLOR_MAP = RISK_COLOR_MAP

    def build_choropleth_map(
        self,
        scored_grid: gpd.GeoDataFrame,
        center: tuple[float, float],
        zoom_start: int = 12,
        use_google_maps: bool = False,
        google_maps_api_key: str | None = None,
        water_bodies: gpd.GeoDataFrame | None = None,
    ) -> folium.Map:
        """
        Construct a Folium map with risk choropleth, drainage lines,
        rainfall heatmap, population, and water body layers.
        """
        m = folium.Map(location=list(center), zoom_start=zoom_start, tiles="OpenStreetMap")

        # Layer 1: Risk classification choropleth (always on)
        add_risk_choropleth_layer(m, scored_grid)

        # Layer 2: Drainage lines (always on — key for justifying risk)
        add_drainage_lines_layer(m)

        # Layer 3: Rainfall heatmap (toggle)
        add_rainfall_heatmap_layer(m, scored_grid)

        # Layer 4: Population density (toggle)
        add_population_density_layer(m, scored_grid)

        # Layer 5: Water bodies (toggle)
        if water_bodies is not None:
            add_water_bodies_layer(m, water_bodies)

        # Layer control
        folium.LayerControl(collapsed=False).add_to(m)

        # Rich popups with risk factor breakdown
        self.add_popup_layer(m, scored_grid)

        return m

    def add_popup_layer(self, folium_map: folium.Map, grid: gpd.GeoDataFrame) -> folium.Map:
        """
        Add click-to-inspect popups. Only adds popups for non-Water cells
        to keep map performance fast for large coastal grids.
        """
        # Precompute max values for bar scaling
        def _max(col):
            return float(grid[col].max()) if col in grid.columns and grid[col].max() > 0 else 1.0

        elev_max = _max("elevation_m")
        rain_max = _max("rainfall_mean_mm")
        twi_max = _max("twi")
        dist_max = _max("dist_water_m")

        fg = folium.FeatureGroup(name="Cell Info (click)", show=False)

        # Limit to max 300 cells for popup performance
        display_grid = grid[grid.risk_class != "Water"].head(300)

        for _, row in display_grid.iterrows():
            risk_class = str(row.get("risk_class", "Low"))
            risk_score = row.get("risk_score", 0)
            risk_color = RISK_COLOR_MAP.get(risk_class, "#999")

            def _bar(value, max_val, reverse=False):
                pct = min(100, int(value / max_val * 100)) if max_val > 0 else 0
                if reverse:
                    pct = 100 - pct
                color = "#e74c3c" if pct > 66 else "#f39c12" if pct > 33 else "#2ecc71"
                return (
                    f'<div style="background:#ddd;border-radius:3px;height:8px;'
                    f'width:100px;display:inline-block">'
                    f'<div style="background:{color};width:{pct}%;height:8px;'
                    f'border-radius:3px"></div></div> <small>{pct}%</small>'
                )

            elev = float(row.get("elevation_m", 0))
            rain = float(row.get("rainfall_mean_mm", 0))
            drain = float(row.get("drainage_capacity", 0.5))
            dist = float(row.get("dist_water_m", 5000))
            twi = float(row.get("twi", 0))
            slope = float(row.get("slope_deg", 0))

            html = f"""<div style="font-family:Arial,sans-serif;font-size:12px;min-width:220px">
  <div style="background:{risk_color};color:white;padding:4px 8px;border-radius:4px;margin-bottom:6px">
    <b>{risk_class} Risk</b> &nbsp; Score: {risk_score:.1f}/100
  </div>
  <table style="width:100%;border-collapse:collapse">
    <tr><td style="padding:2px 4px"><b>🌧 Rainfall</b></td><td>{_bar(rain, rain_max)}</td><td style="padding-left:4px">{rain:.0f}mm</td></tr>
    <tr><td style="padding:2px 4px"><b>⛰ Elevation</b></td><td>{_bar(elev, elev_max, reverse=True)}</td><td style="padding-left:4px">{elev:.0f}m</td></tr>
    <tr><td style="padding:2px 4px"><b>💧 TWI</b></td><td>{_bar(twi, twi_max)}</td><td style="padding-left:4px">{twi:.2f}</td></tr>
    <tr><td style="padding:2px 4px"><b>🚰 Drainage</b></td><td>{_bar(1-drain, 1.0)}</td><td style="padding-left:4px">{drain:.2f}</td></tr>
    <tr><td style="padding:2px 4px"><b>🏞 Dist Water</b></td><td>{_bar(dist, dist_max, reverse=True)}</td><td style="padding-left:4px">{dist:.0f}m</td></tr>
    <tr><td style="padding:2px 4px"><b>📐 Slope</b></td><td>{_bar(slope, 45.0, reverse=True)}</td><td style="padding-left:4px">{slope:.1f}°</td></tr>
  </table>
</div>"""

            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _: {"fillOpacity": 0, "weight": 0},
                popup=folium.Popup(html, max_width=300),
            ).add_to(fg)

        fg.add_to(folium_map)
        return folium_map
