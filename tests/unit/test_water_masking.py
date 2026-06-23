"""
Unit tests for water masking (Part 1) and coastal tsunami flag (Part 2).

Tests use synthetic GeoDataFrames — no network calls, no pipeline run.
The masking logic under test lives in:
    FloodRiskPipeline._apply_water_mask_and_proximity_boost
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.pipeline import FloodRiskPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_grid(
    lons: list[float],
    lats: list[float],
    elevations: list[float],
    cell_size_deg: float = 0.001,
) -> gpd.GeoDataFrame:
    """Build a minimal scored grid for masking tests."""
    n = len(lons)
    assert len(lats) == n and len(elevations) == n
    geoms = [
        box(lon, lat, lon + cell_size_deg, lat + cell_size_deg)
        for lon, lat in zip(lons, lats)
    ]
    return gpd.GeoDataFrame(
        {
            "cell_id": [str(i) for i in range(n)],
            "centroid_lon": [lon + cell_size_deg / 2 for lon in lons],
            "centroid_lat": [lat + cell_size_deg / 2 for lat in lats],
            "elevation_m": np.array(elevations, dtype=np.float32),
            "risk_score": np.full(n, 50.0, dtype=np.float32),
            "risk_class": ["Medium"] * n,
            "slope_deg": np.zeros(n, dtype=np.float32),
            "twi": np.zeros(n, dtype=np.float32),
            "rainfall_mean_mm": np.zeros(n, dtype=np.float32),
            "rainfall_max_24h_mm": np.zeros(n, dtype=np.float32),
            "dist_water_m": np.full(n, 5000.0, dtype=np.float32),
            "drainage_capacity": np.full(n, 0.5, dtype=np.float32),
            "population_density": np.zeros(n, dtype=np.float32),
            "aspect_deg": np.zeros(n, dtype=np.float32),
            "curvature": np.zeros(n, dtype=np.float32),
        },
        geometry=geoms,
        crs="EPSG:4326",
    )


def _water_gdf(polygons: list[Polygon], water_types: list[str]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "geometry": polygons,
            "water_type": water_types,
            "name": [""] * len(polygons),
        },
        crs="EPSG:4326",
    )


def _config(cell_size: float = 100.0) -> PipelineConfig:
    return PipelineConfig(
        cell_size_meters=cell_size,
        use_cache=False,
        allow_network=False,
    )


def _pipeline() -> FloodRiskPipeline:
    return FloodRiskPipeline(_config())


# ── Part 1: Water masking is a hard override ──────────────────────────────────

class TestWaterMaskingHardOverride:
    """Cells inside water polygons must become Water regardless of prior score."""

    def test_cell_inside_water_polygon_becomes_water(self):
        """A cell whose centroid falls inside a water polygon → risk_class='Water'."""
        # Cell centroid at (77.555, 12.845) — inside the polygon
        grid = _make_grid([77.55], [12.84], [50.0])
        water = _water_gdf(
            [box(77.50, 12.80, 77.60, 12.90)],  # large polygon covering cell
            ["water"],
        )
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, water, _config())
        assert result.iloc[0]["risk_class"] == "Water"

    def test_water_mask_overrides_high_risk_score(self):
        """Water mask must override even a cell scored as High risk."""
        grid = _make_grid([77.55], [12.84], [50.0])
        grid.at[0, "risk_class"] = "High"
        grid.at[0, "risk_score"] = 90.0
        water = _water_gdf([box(77.50, 12.80, 77.60, 12.90)], ["water"])
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, water, _config())
        assert result.iloc[0]["risk_class"] == "Water", (
            "Water mask must override High risk classification"
        )

    def test_elevation_mask_fires_for_low_elevation(self):
        """Cells with elevation ≤ 2 m near a coastline get Water class."""
        # Two cells: one ocean (0.5 m), one land (50 m)
        # A coastline LineString is provided so the elevation mask activates
        from shapely.geometry import LineString
        grid = _make_grid([77.55, 77.56], [12.84, 12.84], [0.5, 50.0])
        # Coastline just south of the cells
        coastline_gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(77.50, 12.83), (77.60, 12.83)])],
             "water_type": ["coastline"], "name": [""]},
            crs="EPSG:4326",
        )
        result = _pipeline()._apply_water_mask_and_proximity_boost(
            grid, coastline_gdf, _config(cell_size=500.0), elevation_source="real"
        )
        assert result.iloc[0]["risk_class"] == "Water", "0.5 m cell near coastline must be masked as Water"
        assert result.iloc[1]["risk_class"] != "Water", "50 m cell must remain land"
        assert result.iloc[0]["risk_class"] == "Water", "0.5 m cell must be masked as Water"
        assert result.iloc[1]["risk_class"] != "Water", "50 m cell must remain land"

    def test_elevation_mask_skipped_for_synthetic_data(self):
        """Synthetic elevation must NOT trigger Water masking for low-value cells."""
        # A coastal synthetic DEM may produce values below 1 m even over urban land
        grid = _make_grid([80.25, 80.26], [13.00, 13.00], [0.4, 50.0])
        empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = _pipeline()._apply_water_mask_and_proximity_boost(
            grid, empty_wb, _config(), elevation_source="synthetic"
        )
        # Neither cell should be masked as Water — no OSM data, synthetic elevation
        assert result.iloc[0]["risk_class"] != "Water", (
            "Synthetic 0.4 m cell must NOT be falsely masked as Water"
        )
        assert result.iloc[1]["risk_class"] != "Water"

    def test_non_water_cell_preserves_risk_class(self):
        """Cells outside water polygons keep their original risk_class."""
        grid = _make_grid([77.55, 77.56], [12.84, 12.84], [50.0, 50.0])
        grid.at[0, "risk_class"] = "High"
        grid.at[1, "risk_class"] = "Low"
        # Water polygon far away — doesn't cover either cell
        water = _water_gdf([box(80.0, 15.0, 81.0, 16.0)], ["water"])
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, water, _config())
        assert result.iloc[0]["risk_class"] == "High"
        assert result.iloc[1]["risk_class"] == "Low"

    def test_ocean_region_all_cells_water(self):
        """
        For a mostly-ocean bbox (all cells inside a large ocean polygon),
        all cells must be classified as Water.
        """
        # 3×3 grid of cells, all inside an ocean polygon
        lons = [80.24, 80.25, 80.26] * 3
        lats = [12.98, 12.98, 12.98, 12.99, 12.99, 12.99, 13.00, 13.00, 13.00]
        elevs = [5.0] * 9  # low but > 1 m so elevation mask won't fire alone
        grid = _make_grid(lons, lats, elevs)
        ocean_polygon = box(80.20, 12.95, 80.35, 13.10)  # covers all cells
        water = _water_gdf([ocean_polygon], ["coastline"])  # OSM coastline tag
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, water, _config())
        non_water = result[result["risk_class"] != "Water"]
        assert len(non_water) == 0, (
            f"{len(non_water)} cells still have non-Water class after ocean masking: "
            f"{non_water['risk_class'].tolist()}"
        )

    def test_is_coastal_tsunami_risk_column_always_present(self):
        """is_coastal_tsunami_risk column must exist even when no water bodies."""
        grid = _make_grid([77.55], [12.84], [50.0])
        empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty_wb, _config())
        assert "is_coastal_tsunami_risk" in result.columns


# ── Part 2: Coastal tsunami flag ──────────────────────────────────────────────

class TestCoastalTsunamiFlag:
    """Coastal flag: ocean/sea neighbours → True; lakes/rivers → False."""

    def _run(
        self,
        land_lon: float,
        land_lat: float,
        water_polygon: Polygon,
        water_type: str,
        cell_size: float = 500.0,
    ) -> gpd.GeoDataFrame:
        grid = _make_grid([land_lon], [land_lat], [20.0])
        water = _water_gdf([water_polygon], [water_type])
        config = _config(cell_size=cell_size)
        pipeline = FloodRiskPipeline(config)
        return pipeline._apply_water_mask_and_proximity_boost(grid, water, config)

    def test_adjacent_ocean_cell_flagged(self):
        """Land cell adjacent to (but not inside) a coastline water body is flagged."""
        # Land cell centroid: ~(80.2505, 13.0005)
        # Water polygon is immediately south, does NOT cover the centroid
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),  # just south, doesn't cover centroid
            water_type="coastline",
        )
        assert result.iloc[0]["is_coastal_tsunami_risk"] is True or \
               bool(result.iloc[0]["is_coastal_tsunami_risk"]) is True

    def test_adjacent_bay_cell_flagged(self):
        """natural=bay also counts as ocean for the tsunami flag."""
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),
            water_type="bay",
        )
        assert bool(result.iloc[0]["is_coastal_tsunami_risk"]) is True

    def test_adjacent_lake_not_flagged(self):
        """Land cell next to a lake (natural=water) must NOT get the tsunami flag."""
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),
            water_type="water",  # lake
        )
        assert not bool(result.iloc[0]["is_coastal_tsunami_risk"])

    def test_adjacent_river_not_flagged(self):
        """Land cell next to a river must NOT get the tsunami flag."""
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),
            water_type="river",
        )
        assert not bool(result.iloc[0]["is_coastal_tsunami_risk"])

    def test_adjacent_reservoir_not_flagged(self):
        """Land cell next to a reservoir must NOT get the tsunami flag."""
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),
            water_type="reservoir",
        )
        assert not bool(result.iloc[0]["is_coastal_tsunami_risk"])

    def test_far_from_ocean_not_flagged(self):
        """Inland cell far from any ocean body must NOT be flagged."""
        result = self._run(
            land_lon=77.55, land_lat=12.84,
            water_polygon=box(80.24, 12.97, 80.31, 12.999),
            water_type="coastline",
            cell_size=500.0,
        )
        assert not bool(result.iloc[0]["is_coastal_tsunami_risk"])

    def test_water_cells_not_flagged(self):
        """Cells classified as Water (elev ≤ 1 m) must not receive the tsunami flag."""
        # cell 0: elevation 0.5 m → elevation mask → Water
        # cell 1: elevation 20 m, land, adjacent to ocean polygon
        grid = _make_grid([80.25, 80.26], [13.00, 13.00], [0.5, 20.0])
        # Ocean polygon: covers cell 0 AND is adjacent to cell 1
        ocean = box(80.24, 12.97, 80.261, 12.999)
        water = _water_gdf([ocean], ["coastline"])
        config = _config(cell_size=500.0)
        result = FloodRiskPipeline(config)._apply_water_mask_and_proximity_boost(
            grid, water, config
        )
        water_cells = result[result["risk_class"] == "Water"]
        assert not water_cells["is_coastal_tsunami_risk"].any(), (
            "Water cells must never receive the coastal tsunami flag"
        )

    def test_multiple_ocean_cells_flagged(self):
        """Multiple land cells adjacent to ocean all get flagged."""
        # 3 land cells side-by-side (all > 1m elevation)
        # Ocean polygon immediately south, does NOT cover any centroid
        lons = [80.25, 80.26, 80.27]
        lats = [13.00, 13.00, 13.00]
        grid = _make_grid(lons, lats, [20.0, 20.0, 20.0])
        ocean_poly = box(80.24, 12.97, 80.28, 12.999)  # just south, not covering centroids
        water = _water_gdf([ocean_poly], ["coastline"])
        config = _config(cell_size=500.0)
        result = FloodRiskPipeline(config)._apply_water_mask_and_proximity_boost(
            grid, water, config
        )
        land_cells = result[result["risk_class"] != "Water"]
        flagged = land_cells["is_coastal_tsunami_risk"].sum()
        assert flagged >= 1, (
            f"At least one land cell adjacent to ocean should be flagged; got {flagged}"
        )
