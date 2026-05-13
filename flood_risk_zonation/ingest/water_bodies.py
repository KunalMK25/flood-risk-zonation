"""
OSM water body ingestion for the Flood Risk Zonation System.

Fetches ALL water body types live from Overpass API for any bbox worldwide.
Caches results locally. Falls back to cached files if API unavailable.
"""
from __future__ import annotations
import logging, time
from pathlib import Path
import geopandas as gpd
import requests
from shapely.geometry import box, shape
from flood_risk_zonation.config import BoundingBox

logger = logging.getLogger(__name__)

_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

_QUERY = (
    "[out:json][timeout:60];\n"
    "(\n"
    "  way[\"natural\"=\"water\"]({s},{w},{n},{e});\n"
    "  relation[\"natural\"=\"water\"]({s},{w},{n},{e});\n"
    "  way[\"waterway\"=\"river\"]({s},{w},{n},{e});\n"
    "  way[\"waterway\"=\"canal\"]({s},{w},{n},{e});\n"
    "  way[\"waterway\"=\"stream\"]({s},{w},{n},{e});\n"
    "  way[\"waterway\"=\"drain\"]({s},{w},{n},{e});\n"
    "  way[\"landuse\"=\"reservoir\"]({s},{w},{n},{e});\n"
    "  way[\"landuse\"=\"basin\"]({s},{w},{n},{e});\n"
    "  way[\"natural\"=\"bay\"]({s},{w},{n},{e});\n"
    "  way[\"natural\"=\"coastline\"]({s},{w},{n},{e});\n"
    ");\n"
    "out geom;"
)


def _fetch(query: str) -> dict | None:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "FloodRiskZonation/1.0",
    }
    for mirror in _MIRRORS:
        for _ in range(2):
            try:
                r = requests.post(mirror, data=query.encode("utf-8"), headers=headers, timeout=20)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:
                    time.sleep(5)
                else:
                    time.sleep(2)
            except Exception as exc:
                logger.debug("Mirror %s failed: %s", mirror, exc)
                time.sleep(1)
    return None


def _osm_to_gdf(osm_data: dict, bbox: BoundingBox) -> gpd.GeoDataFrame:
    bbox_poly = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    features = []
    for el in osm_data.get("elements", []):
        if el.get("type") != "way":
            continue
        pts = el.get("geometry", [])
        if len(pts) < 3:
            continue
        coords = [[p["lon"], p["lat"]] for p in pts]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            geom = shape({"type": "Polygon", "coordinates": [coords]}).buffer(0)
            if not geom.is_valid or geom.is_empty:
                continue
            if not geom.intersects(bbox_poly):
                continue
            tags = el.get("tags", {})
            wtype = tags.get("natural", tags.get("waterway", tags.get("landuse", "water")))
            features.append({
                "geometry": geom,
                "water_type": wtype,
                "name": tags.get("name", ""),
            })
        except Exception:
            continue
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(features, crs="EPSG:4326")


def load_water_bodies(bounding_box: BoundingBox, data_dir=None) -> gpd.GeoDataFrame:
    """
    Load ALL water body polygons for the bbox from OSM Overpass API.
    Works for any bbox worldwide. Caches results locally.
    """
    cache_path = None
    if data_dir is not None:
        cache_dir = Path(data_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ck = (
            f"wb_{bounding_box.min_lon:.4f}_{bounding_box.min_lat:.4f}"
            f"_{bounding_box.max_lon:.4f}_{bounding_box.max_lat:.4f}.geojson"
        )
        cache_path = cache_dir / ck
        if cache_path.exists():
            try:
                gdf = gpd.read_file(str(cache_path))
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                logger.info("Water bodies from cache: %d features.", len(gdf))
                return gdf
            except Exception as e:
                logger.warning("Cache read failed: %s", e)

    query = _QUERY.format(
        s=bounding_box.min_lat, w=bounding_box.min_lon,
        n=bounding_box.max_lat, e=bounding_box.max_lon,
    )
    logger.info("Fetching water bodies from Overpass API for %s...", bounding_box)
    osm_data = _fetch(query)

    if osm_data is not None:
        gdf = _osm_to_gdf(osm_data, bounding_box)
        logger.info("Fetched %d water features from Overpass.", len(gdf))
        if cache_path is not None and len(gdf) > 0:
            try:
                gdf.to_file(str(cache_path), driver="GeoJSON")
            except Exception:
                pass
        return gdf

    logger.warning("Overpass unavailable. Returning empty GeoDataFrame.")
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
