"""
Pipeline orchestrator for the Flood Risk Zonation System.

Executes the full DAG: validate → grid → ingest → features → train → score.
Implements a three-tier data fallback strategy (real → partial → synthetic).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.exceptions import FloodRiskError
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS, extract_features
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.ingest.drainage import generate_synthetic_drainage
from flood_risk_zonation.ingest.elevation import generate_synthetic_elevation
from flood_risk_zonation.ingest.population import load_population
from flood_risk_zonation.ingest.rainfall import generate_synthetic_rainfall
from flood_risk_zonation.ingest.water_bodies import load_water_bodies
from flood_risk_zonation.model.trainer import FloodRiskModelTrainer
from flood_risk_zonation.models import FloodRiskResult
from flood_risk_zonation.scoring.scorer import FloodRiskScorer
from flood_risk_zonation.utils.cache import cache_key, get_cache_path, is_cached, load_geodataframe, save_geodataframe
from flood_risk_zonation.utils.validation import validate_bounding_box, validate_config

logger = logging.getLogger(__name__)


class FloodRiskPipeline:
    """
    End-to-end flood risk zonation pipeline.

    Parameters
    ----------
    config : PipelineConfig
        All tunable parameters for this run.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._data_tier: int = 3  # 1=real, 2=partial, 3=synthetic

    def run(self, bounding_box: BoundingBox) -> FloodRiskResult:
        """
        Execute the full pipeline for a given bounding box.

        Returns a FloodRiskResult containing the scored GeoDataFrame
        and a trained model artifact.
        """
        t0 = time.time()
        config = self.config

        # --- Validate inputs ---
        validate_bounding_box(bounding_box)
        validate_config(config)

        # --- Grid generation (with optional cache) ---
        ck = cache_key(bounding_box, config)
        cache_path = get_cache_path(ck + "_grid", config.cache_dir)

        if config.use_cache and is_cached(ck + "_grid", config.cache_dir):
            logger.info("Loading grid from cache: %s", cache_path)
            grid = load_geodataframe(cache_path)
        else:
            logger.info("Generating grid (cell_size=%dm)…", int(config.cell_size_meters))
            grid = generate_grid(bounding_box, config.cell_size_meters)
            if config.use_cache:
                save_geodataframe(grid, cache_path)

        seed = config.random_seed

        # --- Data ingestion — use real data if available, else synthetic fallback ---
        logger.info("Ingesting data...")
        self._data_tier = 3

        # Elevation — search ALL tif files, not just Gottigere
        elev_dir = Path("data/elevation")
        tif_files = list(elev_dir.glob("*.tif"))
        elevation = None
        if tif_files:
            from flood_risk_zonation.ingest.elevation import load_elevation
            for tif in tif_files:
                try:
                    elevation = load_elevation(bounding_box, tif.parent)
                    self._data_tier = 1
                    logger.info("Tier 1: Real SRTM elevation loaded from %s.", tif.name)
                    break
                except Exception:
                    continue
        if elevation is None:
            logger.warning("No SRTM file covers this bbox, using synthetic elevation.")
            elevation = generate_synthetic_elevation(bounding_box, resolution_m=500, seed=seed)

        # Rainfall
        rain_dir = Path("data/rainfall")
        if list(rain_dir.glob("*.tif")):
            from flood_risk_zonation.ingest.rainfall import load_rainfall
            try:
                rainfall = load_rainfall(bounding_box, rain_dir)
                logger.info("Real rainfall loaded.")
            except Exception as e:
                logger.warning("Real rainfall failed (%s), using synthetic.", e)
                rainfall = generate_synthetic_rainfall(bounding_box, resolution_m=1000, seed=seed)
        else:
            rainfall = generate_synthetic_rainfall(bounding_box, resolution_m=1000, seed=seed)

        # Water bodies — fetch live from Overpass API for any bbox worldwide
        # Results are cached locally so subsequent runs are instant
        logger.info("Fetching water bodies from Overpass API...")
        water_bodies = load_water_bodies(bounding_box, data_dir="data/water_bodies")
        logger.info("Water bodies loaded: %d features.", len(water_bodies))
        if len(water_bodies) > 0 and self._data_tier == 3:
            self._data_tier = 2

        population = load_population(bounding_box, data_dir=str(config.cache_dir))
        drainage = generate_synthetic_drainage(grid, seed=seed)
        logger.info("Extracting features for %d cells…", len(grid))
        featured_grid = extract_features(
            grid, elevation, rainfall, water_bodies, population, drainage
        )

        # --- Generate training labels from features ---
        # High risk = bottom 25% elevation + low drainage + close to water + high TWI
        # Use top 25% of risk proxy as "High" — realistic distribution
        X = featured_grid[FEATURE_COLUMNS].copy()
        risk_score_proxy = (
            (featured_grid["rainfall_mean_mm"].values / (featured_grid["rainfall_mean_mm"].max() + 1e-6)) * 0.25
            + (1 - featured_grid["elevation_m"].values / (featured_grid["elevation_m"].max() + 1e-6)) * 0.30
            + (featured_grid["twi"].values / (featured_grid["twi"].max() + 1e-6)) * 0.20
            + (1 - featured_grid["drainage_capacity"].values) * 0.15
            + (1 - featured_grid["dist_water_m"].values / (featured_grid["dist_water_m"].max() + 1e-6)) * 0.10
        )
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-0.05, 0.05, len(featured_grid))
        y_continuous = np.clip(risk_score_proxy + noise, 0, 1)
        # Top 25% = High risk (1), rest = Low risk (0) — realistic for flood susceptibility
        threshold = float(np.percentile(y_continuous, 75))
        y = pd.Series((y_continuous >= threshold).astype(int), name="high_risk")

        # Safety: ensure at least 2 classes
        if y.nunique() < 2:
            threshold = float(np.median(y_continuous))
            y = pd.Series((y_continuous >= threshold).astype(int), name="high_risk")

        # --- Model training ---
        logger.info("Training model (%s)…", config.model_type)
        trainer = FloodRiskModelTrainer(
            model_type=config.model_type,
            n_estimators=config.rf_n_estimators,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=seed,
        )
        training_result = trainer.train(X, y, cv_folds=config.cv_folds)

        # --- Risk scoring ---
        logger.info("Scoring grid…")
        scorer = FloodRiskScorer()
        scorer.p_min = trainer.p_min
        scorer.p_max = trainer.p_max
        thresholds = {"low_max": config.low_threshold, "medium_max": config.medium_threshold}
        scored_grid = scorer.score_grid(featured_grid, training_result.model, FEATURE_COLUMNS, thresholds)

        # --- Post-processing: water masking + proximity boosting ---
        scored_grid = self._apply_water_mask_and_proximity_boost(
            scored_grid, water_bodies, config
        )

        duration = time.time() - t0
        logger.info("Pipeline complete in %.1fs. Cells: %d", duration, len(scored_grid))

        return FloodRiskResult(
            scored_grid=scored_grid,
            training_result=training_result,
            bounding_box=bounding_box,
            config=config,
            pipeline_duration_seconds=duration,
            cell_count=len(scored_grid),
        )

    def _apply_water_mask_and_proximity_boost(
        self,
        scored_grid: gpd.GeoDataFrame,
        water_bodies: gpd.GeoDataFrame,
        config,
    ) -> gpd.GeoDataFrame:
        """
        Global water masking — works for any region worldwide:

        1. ELEVATION MASK: cells with elevation <= 1m are ocean/sea/tidal → Water
        2. OSM POLYGON MASK: cells whose centroid is inside any OSM water polygon → Water
        3. PROXIMITY BOOST: cells within 0.6× cell_size of any water boundary → boosted risk
        """
        from shapely.geometry import Point
        from shapely.ops import unary_union

        result = scored_grid.copy()

        # ── Step 1: Elevation-based sea/ocean mask ────────────────────────────
        # SRTM assigns 0 or negative values to ocean cells.
        # Use <= 1m to catch tidal flats and coastal cells too.
        if "elevation_m" in result.columns:
            sea_mask = result["elevation_m"].values <= 1.0
            n_sea = int(sea_mask.sum())
            if n_sea > 0:
                result.loc[sea_mask, "risk_class"] = "Water"
                result.loc[sea_mask, "risk_score"] = -1.0
                logger.info("Elevation mask: %d sea/ocean cells (elev <= 1m) → Water.", n_sea)

        # ── Step 2 & 3: OSM polygon mask + proximity boost ───────────────────
        if water_bodies is None or len(water_bodies) == 0:
            return result

        try:
            wb_m = water_bodies.to_crs("EPSG:3857")
            cleaned = wb_m.geometry.buffer(0)
            valid = cleaned[cleaned.is_valid & ~cleaned.is_empty]
            if len(valid) == 0:
                return result
            water_union = unary_union(valid.values)

            centroid_pts_m = gpd.GeoSeries(
                [Point(row.centroid_lon, row.centroid_lat) for _, row in result.iterrows()],
                crs="EPSG:4326",
            ).to_crs("EPSG:3857")

            proximity_m = config.cell_size_meters * 0.6
            already_water = result["risk_class"].values == "Water"

            osm_water = np.array([
                (not already_water[i]) and water_union.contains(pt)
                for i, pt in enumerate(centroid_pts_m)
            ])
            proximity = np.array([
                (not already_water[i]) and (not osm_water[i])
                and pt.distance(water_union) <= proximity_m
                for i, pt in enumerate(centroid_pts_m)
            ])

            # Apply OSM water mask
            result.loc[osm_water, "risk_class"] = "Water"
            result.loc[osm_water, "risk_score"] = -1.0

            # Apply proximity boost
            boost_floor = config.low_threshold + 5.0
            current = result.loc[proximity, "risk_score"].values
            result.loc[proximity, "risk_score"] = np.maximum(current, boost_floor)
            for idx in result.index[proximity]:
                s = result.at[idx, "risk_score"]
                result.at[idx, "risk_class"] = "High" if s > config.medium_threshold else "Medium"

            logger.info(
                "Water mask complete: %d elevation, %d OSM polygon, %d proximity-boosted.",
                int(already_water.sum()), int(osm_water.sum()), int(proximity.sum()),
            )

        except Exception as e:
            logger.warning("Water mask failed (%s), elevation mask still applied.", e)

        return result

    def run_stage(self, stage_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a single named pipeline stage."""
        stages = {
            "grid": generate_grid,
            "elevation": generate_synthetic_elevation,
            "rainfall": generate_synthetic_rainfall,
        }
        if stage_name not in stages:
            raise ValueError(f"Unknown stage: {stage_name}. Available: {list(stages)}")
        return stages[stage_name](*args, **kwargs)
