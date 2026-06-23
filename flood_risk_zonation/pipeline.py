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

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.exceptions import FloodRiskError
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS, extract_features
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.ingest.drainage import generate_synthetic_drainage
from flood_risk_zonation.ingest.elevation import generate_synthetic_elevation
from flood_risk_zonation.ingest.population import load_population
from flood_risk_zonation.ingest.rainfall import generate_synthetic_rainfall
from flood_risk_zonation.ingest.water_bodies import load_water_bodies
from flood_risk_zonation.models import AnalysisResult, FloodRiskResult
from flood_risk_zonation.scoring.scorer import FloodRiskScorer
from flood_risk_zonation.scoring.susceptibility import WeightedSusceptibilityModel, RandomForestSusceptibilityModel, EnsembleSusceptibilityModel
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
            grid = generate_grid(
                bounding_box,
                config.cell_size_meters,
                max_cells=config.max_grid_cells,
            )
            if config.use_cache:
                save_geodataframe(grid, cache_path)

        seed = config.random_seed

        # --- Data ingestion — use real data if available, else synthetic fallback ---
        logger.info("Ingesting data...")
        self._data_tier = 3
        provenance: dict[str, str] = {}

        # Elevation — search ALL tif files, not just Gottigere
        elev_dir = Path("data/elevation")
        elevation = None
        if elev_dir.exists():
            from flood_risk_zonation.ingest.elevation import load_elevation
            try:
                elevation = load_elevation(bounding_box, elev_dir)
                logger.info("Real elevation loaded from %s.", elevation.source)
            except FloodRiskError as exc:
                logger.warning("Real elevation unavailable (%s).", exc)
        if elevation is None:
            logger.warning("No SRTM file covers this bbox, using synthetic elevation.")
            elevation = generate_synthetic_elevation(bounding_box, resolution_m=500, seed=seed)
        provenance["elevation"] = elevation.source

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
        provenance["rainfall"] = rainfall.source

        water_bodies = load_water_bodies(
            bounding_box,
            data_dir="data/water_bodies",
            allow_network=config.allow_network,
        )
        logger.info("Water bodies loaded: %d features.", len(water_bodies))
        provenance["water_bodies"] = water_bodies.attrs.get("source", "unavailable")

        population = load_population(bounding_box, data_dir=str(config.cache_dir))
        drainage = generate_synthetic_drainage(grid, seed=seed)
        provenance["population"] = population.source
        provenance["drainage"] = "synthetic"

        core_real = [
            provenance["elevation"] != "synthetic",
            provenance["rainfall"] != "synthetic",
            provenance["water_bodies"] in {"osm_overpass", "osm_cache"},
        ]
        self._data_tier = 1 if all(core_real) else (2 if any(core_real) else 3)
        logger.info("Extracting features for %d cells…", len(grid))
        featured_grid = extract_features(
            grid, elevation, rainfall, water_bodies, population, drainage
        )

        # --- Susceptibility model ---
        # WSI: transparent weighted index, no training needed.
        # RF: Random Forest trained on WSI pseudo-labels with 5-fold CV.
        # Ensemble (default): blends WSI + RF, reports full CV metrics.
        X = featured_grid[FEATURE_COLUMNS].copy()
        model_type = getattr(config, "model_type", "ensemble")

        if model_type == "ensemble":
            logger.info("Training Ensemble (WSI + RF) susceptibility model…")
            model = EnsembleSusceptibilityModel(
                n_estimators=config.rf_n_estimators,
                cv_folds=config.cv_folds,
                random_state=config.random_seed,
            ).fit(X)
            analysis_result = AnalysisResult(
                model=model,
                feature_names=list(model.feature_names),
                feature_importances=model.feature_importances,
                method="ensemble",
                validation_note=(
                    f"Ensemble (WSI + RF blend). "
                    f"5-fold CV — AUC: {model.mean_cv_auc:.3f}, "
                    f"F1: {model.mean_cv_f1:.3f}, "
                    f"Accuracy: {model.mean_cv_accuracy:.3f}. "
                    "Labels derived from WSI; not calibrated against observed flood events."
                ),
                mean_cv_auc=model.mean_cv_auc,
                mean_cv_f1=model.mean_cv_f1,
                mean_cv_accuracy=model.mean_cv_accuracy,
                mean_cv_precision=model.mean_cv_precision,
                mean_cv_recall=model.mean_cv_recall,
                cv_auc_scores=model.cv_auc_scores,
                cv_f1_scores=model.cv_f1_scores,
                cv_accuracy_scores=model.cv_accuracy_scores,
                cv_precision_scores=model.cv_precision_scores,
                cv_recall_scores=model.cv_recall_scores,
            )
        elif model_type == "random_forest":
            logger.info("Training Random Forest susceptibility model…")
            model = RandomForestSusceptibilityModel(
                n_estimators=config.rf_n_estimators,
                cv_folds=config.cv_folds,
                random_state=config.random_seed,
            ).fit(X)
            analysis_result = AnalysisResult(
                model=model,
                feature_names=list(model.feature_names),
                feature_importances=model.feature_importances,
                method="random_forest",
                validation_note=(
                    f"Random Forest trained on WSI pseudo-labels. "
                    f"5-fold CV — AUC: {model.mean_cv_auc:.3f}, F1: {model.mean_cv_f1:.3f}. "
                    "Labels derived from WSI; not calibrated against observed flood events."
                ),
                mean_cv_auc=model.mean_cv_auc,
                mean_cv_f1=model.mean_cv_f1,
                cv_auc_scores=model.cv_auc_scores,
                cv_f1_scores=model.cv_f1_scores,
            )
        else:
            # Weighted Susceptibility Index (fully transparent, no training)
            model = WeightedSusceptibilityModel().fit(X)
            analysis_result = AnalysisResult(
                model=model,
                feature_names=list(model.feature_names),
                feature_importances=model.feature_importances,
                method="weighted_susceptibility_index",
                validation_note=(
                    "Relative susceptibility index; not calibrated against observed flood events."
                ),
            )

        # --- Risk scoring ---
        logger.info("Scoring grid…")
        scorer = FloodRiskScorer()
        scorer.p_min = 0.0
        scorer.p_max = 1.0
        thresholds = {"low_max": config.low_threshold, "medium_max": config.medium_threshold}
        scored_grid = scorer.score_grid(featured_grid, model, FEATURE_COLUMNS, thresholds)

        # --- Post-processing: water masking + proximity boosting ---
        scored_grid = self._apply_water_mask_and_proximity_boost(
            scored_grid, water_bodies, config,
            elevation_source=provenance.get("elevation", "synthetic"),
        )

        duration = time.time() - t0
        logger.info("Pipeline complete in %.1fs. Cells: %d", duration, len(scored_grid))

        return FloodRiskResult(
            scored_grid=scored_grid,
            analysis_result=analysis_result,
            bounding_box=bounding_box,
            config=config,
            pipeline_duration_seconds=duration,
            cell_count=len(scored_grid),
            data_provenance=provenance,
            data_tier=self._data_tier,
        )

    def _apply_water_mask_and_proximity_boost(
        self,
        scored_grid: gpd.GeoDataFrame,
        water_bodies: gpd.GeoDataFrame,
        config,
        elevation_source: str = "real",
    ) -> gpd.GeoDataFrame:
        """
        Post-scoring pipeline:

        1. ELEVATION MASK  — cells with elevation ≤ 1 m (SRTM ocean/tidal) → Water
                             Skipped when elevation_source is "synthetic" or
                             "offline_sample" to avoid false positives.
        2. OSM AREA MASK   — cells whose *centroid* lies inside an OSM area water
                             body (lake, reservoir, bay, coastline) → Water.
                             Only genuine area features qualify; linear features
                             (drains, streams, rivers, canals) NEVER mask cells
                             as Water — they only trigger the proximity boost.
        3. PROXIMITY BOOST  — land cells whose centroid is within 0.6 × cell_size
                              of any water geometry (including linear) → boosted risk
        4. COASTAL FLAG     — land cells within 1.5 × cell_size of ocean/sea
                              geometry get ``is_coastal_tsunami_risk = True``
        """
        from shapely.geometry import Point
        from shapely.ops import unary_union

        result = scored_grid.copy()

        # ── Initialise the coastal tsunami flag column ────────────────────────
        result["is_coastal_tsunami_risk"] = False

        # ── Step 1: Elevation-based sea/ocean mask ────────────────────────────
        synthetic_sources = {"synthetic", "offline_sample"}
        apply_elevation_mask = elevation_source not in synthetic_sources

        if apply_elevation_mask and "elevation_m" in result.columns:
            sea_mask = result["elevation_m"].values <= 1.0
            n_sea = int(sea_mask.sum())
            if n_sea > 0:
                result.loc[sea_mask, "risk_class"] = "Water"
                result.loc[sea_mask, "risk_score"] = 0.0
                logger.info("Elevation mask: %d sea/ocean cells (elev ≤ 1 m) → Water.", n_sea)
        elif not apply_elevation_mask:
            logger.debug("Elevation mask skipped (source: %s).", elevation_source)

        # ── Steps 2, 3 & 4: OSM-based masking, boost, and coastal flag ───────
        if water_bodies is None or len(water_bodies) == 0:
            return result

        try:
            wb_m = water_bodies.to_crs("EPSG:3857")

            # Water body types that represent genuine area water bodies
            # (i.e. features that actually cover 2D area on the ground).
            # Drains, streams, rivers, canals are linear features — they
            # should boost adjacent cells' risk but NOT mask them as Water.
            AREA_WATER_TYPES = {"water", "reservoir", "basin", "bay", "coastline", "sea", "ocean"}
            OCEAN_TYPES = {"coastline", "bay", "sea", "ocean"}

            area_geoms: list = []   # for Water-mask step 2
            ocean_geoms: list = []  # for coastal tsunami flag step 4
            all_valid_geoms: list = []  # for proximity boost step 3 (all types)

            for _, wb_row in wb_m.iterrows():
                geom = wb_row.geometry
                if geom is None or geom.is_empty:
                    continue
                geom = geom if geom.is_valid else geom.buffer(0)
                if geom.is_empty or not geom.is_valid:
                    continue
                wtype = str(wb_row.get("water_type", "")).lower()
                all_valid_geoms.append(geom)
                if wtype in AREA_WATER_TYPES:
                    area_geoms.append(geom)
                if wtype in OCEAN_TYPES:
                    ocean_geoms.append(geom)

            if not all_valid_geoms:
                return result

            # Union of ALL water geometries (including linear) — for proximity boost
            water_union = unary_union(all_valid_geoms)

            # Union of AREA water bodies only — for Water-mask centroid check
            # We deliberately use centroid-point test here (not cell-polygon
            # intersection) to avoid thin drain/stream slivers that were stored
            # as closed polygons from earlier OSM fetches marking every
            # adjacent cell as Water.
            area_polygon_geoms = [
                g for g in area_geoms
                if g.geom_type in {"Polygon", "MultiPolygon"}
            ]
            area_polygon_union = unary_union(area_polygon_geoms) if area_polygon_geoms else None

            # Ocean union for the coastal tsunami flag
            ocean_union = unary_union(ocean_geoms) if ocean_geoms else None

            # Build centroid GeoSeries in EPSG:3857
            centroid_pts_m = gpd.GeoSeries(
                [Point(row.centroid_lon, row.centroid_lat) for _, row in result.iterrows()],
                crs="EPSG:4326",
            ).to_crs("EPSG:3857")

            proximity_m = config.cell_size_meters * 0.6
            already_water = result["risk_class"].values == "Water"

            # Step 2: centroid lies inside an area water polygon
            # Using centroid point (not full cell polygon) prevents thin
            # drain/stream sliver polygons from falsely masking adjacent cells.
            osm_water = np.zeros(len(result), dtype=bool)
            if area_polygon_union is not None:
                for i, pt in enumerate(centroid_pts_m):
                    if not already_water[i]:
                        try:
                            if area_polygon_union.contains(pt):
                                osm_water[i] = True
                        except Exception:
                            pass

            # Step 3: centroid within proximity of ANY water geometry (incl. linear)
            proximity = np.zeros(len(result), dtype=bool)
            for i, pt in enumerate(centroid_pts_m):
                if not already_water[i] and not osm_water[i]:
                    if pt.distance(water_union) <= proximity_m:
                        proximity[i] = True

            # Apply OSM water mask (hard override)
            result.loc[osm_water, "risk_class"] = "Water"
            result.loc[osm_water, "risk_score"] = 0.0

            # Apply proximity boost to land cells near water
            boost_floor = config.low_threshold + 5.0
            current = result.loc[proximity, "risk_score"].values
            result.loc[proximity, "risk_score"] = np.maximum(current, boost_floor)
            for idx in result.index[proximity]:
                s = result.at[idx, "risk_score"]
                result.at[idx, "risk_class"] = "High" if s > config.medium_threshold else "Medium"

            # ── Step 4: Coastal tsunami flag ──────────────────────────────────
            if ocean_union is not None:
                coastal_distance_m = config.cell_size_meters * 1.5
                now_water = result["risk_class"].values == "Water"
                for i, pt in enumerate(centroid_pts_m):
                    if not now_water[i]:
                        if pt.distance(ocean_union) <= coastal_distance_m:
                            result.iloc[i, result.columns.get_loc("is_coastal_tsunami_risk")] = True

            n_water_final = int((result["risk_class"] == "Water").sum())
            n_coastal = int(result["is_coastal_tsunami_risk"].sum())
            logger.info(
                "Water mask complete: %d elevation, %d OSM polygon, "
                "%d proximity-boosted, %d coastal-flagged.",
                int((result["risk_class"] == "Water").sum() - int(osm_water.sum())) if apply_elevation_mask else 0,
                int(osm_water.sum()),
                int(proximity.sum()),
                n_coastal,
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
