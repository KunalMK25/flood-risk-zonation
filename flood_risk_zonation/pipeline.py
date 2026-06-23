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
        scored_grid,
        water_bodies,
        config,
        elevation_source='real',
    ):
        from shapely.geometry import Point, box as shapely_box
        from shapely.ops import unary_union

        result = scored_grid.copy()
        result['is_coastal_tsunami_risk'] = False
        result['water_mask_reason'] = ''
        result['water_coverage_pct'] = 0.0

        OCEAN_TYPES = {'coastline', 'bay', 'sea', 'ocean'}
        AREA_WATER_TYPES = {'water', 'reservoir', 'basin', 'bay', 'sea', 'ocean', 'coastline'}
        LINEAR_BOOST_TYPES = {'river', 'canal', 'stream', 'drain', 'ditch'}

        area_water_geoms = []
        coastline_lines = []
        boost_geoms = []
        ocean_area_geoms = []

        if water_bodies is not None and len(water_bodies) > 0:
            wb = water_bodies.copy()
            if wb.crs and str(wb.crs).upper() != 'EPSG:4326':
                wb = wb.to_crs('EPSG:4326')
            for _, row in wb.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                geom = geom if geom.is_valid else geom.buffer(0)
                if geom.is_empty or not geom.is_valid:
                    continue
                wtype = str(row.get('water_type', '')).lower()
                boost_geoms.append(geom)
                if geom.geom_type in {'LineString', 'MultiLineString'}:
                    if wtype == 'coastline' or wtype in OCEAN_TYPES:
                        coastline_lines.append(geom)
                else:
                    if wtype in AREA_WATER_TYPES:
                        area_water_geoms.append(geom)
                        if wtype in OCEAN_TYPES:
                            ocean_area_geoms.append(geom)
                    elif wtype not in LINEAR_BOOST_TYPES:
                        area_water_geoms.append(geom)

        lon_min = float(result['centroid_lon'].min()) - 0.001
        lon_max = float(result['centroid_lon'].max()) + 0.001
        lat_min = float(result['centroid_lat'].min()) - 0.001
        lat_max = float(result['centroid_lat'].max()) + 0.001
        bbox_poly = shapely_box(lon_min, lat_min, lon_max, lat_max)

        # Build ocean polygon from coastline using RIGHT-SIDE buffer
        # OSM coastline convention: land is LEFT, sea is RIGHT of line direction.
        # We buffer the line on the right side by sweeping a wide band seaward,
        # then intersect with bbox to get the ocean area.
        ocean_polygon = None
        if coastline_lines:
            try:
                import geopandas as _gpd_c
                coast_union = unary_union(coastline_lines)
                # Convert to metric CRS for buffering
                coast_gdf = _gpd_c.GeoDataFrame(geometry=[coast_union], crs='EPSG:4326').to_crs('EPSG:3857')
                coast_m = coast_gdf.geometry.iloc[0]
                # Wide buffer (5x cell size) to cover the full ocean area in bbox
                wide_buf_m = config.cell_size_meters * 5.0
                ocean_buf_m = coast_m.buffer(wide_buf_m, single_sided=True)
                # Back to 4326
                ocean_buf_gdf = _gpd_c.GeoDataFrame(geometry=[ocean_buf_m], crs='EPSG:3857').to_crs('EPSG:4326')
                ocean_buf_4326 = ocean_buf_gdf.geometry.iloc[0]
                # Clip to bbox
                ocean_candidate = ocean_buf_4326.intersection(bbox_poly)
                if not ocean_candidate.is_empty:
                    # Verify: the ocean candidate should contain FEWER grid centroids than land
                    n_in = sum(1 for _, r in result.iterrows()
                               if ocean_candidate.contains(Point(r['centroid_lon'], r['centroid_lat'])))
                    n_total = len(result)
                    # If candidate contains more than 80% of centroids, it's likely wrong side
                    # Try the opposite side
                    if n_in > n_total * 0.8:
                        ocean_buf_m2 = coast_m.buffer(-wide_buf_m, single_sided=True)
                        ocean_buf_gdf2 = _gpd_c.GeoDataFrame(geometry=[ocean_buf_m2], crs='EPSG:3857').to_crs('EPSG:4326')
                        ocean_candidate2 = ocean_buf_gdf2.geometry.iloc[0].intersection(bbox_poly)
                        n_in2 = sum(1 for _, r in result.iterrows()
                                    if ocean_candidate2.contains(Point(r['centroid_lon'], r['centroid_lat'])))
                        if n_in2 < n_in:
                            ocean_candidate = ocean_candidate2
                    ocean_polygon = ocean_candidate
                    ocean_area_geoms.append(ocean_polygon)
                    boost_geoms.append(ocean_polygon)
                    logger.info('Ocean polygon (right-side buffer): covers ~%d cells.', 
                                sum(1 for _, r in result.iterrows()
                                    if ocean_polygon.contains(Point(r['centroid_lon'], r['centroid_lat']))))
            except Exception as exc:
                logger.warning('Ocean polygon (right-side buffer) failed: %s', exc)

        # Pure ocean bbox: no OSM features + low elevation
        if not coastline_lines and not area_water_geoms and 'elevation_m' in result.columns:
            finite_elev = result['elevation_m'].values[np.isfinite(result['elevation_m'].values)]
            if len(finite_elev) > 0 and float(np.percentile(finite_elev, 90)) <= 10.0:
                ocean_polygon = bbox_poly
                area_water_geoms.append(bbox_poly)
                ocean_area_geoms.append(bbox_poly)
                boost_geoms.append(bbox_poly)
                logger.info('Pure ocean bbox detected.')

        # Step 1: coverage mask for OSM area water bodies (lakes/ponds/reservoirs)
        if area_water_geoms:
            try:
                import geopandas as _gpd
                water_union_4326 = unary_union(area_water_geoms)
                _wdf = _gpd.GeoDataFrame(geometry=[water_union_4326], crs='EPSG:4326')
                water_union_m = _wdf.to_crs('EPSG:3857').geometry.iloc[0]
                grid_m = _gpd.GeoDataFrame(result, geometry='geometry', crs='EPSG:4326').to_crs('EPSG:3857')
                coverage_water = np.zeros(len(result), dtype=bool)
                coverage_pct = np.zeros(len(result), dtype=float)
                for i, cell_geom in enumerate(grid_m.geometry):
                    if cell_geom is None or cell_geom.is_empty:
                        continue
                    try:
                        inter = cell_geom.intersection(water_union_m)
                        if inter.is_empty:
                            continue
                        pct = inter.area / cell_geom.area if cell_geom.area > 0 else 0
                        coverage_pct[i] = pct
                        if pct >= 0.60:
                            coverage_water[i] = True
                    except Exception:
                        continue
                if coverage_water.any():
                    result.loc[coverage_water, 'risk_class'] = 'Water'
                    result.loc[coverage_water, 'risk_score'] = 0.0
                    result.loc[coverage_water, 'water_mask_reason'] = 'coverage'
                    result['water_coverage_pct'] = (coverage_pct * 100).round(1)
                    logger.info('Coverage mask: %d cells -> Water.', int(coverage_water.sum()))
                else:
                    result['water_coverage_pct'] = 0.0
            except Exception as exc:
                logger.warning('Coverage mask failed: %s', exc)

        # Step 1b: ocean centroid mask - cells whose centroid is inside ocean polygon
        if ocean_polygon is not None and not ocean_polygon.is_empty:
            try:
                already_water = result['risk_class'].values == 'Water'
                # Shrink ocean polygon by 0.5 cell to avoid coast straddlers
                shrink_deg = (config.cell_size_meters * 0.5) / 111_320.0
                try:
                    ocean_shrunk = ocean_polygon.buffer(-shrink_deg)
                except Exception:
                    ocean_shrunk = ocean_polygon
                if ocean_shrunk and not ocean_shrunk.is_empty:
                    ocean_water = np.zeros(len(result), dtype=bool)
                    for i, (_, r) in enumerate(result.iterrows()):
                        if already_water[i]:
                            continue
                        try:
                            if ocean_shrunk.contains(Point(r['centroid_lon'], r['centroid_lat'])):
                                ocean_water[i] = True
                        except Exception:
                            pass
                    if ocean_water.any():
                        result.loc[ocean_water, 'risk_class'] = 'Water'
                        result.loc[ocean_water, 'risk_score'] = 0.0
                        result.loc[ocean_water, 'water_mask_reason'] = 'ocean'
                        logger.info('Ocean centroid mask: %d cells -> Water.', int(ocean_water.sum()))
            except Exception as exc:
                logger.warning('Ocean centroid mask failed: %s', exc)

        # Step 2: elevation fallback (real SRTM only)
        if elevation_source not in {'synthetic', 'offline_sample'} and 'elevation_m' in result.columns:
            already_water = result['risk_class'].values == 'Water'
            sea_mask = (result['elevation_m'].values <= 2.0) & ~already_water
            if sea_mask.sum() > 0:
                result.loc[sea_mask, 'risk_class'] = 'Water'
                result.loc[sea_mask, 'risk_score'] = 0.0
                result.loc[sea_mask, 'water_mask_reason'] = 'elevation'
                logger.info('Elevation fallback: %d cells -> Water.', int(sea_mask.sum()))

        # Steps 3 & 4: proximity boost and coastal flag
        if not boost_geoms:
            return result
        try:
            import geopandas as _gpd2
            boost_union_m = unary_union(
                _gpd2.GeoDataFrame(geometry=boost_geoms, crs='EPSG:4326')
                .to_crs('EPSG:3857').geometry.tolist()
            )
            ocean_union_m = None
            if ocean_area_geoms:
                ocean_union_m = unary_union(
                    _gpd2.GeoDataFrame(geometry=ocean_area_geoms, crs='EPSG:4326')
                    .to_crs('EPSG:3857').geometry.tolist()
                )
            centroid_pts_m = gpd.GeoSeries(
                [Point(r.centroid_lon, r.centroid_lat) for _, r in result.iterrows()],
                crs='EPSG:4326',
            ).to_crs('EPSG:3857')
            proximity_m = config.cell_size_meters * 0.6
            now_water = result['risk_class'].values == 'Water'
            proximity = np.zeros(len(result), dtype=bool)
            for i, pt in enumerate(centroid_pts_m):
                if not now_water[i]:
                    try:
                        if pt.distance(boost_union_m) <= proximity_m:
                            proximity[i] = True
                    except Exception:
                        pass
            boost_floor = config.low_threshold + 5.0
            current = result.loc[proximity, 'risk_score'].values
            result.loc[proximity, 'risk_score'] = np.maximum(current, boost_floor)
            for idx in result.index[proximity]:
                s = result.at[idx, 'risk_score']
                result.at[idx, 'risk_class'] = 'High' if s > config.medium_threshold else 'Medium'
            if ocean_union_m is not None:
                coastal_m = config.cell_size_meters * 1.5
                now_water2 = result['risk_class'].values == 'Water'
                for i, pt in enumerate(centroid_pts_m):
                    if not now_water2[i]:
                        try:
                            if pt.distance(ocean_union_m) <= coastal_m:
                                result.iloc[i, result.columns.get_loc('is_coastal_tsunami_risk')] = True
                        except Exception:
                            pass
            logger.info('Water mask done: %d Water, %d coastal.',
                        int((result['risk_class']=='Water').sum()),
                        int(result['is_coastal_tsunami_risk'].sum()))
        except Exception as exc:
            logger.warning('Proximity/coastal step failed: %s', exc)
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
