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

        1. WATER COVERAGE MASK — cells with ≥ 60% area covered by OSM water
           geometries (any type including ocean derived from coastline) → Water.
           For ocean: the sea is constructed as bbox minus land polygons from
           the coastline, giving a proper ocean polygon for intersection.
        2. ELEVATION MASK — remaining cells with elevation ≤ 3 m → Water
           (catches ocean cells not covered by OSM data).
        3. PROXIMITY BOOST — land cells whose centroid is within 0.6 × cell_size
           of any water geometry → boosted risk.
        4. COASTAL FLAG — land cells within 1.5 × cell_size of ocean/sea
           geometry get ``is_coastal_tsunami_risk = True``.
        """
        from shapely.geometry import Point, box as shapely_box
        from shapely.ops import unary_union

        result = scored_grid.copy()
        result["is_coastal_tsunami_risk"] = False
        if "water_mask_reason" not in result.columns:
            result["water_mask_reason"] = ""

        synthetic_sources = {"synthetic", "offline_sample"}
        apply_elevation_mask = elevation_source not in synthetic_sources

        # ── Build water geometry sets ─────────────────────────────────────────
        OCEAN_TYPES = {"coastline", "bay", "sea", "ocean"}
        AREA_WATER_TYPES = {"water", "reservoir", "basin", "bay", "sea", "ocean"}
        # Linear features that contribute to proximity boost but NOT area mask
        LINEAR_TYPES = {"river", "canal", "stream", "drain", "ditch"}

        area_water_geoms: list = []    # for coverage mask (step 1)
        ocean_line_geoms: list = []    # coastline LineStrings → converted to ocean polygon
        linear_geoms: list = []        # for proximity boost only
        ocean_area_geoms: list = []    # ocean polygons for tsunami flag

        if water_bodies is not None and len(water_bodies) > 0:
            wb_4326 = water_bodies.copy()
            if wb_4326.crs and str(wb_4326.crs).upper() != "EPSG:4326":
                wb_4326 = wb_4326.to_crs("EPSG:4326")

            for _, wb_row in wb_4326.iterrows():
                geom = wb_row.geometry
                if geom is None or geom.is_empty:
                    continue
                geom = geom if geom.is_valid else geom.buffer(0)
                if geom.is_empty or not geom.is_valid:
                    continue
                wtype = str(wb_row.get("water_type", "")).lower()

                if geom.geom_type in {"LineString", "MultiLineString"}:
                    if wtype in OCEAN_TYPES or wtype in {"coastline"}:
                        ocean_line_geoms.append(geom)
                    else:
                        linear_geoms.append(geom)
                elif geom.geom_type in {"Polygon", "MultiPolygon"}:
                    if wtype in OCEAN_TYPES:
                        area_water_geoms.append(geom)
                        ocean_area_geoms.append(geom)
                    elif wtype in AREA_WATER_TYPES:
                        area_water_geoms.append(geom)
                    elif wtype in LINEAR_TYPES:
                        # Buffered river/canal polygon — use for proximity only
                        linear_geoms.append(geom)
                    else:
                        area_water_geoms.append(geom)

        # ── Construct ocean polygon from coastline LineStrings ────────────────
        # OSM coastline convention: sea is to the RIGHT of the line direction.
        # Best practical approach: take the bbox rectangle and subtract any
        # land polygons that can be derived from the coastline ring.
        # If we can't close the coastline ring, fall back to a wide buffer.
        bbox_poly_4326 = shapely_box(
            config.cache_dir and float(result["centroid_lon"].min()) - 0.001 or result["centroid_lon"].min() - 0.001,
            result["centroid_lat"].min() - 0.001,
            result["centroid_lon"].max() + 0.001,
            result["centroid_lat"].max() + 0.001,
        )

        ocean_polygon = None
        if ocean_line_geoms:
            try:
                from shapely.ops import polygonize, split
                coastline_union = unary_union(ocean_line_geoms)

                # Try to close the coastline ring using the bbox boundary
                # Split the bbox with the coastline and take the smaller fragment
                # (the ocean side — the fragment that doesn't contain the centroid
                # of the main land mass)
                try:
                    from shapely.ops import split as shapely_split
                    fragments = list(polygonize(
                        unary_union([coastline_union, bbox_poly_4326.boundary])
                    ))
                    if fragments:
                        # The ocean fragment is the one whose centroid has lower elevation
                        # or is closest to the coastline line
                        # Heuristic: take the fragment(s) on the seaward side
                        # — identified as fragments NOT containing the grid centroid cluster
                        grid_center = Point(
                            result["centroid_lon"].mean(),
                            result["centroid_lat"].mean(),
                        )
                        land_frag = None
                        for frag in fragments:
                            if frag.contains(grid_center):
                                land_frag = frag
                                break
                        if land_frag is not None:
                            ocean_polygon = bbox_poly_4326.difference(land_frag)
                        else:
                            # All fragments are partial — union the ones not containing the center
                            sea_frags = [f for f in fragments if not f.contains(grid_center)]
                            if sea_frags:
                                ocean_polygon = unary_union(sea_frags)
                except Exception:
                    pass

                # Fallback: wide buffer around coastline on the seaward side
                if ocean_polygon is None or ocean_polygon.is_empty:
                    # Use bbox minus a buffer inland from the coastline
                    coast_buffer_deg = (config.cell_size_meters * 5) / 111_320.0
                    ocean_polygon = bbox_poly_4326.difference(
                        coastline_union.buffer(coast_buffer_deg)
                    )

                if ocean_polygon and not ocean_polygon.is_empty:
                    area_water_geoms.append(ocean_polygon)
                    ocean_area_geoms.append(ocean_polygon)
                    logger.info("Ocean polygon constructed from coastline LineStrings.")

            except Exception as exc:
                logger.warning("Ocean polygon construction failed: %s", exc)

        # ── Step 1: Water coverage mask (≥ 60% of cell area = water) ─────────
        WATER_COVERAGE_THRESHOLD = 0.60

        if area_water_geoms:
            water_union_4326 = unary_union(area_water_geoms)
            grid_proj = result.copy().to_crs("EPSG:3857")
            water_union_3857 = water_bodies.to_crs("EPSG:3857") if water_bodies is not None and len(water_bodies) > 0 else None

            # Reproject water union to metric CRS for area calculation
            try:
                import geopandas as _gpd
                _wdf = _gpd.GeoDataFrame(geometry=[water_union_4326], crs="EPSG:4326")
                water_union_m = _wdf.to_crs("EPSG:3857").geometry.iloc[0]
            except Exception:
                water_union_m = None

            coverage_water = np.zeros(len(result), dtype=bool)
            coverage_pct = np.zeros(len(result), dtype=float)

            if water_union_m is not None:
                for i, row in enumerate(grid_proj.itertuples()):
                    cell_geom = row.geometry
                    if cell_geom is None or cell_geom.is_empty:
                        continue
                    try:
                        intersection = cell_geom.intersection(water_union_m)
                        if intersection.is_empty:
                            continue
                        cell_area = cell_geom.area
                        if cell_area <= 0:
                            continue
                        pct = intersection.area / cell_area
                        coverage_pct[i] = pct
                        if pct >= WATER_COVERAGE_THRESHOLD:
                            coverage_water[i] = True
                    except Exception:
                        continue

            # Determine dominant water type for each covered cell
            if coverage_water.any():
                result.loc[coverage_water, "risk_class"] = "Water"
                result.loc[coverage_water, "risk_score"] = 0.0
                result.loc[coverage_water, "water_mask_reason"] = "coverage"
                # Store coverage percentage for popup display
                result["water_coverage_pct"] = (coverage_pct * 100).round(1)
                logger.info(
                    "Coverage mask: %d cells (≥%.0f%% water area) → Water.",
                    int(coverage_water.sum()), WATER_COVERAGE_THRESHOLD * 100,
                )
            else:
                result["water_coverage_pct"] = 0.0

        else:
            result["water_coverage_pct"] = 0.0

        # ── Step 2: Elevation mask for remaining cells ────────────────────────
        if apply_elevation_mask and "elevation_m" in result.columns:
            already_water = result["risk_class"].values == "Water"
            sea_mask = (result["elevation_m"].values <= 3.0) & ~already_water
            n_sea = int(sea_mask.sum())
            if n_sea > 0:
                result.loc[sea_mask, "risk_class"] = "Water"
                result.loc[sea_mask, "risk_score"] = 0.0
                result.loc[sea_mask, "water_mask_reason"] = "elevation"
                logger.info("Elevation mask: %d cells (elev ≤ 3 m) → Water.", n_sea)

        # ── Steps 3 & 4: Proximity boost and coastal flag ─────────────────────
        if water_bodies is None or len(water_bodies) == 0:
            return result

        try:
            all_geoms_m: list = []
            wb_m = water_bodies.to_crs("EPSG:3857")
            for _, wb_row in wb_m.iterrows():
                geom = wb_row.geometry
                if geom is None or geom.is_empty:
                    continue
                geom = geom if geom.is_valid else geom.buffer(0)
                if not geom.is_empty and geom.is_valid:
                    all_geoms_m.append(geom)

            if ocean_polygon is not None and not ocean_polygon.is_empty:
                try:
                    import geopandas as _gpd2
                    _odf = _gpd2.GeoDataFrame(geometry=[ocean_polygon], crs="EPSG:4326")
                    ocean_m = _odf.to_crs("EPSG:3857").geometry.iloc[0]
                    all_geoms_m.append(ocean_m)
                except Exception:
                    pass

            if not all_geoms_m:
                return result

            water_union_all_m = unary_union(all_geoms_m)

            # Ocean union in metric CRS for coastal flag
            ocean_geoms_m: list = []
            for _, wb_row in wb_m.iterrows():
                wtype = str(wb_row.get("water_type", "")).lower()
                if wtype in OCEAN_TYPES:
                    g = wb_row.geometry
                    if g is not None and not g.is_empty:
                        ocean_geoms_m.append(g if g.is_valid else g.buffer(0))
            if ocean_polygon is not None and not ocean_polygon.is_empty:
                try:
                    ocean_geoms_m.append(ocean_m)
                except Exception:
                    pass
            ocean_union_m = unary_union(ocean_geoms_m) if ocean_geoms_m else None

            centroid_pts_m = gpd.GeoSeries(
                [Point(row.centroid_lon, row.centroid_lat) for _, row in result.iterrows()],
                crs="EPSG:4326",
            ).to_crs("EPSG:3857")

            proximity_m = config.cell_size_meters * 0.6
            now_water = result["risk_class"].values == "Water"

            # Step 3: proximity boost for land cells near any water
            proximity = np.zeros(len(result), dtype=bool)
            for i, pt in enumerate(centroid_pts_m):
                if not now_water[i]:
                    try:
                        if pt.distance(water_union_all_m) <= proximity_m:
                            proximity[i] = True
                    except Exception:
                        pass

            boost_floor = config.low_threshold + 5.0
            current = result.loc[proximity, "risk_score"].values
            result.loc[proximity, "risk_score"] = np.maximum(current, boost_floor)
            for idx in result.index[proximity]:
                s = result.at[idx, "risk_score"]
                result.at[idx, "risk_class"] = "High" if s > config.medium_threshold else "Medium"

            # Step 4: coastal tsunami flag
            if ocean_union_m is not None:
                coastal_distance_m = config.cell_size_meters * 1.5
                now_water2 = result["risk_class"].values == "Water"
                for i, pt in enumerate(centroid_pts_m):
                    if not now_water2[i]:
                        try:
                            if pt.distance(ocean_union_m) <= coastal_distance_m:
                                result.iloc[i, result.columns.get_loc("is_coastal_tsunami_risk")] = True
                        except Exception:
                            pass

            n_water_final = int((result["risk_class"] == "Water").sum())
            n_coastal = int(result["is_coastal_tsunami_risk"].sum())
            logger.info(
                "Water mask complete: %d total Water cells, %d coastal-flagged.",
                n_water_final, n_coastal,
            )

        except Exception as e:
            logger.warning("Proximity/coastal step failed (%s).", e)

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
