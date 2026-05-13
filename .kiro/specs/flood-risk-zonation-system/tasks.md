# Implementation Plan: Machine Learning-Based Flood Risk Zonation System

## Overview

This plan converts the flood risk zonation system design into incremental coding tasks. Each task builds on the previous, starting with project scaffolding and core data models, progressing through data ingestion, feature engineering, ML training, risk scoring, visualization, and the Streamlit application. Property-based tests (Hypothesis) are placed immediately after the components they validate to catch errors early. The implementation language is Python throughout, matching the design document.

## Tasks

- [x] 1. Project scaffolding, configuration, and core data models
  - Create the full directory structure: `flood_risk_zonation/`, `ingest/`, `grid/`, `features/`, `model/`, `scoring/`, `visualization/`, `utils/`, `tests/unit/`, `tests/integration/`, `tests/property/`, `model/artifacts/`, `data/cache/`
  - Add `__init__.py` files to every package directory
  - Create `requirements.txt` (or `pyproject.toml`) pinning: streamlit>=1.35, folium>=0.17, scikit-learn>=1.4, lightgbm>=4.3, geopandas>=0.14, rasterio>=1.3, pandas>=2.0, numpy>=1.26, pyproj>=3.6, joblib>=1.3, matplotlib>=3.8, seaborn>=0.13, hypothesis>=6.100, pytest>=8.0, pytest-cov
  - Create `config.py` with the `PipelineConfig` dataclass (all fields from design §4.1), `BoundingBox` frozen dataclass with `__post_init__` validation, `area_km2` property, and `center` property
  - Create `flood_risk_zonation/exceptions.py` with the full exception hierarchy: `FloodRiskError`, `DataIngestionError`, `DataAlignmentError`, `FeatureExtractionError`, `ModelTrainingError`, `ScoringError`, `ConfigurationError`
  - Create `utils/validation.py` with `validate_bounding_box` and `validate_config` functions (exact logic from design §6.4)
  - Create `TrainingResult` and `FloodRiskResult` dataclasses in `config.py` or a dedicated `models.py`
  - Create `RasterDataset` and `RainfallDataset` dataclasses (design §4.2)
  - _Requirements: 7.3, 8.1, 8.2_

  - [x] 1.1 Write unit tests for BoundingBox and PipelineConfig validation
    - Test `BoundingBox` raises `ConfigurationError` for `min_lon >= max_lon`, `min_lat >= max_lat`, out-of-WGS84-range coordinates
    - Test `PipelineConfig` raises `ConfigurationError` for `cell_size_meters <= 0` and `low_threshold >= medium_threshold`
    - Test `BoundingBox.area_km2` and `BoundingBox.center` return correct values for known inputs
    - _Requirements: 7.3, 8.1, 8.2_

- [x] 2. Input validation utilities and property test for invalid input rejection
  - Implement `validate_bounding_box` and `validate_config` in `utils/validation.py` with all checks from design §6.4
  - Wire validation calls into `BoundingBox.__post_init__` and `PipelineConfig.__post_init__`
  - Create `utils/crs.py` with CRS utility helpers (reproject raster, reproject GeoDataFrame, degrees-to-metres conversion at a given latitude)
  - _Requirements: 7.3, 8.1, 8.2_

  - [x] 2.1 Write property test for invalid input rejection (Property 17)
    - **Property 17: Invalid Input Rejection**
    - **Validates: Requirements 7.3, 8.1, 8.2**
    - Use Hypothesis to generate invalid `BoundingBox` inputs (`min_lon >= max_lon`, `min_lat >= max_lat`, out-of-range coordinates) and assert `ConfigurationError` is raised
    - Use Hypothesis to generate invalid `PipelineConfig` inputs (`cell_size_meters <= 0`, `low_threshold >= medium_threshold`) and assert `ConfigurationError` is raised

- [x] 3. Grid generation engine
  - Implement `grid/generator.py` with `generate_grid(bounding_box, cell_size_meters, crs)` function
  - Convert `cell_size_meters` to degrees using latitude-dependent conversion (`cell_deg = cell_size_meters / (111_320 * cos(lat_rad))`)
  - Generate a regular rectangular grid of Shapely `Polygon` cells covering the full bounding box
  - Assign `cell_id` as `"{row_idx}_{col_idx}"` strings, compute `centroid_lat` and `centroid_lon` for each cell
  - Return a `gpd.GeoDataFrame` with columns: `cell_id`, `geometry`, `centroid_lat`, `centroid_lon`
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.1 Write unit tests for grid generator edge cases
    - Test that a 1×1 degree bounding box with 500m cells produces the expected approximate cell count
    - Test that `cell_id` values are all unique strings
    - Test that all cell geometries are valid Shapely polygons
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.2 Write property test for grid coverage completeness (Property 3)
    - **Property 3: Grid Coverage Completeness**
    - **Validates: Requirements 2.1**
    - Use Hypothesis `valid_bounding_boxes` strategy; assert union of all cell geometries contains the bounding box polygon

  - [x] 3.3 Write property test for grid cell size accuracy (Property 4)
    - **Property 4: Grid Cell Size Accuracy**
    - **Validates: Requirements 2.2**
    - For any valid bounding box and positive `cell_size_meters`, assert every cell area is within ±10% of `cell_size_meters²`

  - [x] 3.4 Write property test for grid cell uniqueness and non-overlap (Property 5)
    - **Property 5: Grid Cell Uniqueness and Non-Overlap**
    - **Validates: Requirements 2.3, 2.4**
    - Assert all `cell_id` values are unique; assert pairwise intersection area of any two distinct cells is zero within floating-point tolerance

- [x] 4. Caching and serialization utilities
  - Implement `utils/cache.py` with `save_geodataframe(gdf, path)` and `load_geodataframe(path)` using GeoParquet format
  - Implement `cache_key(bounding_box, config)` function that produces a deterministic string key from the inputs
  - Implement `is_cached(key, cache_dir)` and `get_cache_path(key, cache_dir)` helpers
  - _Requirements: 7.2_

- [x] 5. Data ingestion — elevation (SRTM)
  - Implement `ingest/elevation.py` with `load_elevation(bounding_box, data_dir)` that reads a local SRTM GeoTIFF file clipped to the bounding box using rasterio
  - Implement `resample_raster(raster_dataset, target_resolution_m)` to resample to a target resolution using bilinear interpolation
  - Implement `reproject_raster(raster_dataset, target_crs)` in `utils/crs.py` that reprojects a `RasterDataset` to the target CRS and returns a new `RasterDataset`
  - Raise `DataIngestionError` if the file is not found or the bounding box falls outside SRTM coverage
  - _Requirements: 1.1_

  - [x] 5.1 Write property test for raster reprojection CRS preservation (Property 1)
    - **Property 1: Raster Reprojection Preserves Target CRS**
    - **Validates: Requirements 1.1**
    - Use Hypothesis to generate source latitudes; create a synthetic `RasterDataset` in a UTM CRS, reproject to WGS84, assert output CRS equals EPSG:4326 and all values are finite

- [x] 6. Data ingestion — rainfall (GPM/IMD) and missing value imputation
  - Implement `ingest/rainfall.py` with `load_rainfall(bounding_box, data_dir)` that reads GPM IMERG or IMD gridded NetCDF/GeoTIFF files and returns a `RainfallDataset`
  - Implement `impute_missing_values(array)` in `utils/validation.py` (or a new `utils/imputation.py`) using spatial nearest-neighbour or mean imputation to fill NaN values
  - Fall back to `generate_synthetic_rainfall(bounding_box, seed)` (using `scipy.ndimage.gaussian_filter` on a random array) when real data is unavailable, logging a warning
  - _Requirements: 1.2_

  - [x] 6.1 Write property test for missing value imputation completeness (Property 2)
    - **Property 2: Missing Value Imputation Completeness**
    - **Validates: Requirements 1.2**
    - Use Hypothesis `st.lists(st.floats(allow_nan=True), min_size=4, max_size=100)`; reshape to 2D array, apply imputation, assert output contains no NaN values

- [x] 7. Data ingestion — OSM water bodies, population density, and synthetic drainage
  - Implement `ingest/water_bodies.py` with `load_water_bodies(bounding_box)` that reads a local OSM GeoJSON/Shapefile of water body polygons clipped to the bounding box; return a `gpd.GeoDataFrame`
  - Implement `ingest/population.py` with `load_population(bounding_box, data_dir)` that reads a population density raster (e.g., WorldPop GeoTIFF) and returns a `RasterDataset`
  - Implement `ingest/drainage.py` with `generate_synthetic_drainage(grid, seed)` that assigns drainage capacity scores `[0, 1]` inversely correlated with population density (design §4.3)
  - Implement `generate_synthetic_elevation(bounding_box, resolution_m, base_elevation_m, relief_m, seed)` in `ingest/elevation.py` using Perlin noise (via the `noise` library) for demo/test mode
  - _Requirements: 1.1, 1.2, 3.3, 3.4_

- [x] 8. Terrain feature computation (slope, TWI, aspect, curvature)
  - Implement `features/terrain.py` with:
    - `compute_slope(dem_array, cell_size_m)` using Horn (1981) method via `np.gradient`; return degrees
    - `compute_twi(dem_array, cell_size_m)` using D8 flow accumulation approximation; apply `ε = 1e-6` to `tan(β)` denominator; return finite array
    - `compute_aspect(dem_array)` returning degrees 0–360 clockwise from north
    - `compute_curvature(dem_array, cell_size_m)` returning plan curvature
  - All functions must handle flat terrain (all-zero DEM) without producing NaN or infinity
  - _Requirements: 3.1, 3.2_

  - [x] 8.1 Write unit tests for terrain feature edge cases
    - Test `compute_twi` on a flat DEM (all zeros) produces all-finite values
    - Test `compute_slope` on a known inclined plane produces the expected angle
    - Test `compute_aspect` on a DEM sloping due north returns 0°
    - _Requirements: 3.1, 3.2_

  - [x] 8.2 Write property test for TWI formula correctness (Property 7)
    - **Property 7: TWI Formula Correctness**
    - **Validates: Requirements 3.2**
    - Use Hypothesis to generate DEM arrays with known slope values; assert computed TWI equals `ln(A / (tan(β) + 1e-6))` within floating-point precision; assert flat terrain produces finite TWI

- [x] 9. Hydrological feature computation (distance to water, drainage density)
  - Implement `features/hydrological.py` with:
    - `compute_distance_to_water(grid, water_bodies)` using GeoPandas `STRtree` spatial index; cap at 10,000m; return non-negative float array
    - `compute_drainage_density(grid, drainage_data)` that assigns the drainage capacity score from the `DrainageDataset` to each grid cell
  - Handle the edge case of empty water bodies GeoDataFrame (return max distance 10,000m for all cells)
  - _Requirements: 3.3, 3.4_

  - [x] 9.1 Write unit tests for hydrological features
    - Test `compute_distance_to_water` returns 0 for a cell centroid that lies on a water body boundary
    - Test `compute_distance_to_water` returns 10,000m when no water bodies are present
    - _Requirements: 3.3_

  - [x] 9.2 Write property test for distance to water non-negativity (Property 8)
    - **Property 8: Distance to Water Non-Negativity**
    - **Validates: Requirements 3.3**
    - Use Hypothesis to generate arbitrary grid centroids and water body geometries (including empty set); assert all computed distances are >= 0

- [x] 10. Rainfall feature extraction per grid cell
  - Implement `features/rainfall_features.py` with `extract_rainfall_features(grid, rainfall_dataset)` that spatially samples `mean_annual_mm` and `max_24h_mm` from the `RainfallDataset` raster at each grid cell centroid using bilinear interpolation
  - Handle cells that fall outside the rainfall raster extent by filling with the dataset mean
  - _Requirements: 3.1, 3.4_

- [x] 11. Feature assembly, normalization, and full feature extractor
  - Implement `features/extractor.py` with `extract_features(grid, elevation_raster, rainfall_data, water_bodies, population_raster, drainage_data)` that:
    - Calls all terrain, hydrological, and rainfall feature functions
    - Assembles results into the grid GeoDataFrame with all 10 `FEATURE_COLUMNS`
    - Applies log-scaling to `population_density` and `dist_water_m`
    - Runs `impute_missing_values` on each feature column to ensure no NaN remains
    - Validates all features are within physically valid ranges (design §5, Property 6)
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 11.1 Write property test for feature extraction validity (Property 6)
    - **Property 6: Feature Extraction Produces Valid, Complete Feature Matrices**
    - **Validates: Requirements 3.1, 3.4**
    - Use Hypothesis to generate synthetic dataset combinations; assert output contains no NaN, no infinite values, and all features are within physically valid ranges

- [x] 12. Checkpoint — core data pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. ML model trainer (Random Forest primary, LightGBM optional)
  - Implement `model/trainer.py` with `FloodRiskModelTrainer` class:
    - `__init__(model_type)` initializing a `RandomForestClassifier` (n_estimators=200, min_samples_leaf=5, random_state from config) or `LGBMClassifier`
    - `train(X, y, cv_folds=5)` using `StratifiedKFold` cross-validation; compute AUC-ROC and F1 per fold; return `TrainingResult` with fitted model, `feature_importances`, `cv_scores`, `mean_cv_auc`, `mean_cv_f1`, `training_timestamp`
    - `save(path)` serializing the full trainer instance (model + calibration params) with `joblib.dump`
    - `load(path)` classmethod deserializing with `joblib.load`
  - Raise `ModelTrainingError` if fewer than 50 samples or all samples belong to one class
  - _Requirements: 4.1, 4.2, 4.3_

  - [x] 13.1 Write unit tests for model trainer
    - Test `train` raises `ModelTrainingError` with < 50 samples
    - Test `train` raises `ModelTrainingError` when all labels are the same class
    - Test `train` returns `TrainingResult` with `mean_cv_auc` between 0 and 1
    - Test `save` and `load` produce identical predictions on the same input
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 13.2 Write property test for model serialization round-trip (Property 9)
    - **Property 9: Model Serialization Round-Trip**
    - **Validates: Requirements 4.2**
    - Use Hypothesis to generate valid feature matrices; train a model, serialize to a temp path, deserialize, assert predictions are bit-for-bit identical

  - [x] 13.3 Write property test for predicted probabilities validity (Property 10)
    - **Property 10: Predicted Probabilities Are Valid**
    - **Validates: Requirements 4.4**
    - Use Hypothesis to generate valid feature matrices (no NaN, no inf, all features in valid ranges); assert `predict_proba` output has all values in [0, 1] and each row sums to 1.0 within 1e-9

- [x] 14. ML model predictor
  - Implement `model/predictor.py` with `FloodRiskPredictor` class:
    - `predict(X, model)` that calls `model.predict_proba(X)` and returns the probability of the high-risk class
    - `get_feature_importance(model, feature_names)` returning a `dict[str, float]` sorted by importance descending
  - _Requirements: 4.3, 4.4_

- [x] 15. Risk scorer — normalization and classification
  - Implement `scoring/scorer.py` with `FloodRiskScorer` class:
    - `normalize_scores(raw_probabilities, p_min, p_max)` applying min-max scaling: `(p - p_min) / (p_max - p_min) * 100`; clip output to [0, 100]; store `p_min` and `p_max` as instance attributes calibrated from training distribution (1st and 99th percentile)
    - `classify(scores, thresholds)` applying threshold rules: `<= low_max → "Low"`, `<= medium_max → "Medium"`, `> medium_max → "High"`; return `np.ndarray` of string labels
    - `score_grid(grid, model, feature_columns)` orchestrating predict → normalize → classify and appending `risk_score` and `risk_class` columns to the grid GeoDataFrame
  - _Requirements: 5.1, 5.2, 5.3_

  - [x] 15.1 Write unit tests for scorer
    - Test `classify` with known values: `[0.0, 33.0, 33.1, 66.0, 66.1, 100.0]` → `["Low", "Low", "Medium", "Medium", "High", "High"]`
    - Test `normalize_scores` maps min input to 0 and max input to 100
    - Test `score_grid` appends `risk_score` and `risk_class` columns with no NaN values
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 15.2 Write property test for risk score normalization bounds (Property 11)
    - **Property 11: Risk Score Normalization Bounds**
    - **Validates: Requirements 5.1**
    - Use Hypothesis `st.arrays(dtype=float, shape=st.integers(1, 200), elements=st.floats(0.0, 1.0))`; assert all output values are in [0, 100]; assert min output is 0 and max is 100 when input spans the calibration range

  - [x] 15.3 Write property test for risk classification threshold correctness (Property 12)
    - **Property 12: Risk Classification Threshold Correctness**
    - **Validates: Requirements 5.2, 5.3**
    - Use Hypothesis `score=st.floats(0.0, 100.0)`, `low_max=st.floats(1.0, 49.0)`, `medium_max=st.floats(51.0, 99.0)` with `assume(low_max < medium_max)`; assert exactly one label is produced per score matching the threshold rules

- [x] 16. Checkpoint — ML pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Pipeline orchestrator
  - Implement `pipeline.py` with `FloodRiskPipeline` class:
    - `__init__(config)` storing config and initializing component instances
    - `run(bounding_box)` executing the full DAG: validate → generate grid → ingest data (with cache check) → extract features → train model → score grid → return `FloodRiskResult`
    - `run_stage(stage_name, *args, **kwargs)` for incremental execution
    - Use `config.random_seed` for all random operations to ensure determinism
    - Implement the three-tier fallback strategy (Tier 1 real data → Tier 2 partial → Tier 3 synthetic) with logging
    - Cache intermediate `GeoDataFrame` results using `utils/cache.py` when `config.use_cache=True`
  - _Requirements: 7.1, 7.2, 7.3_

  - [x] 17.1 Write integration test for full pipeline with synthetic data
    - Test `FloodRiskPipeline.run` with a small bounding box (0.5° × 0.5°), `cell_size_meters=1000`, `use_cache=False`
    - Assert `result.cell_count > 0`
    - Assert `risk_score` column exists with all values in [0, 100]
    - Assert `risk_class` column exists with all values in `{"Low", "Medium", "High"}`
    - Assert `result.training_result.mean_cv_auc >= 0.70`
    - _Requirements: 7.1_

  - [x] 17.2 Write property test for pipeline output completeness (Property 15)
    - **Property 15: Pipeline Output Completeness**
    - **Validates: Requirements 7.1**
    - Use Hypothesis `valid_bounding_boxes` strategy with synthetic data mode; assert every cell in `scored_grid` has non-null `risk_score` in [0, 100] and non-null `risk_class` in `{"Low", "Medium", "High"}`

  - [x] 17.3 Write property test for pipeline determinism (Property 16)
    - **Property 16: Pipeline Determinism**
    - **Validates: Requirements 7.2**
    - Use Hypothesis `valid_bounding_boxes` strategy; run pipeline twice with identical `random_seed`; assert `scored_grid` DataFrames are identical row-by-row for `risk_score` and `risk_class`

- [x] 18. Visualization — Folium map builder and layer construction
  - Implement `visualization/map_builder.py` with `FloodRiskMapBuilder` class:
    - `RISK_COLOR_MAP = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}`
    - `build_choropleth_map(scored_grid, center, zoom_start, use_google_maps, google_maps_api_key)` constructing a `folium.Map` with:
      - Risk classification choropleth layer (color-coded polygons using `RISK_COLOR_MAP`)
      - Risk score continuous heatmap layer
      - Population density overlay
      - Water bodies overlay
      - `folium.LayerControl` toggle
    - Fall back to OpenStreetMap tiles if Google Maps API key is invalid, logging a warning
  - Implement `visualization/layers.py` with individual layer builder functions for each overlay
  - _Requirements: 6.1_

  - [x] 18.1 Write unit tests for map builder
    - Test `build_choropleth_map` returns a `folium.Map` instance
    - Test that the map contains a `LayerControl` element
    - Test that each risk class polygon uses the correct color from `RISK_COLOR_MAP`
    - _Requirements: 6.1_

  - [x] 18.2 Write property test for visualization color mapping correctness (Property 13)
    - **Property 13: Visualization Color Mapping Correctness**
    - **Validates: Requirements 6.1**
    - Use Hypothesis to generate scored GeoDataFrames with arbitrary `risk_class` assignments from `{"Low", "Medium", "High"}`; assert every cell's fill color in the choropleth exactly matches `RISK_COLOR_MAP[risk_class]`

- [x] 19. Visualization — popup layer and export
  - Implement `add_popup_layer(folium_map, grid)` in `visualization/map_builder.py` that adds click-to-inspect popups to each cell showing: risk score, risk class, elevation, slope, TWI, mean annual rainfall, distance to water, drainage capacity
  - Implement `visualization/export.py` with:
    - `export_html(folium_map, output_path)` saving the map as a self-contained HTML file
    - `export_geojson(scored_grid, output_path)` saving the scored GeoDataFrame as GeoJSON
    - `export_csv(scored_grid, output_path)` saving feature columns and risk outputs as CSV
  - _Requirements: 6.2, 6.3_

  - [x] 19.1 Write unit tests for popup content and export
    - Test `add_popup_layer` popup HTML contains all required fields: risk score, risk class, elevation, slope, TWI, rainfall, distance to water, drainage capacity
    - Test `export_html` creates a file at the specified path
    - Test `export_geojson` creates a valid GeoJSON file with all grid cells
    - Test `export_csv` creates a CSV with the correct column headers
    - _Requirements: 6.2, 6.3_

  - [x] 19.2 Write property test for popup content completeness (Property 14)
    - **Property 14: Popup Content Completeness**
    - **Validates: Requirements 6.2**
    - Use Hypothesis to generate grid cells with all feature columns populated; assert popup HTML contains all 8 required fields and is non-empty for every cell

- [x] 20. Checkpoint — visualization and export complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 21. Streamlit web application
  - Implement `app.py` with the full Streamlit UI layout (design §3.9):
    - **Sidebar**: region selection (bounding box inputs or city name lookup via geocoding), grid resolution selector (250m / 500m / 1000m), rainfall scenario selector (Historical / Custom mm), model selector (Random Forest / LightGBM), classification threshold sliders (Low/Medium/High), "Run Analysis" button
    - **Tab 1 — Interactive Map**: render the Folium choropleth map using `streamlit_folium` or `st.components.v1.html`
    - **Tab 2 — Risk Statistics**: bar chart and pie chart of cell counts per risk class using Matplotlib/Seaborn
    - **Tab 3 — Feature Importance**: horizontal bar chart of `TrainingResult.feature_importances`
    - **Tab 4 — Data Table**: filterable `st.dataframe` of the scored grid with download buttons for GeoJSON and CSV export
    - **Tab 5 — Methodology**: static markdown documentation of the pipeline methodology
    - Display active data tier (Tier 1/2/3) and pipeline duration in the UI
    - Catch all `FloodRiskError` subclasses and display user-friendly error messages via `st.error`
  - _Requirements: 6.1, 6.2, 6.3, 7.1_

  - [x] 21.1 Write integration test for Streamlit app pipeline invocation
    - Test that calling `FloodRiskPipeline.run` from the app's analysis function with a valid bounding box returns a `FloodRiskResult` without raising exceptions
    - Test that `FloodRiskError` subclasses are caught and do not propagate as unhandled exceptions
    - _Requirements: 7.1_

- [x] 22. Integration test — model training AUC threshold
  - Implement `tests/integration/test_model_training.py`:
    - `test_model_achieves_minimum_auc_on_synthetic_data`: generate 500 synthetic training samples, train with 5-fold CV, assert `mean_cv_auc >= 0.70`
    - `test_model_feature_importances_sum_to_one`: assert feature importance values sum to approximately 1.0
  - _Requirements: 4.1, 4.3_

- [x] 23. Integration test — export pipeline
  - Implement `tests/integration/test_export.py`:
    - `test_html_export_creates_valid_file`: run pipeline on small bbox, export HTML, assert file exists and contains `<html>` tag
    - `test_geojson_export_is_valid`: export GeoJSON, parse with `json.loads`, assert `type == "FeatureCollection"` and `features` list is non-empty
    - `test_csv_export_has_correct_columns`: export CSV, read with pandas, assert all `FEATURE_COLUMNS` plus `risk_score` and `risk_class` are present
  - _Requirements: 6.3_

- [x] 24. Final checkpoint — all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Property tests use Hypothesis with `@settings(max_examples=100)` minimum; CI should use `max_examples=500`
- All 17 correctness properties from the design document are covered by property test sub-tasks (Properties 1–17)
- The implementation language is Python throughout, matching the design document
- Synthetic data generation (Tier 3) must be fully functional before real data ingestion is attempted, enabling offline development and testing
- The `random_seed` in `PipelineConfig` must be threaded through all stochastic operations (model training, synthetic data generation) to guarantee pipeline determinism (Property 16)
- Run tests with: `pytest tests/ -v --hypothesis-seed=0`
- Run property tests only: `pytest tests/property/ -v`
- Run with coverage: `pytest tests/ --cov=flood_risk_zonation --cov-report=html`
