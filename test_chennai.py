import logging
logging.basicConfig(level=logging.INFO)
from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.pipeline import FloodRiskPipeline
bbox = BoundingBox(min_lon=80.24, min_lat=12.98, max_lon=80.31, max_lat=13.05)
config = PipelineConfig(cell_size_meters=500, rf_n_estimators=50, cv_folds=3, use_cache=False)
pipeline = FloodRiskPipeline(config)
result = pipeline.run(bbox)
dist = result.risk_distribution
print("Distribution:", dist)
print("Elevation range:", result.scored_grid.elevation_m.min(), "-", result.scored_grid.elevation_m.max())
print("Water cells:", dist.get("Water", 0))
