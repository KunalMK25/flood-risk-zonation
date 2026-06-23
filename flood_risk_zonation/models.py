"""
Core data model dataclasses for the Flood Risk Zonation System.

These are plain dataclasses (no validation logic) that act as typed
containers passed between pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class RasterDataset:
    """
    A single-band raster dataset loaded from a GeoTIFF or similar source.

    Attributes
    ----------
    array : np.ndarray
        2-D float32 array of raster values.
    transform : Any
        Rasterio Affine transform mapping pixel coordinates to CRS coordinates.
    crs : Any
        PyProj CRS object describing the coordinate reference system.
    nodata : Optional[float]
        Sentinel value representing missing / no-data pixels.
    source : str
        Human-readable provenance string (e.g. file path or dataset name).
    """

    array: np.ndarray
    transform: Any  # rasterio.transform.Affine
    crs: Any        # pyproj.CRS
    nodata: Optional[float]
    source: str


@dataclass
class RainfallDataset:
    """
    Gridded rainfall statistics derived from GPM IMERG or IMD data.

    Attributes
    ----------
    mean_annual_mm : np.ndarray
        2-D array of mean annual rainfall in millimetres.
    max_24h_mm : np.ndarray
        2-D array of maximum 24-hour rainfall in millimetres.
    transform : Any
        Rasterio Affine transform.
    crs : Any
        PyProj CRS object.
    temporal_range : tuple
        (start_date, end_date) of the underlying data record.
    source : str
        Provenance string, e.g. "IMD", "NASA_GPM", or "synthetic".
    """

    mean_annual_mm: np.ndarray
    max_24h_mm: np.ndarray
    transform: Any
    crs: Any
    temporal_range: tuple
    source: str


@dataclass
class DrainageDataset:
    """
    Per-cell synthetic drainage capacity scores.

    Attributes
    ----------
    capacity_scores : np.ndarray
        1-D array of drainage capacity scores in [0, 1] — one per grid cell.
        1.0 = excellent drainage; 0.0 = no drainage capacity.
    cell_ids : list[str]
        Ordered list of cell identifiers matching capacity_scores.
    """

    capacity_scores: np.ndarray  # per-cell scores in [0, 1]
    cell_ids: list[str]


@dataclass
class AnalysisResult:
    """Metadata and artefacts produced by a susceptibility analysis."""

    model: Any
    feature_names: list[str]
    feature_importances: dict[str, float]
    method: str
    validation_note: str
    # CV metrics — populated for RF and Ensemble methods
    mean_cv_auc: Optional[float] = None
    mean_cv_f1: Optional[float] = None
    mean_cv_accuracy: Optional[float] = None
    mean_cv_precision: Optional[float] = None
    mean_cv_recall: Optional[float] = None
    cv_auc_scores: Optional[list] = None
    cv_f1_scores: Optional[list] = None
    cv_accuracy_scores: Optional[list] = None
    cv_precision_scores: Optional[list] = None
    cv_recall_scores: Optional[list] = None


@dataclass
class TrainingResult:
    """
    Artefacts produced by a completed model training run.

    Attributes
    ----------
    model : Any
        Fitted scikit-learn or LightGBM estimator.
    feature_names : list[str]
        Ordered list of feature column names used during training.
    feature_importances : dict[str, float]
        Mapping of feature name → importance score (sums to ~1.0 for RF).
    cv_scores : dict[str, list[float]]
        Per-fold scores keyed by metric name, e.g. {"auc": [...], "f1": [...]}.
    mean_cv_auc : float
        Mean AUC-ROC across all cross-validation folds.
    mean_cv_f1 : float
        Mean F1 score across all cross-validation folds.
    training_timestamp : Any
        datetime object recording when training completed.
    """

    model: Any
    feature_names: list[str]
    feature_importances: dict[str, float]
    cv_scores: dict[str, list[float]]
    mean_cv_auc: float
    mean_cv_f1: float
    training_timestamp: Any  # datetime


@dataclass
class FloodRiskResult:
    """
    Complete output of a flood risk pipeline run.

    Attributes
    ----------
    scored_grid : Any
        gpd.GeoDataFrame with all grid cells, feature columns, risk_score,
        and risk_class populated.
    training_result : TrainingResult
        Model training artefacts from this run.
    bounding_box : BoundingBox
        Geographic extent that was analysed.
    config : PipelineConfig
        Configuration used for this run.
    pipeline_duration_seconds : float
        Wall-clock time for the full pipeline execution.
    cell_count : int
        Total number of grid cells in scored_grid.
    """

    scored_grid: Any  # gpd.GeoDataFrame
    analysis_result: AnalysisResult
    bounding_box: Any  # BoundingBox — avoid circular import at module level
    config: Any        # PipelineConfig
    pipeline_duration_seconds: float
    cell_count: int
    data_provenance: dict[str, str] = field(default_factory=dict)
    data_tier: int = 3

    @property
    def training_result(self) -> AnalysisResult:
        """Backward-compatible alias for callers using the pre-0.2 API."""
        return self.analysis_result

    @property
    def risk_distribution(self) -> dict[str, int]:
        """Return count of cells per risk class."""
        return self.scored_grid["risk_class"].value_counts().to_dict()

    @property
    def high_risk_cells(self) -> Any:
        """Return a GeoDataFrame containing only High-risk cells."""
        return self.scored_grid[self.scored_grid["risk_class"] == "High"]
