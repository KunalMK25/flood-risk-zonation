"""
Elevation data ingestion for the Flood Risk Zonation System.

Provides loaders for real SRTM GeoTIFF files and synthetic DEM generation
for demo/test mode using Perlin noise.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.exceptions import DataIngestionError
from flood_risk_zonation.models import RasterDataset

logger = logging.getLogger(__name__)


def load_elevation(bounding_box: BoundingBox, data_dir: Path | str) -> RasterDataset:
    """
    Load a SRTM elevation GeoTIFF clipped to the bounding box.

    Searches data_dir for any .tif or .tiff file. Raises DataIngestionError
    if no file is found or the bounding box falls outside coverage.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent to clip to.
    data_dir : Path | str
        Directory containing SRTM GeoTIFF files.

    Returns
    -------
    RasterDataset
        Elevation raster clipped to the bounding box.

    Raises
    ------
    DataIngestionError
        If no GeoTIFF file is found or the bbox is outside coverage.
    """
    data_dir = Path(data_dir)
    tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

    if not tif_files:
        raise DataIngestionError(
            f"No SRTM GeoTIFF files found in {data_dir}. "
            "Use generate_synthetic_elevation() for demo/test mode."
        )

    tif_path = tif_files[0]
    try:
        with rasterio.open(tif_path) as src:
            # Check coverage
            bounds = src.bounds
            if (bounding_box.max_lon < bounds.left or bounding_box.min_lon > bounds.right
                    or bounding_box.max_lat < bounds.bottom or bounding_box.min_lat > bounds.top):
                raise DataIngestionError(
                    f"Bounding box {bounding_box} falls outside SRTM coverage {bounds}."
                )

            from rasterio.windows import from_bounds as window_from_bounds
            window = window_from_bounds(
                bounding_box.min_lon, bounding_box.min_lat,
                bounding_box.max_lon, bounding_box.max_lat,
                src.transform,
            )
            array = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
            crs = src.crs
            nodata = src.nodata

    except rasterio.errors.RasterioIOError as exc:
        raise DataIngestionError(f"Failed to read {tif_path}: {exc}") from exc

    return RasterDataset(
        array=array,
        transform=transform,
        crs=crs,
        nodata=nodata,
        source=str(tif_path),
    )


def resample_raster(raster_dataset: RasterDataset, target_resolution_m: float) -> RasterDataset:
    """
    Resample a RasterDataset to a target resolution using bilinear interpolation.

    Parameters
    ----------
    raster_dataset : RasterDataset
        Source raster to resample.
    target_resolution_m : float
        Target resolution in metres.

    Returns
    -------
    RasterDataset
        Resampled raster at the target resolution.
    """
    src_array = raster_dataset.array
    src_transform = raster_dataset.transform
    src_height, src_width = src_array.shape

    # Current pixel size in degrees (approximate)
    pixel_size_deg = abs(src_transform.a)
    # Convert target resolution to degrees (approximate at equator)
    target_deg = target_resolution_m / 111_320.0

    scale_factor = pixel_size_deg / target_deg
    new_height = max(1, int(src_height * scale_factor))
    new_width = max(1, int(src_width * scale_factor))

    dst_array = np.empty((new_height, new_width), dtype=np.float32)

    # Compute new transform
    left = src_transform.c
    top = src_transform.f
    right = left + src_width * src_transform.a
    bottom = top + src_height * src_transform.e

    new_transform = from_bounds(left, bottom, right, top, new_width, new_height)

    rasterio.warp.reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=raster_dataset.crs,
        dst_transform=new_transform,
        dst_crs=raster_dataset.crs,
        resampling=Resampling.bilinear,
    )

    return RasterDataset(
        array=dst_array,
        transform=new_transform,
        crs=raster_dataset.crs,
        nodata=raster_dataset.nodata,
        source=raster_dataset.source,
    )


def generate_synthetic_elevation(
    bounding_box: BoundingBox,
    resolution_m: float = 30.0,
    base_elevation_m: float = 50.0,
    relief_m: float = 100.0,
    seed: int = 42,
) -> RasterDataset:
    """
    Generate a synthetic DEM using random noise smoothed with a Gaussian filter
    to simulate realistic terrain with valleys, ridges, and flat plains.

    Falls back gracefully if the 'noise' library is not installed.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent.
    resolution_m : float
        Pixel resolution in metres.
    base_elevation_m : float
        Mean elevation of the synthetic terrain.
    relief_m : float
        Peak-to-peak elevation range.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    RasterDataset
        Synthetic elevation raster.
    """
    from scipy.ndimage import gaussian_filter

    deg_per_m = 1.0 / 111_320.0
    resolution_deg = resolution_m * deg_per_m

    width_deg = bounding_box.max_lon - bounding_box.min_lon
    height_deg = bounding_box.max_lat - bounding_box.min_lat

    ncols = max(2, int(width_deg / resolution_deg))
    nrows = max(2, int(height_deg / resolution_deg))

    rng = np.random.default_rng(seed)
    raw = rng.random((nrows, ncols)).astype(np.float32)
    # Smooth to create realistic terrain
    smoothed = gaussian_filter(raw, sigma=max(1, min(nrows, ncols) // 10))
    # Scale to [base, base + relief]
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-9)
    array = (base_elevation_m + smoothed * relief_m).astype(np.float32)

    transform = from_bounds(
        bounding_box.min_lon, bounding_box.min_lat,
        bounding_box.max_lon, bounding_box.max_lat,
        ncols, nrows,
    )

    from rasterio.crs import CRS
    crs = CRS.from_epsg(4326)

    return RasterDataset(
        array=array,
        transform=transform,
        crs=crs,
        nodata=None,
        source="synthetic",
    )
