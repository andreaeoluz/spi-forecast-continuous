"""loader.py - Climate raster loading."""

import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from typing import Dict

from config import ExperimentConfig
from .preprocessing import downsample_block_mean, build_valid_mask


def load_region_timeseries(
    base_path: Path,
    config: ExperimentConfig,
) -> Dict:
    """
    Load the complete climate time series for a region.

    The downsampling factor is region-specific, based on
    config.data.downsample_config.

    Args:
        base_path: Base directory containing the region raster folders.
        config: Experiment configuration.

    Returns:
        Dict with data, years, months, metadata, and valid_mask.
    """
    region = config.region
    num_bands = config.num_bands

    factor_h, factor_w = config.get_downsample(region)

    region_path = base_path / region
    # Sort chronologically by the (year, month) parsed from the file name,
    # not lexicographically: a non-zero-padded month ("_1980_10" before
    # "_1980_2") would otherwise silently scramble the time axis.
    files = sorted(
        region_path.glob(f"{region}_*.tif"),
        key=lambda f: (int(f.stem.split("_")[-2]), int(f.stem.split("_")[-1])),
    )

    if not files:
        raise FileNotFoundError(f"No files found in {region_path}")

    data_list = []
    years = []
    months = []
    metadata = None

    print(f"\n📂 Loading time series for region: {region}")
    print(f"   Files found: {len(files)}")
    print(f"   Downsampling: {factor_h}x{factor_w}")

    ds_info = config.get_downsample_info(region)
    print(f"   {ds_info['preservation_estimate']}")

    for f in files:
        parts = f.stem.split("_")
        year = int(parts[-2])
        month = int(parts[-1])

        with rasterio.open(f) as src:
            arr = src.read().astype(np.float32)
            arr = np.transpose(arr, (1, 2, 0))

            if arr.shape[-1] != num_bands:
                raise ValueError(
                    f"Bands in {f.name}: {arr.shape[-1]} != {num_bands}"
                )

            arr_ds = downsample_block_mean(arr, factor_h, factor_w)

            if metadata is None:
                metadata = {
                    "crs": src.crs,
                    "transform": Affine(
                        src.transform.a * factor_w,
                        src.transform.b,
                        src.transform.c,
                        src.transform.d,
                        src.transform.e * factor_h,
                        src.transform.f
                    ),
                    "height": arr_ds.shape[0],
                    "width": arr_ds.shape[1],
                }

        data_list.append(arr_ds)
        years.append(year)
        months.append(month)

    # Every temporal index in the pipeline assumes consecutive months.
    ym = np.array(years) * 12 + np.array(months) - 1
    gaps = np.where(np.diff(ym) != 1)[0]
    if len(gaps) > 0:
        i = gaps[0]
        raise ValueError(
            f"Non-consecutive monthly rasters in {region_path}: "
            f"{years[i]}-{months[i]:02d} is followed by {years[i + 1]}-{months[i + 1]:02d}"
        )

    data_stack = np.stack(data_list, axis=0)

    print("\n📊 Data loaded:")
    print(f"   Shape: {data_stack.shape}")
    print(f"   Timesteps: {data_stack.shape[0]}")
    print(f"   Height: {data_stack.shape[1]}")
    print(f"   Width: {data_stack.shape[2]}")
    print(f"   Bands: {data_stack.shape[3]}")

    # The mask is derived from the training period only (train_gs) and then
    # frozen for validation and test, so no evaluation-period data enters
    # this preprocessing decision. (In this dataset each pixel is either
    # finite in every month or almost never, so the result is identical to
    # a full-record mask; restricting it keeps the protocol leakage-free.)
    train_end = config.split.ym_to_int(config.split.train_gs[1])
    train_idx = ym <= train_end
    valid_mask = build_valid_mask(data_stack[train_idx], min_valid_ratio=config.data.min_valid_ratio)

    total_pixels = valid_mask.size
    valid_pixels = valid_mask.sum()

    print("\n📊 Validity mask:")
    print(f"   Total pixels: {total_pixels:,}")
    print(f"   Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
    print(f"   Invalid pixels: {total_pixels - valid_pixels:,} ({(total_pixels - valid_pixels)/total_pixels:.1%})")

    return {
        "data": data_stack.astype(np.float32),
        "years": np.array(years),
        "months": np.array(months),
        "metadata": metadata,
        "valid_mask": valid_mask,
    }
