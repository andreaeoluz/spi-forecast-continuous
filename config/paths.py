"""paths.py - Project path management."""

import sys
from pathlib import Path
from typing import Dict, Optional

IS_WINDOWS = sys.platform == "win32"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if IS_WINDOWS:
    DATA_BASE_PATH = Path("C:/Users/User/Desktop/projects_tese/rasters")
else:
    DATA_BASE_PATH = Path("/home/andrea/projects/rasters")


def get_data_path() -> Path:
    """Return the raw raster data directory."""
    return DATA_BASE_PATH


def get_output_path() -> Path:
    """Return the base output directory."""
    return PROJECT_ROOT / "outputs"


def get_paths(
    region: str,
    spi_scale: Optional[int] = None,
    p: Optional[int] = None,
    q: Optional[int] = None,
    model_type: Optional[str] = None,
    split: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Build and create the directory layout for a region/experiment.

    Args:
        region: Region name (matches the raw data folder name).
        spi_scale: SPI accumulation scale (e.g. 3 for SPI-3, 12 for SPI-12).
            Used only to namespace outputs by scale, exactly like the binary
            framework namespaced them by drought threshold.
        p: Historical window length.
        q: Forecast horizon.
        model_type: 'pretrained' or 'scratch'.
        split: 'test' or 'validation'.

    Returns:
        Dict mapping logical names to paths.
    """
    output_dir = get_output_path() / region
    scale_str = f"spi{int(spi_scale)}" if spi_scale is not None else "spi3"

    paths: Dict[str, Optional[Path]] = {
        "base": output_dir,
        "output_dir": output_dir,
        "autoencoder_dir": output_dir / "autoencoder",
        "spi_cache_dir": output_dir / "spi_cache",
        "analysis_dir": output_dir / "analysis",
        "grid_search_dir": output_dir / "grid_search",
        "grid_search_results": output_dir / "grid_search" / "results",
    }

    grid_base = output_dir / "grid_search" / scale_str / "models"
    paths["grid_search_pretrained"] = grid_base / "pretrained"
    paths["grid_search_scratch"] = grid_base / "scratch"

    if p is not None and q is not None and model_type is not None:
        inf_dir = output_dir / "inferences" / scale_str / f"p{p}_q{q}" / model_type
        split_name = split if split else "test"
        paths["inference_dir"] = inf_dir / split_name
        paths["inference_pred"] = paths["inference_dir"] / "pred"
        paths["inference_truth"] = paths["inference_dir"] / "truth"

        analysis_dir = output_dir / "analysis" / scale_str / f"p{p}_q{q}" / model_type
        paths["analysis_metrics"] = analysis_dir / "metrics"
        paths["analysis_spatial"] = analysis_dir / "spatial"
        paths["analysis_examples"] = analysis_dir / "examples"

    essential_keys = [
        "base", "output_dir", "autoencoder_dir", "spi_cache_dir",
        "analysis_dir", "grid_search_dir", "grid_search_results",
        "grid_search_pretrained", "grid_search_scratch"
    ]

    for key in essential_keys:
        if key in paths and paths[key] is not None:
            paths[key].mkdir(parents=True, exist_ok=True)

    for key in ["analysis_metrics", "analysis_spatial", "analysis_examples"]:
        if key in paths and paths[key] is not None:
            paths[key].mkdir(parents=True, exist_ok=True)

    paths["models_dir"] = paths["autoencoder_dir"]
    paths["results_dir"] = paths["grid_search_results"]
    paths["analysis"] = paths["analysis_dir"]

    return paths  # type: ignore


def find_model_path(
    region: str,
    p: int,
    q: int,
    model_type: str,
    spi_scale: Optional[int] = None,
) -> Optional[Path]:
    """
    Locate a trained model checkpoint on disk.

    Args:
        region: Region name.
        p: Historical window length.
        q: Forecast horizon.
        model_type: 'pretrained' or 'scratch'.
        spi_scale: SPI accumulation scale.

    Returns:
        Path to the checkpoint, or None if not found.
    """
    paths = get_paths(region, spi_scale=spi_scale)
    model_filename = f"model_p{p}_q{q}.pth"

    candidates = []

    key = "grid_search_pretrained" if model_type == "pretrained" else "grid_search_scratch"
    if key in paths and paths[key] is not None:
        candidates.append(paths[key] / model_filename)

    gs_dir = paths.get("grid_search_dir")
    if gs_dir is not None:
        candidates.append(gs_dir / model_type / model_filename)

    ae_dir = paths.get("autoencoder_dir")
    if ae_dir is not None:
        candidates.append(ae_dir / model_filename)

    for path in candidates:
        if path is not None and path.exists():
            return path

    return None
