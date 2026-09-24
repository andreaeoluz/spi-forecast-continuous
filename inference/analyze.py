"""analyze.py - Analysis of continuous SPI inference results.

Derived from the binary framework's inference/analyze.py. Removed:
confusion matrices/maps (TP/FP/FN), CSI maps, calibration curves/ECE
(there is no probability to calibrate), and drought-event-area timeseries.
Replaced with continuous-SPI-appropriate diagnostics: per-timestep
WI/RMSE/MAE, spatial mean-error (bias) and RMSE maps, observed/predicted/
error map panels for the best- and worst-agreement timesteps, all using a
consistent diverging SPI color scale ([-3, 3] by default).

Usage (build the path from region/p/q/model-type, matching config.paths):
    python -m inference.analyze --region Sul --p 3 --q 1 --model-type pretrained

Usage (point directly at a "test"-style directory containing pred/, truth/):
    python -m inference.analyze --pred-dir outputs/Sul/inferences/spi3/p3_q1/pretrained/test
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from evaluation.metrics import compute_regression_metrics

SPI_VMIN, SPI_VMAX = -3.0, 3.0


def run_analysis(
    pred_dir: Optional[Path] = None,
    valid_mask_path: Optional[Path] = None,
    generate_spatial: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Run analysis on continuous SPI inference results.

    Args:
        pred_dir: Directory containing prediction results (pred/, truth/).
        valid_mask_path: Path to a validity mask (optional).
        generate_spatial: Whether to generate spatial figures.

    Returns:
        Tuple of (metrics DataFrame, output directory).
    """
    print("\n" + "=" * 70)
    print("ANALYZING CONTINUOUS SPI INFERENCE RESULTS")
    print("=" * 70)

    if pred_dir is None:
        pred_dir = _find_prediction_dir()
        if pred_dir is None:
            print("Prediction directory not found")
            print("   Use --pred-dir, or --region/--p/--q/--model-type, to specify it")
            return None, None

    print(f"\nAnalyzing: {pred_dir}")

    pred_path = pred_dir / "pred"
    truth_path = pred_dir / "truth"

    if not pred_path.exists() or not truth_path.exists():
        print(f"pred/ or truth/ not found in {pred_dir}")
        return None, None

    pred_stack, dates = _load_raster_stack(pred_path)
    obs_stack, _ = _load_raster_stack(truth_path)

    if pred_stack is None or obs_stack is None:
        print("Error loading stacks")
        return None, None

    T = min(len(pred_stack), len(obs_stack))
    pred_stack, obs_stack = pred_stack[:T], obs_stack[:T]
    dates = dates[:T]

    print(f"  Timesteps: {T}")
    print(f"  Grid: {pred_stack.shape[1]} x {pred_stack.shape[2]}")

    sample_file = sorted(truth_path.glob("*.tif"))[0]
    valid_mask = _load_valid_mask(sample_file, valid_mask_path)
    print(f"  Valid pixels: {valid_mask.sum():,} / {valid_mask.size:,} "
          f"({100 * valid_mask.sum() / valid_mask.size:.1f}%)")

    metrics_list = []
    mean_obs, mean_pred = [], []

    for t in range(T):
        m = compute_regression_metrics(pred_stack[t], obs_stack[t], mask=valid_mask)
        metrics_list.append({
            "date": dates[t],
            "wi": m["wi"],
            "rmse": m["rmse"],
            "mae": m["mae"],
            "n_valid": m["n_valid"],
        })
        mean_obs.append(float(np.nanmean(obs_stack[t][valid_mask])))
        mean_pred.append(float(np.nanmean(pred_stack[t][valid_mask])))

    df = pd.DataFrame(metrics_list)

    out_dir = pred_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "metrics_timeseries.csv", index=False)
    print(f"  Metrics saved: {out_dir / 'metrics_timeseries.csv'}")

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  WI:   {df['wi'].mean():.4f} (+/-{df['wi'].std():.4f})")
    print(f"  RMSE: {df['rmse'].mean():.4f} (+/-{df['rmse'].std():.4f})")
    print(f"  MAE:  {df['mae'].mean():.4f} (+/-{df['mae'].std():.4f})")
    print(f"\n  Best WI:  {df['wi'].max():.4f} ({df.loc[df['wi'].idxmax(), 'date']})")
    print(f"  Worst WI: {df['wi'].min():.4f} ({df.loc[df['wi'].idxmin(), 'date']})")

    # Overall metrics computed over the full pooled sample (not the mean of
    # per-timestep metrics, which is a different - looser - aggregation).
    overall = compute_regression_metrics(pred_stack, obs_stack, mask=valid_mask)

    summary: Dict[str, Any] = {
        "directory": str(pred_dir),
        "n_timesteps": T,
        "valid_pixels": int(valid_mask.sum()),
        "overall_metrics": {"wi": overall["wi"], "rmse": overall["rmse"], "mae": overall["mae"]},
        "per_timestep_metrics": {
            m: {"mean": float(df[m].mean()), "std": float(df[m].std())}
            for m in ["wi", "rmse", "mae"]
        },
        "best_wi": {"date": str(df.loc[df['wi'].idxmax(), 'date']), "value": float(df['wi'].max())},
        "worst_wi": {"date": str(df.loc[df['wi'].idxmin(), 'date']), "value": float(df['wi'].min())},
    }

    if generate_spatial:
        try:
            lon, lat = _get_geo_coords(sample_file)
        except Exception as exc:
            print(f"  (skipping spatial outputs: could not read geo-coordinates: {exc})")
            lon = lat = None

        if lon is not None:
            mean_obs_map = np.full(obs_stack[0].shape, np.nan, dtype=np.float64)
            mean_pred_map = np.full(pred_stack[0].shape, np.nan, dtype=np.float64)
            bias_map = np.full(obs_stack[0].shape, np.nan, dtype=np.float64)
            rmse_map = np.full(obs_stack[0].shape, np.nan, dtype=np.float64)

            obs_valid = obs_stack[:, valid_mask]
            pred_valid = pred_stack[:, valid_mask]
            error_valid = pred_valid - obs_valid

            mean_obs_map[valid_mask] = np.nanmean(obs_valid, axis=0)
            mean_pred_map[valid_mask] = np.nanmean(pred_valid, axis=0)
            bias_map[valid_mask] = np.nanmean(error_valid, axis=0)
            rmse_map[valid_mask] = np.sqrt(np.nanmean(error_valid ** 2, axis=0))

            print("  Saving spatial GeoTIFFs and figures...")
            _save_map_as_geotiff(mean_obs_map, sample_file, out_dir / "mean_observed_spi.tif")
            _save_map_as_geotiff(mean_pred_map, sample_file, out_dir / "mean_predicted_spi.tif")
            _save_map_as_geotiff(bias_map, sample_file, out_dir / "bias_map.tif")
            _save_map_as_geotiff(rmse_map, sample_file, out_dir / "rmse_map.tif")

            _plot_regional_mean_timeseries(dates, mean_obs, mean_pred, out_dir / "regional_mean_spi_timeseries.png")
            _plot_observed_vs_predicted_maps(mean_obs_map, mean_pred_map, lon, lat, out_dir / "mean_spi_maps.png")
            _plot_error_map(bias_map, lon, lat, out_dir / "bias_map.png")
            _plot_rmse_map(rmse_map, lon, lat, out_dir / "rmse_map.png")
            _plot_skill_timeseries(df, out_dir / "skill_timeseries.png")
            _plot_example_predictions(obs_stack, pred_stack, dates, df, lon, lat,
                                       out_dir / "best_predictions.png", n=3, best=True)
            _plot_example_predictions(obs_stack, pred_stack, dates, df, lon, lat,
                                       out_dir / "worst_predictions.png", n=3, best=False)

    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis complete! Results saved to: {out_dir}")
    print("=" * 70)

    return df, out_dir


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_prediction_dir() -> Optional[Path]:
    """Auto-discover a prediction directory under the project's outputs/ root."""
    base_paths = [
        Path("outputs"),
    ]

    for base in base_paths:
        if not base.exists():
            continue
        for region_dir in sorted(base.iterdir()):
            if not region_dir.is_dir():
                continue
            inf_dir = region_dir / "inferences"
            if not inf_dir.exists():
                continue
            for scale_dir in sorted(inf_dir.glob("spi*")):
                for p_dir in sorted(scale_dir.glob("p*_q*")):
                    for model_type in ("pretrained", "scratch"):
                        model_dir = p_dir / model_type
                        test_dir = model_dir / "test"
                        if test_dir.exists() and (test_dir / "pred").exists():
                            return test_dir
    return None


def _resolve_pred_dir(region: str, p: int, q: int, model_type: str,
                      spi_scale: Optional[int] = None, split: str = "test") -> Path:
    """Build the prediction directory for a given (region, p, q, model_type),
    using the same layout as config.paths.get_paths."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.paths import get_paths

    paths = get_paths(region, spi_scale=spi_scale, p=p, q=q, model_type=model_type, split=split)
    return paths["inference_dir"]


def _load_raster_stack(folder: Path) -> Tuple[Optional[np.ndarray], list]:
    """Load all rasters in a folder into a single stack, sorted by filename.
    Values are continuous SPI - no [0, 1] clipping (that was a binary-
    probability convention that does not apply here)."""
    try:
        import rasterio
    except ImportError:
        return None, []

    files = sorted(folder.glob("*.tif"))
    if not files:
        return None, []

    stack = []
    dates = []

    for f in files:
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            stack.append(data)

        parts = f.stem.split("_")
        if len(parts) >= 3:
            dates.append(f"{parts[1]}-{parts[2]}")
        else:
            dates.append(f.stem)

    return np.array(stack), dates


def _load_valid_mask(truth_file: Path, valid_mask_path: Optional[Path] = None) -> np.ndarray:
    """Load the validity mask, either from a dedicated file or from a truth raster's nodata."""
    try:
        import rasterio
    except ImportError:
        return np.ones((1, 1), dtype=bool)

    if valid_mask_path and valid_mask_path.exists():
        with rasterio.open(valid_mask_path) as src:
            return src.read(1).astype(bool)

    with rasterio.open(truth_file) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            return (data != nodata) & ~np.isnan(data)
        return ~np.isnan(data)


def _get_geo_coords(sample_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract longitude/latitude grids (height x width) from a GeoTIFF's transform."""
    import rasterio

    with rasterio.open(sample_file) as src:
        transform = src.transform
        width = src.width
        height = src.height

        left = transform[2]
        top = transform[5]
        right = left + transform[0] * width
        bottom = top + transform[4] * height

        lon_1d = np.linspace(left, right, width)
        lat_1d = np.linspace(top, bottom, height)
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    return lon_grid, lat_grid


def _save_map_as_geotiff(data: np.ndarray, reference_file: Path, output_path: Path) -> None:
    """Save a 2D array as a GeoTIFF, using a reference file for metadata."""
    import rasterio

    with rasterio.open(reference_file) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1)

    data_out = np.where(np.isnan(data), profile.get('nodata', -9999.0), data)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data_out.astype(np.float32), 1)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def _plot_regional_mean_timeseries(dates, obs_mean, pred_mean, out_path: Path) -> None:
    """Plot observed vs. predicted region-mean SPI over time."""
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    plt.figure(figsize=(14, 5))
    plt.plot(dates, obs_mean, label="Observed SPI", marker='o', markersize=3, linewidth=1)
    plt.plot(dates, pred_mean, label="Predicted SPI", marker='s', markersize=3, linewidth=1)
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.xlabel("Date")
    plt.ylabel("Region-mean SPI")
    plt.title("Region-Mean SPI: Observed vs. Predicted")
    plt.legend()
    plt.grid(alpha=0.3)

    if len(obs_mean) > 1 and len(pred_mean) > 1:
        corr, _ = pearsonr(obs_mean, pred_mean)
        plt.text(0.02, 0.98, f"Corr: {corr:.3f}",
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_observed_vs_predicted_maps(obs_map, pred_map, lon, lat, out_path: Path) -> None:
    """Plot the temporal-mean observed and predicted SPI maps, sharing a
    consistent diverging color scale so the two are directly comparable."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in zip(axes, [obs_map, pred_map],
                               ["Mean Observed SPI", "Mean Predicted SPI"]):
        im = ax.imshow(data, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       origin="upper", cmap="RdBu", vmin=SPI_VMIN, vmax=SPI_VMAX)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3, linestyle="--")
        plt.colorbar(im, ax=ax, fraction=0.046, label="SPI")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_error_map(bias_map, lon, lat, out_path: Path) -> None:
    """Plot the spatial mean prediction error (Predicted - Observed)."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    vmax = float(np.nanmax(np.abs(bias_map))) if np.isfinite(bias_map).any() else 1.0
    vmax = max(vmax, 1e-3)
    im = plt.imshow(bias_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    extent=[lon.min(), lon.max(), lat.min(), lat.max()], origin="upper")
    plt.title("Mean Prediction Error (Predicted - Observed SPI)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(alpha=0.3, linestyle="--")
    plt.colorbar(im, fraction=0.046, label="Error (SPI units)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_rmse_map(rmse_map, lon, lat, out_path: Path) -> None:
    """Plot the spatial (per-pixel, temporally-aggregated) RMSE map."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    vmax = float(np.nanmax(rmse_map)) if np.isfinite(rmse_map).any() else 1.0
    im = plt.imshow(rmse_map, vmin=0, vmax=vmax,
                    extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    origin="upper", cmap="viridis")
    plt.title("Spatial RMSE (SPI units)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(alpha=0.3, linestyle="--")
    plt.colorbar(im, fraction=0.046, label="RMSE")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_skill_timeseries(df_metrics: pd.DataFrame, out_path: Path) -> None:
    """Plot the temporal evolution of WI, RMSE, and MAE."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(df_metrics.index, df_metrics.wi, label="WI", marker='o', markersize=3, color='green')
    axes[0].set_xticks(range(len(df_metrics.date)))
    axes[0].set_xticklabels(df_metrics.date, rotation=45, ha='right')
    axes[0].set_ylabel("WI")
    axes[0].set_title("Willmott's Index of Agreement over time")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)

    axes[1].plot(df_metrics.index, df_metrics.rmse, label="RMSE", marker='x', markersize=3, color='purple')
    axes[1].plot(df_metrics.index, df_metrics.mae, label="MAE", marker='+', markersize=3, color='orange')
    axes[1].set_xticks(range(len(df_metrics.date)))
    axes[1].set_xticklabels(df_metrics.date, rotation=45, ha='right')
    axes[1].set_ylabel("SPI units")
    axes[1].set_title("RMSE / MAE over time")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_example_predictions(
    obs_stack, pred_stack, dates, df_metrics: pd.DataFrame, lon, lat,
    out_path: Path, n: int = 3, best: bool = True,
) -> None:
    """Plot observed / predicted / error SPI map panels for the n
    timesteps with the best (or worst) WI."""
    import matplotlib.pyplot as plt

    selected = df_metrics.nlargest(n, "wi") if best else df_metrics.nsmallest(n, "wi")
    n_plots = len(selected)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 3, figsize=(15, 4.5 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    for i, row in enumerate(selected.itertuples()):
        idx = dates.index(row.date) if row.date in dates else None
        if idx is None:
            continue

        obs_i, pred_i = obs_stack[idx], pred_stack[idx]
        error_i = pred_i - obs_i
        err_vmax = max(float(np.nanmax(np.abs(error_i))), 1e-3) if np.isfinite(error_i).any() else 1.0

        im0 = axes[i, 0].imshow(obs_i, cmap="RdBu", vmin=SPI_VMIN, vmax=SPI_VMAX, extent=extent, origin="upper")
        axes[i, 0].set_title(f"Observed SPI\n{row.date}")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

        im1 = axes[i, 1].imshow(pred_i, cmap="RdBu", vmin=SPI_VMIN, vmax=SPI_VMAX, extent=extent, origin="upper")
        axes[i, 1].set_title(f"Predicted SPI\n{row.date} (WI={row.wi:.3f})")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        im2 = axes[i, 2].imshow(error_i, cmap="RdBu_r", vmin=-err_vmax, vmax=err_vmax, extent=extent, origin="upper")
        axes[i, 2].set_title(f"Error (Pred - Obs)\nRMSE={row.rmse:.3f}")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

    title = "Best-Agreement Predictions (highest WI)" if best else "Worst-Agreement Predictions (lowest WI)"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze continuous SPI inference results (WI/RMSE/MAE, spatial maps)."
    )
    parser.add_argument("--pred-dir", type=str, default=None,
                       help="Directory containing pred/, truth/. "
                            "If omitted, use --region/--p/--q/--model-type instead.")
    parser.add_argument("--region", type=str, default=None, help="Region (e.g. Sul, Norte)")
    parser.add_argument("--p", type=int, default=None, help="Historical window length")
    parser.add_argument("--q", type=int, default=None, help="Forecast horizon")
    parser.add_argument("--model-type", type=str, default=None, choices=["pretrained", "scratch"],
                       help="Model type")
    parser.add_argument("--spi-scale", type=int, default=3,
                       help="SPI accumulation scale used for this run [default: 3]; only used to "
                            "locate the output folder (as 'spi<scale>'), not the metric computation.")
    parser.add_argument("--valid-mask", type=str, default=None,
                       help="Path to a validity mask GeoTIFF (optional)")
    parser.add_argument("--no-spatial", action="store_true",
                       help="Skip spatial maps/GeoTIFFs (metrics CSV only)")

    args = parser.parse_args()

    if args.pred_dir is not None:
        resolved_dir = Path(args.pred_dir)
    elif args.region and args.p is not None and args.q is not None and args.model_type:
        resolved_dir = _resolve_pred_dir(
            args.region, args.p, args.q, args.model_type, spi_scale=args.spi_scale
        )
    else:
        resolved_dir = None  # falls back to auto-discovery inside run_analysis

    run_analysis(
        pred_dir=resolved_dir,
        valid_mask_path=Path(args.valid_mask) if args.valid_mask else None,
        generate_spatial=not args.no_spatial,
    )
