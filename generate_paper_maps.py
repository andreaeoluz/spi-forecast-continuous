#!/usr/bin/env python3
"""
generate_paper_maps.py - Publication-quality spatial maps for the continuous
SPI regression paper (Paper.Regression/images/), analogous to the binary
project's generate_false_alarm_maps.py.

For each region's best test-period configuration (matching
results_regression.tex Table~\\ref{tab:spatial_summary}), this produces:

  - spi_class_maps_<region>.png: temporal-mean observed vs. predicted SPI,
    classified into McKee et al. (1993)'s seven SPI severity categories
    rather than shown on a continuous color scale, since a discrete
    drought-class map (as used operationally, e.g. the US/South American
    Drought Monitor style) is more directly interpretable as a "drought
    map" than a raw SPI gradient.
  - bias_map_<region>.png / rmse_map_<region>.png: the same continuous
    spatial mean-error and RMSE maps inference/analyze.py produces during a
    normal run, regenerated here with the paper-specific styling fixes
    below (that framework script keeps a title and an automatic colorbar,
    which is appropriate for routine diagnostics but not for a paper figure
    whose caption already gives that context).

Applies the same three fixes validated on the binary project's confusion/
bias maps (see drought_forecast_binary/generate_false_alarm_maps.py):
  1. No in-image title/description - region/config belongs in the LaTeX
     caption instead.
  2. Every raster is cropped to the bounding box of its actually-valid
     pixels before plotting: the raw grid can carry an all-invalid margin
     that both distorts the true H/W aspect ratio and (since a fully-masked
     row/column renders transparent, not the Axes facecolor) leaves an
     unexplained gap between the map and its colorbar if left in.
  3. Colorbar height is made to exactly match the map height: the two-panel
     SPI-class figure uses box_aspect + a 'compressed' layout (reliable for
     >=2 Axes sharing a row); the lone-Axes bias/RMSE maps instead use
     fully explicit Axes positions in inches, since 'compressed' layout was
     found unreliable for a single Axes + colorbar pair.

Unlike the binary project's rasters, these GeoTIFFs carry a real
geotransform, so longitude/latitude tick labels are kept (accounted for
after cropping) rather than stripped.

Usage: python generate_paper_maps.py
Output: Paper.Regression/images/{spi_class_maps,bias_map,rmse_map}_
        {north,northeast,centerwest,southeast,south}.png
"""
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

REPO_ROOT = Path(__file__).parent
OUTPUTS = REPO_ROOT / "outputs"
OUT_DIR = REPO_ROOT / "Paper.Regression" / "images"


def apply_publication_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Each region's best test-period configuration (q=1 fixed), matching
# Paper.Regression/results_regression.tex Table~\ref{tab:spatial_summary}.
# Runs of 2026-09-28 (train-only SPI fit). Selection rule of the paper: the
# strategy with the higher mean test-period WI over p (test_period.csv), then
# the p with the highest test WI within it. Center-West is a tie on mean WI
# (0.77380 scratch vs 0.77379 TL); the rule picks scratch.
CASE_STUDIES = [
    ("Norte", "north", "scratch", 9),
    ("Nordeste", "northeast", "scratch", 9),
    ("Centro-Oeste", "centerwest", "scratch", 3),
    ("Sudeste", "southeast", "pretrained", 12),
    ("Sul", "south", "scratch", 12),
]

# McKee et al. (1993) SPI severity classes.
SPI_CLASS_BOUNDS = [-6.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 6.0]
SPI_CLASS_NAMES = [
    "Extremely dry", "Severely dry", "Moderately dry", "Near normal",
    "Moderately wet", "Very wet", "Extremely wet",
]
SPI_CLASS_CMAP = plt.get_cmap("RdBu", 7)
SPI_CLASS_NORM = mcolors.BoundaryNorm(SPI_CLASS_BOUNDS, SPI_CLASS_CMAP.N)
SPI_CLASS_MIDPOINTS = [
    (SPI_CLASS_BOUNDS[i] + SPI_CLASS_BOUNDS[i + 1]) / 2 for i in range(7)
]
SPI_CLASS_MIDPOINTS[0], SPI_CLASS_MIDPOINTS[-1] = -2.7, 2.7  # keep the open-ended bins' labels near their band


def load_stack(folder: Path):
    files = sorted(folder.glob("*.tif"))
    stack = []
    for f in files:
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            stack.append(data)
    return np.array(stack)


def crop_to_valid(valid_mask, *arrays):
    """Crop valid_mask and every array in `arrays` to the bounding box of
    actually-valid pixels (module docstring, fix #2)."""
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    r0, r1 = np.nonzero(rows)[0][[0, -1]]
    c0, c1 = np.nonzero(cols)[0][[0, -1]]
    cropped = [a[..., r0:r1 + 1, c0:c1 + 1] for a in arrays]
    return valid_mask[r0:r1 + 1, c0:c1 + 1], r0, r1, c0, c1, cropped


def cropped_extent(sample_file: Path, r0, r1, c0, c1):
    """[left, right, bottom, top] extent for imshow, in the cropped raster's
    true geographic coordinates."""
    with rasterio.open(sample_file) as src:
        t = src.transform
    lon0 = t[2] + c0 * t[0]
    lon1 = t[2] + (c1 + 1) * t[0]
    lat0 = t[5] + (r1 + 1) * t[4]
    lat1 = t[5] + r0 * t[4]
    return [lon0, lon1, lat0, lat1]


def add_axes_in(fig, x_in, y_in, w_in, h_in, fig_w_in, fig_h_in):
    """Add an Axes at an explicit position/size given in inches."""
    return fig.add_axes([x_in / fig_w_in, y_in / fig_h_in, w_in / fig_w_in, h_in / fig_h_in])


REGION_LABELS = {
    "north": "North", "northeast": "Northeast", "centerwest": "Center-West",
    "southeast": "Southeast", "south": "South",
}


def plot_rmse_panel(regions_data, out_dir: Path) -> None:
    """Single multipanel figure combining all five regions' RMSE maps, on
    one shared 0..vmax color scale (all regions report RMSE in the same SPI
    units and a comparable magnitude range, so a shared scale makes RMSE
    directly comparable across regions rather than each panel auto-scaling
    to its own range)."""
    n = len(regions_data)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    rmse_vmax = max(float(np.nanmax(d["masked_rmse"])) for d in regions_data)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.4 * n_rows), layout="compressed")
    axes = axes.flatten()
    im = None
    for ax, d in zip(axes, regions_data):
        ax.set_box_aspect(d["aspect"])
        im = ax.imshow(d["masked_rmse"], cmap="viridis", vmin=0, vmax=rmse_vmax,
                        extent=d["extent"], origin="upper", interpolation="nearest")
        ax.set_title(REGION_LABELS[d["region_slug"]], fontsize=15, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    for ax in axes[len(regions_data):]:
        fig.delaxes(ax)

    cb = fig.colorbar(im, ax=axes[:len(regions_data)].tolist(), fraction=0.046, pad=0.03,
                       label="RMSE (SPI units)")
    cb.ax.tick_params(labelsize=12)

    out = out_dir / "rmse_maps_panel"
    plt.savefig(f"{out}.png")
    plt.savefig(f"{out}.pdf")
    plt.close()
    print(f"Saved: {out}.png / .pdf")


def plot_spi_class_panel(regions_data, out_dir: Path) -> None:
    """Single multipanel figure combining all five regions' Observed vs.
    Predicted SPI severity classifications, one shared McKee (1993)
    7-category colorbar (the same fixed categorical scale applies to every
    region, so unlike the RMSE panel there is no scaling choice to make)."""
    n = len(regions_data)
    fig, axes = plt.subplots(n, 2, figsize=(9.6, 4.3 * n), layout="compressed")
    im = None
    for row, d in enumerate(regions_data):
        for col, (data, label) in enumerate(zip([d["mean_obs"], d["mean_pred"]], ["Observed", "Predicted"])):
            ax = axes[row, col]
            ax.set_box_aspect(d["aspect"])
            im = ax.imshow(np.ma.masked_where(~d["valid_mask"], data), cmap=SPI_CLASS_CMAP,
                            norm=SPI_CLASS_NORM, extent=d["extent"], origin="upper",
                            interpolation="nearest")
            if row == 0:
                ax.set_title(label, fontsize=15, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Latitude")
                ax.text(0.02, 0.98, REGION_LABELS[d["region_slug"]], transform=ax.transAxes,
                         fontsize=13, fontweight="bold", verticalalignment="top",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
            if row == n - 1:
                ax.set_xlabel("Longitude")

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.04, pad=0.03)
    cb.set_ticks(SPI_CLASS_MIDPOINTS)
    cb.set_ticklabels(SPI_CLASS_NAMES, fontsize=12)

    out = out_dir / "spi_class_maps_panel"
    plt.savefig(f"{out}.png")
    plt.savefig(f"{out}.pdf")
    plt.close()
    print(f"Saved: {out}.png / .pdf")


def main():
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    regions_data = []

    for region_pt, region_slug, mtype, p in CASE_STUDIES:
        base = OUTPUTS / region_pt / "inferences" / "spi3" / f"p{p}_q1" / mtype / "test"
        pred_stack = load_stack(base / "pred")
        truth_stack = load_stack(base / "truth")

        sample_file = sorted((base / "truth").glob("*.tif"))[0]
        with rasterio.open(sample_file) as src:
            nodata = src.nodata
            sample = src.read(1)
            valid_mask = sample != nodata if nodata is not None else np.isfinite(sample)

        T = min(len(pred_stack), len(truth_stack))
        pred_stack, truth_stack = pred_stack[:T], truth_stack[:T]

        valid_mask, r0, r1, c0, c1, (pred_stack, truth_stack) = crop_to_valid(
            valid_mask, pred_stack, truth_stack
        )
        H, W = valid_mask.shape
        aspect = H / W
        extent = cropped_extent(sample_file, r0, r1, c0, c1)

        masked = lambda a: np.ma.masked_where(~valid_mask, a)  # noqa: E731

        mean_obs = np.nanmean(truth_stack, axis=0)
        mean_pred = np.nanmean(pred_stack, axis=0)
        bias_map = np.nanmean(pred_stack - truth_stack, axis=0)
        rmse_map = np.sqrt(np.nanmean((pred_stack - truth_stack) ** 2, axis=0))

        # --- SPI class maps (Observed vs. Predicted, McKee 1993, 7 classes) ---
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), layout="compressed")
        for ax, data, label in zip(axes, [mean_obs, mean_pred], ["Observed", "Predicted"]):
            ax.set_box_aspect(aspect)
            im = ax.imshow(masked(data), cmap=SPI_CLASS_CMAP, norm=SPI_CLASS_NORM,
                            extent=extent, origin="upper", interpolation="nearest")
            ax.set_title(label, fontsize=15, fontweight="bold")
            ax.set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.046, pad=0.03)
        cb.set_ticks(SPI_CLASS_MIDPOINTS)
        cb.set_ticklabels(SPI_CLASS_NAMES, fontsize=12)
        out = OUT_DIR / f"spi_class_maps_{region_slug}"
        plt.savefig(f"{out}.png")
        plt.savefig(f"{out}.pdf")
        plt.close()
        print(f"Saved: {out}.png / .pdf")

        # --- bias map (mean Predicted - Observed SPI) ---
        map_h_in, cbar_w_in, cbar_gap_in = 4.6, 0.18, 0.10
        left_margin_in, bottom_margin_in, right_label_in = 0.75, 0.62, 1.05
        map_w_in = map_h_in / aspect
        fig_w = left_margin_in + map_w_in + cbar_gap_in + cbar_w_in + right_label_in
        fig_h = bottom_margin_in + map_h_in + 0.12

        fig = plt.figure(figsize=(fig_w, fig_h))
        vmax = float(np.nanmax(np.abs(masked(bias_map))))
        ax = add_axes_in(fig, left_margin_in, bottom_margin_in, map_w_in, map_h_in, fig_w, fig_h)
        # RdBu_r (not RdBu): positive bias (over-prediction, wetter than
        # observed) reads as warm/red, negative bias (under-prediction,
        # drier than observed) as cool/blue.
        im = ax.imshow(masked(bias_map), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        extent=extent, origin="upper", interpolation="nearest")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cax = add_axes_in(fig, left_margin_in + map_w_in + cbar_gap_in, bottom_margin_in,
                           cbar_w_in, map_h_in, fig_w, fig_h)
        cb = fig.colorbar(im, cax=cax, label="Bias (Predicted $-$ Observed SPI)")
        cb.ax.tick_params(labelsize=12)
        domain_mean_bias = float(np.nanmean(masked(bias_map)))
        print(f"  [{region_slug}] domain-mean bias (P-O) = {domain_mean_bias:+.4f}")

        out = OUT_DIR / f"bias_map_{region_slug}"
        plt.savefig(f"{out}.png")
        plt.savefig(f"{out}.pdf")
        plt.close()
        print(f"Saved: {out}.png / .pdf")

        # --- RMSE map ---
        fig = plt.figure(figsize=(fig_w, fig_h))
        rmse_vmax = float(np.nanmax(masked(rmse_map)))
        ax = add_axes_in(fig, left_margin_in, bottom_margin_in, map_w_in, map_h_in, fig_w, fig_h)
        im = ax.imshow(masked(rmse_map), cmap="viridis", vmin=0, vmax=rmse_vmax,
                        extent=extent, origin="upper", interpolation="nearest")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cax = add_axes_in(fig, left_margin_in + map_w_in + cbar_gap_in, bottom_margin_in,
                           cbar_w_in, map_h_in, fig_w, fig_h)
        cb = fig.colorbar(im, cax=cax, label="RMSE (SPI units)")
        cb.ax.tick_params(labelsize=12)
        out = OUT_DIR / f"rmse_map_{region_slug}"
        plt.savefig(f"{out}.png")
        plt.savefig(f"{out}.pdf")
        plt.close()
        print(f"Saved: {out}.png / .pdf")

        regions_data.append(dict(
            region_slug=region_slug, mean_obs=mean_obs, mean_pred=mean_pred,
            masked_rmse=masked(rmse_map), valid_mask=valid_mask, aspect=aspect, extent=extent,
        ))

    plot_rmse_panel(regions_data, OUT_DIR)
    plot_spi_class_panel(regions_data, OUT_DIR)


if __name__ == "__main__":
    main()
