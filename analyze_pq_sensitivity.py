#!/usr/bin/env python3
"""
analyze_pq_sensitivity.py - Analysis of p and q parameter sensitivity.

Derived from the binary framework's analyze_pq_sensitivity.py. Analyzes
grid search results to evaluate model sensitivity to p (history length)
and q (forecast horizon), comparing Transfer Learning vs. Scratch, for
continuous SPI regression. MCC/CSI/F1/precision/recall have been replaced
throughout with WI (primary, higher is better), RMSE and MAE (secondary,
lower is better).

Generates:
1. WI/RMSE heatmaps for each region
2. p/q sensitivity plots
3. Transfer Learning vs. Scratch comparison
4. LaTeX tables for the paper
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from utils.logger import Logger, Colors

# Metrics where a LOWER value is better (everything else is "higher is better").
LOWER_IS_BETTER = {"rmse", "mae"}


class PQSensitivityAnalyzer:
    """Analyzes p and q parameter sensitivity from grid search results."""

    REGIONS = ["South", "Southeast", "Northeast", "Center-West", "North"]

    REGION_MAPPING = {
        "South": "Sul",
        "Southeast": "Sudeste",
        "Northeast": "Nordeste",
        "Center-West": "Centro-Oeste",
        "North": "Norte",
    }

    SPI_SCALE = 3

    REGION_COLORS = {
        "South": "#2E86AB",
        "Southeast": "#A23B72",
        "Northeast": "#F18F01",
        "Center-West": "#73AB84",
        "North": "#C73E1D",
    }

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.logger = Logger()
        self.results = {}
        self.metrics = ["wi", "rmse", "mae"]

    # ========================================================================
    # LOADING METHODS
    # ========================================================================

    def load_all_results(self) -> None:
        """Loads all grid search results for all regions."""
        self.logger.header("📊 LOADING GRID SEARCH RESULTS")

        for region_en in self.REGIONS:
            region_pt = self.REGION_MAPPING[region_en]
            self.logger.info(f"\n  Loading {region_en} ({region_pt})...")
            region_loaded = False

            for tl in [False, True]:
                suffix = f"spi{self.SPI_SCALE}_tl_{tl}"
                results_dir = self.base_dir / region_pt / "grid_search" / "results" / suffix

                excel_file = results_dir / f"grid_search_results_{suffix}.xlsx"

                if excel_file.exists():
                    df = pd.read_excel(excel_file, sheet_name="All Results")
                    df['region'] = region_en
                    df['region_pt'] = region_pt
                    df['transfer_learning'] = tl

                    self.results[(region_en, tl)] = df
                    self.logger.success(f"    ✅ TL={tl}: {len(df)} combinations")
                    region_loaded = True
                else:
                    self.logger.warning(f"    ⚠️  TL={tl}: file not found ({excel_file})")
                    self.results[(region_en, tl)] = None

            if not region_loaded:
                self.logger.warning(f"  ⚠️  No files found for {region_en}")

        self.logger.success("\n✅ All results loaded")

    # ========================================================================
    # AGGREGATION METHODS
    # ========================================================================

    def get_aggregated_results(self, metric: str = "wi") -> pd.DataFrame:
        """Aggregates results by region, p, q and TL."""
        rows = []

        for (region_en, tl), df in self.results.items():
            if df is None:
                continue

            col = f"best_{metric}"
            if col not in df.columns:
                continue

            for _, row in df.iterrows():
                rows.append({
                    "region": row.get("region", region_en),
                    "region_pt": row.get("region_pt", self.REGION_MAPPING.get(region_en, region_en)),
                    "p": row["p"],
                    "q": row["q"],
                    "transfer_learning": row.get("transfer_learning", tl),
                    metric: row[col],
                    "wi": row.get("best_wi", np.nan),
                    "rmse": row.get("best_rmse", np.nan),
                    "mae": row.get("best_mae", np.nan),
                    "best_epoch": row.get("best_epoch", np.nan),
                })

        return pd.DataFrame(rows)

    def get_best_parameters(self, metric: str = "wi") -> pd.DataFrame:
        """Finds best parameters by region and TL."""
        df = self.get_aggregated_results(metric)
        if df.empty:
            return df
        lower_better = metric in LOWER_IS_BETTER
        idx_fn = (lambda g: g[metric].idxmin()) if lower_better else (lambda g: g[metric].idxmax())
        best = df.loc[df.groupby(["region", "transfer_learning"]).apply(idx_fn)]
        return best.reset_index(drop=True)

    def _is_better(self, metric: str, a: float, b: float) -> bool:
        """True if value `a` is better than `b` for the given metric."""
        if metric in LOWER_IS_BETTER:
            return a < b
        return a > b

    # ========================================================================
    # LATEX TABLE GENERATION
    # ========================================================================

    def generate_latex_tables(self, metric: str = "wi", output_dir: Path = None) -> None:
        """Generates LaTeX tables with results."""
        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity" / "latex"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)
        best = self.get_best_parameters(metric)

        if df.empty:
            self.logger.warning(f"⚠️  No data for LaTeX tables ({metric})")
            return

        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Best p and q parameters by region}")
        latex.append("\\label{tab:best_params}")
        latex.append("\\small")
        latex.append("\\begin{tabular}{lcccccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Region} & \\textbf{TL} & \\textbf{Best p} & \\textbf{Best q} & ")
        latex.append(f"\\textbf{{{metric.upper()}}} & \\textbf{{RMSE}} & \\textbf{{MAE}} \\\\")
        latex.append("\\midrule")

        for _, row in best.iterrows():
            region = row["region"]
            tl = "Yes" if row["transfer_learning"] else "No"
            p, q = int(row["p"]), int(row["q"])
            latex.append(
                f"{region} & {tl} & {p} & {q} & "
                f"{row[metric]:.4f} & {row.get('rmse', np.nan):.4f} & {row.get('mae', np.nan):.4f} \\\\"
            )

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        with open(output_dir / f"best_params_{metric}.tex", "w") as f:
            f.write("\n".join(latex))

        # Sensitivity matrix by region (TL only)
        for region_en in self.REGIONS:
            subset = df[(df["region"] == region_en) & (df["transfer_learning"] == True)]
            if subset.empty:
                continue

            pivot = subset.pivot_table(values=metric, index="p", columns="q", aggfunc="mean").round(4)
            if pivot.empty:
                continue

            lower_better = metric in LOWER_IS_BETTER
            latex = []
            latex.append("\\begin{table}[htbp]")
            latex.append("\\centering")
            latex.append(f"\\caption{{p/q sensitivity - {region_en} (with TL)}}")
            latex.append(f"\\label{{tab:sensitivity_{region_en.lower()}}}")
            latex.append("\\small")
            latex.append("\\begin{tabular}{l" + "c" * len(pivot.columns) + "}")
            latex.append("\\toprule")
            header = ["p $\\backslash$ q"] + [str(q) for q in pivot.columns]
            latex.append(" & ".join(header) + " \\\\")
            latex.append("\\midrule")

            for p in pivot.index:
                row = [str(p)] + [f"{pivot.loc[p, q]:.4f}" for q in pivot.columns]
                best_val = pivot.loc[p].min() if lower_better else pivot.loc[p].max()
                for i, q in enumerate(pivot.columns):
                    if pivot.loc[p, q] == best_val:
                        row[i + 1] = f"\\textbf{{{row[i + 1]}}}"
                latex.append(" & ".join(row) + " \\\\")

            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
            latex.append("\\end{table}")

            with open(output_dir / f"sensitivity_{region_en.lower()}_{metric}.tex", "w") as f:
                f.write("\n".join(latex))

        # TL vs Scratch comparison
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Transfer Learning vs Scratch comparison}")
        latex.append("\\label{tab:tl_comparison}")
        latex.append("\\small")
        latex.append("\\begin{tabular}{lcccccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Region} & \\textbf{Best (Scratch)} & ")
        latex.append("\\textbf{Best (TL)} & \\textbf{Gain} & ")
        latex.append("\\textbf{Improvements (\\%)} & \\textbf{Best p/q (TL)} \\\\")
        latex.append("\\midrule")

        lower_better = metric in LOWER_IS_BETTER
        for region_en in self.REGIONS:
            subset = df[df["region"] == region_en]
            if subset.empty:
                continue

            scratch = subset[subset["transfer_learning"] == False]
            tl = subset[subset["transfer_learning"] == True]
            if scratch.empty or tl.empty:
                continue

            best_scratch = scratch.loc[scratch[metric].idxmin() if lower_better else scratch[metric].idxmax()]
            best_tl = tl.loc[tl[metric].idxmin() if lower_better else tl[metric].idxmax()]

            gain = best_tl[metric] - best_scratch[metric]
            if lower_better:
                gain = -gain  # positive gain always means "TL improved things"

            merged = pd.merge(
                scratch[["p", "q", metric]], tl[["p", "q", metric]],
                on=["p", "q"], suffixes=("_scratch", "_tl")
            )

            if not merged.empty:
                if lower_better:
                    improvements = (merged[f"{metric}_tl"] < merged[f"{metric}_scratch"]).mean() * 100
                else:
                    improvements = (merged[f"{metric}_tl"] > merged[f"{metric}_scratch"]).mean() * 100
            else:
                improvements = 0.0

            latex.append(
                f"{region_en} & {best_scratch[metric]:.4f} & "
                f"{best_tl[metric]:.4f} & {gain:+.4f} & "
                f"{improvements:.1f}\\% & "
                f"{int(best_tl['p'])}/{int(best_tl['q'])} \\\\"
            )

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        with open(output_dir / f"tl_comparison_{metric}.tex", "w") as f:
            f.write("\n".join(latex))

        self.logger.success(f"✅ LaTeX tables saved to: {output_dir}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    def print_summary(self, metric: str = "wi") -> None:
        """Prints analysis summary."""
        self.logger.header(f"📊 SENSITIVITY ANALYSIS SUMMARY - {metric.upper()}")

        df = self.get_aggregated_results(metric)
        best = self.get_best_parameters(metric)

        if df.empty:
            self.logger.warning("⚠️  No data available")
            return

        print(f"\n{Colors.BOLD}Best Parameters by Region:{Colors.RESET}")
        print(f"  {'Region':<15} {'TL':<6} {'p':<4} {'q':<4} {metric.upper():<8} {'RMSE':<8} {'MAE':<8}")
        print(f"  {'─' * 60}")

        for _, row in best.iterrows():
            tl = "Yes" if row["transfer_learning"] else "No"
            print(
                f"  {row['region']:<15} {tl:<6} {int(row['p']):<4} {int(row['q']):<4} "
                f"{row[metric]:<8.4f} {row.get('rmse', np.nan):<8.4f} {row.get('mae', np.nan):<8.4f}"
            )

        print(f"\n{Colors.BOLD}Best Overall Configuration:{Colors.RESET}")
        if not best.empty:
            lower_better = metric in LOWER_IS_BETTER
            best_overall = best.loc[best[metric].idxmin() if lower_better else best[metric].idxmax()]
            print(f"  Region: {best_overall['region']}")
            print(f"  TL: {'Yes' if best_overall['transfer_learning'] else 'No'}")
            print(f"  p: {int(best_overall['p'])}")
            print(f"  q: {int(best_overall['q'])}")
            print(f"  {metric.upper()}: {best_overall[metric]:.4f}")

    # ========================================================================
    # PLOTTING METHODS
    # ========================================================================

    PUBLICATION_STYLE = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 16,
        'axes.labelsize': 17,
        'axes.titlesize': 18,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 14,
        'figure.titlesize': 19,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
    }

    def apply_publication_style(self):
        plt.rcParams.update(self.PUBLICATION_STYLE)
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.3)

    def _save_figure(self, fig, output_path: Path) -> None:
        """Save figure in PNG (300 DPI) and PDF formats."""
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def plot_heatmaps(self, metric: str = "wi", output_dir: Path = None) -> None:
        """Generate p/q sensitivity heatmaps. Colormap direction follows
        whether the metric is higher-is-better (WI) or lower-is-better
        (RMSE/MAE).

        All 10 region/strategy panels share one standardized color scale and
        one colorbar (previously each panel computed its own vmin/vmax while
        only a single panel's colorbar was drawn, so the other nine panels'
        colors did not correspond to the legend shown)."""
        self.apply_publication_style()

        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)
        if df.empty:
            self.logger.warning(f"⚠️  No data for {metric} heatmaps")
            return

        lower_better = metric in LOWER_IS_BETTER
        # cividis: sequential and colorblind-safe, and it keeps its contrast in
        # grayscale print (RdYlGn's red-green pair does neither); reversed for
        # lower-is-better metrics so the better value is always the light end.
        cmap = 'cividis_r' if lower_better else 'cividis'

        # Shared scale across every panel: WI is anchored at its natural
        # floor (0 = no skill); RMSE/MAE have no fixed ceiling, so both
        # bounds are taken from the observed data range.
        vmin = 0.0 if not lower_better else df[metric].min()
        vmax = df[metric].max()

        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        im = None

        for idx, region_en in enumerate(self.REGIONS):
            for tl_idx, tl in enumerate([False, True]):
                ax = axes[idx, tl_idx]

                subset = df[(df["region"] == region_en) & (df["transfer_learning"] == tl)]
                if subset.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                             transform=ax.transAxes, fontsize=16, style='italic')
                    continue

                pivot = subset.pivot_table(values=metric, index="p", columns="q", aggfunc="mean")
                pivot = pivot.fillna(0)

                im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                                interpolation='nearest')

                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, fontsize=15)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=15)
                ax.set_xlabel('q (forecast horizon)', fontsize=17)
                ax.set_ylabel('p (history length)', fontsize=17)

                tl_label = 'TL' if tl else 'Scratch'
                ax.text(0.02, 0.98, f'{region_en} - {tl_label}',
                        transform=ax.transAxes, fontsize=15, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        fig.subplots_adjust(left=0.08, right=0.89, top=0.98, bottom=0.05, wspace=0.35, hspace=0.35)
        if im is not None:
            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(metric.upper(), fontsize=17)
            cbar.ax.tick_params(labelsize=15)

        output_path = output_dir / f'heatmaps_{metric}.png'
        self._save_figure(fig, output_path)

        self.logger.success(f"✅ Heatmaps saved to: {output_path} (+ .pdf)")

    REGION_SLUGS = {
        "North": "north", "Northeast": "northeast", "Center-West": "centerwest",
        "Southeast": "southeast", "South": "south",
    }

    def plot_heatmaps_per_region(self, metric: str = "wi", output_dir: Path = None) -> None:
        """Same p/q sensitivity heatmaps as plot_heatmaps, but saved as one
        compact two-panel (Scratch vs. TL) figure per region instead of a
        single 5x2 grid. The single combined figure occupies a full page at
        publication size; splitting it into five smaller per-region figures,
        laid out with LaTeX subfigures, keeps the same content and the same
        shared color scale (for cross-region comparability) while letting
        each panel be shown at a readable size without dominating a page."""
        self.apply_publication_style()

        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)
        if df.empty:
            self.logger.warning(f"⚠️  No data for {metric} per-region heatmaps")
            return

        lower_better = metric in LOWER_IS_BETTER
        # cividis: sequential and colorblind-safe, and it keeps its contrast in
        # grayscale print (RdYlGn's red-green pair does neither); reversed for
        # lower-is-better metrics so the better value is always the light end.
        cmap = 'cividis_r' if lower_better else 'cividis'
        vmin = 0.0 if not lower_better else df[metric].min()
        vmax = df[metric].max()

        # Drawn at its printed size (the paper's text width), with each
        # cell's value written in it, so the grid can be read without
        # relying on the color scale alone.
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for region_en in self.REGIONS:
            fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7))
            im = None
            for tl_idx, tl in enumerate([False, True]):
                ax = axes[tl_idx]
                ax.grid(False)
                subset = df[(df["region"] == region_en) & (df["transfer_learning"] == tl)]
                if subset.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                             transform=ax.transAxes, fontsize=8, style='italic')
                    continue

                pivot = subset.pivot_table(values=metric, index="p", columns="q", aggfunc="mean")
                pivot = pivot.fillna(0)

                im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                                interpolation='nearest')
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        v = pivot.values[i, j]
                        r, g, b, _ = cmap_obj(norm(v))
                        txt = 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else 'black'
                        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color=txt)

                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, fontsize=7.5)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=7.5)
                ax.tick_params(length=2)
                ax.set_xlabel('q (forecast horizon)', fontsize=8)
                if tl_idx == 0:
                    ax.set_ylabel('p (history length)', fontsize=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                tl_label = 'TL' if tl else 'Scratch'
                ax.set_title(tl_label, fontsize=9, fontweight='bold', pad=4)

            fig.subplots_adjust(left=0.09, right=0.88, top=0.89, bottom=0.17, wspace=0.18)
            if im is not None:
                cbar_ax = fig.add_axes([0.905, 0.17, 0.022, 0.72])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label(metric.upper(), fontsize=8)
                cbar.ax.tick_params(labelsize=7, length=2)
                cbar.outline.set_linewidth(0.5)

            slug = self.REGION_SLUGS[region_en]
            output_path = output_dir / f'heatmap_{slug}_{metric}.png'
            self._save_figure(fig, output_path)

        self.logger.success(f"✅ Per-region heatmaps saved to: {output_dir} (heatmap_<region>_{metric}.png/.pdf)")

    def plot_sensitivity_analysis(self, metric: str = "wi", output_dir: Path = None) -> None:
        """Generate sensitivity analysis plots (p, q, and TL gain)."""
        self.apply_publication_style()

        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)
        if df.empty:
            self.logger.warning(f"⚠️  No data for {metric} sensitivity analysis")
            return

        lower_better = metric in LOWER_IS_BETTER
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax = axes[0, 0]
        for region_en in self.REGIONS:
            for tl, style in [(False, '--'), (True, '-')]:
                subset = df[(df["region"] == region_en) & (df["transfer_learning"] == tl)]
                if subset.empty:
                    continue
                grouped = subset.groupby("p")[metric].mean()
                if not grouped.empty:
                    label = f'{region_en} {"(TL)" if tl else "(Scratch)"}'
                    ax.plot(grouped.index, grouped.values, style, label=label,
                            color=self.REGION_COLORS[region_en],
                            linewidth=2.0 if tl else 1.2, markersize=5)
        ax.set_xlabel('p (history length)', fontsize=14)
        ax.set_ylabel(metric.upper(), fontsize=14)
        ax.legend(loc='best', fontsize=10, framealpha=0.85, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=15)

        ax = axes[0, 1]
        for region_en in self.REGIONS:
            for tl, style in [(False, '--'), (True, '-')]:
                subset = df[(df["region"] == region_en) & (df["transfer_learning"] == tl)]
                if subset.empty:
                    continue
                grouped = subset.groupby("q")[metric].mean()
                if not grouped.empty:
                    label = f'{region_en} {"(TL)" if tl else "(Scratch)"}'
                    ax.plot(grouped.index, grouped.values, style, label=label,
                            color=self.REGION_COLORS[region_en],
                            linewidth=2.0 if tl else 1.2, markersize=5)
        ax.set_xlabel('q (forecast horizon)', fontsize=14)
        ax.set_ylabel(metric.upper(), fontsize=14)
        ax.legend(loc='best', fontsize=10, framealpha=0.85, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=15)

        ax = axes[1, 0]
        gains = []
        for region_en in self.REGIONS:
            df_scratch = df[(df["region"] == region_en) & (df["transfer_learning"] == False)]
            df_tl = df[(df["region"] == region_en) & (df["transfer_learning"] == True)]
            if df_scratch.empty or df_tl.empty:
                continue
            for p in df["p"].unique():
                scratch_p = df_scratch[df_scratch["p"] == p][metric].mean()
                tl_p = df_tl[df_tl["p"] == p][metric].mean()
                if not np.isnan(scratch_p) and not np.isnan(tl_p):
                    gain = (scratch_p - tl_p) if lower_better else (tl_p - scratch_p)
                    gains.append({"region": region_en, "p": p, "gain": gain})

        gains_df = pd.DataFrame(gains)
        if not gains_df.empty:
            for region_en in self.REGIONS:
                subset = gains_df[gains_df["region"] == region_en]
                if not subset.empty:
                    ax.plot(subset["p"], subset["gain"], 'o-', label=region_en,
                            color=self.REGION_COLORS[region_en], linewidth=1.8, markersize=7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.2)
        ax.set_xlabel('p (history length)', fontsize=14)
        ax.set_ylabel(f'TL gain ({metric.upper()}, positive = TL better)', fontsize=14)
        ax.legend(loc='best', fontsize=10, framealpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=15)

        ax = axes[1, 1]
        data_by_region, labels = [], []
        for region_en in self.REGIONS:
            subset = df[(df["region"] == region_en) & (df["transfer_learning"] == True)]
            if not subset.empty:
                data_by_region.append(subset[metric].values)
                labels.append(region_en)
        if data_by_region:
            bp = ax.boxplot(data_by_region, labels=labels, patch_artist=True,
                             widths=0.6, medianprops=dict(color='black', linewidth=2.0))
            for patch, region_en in zip(bp['boxes'], self.REGIONS[:len(data_by_region)]):
                patch.set_facecolor(self.REGION_COLORS[region_en])
                patch.set_alpha(0.65)
                patch.set_edgecolor('black')
                patch.set_linewidth(0.8)
        ax.set_xlabel('Region', fontsize=14)
        ax.set_ylabel(metric.upper(), fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=15)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.3)

        output_path = output_dir / f'sensitivity_analysis_{metric}.png'
        self._save_figure(fig, output_path)

        self.logger.success(f"✅ Sensitivity analysis saved to: {output_path} (+ .pdf)")

    def plot_tl_comparison(self, metric: str = "wi", output_dir: Path = None) -> None:
        """Generate TL vs Scratch scatter comparison plots."""
        self.apply_publication_style()

        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_aggregated_results(metric)
        if df.empty:
            self.logger.warning(f"⚠️  No data for {metric} TL comparison")
            return

        lower_better = metric in LOWER_IS_BETTER

        # Drawn at its printed size (about the text width of the paper), so
        # fonts reach the page at their nominal size instead of being shrunk
        # ~3x from an 18-inch canvas. Regions follow the order of the paper's
        # tables; the sixth cell holds the reading key.
        order = [r for r in ("North", "Northeast", "Center-West", "Southeast", "South")
                 if r in self.REGIONS]
        plt.rcParams.update({
            'font.size': 8, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5,
            'ytick.labelsize': 7.5, 'axes.linewidth': 0.8,
            'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        })
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4))
        axes = axes.flatten()

        for idx, region_en in enumerate(order):
            ax = axes[idx]

            subset = df[df["region"] == region_en]
            if subset.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                         transform=ax.transAxes, fontsize=16, style='italic')
                continue

            scratch = subset[subset["transfer_learning"] == False]
            tl = subset[subset["transfer_learning"] == True]
            if scratch.empty or tl.empty:
                ax.text(0.5, 0.5, 'Incomplete data', ha='center', va='center',
                         transform=ax.transAxes, fontsize=16, style='italic')
                continue

            merged = pd.merge(
                scratch[["p", "q", metric]], tl[["p", "q", metric]],
                on=["p", "q"], suffixes=("_scratch", "_tl")
            )
            if merged.empty:
                ax.text(0.5, 0.5, 'No common combinations', ha='center', va='center',
                         transform=ax.transAxes, fontsize=16, style='italic')
                continue

            ax.scatter(merged[f"{metric}_scratch"], merged[f"{metric}_tl"], alpha=0.8, s=22,
                       color=self.REGION_COLORS[region_en], edgecolors='black', linewidth=0.5,
                       zorder=3)

            min_val = min(merged[f"{metric}_scratch"].min(), merged[f"{metric}_tl"].min())
            max_val = max(merged[f"{metric}_scratch"].max(), merged[f"{metric}_tl"].max())
            pad = max(0.02, 0.05 * (max_val - min_val))
            min_val, max_val = min_val - pad, max_val + pad
            ax.plot([min_val, max_val], [min_val, max_val], color='#555555', linestyle='--',
                    alpha=0.7, linewidth=0.8, zorder=1)

            ax.set_title(region_en, fontsize=9, fontweight='bold', pad=3)

            diff = merged[f"{metric}_tl"] - merged[f"{metric}_scratch"]
            mean_gain = (-diff.mean()) if lower_better else diff.mean()
            better_count = (diff < 0).sum() if lower_better else (diff > 0).sum()
            total_count = len(merged)

            # Colored by sign; placed top-left, the corner the points leave free
            gain_color = "#009E73" if mean_gain > 0 else "#D55E00"
            stats_text = f"Δ = {mean_gain:+.4f}\n{better_count}/{total_count} better"
            ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, fontsize=7.5,
                     color=gain_color, fontweight='bold', zorder=4,
                     horizontalalignment='left', verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                               edgecolor=gain_color, alpha=0.9))

            ax.set_xlabel(f'Scratch {metric.upper()}', labelpad=1.5)
            ax.set_ylabel(f'TL {metric.upper()}', labelpad=1.5)
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linewidth=0.5)

        # Sixth cell: reading key instead of an empty slot
        key = axes[len(order)]
        key.axis('off')
        key.text(0.02, 0.5,
                 "Each point is one (p, q)\nconfiguration (20 per region).\n\n"
                 f"{'Below' if lower_better else 'Above'} the dashed line (y = x):\n"
                 "TL better than scratch.\n\n"
                 "Δ: mean TL − scratch\n"
                 f"({'sign flipped, ' if lower_better else ''}positive = TL better).\n"
                 "n/20 better: configurations\nwhere TL is better.",
                 transform=key.transAxes, fontsize=7.5, va='center', ha='left', linespacing=1.3)
        for idx in range(len(order) + 1, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)

        output_path = output_dir / f'tl_comparison_{metric}.png'
        self._save_figure(fig, output_path)

        self.logger.success(f"✅ TL comparison saved to: {output_path} (+ .pdf)")

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    def run(self, output_dir: Path = None) -> None:
        """Executes complete analysis."""
        if output_dir is None:
            output_dir = self.base_dir / "analysis" / "pq_sensitivity"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.header("🔍 P/Q SENSITIVITY ANALYSIS")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Regions: {', '.join(self.REGIONS)}")
        print(f"  SPI scale: {self.SPI_SCALE}")

        self.load_all_results()

        has_data = any(df is not None for df in self.results.values())
        if not has_data:
            self.logger.error("❌ No data loaded. Please check the base directory.")
            return

        for metric in ["wi", "rmse"]:
            self.logger.section(f"📊 Analyzing {metric.upper()}")
            self.plot_heatmaps(metric, output_dir)
            self.plot_sensitivity_analysis(metric, output_dir)
            self.plot_tl_comparison(metric, output_dir)
            self.generate_latex_tables(metric, output_dir)

        self.print_summary("wi")

        self.logger.header("✅ ANALYSIS COMPLETED")
        self.logger.info(f"  Results saved to: {output_dir}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze p/q sensitivity from grid search results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_pq_sensitivity.py --base-dir ./outputs
  python analyze_pq_sensitivity.py --base-dir ./outputs --regions Sul Sudeste
  python analyze_pq_sensitivity.py --base-dir ./outputs --output-dir ./analysis
        """
    )

    parser.add_argument("--base-dir", type=str, default="./outputs",
                        help="Base directory with grid search results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for analysis")
    parser.add_argument("--spi-scale", type=int, default=3,
                        help="SPI accumulation scale used by the grid search runs [default: 3]")
    parser.add_argument("--regions", type=str, nargs="+", default=None,
                        help="Regions to analyze (default: all)")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    analyzer = PQSensitivityAnalyzer(base_dir)
    analyzer.SPI_SCALE = args.spi_scale

    if args.regions:
        analyzer.REGIONS = args.regions

    analyzer.run(output_dir)


if __name__ == "__main__":
    main()