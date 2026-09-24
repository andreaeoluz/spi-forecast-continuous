"""grid_search.py - p x q x {Transfer Learning, Scratch} grid search with
result tracking, for continuous SPI regression.

Derived from the binary framework's GridSearch. Removed entirely:
class-imbalance handling (weighted sampling), extreme-drought augmentation,
decision-threshold optimization, and rolling-origin multi-split evaluation
(out of scope for this migration - see the project README). Everything
else - the p x q sweep, the Transfer Learning vs. Scratch comparison, the
normalizer-reuse logic that keeps a transferred encoder's input
distribution consistent with pretraining, and the results/report writers -
is preserved, with metrics swapped from CSI/MCC to WI/RMSE/MAE.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from config import ExperimentConfig, get_paths
from data import load_region_timeseries, ClimateNormalizer, ClimateDataset, load_spi_cache
from models import SPIPredictor
from training import SPIPredictorTrainer
from utils import set_reproducible_seeds
from utils.logger import Logger, Colors


class GridSearch:
    """
    Grid search for hyperparameter optimization.

    Performs a systematic search over p (history length) and q (forecast
    horizon), evaluating continuous SPI regression models using the
    specified optimization metric (default: validation WI, maximized).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        base_data_path: Path,
        optimization_metric: str = None,
        load_attention: bool = True,
    ):
        self.config = config
        self.base_data_path = base_data_path
        self.spi_scale = config.spi.scale
        self.use_transfer_learning = config.use_transfer_learning
        self.logger = Logger()

        # Ablation knob: when True (default, unchanged behavior), the
        # DualTemporalAttention weights are transferred from the autoencoder
        # along with the encoder. When False, only the encoder is transferred
        # and attention starts randomly initialized.
        self.load_attention = load_attention

        self.optimization_metric = (
            optimization_metric or config.optimization.primary_metric
        )
        self.secondary_metric = config.optimization.secondary_metric

        self.suffix = f"spi{self.spi_scale}_tl_{self.use_transfer_learning}"
        if self.use_transfer_learning and not self.load_attention:
            self.suffix += "_noattn"
        self.paths = get_paths(config.region, spi_scale=self.spi_scale)

        self.results_dir = self.paths["grid_search_results"] / self.suffix
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None

        set_reproducible_seeds(config.random_seed)

        self.logger.info(f"✅ GridSearch initialized for region: {config.region}")
        self.logger.info(f"   Primary metric: {self.optimization_metric.upper()}")
        self.logger.info(f"   Secondary metric: {self.secondary_metric.upper()}")

    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================

    def _print_header(self, text: str, char: str = "=", width: int = 70) -> None:
        """Print a formatted header."""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{char * width}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}{text:^{width}}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}{char * width}{Colors.RESET}")

    def _print_section(self, text: str, char: str = "-", width: int = 70) -> None:
        """Print a formatted section title."""
        print(f"\n{Colors.CYAN}{char * width}{Colors.RESET}")
        print(f"{Colors.CYAN}  {text}{Colors.RESET}")
        print(f"{Colors.CYAN}{char * width}{Colors.RESET}")

    def _print_progress(self, current: int, total: int, p: int, q: int) -> None:
        """Print a progress bar for the current combination."""
        bar_len = 30
        percent = current / total
        filled = int(bar_len * percent)
        bar = "█" * filled + "░" * (bar_len - filled)

        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]

        print(f"\n┌{'─' * 68}┐")
        print(f"│ {Colors.BOLD}[{current:>3}/{total}]{Colors.RESET} {bar} {percent:>6.1%} │")
        print(f"│ {'─' * 68} │")
        print(f"│ p={p:>2}  q={q:>2}  ⏱  {elapsed_str} │")
        print(f"└{'─' * 68}┘")

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_data(self) -> Dict[str, Any]:
        """Load and preprocess data for the grid search."""
        self._print_section("📂 LOADING DATA")

        out = load_region_timeseries(self.base_data_path, self.config)

        data = out["data"]
        years = out["years"]
        months = out["months"]
        valid_mask = out["valid_mask"]

        self._validate_mask(valid_mask, data)

        split_info = self.config.split.get_end_indices_gs()
        time_idx = np.array([y * 12 + (m - 1) for y, m in zip(years, months)])

        train_mask = time_idx <= split_info["train_end"]
        val_mask = (time_idx > split_info["train_end"]) & (time_idx <= split_info["val_end"])

        data_train = data[train_mask]
        data_val = data[val_mask]
        months_train = months[train_mask]
        months_val = months[val_mask]

        train_period = (
            f"{years[train_mask][0]}-{months[train_mask][0]:02d} "
            f"to {years[train_mask][-1]}-{months[train_mask][-1]:02d}"
        )
        val_period = (
            f"{years[val_mask][0]}-{months[val_mask][0]:02d} "
            f"to {years[val_mask][-1]}-{months[val_mask][-1]:02d}"
        )

        print(f"  {Colors.GREEN}Training{Colors.RESET}    : {len(data_train):>4} months  ({train_period})")
        print(f"  {Colors.CYAN}Validation{Colors.RESET}  : {len(data_val):>4} months  ({val_period})")

        # ================================================================
        # NORMALIZER
        # ================================================================
        # When transferring the encoder, it MUST see data normalized the
        # exact same way it was pretrained on - reuse the autoencoder's own
        # normalizer instead of fitting a new one, so the encoder's input
        # distribution matches pretraining. Fitting fresh here for a
        # transfer-learning run would silently corrupt the exact normalizer
        # the encoder was pretrained with for every later consumer of that
        # file (evaluate_autoencoder.py, inference, other grid search runs).
        autoencoder_normalizer_path = self.paths["autoencoder_dir"] / "normalizer.json"

        if self.use_transfer_learning and autoencoder_normalizer_path.exists():
            normalizer = ClimateNormalizer.load(autoencoder_normalizer_path)
            print("  ✅ Reusing pretrained autoencoder's normalizer "
                  f"(required for a correct transfer): {autoencoder_normalizer_path}")
            backup_path = self.paths["grid_search_dir"] / "normalizer_pretrained.json"
        else:
            if self.use_transfer_learning:
                self.logger.warning(
                    "  ⚠️  Transfer learning requested but no autoencoder normalizer "
                    f"found at {autoencoder_normalizer_path} - fitting a fresh one; "
                    "the encoder's input distribution will not exactly match pretraining."
                )
            normalizer = ClimateNormalizer(self.config.data.bands)
            normalizer.fit(data_train, months_train, valid_mask)
            print("  ✅ Normalizer fit fresh for this scratch run "
                  f"(NOT written to {autoencoder_normalizer_path} - that file belongs "
                  "to the autoencoder)")
            backup_path = self.paths["grid_search_dir"] / "normalizer_scratch.json"

        data_train_norm = normalizer.transform(data_train, months_train, valid_mask)
        data_val_norm = normalizer.transform(data_val, months_val, valid_mask)

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        normalizer.save(backup_path)
        print(f"  📄 Normalizer copy saved to: {backup_path}")

        spi, _ = load_spi_cache(self.spi_scale, self.paths["spi_cache_dir"])
        spi_train = spi[train_mask] if spi is not None else None
        spi_val = spi[val_mask] if spi is not None else None

        return {
            "data_train": np.nan_to_num(data_train_norm, nan=0.0),
            "data_val": np.nan_to_num(data_val_norm, nan=0.0),
            "months_train": months_train,
            "months_val": months_val,
            "spi_train": spi_train,
            "spi_val": spi_val,
            "valid_mask": valid_mask,
        }

    def _validate_mask(self, valid_mask: np.ndarray, data: np.ndarray) -> None:
        """Validate the validity mask and log statistics."""
        if valid_mask is None:
            self.logger.warning("⚠️  No validity mask provided")
            return

        total_pixels = valid_mask.size
        valid_pixels = valid_mask.sum()
        invalid_pixels = total_pixels - valid_pixels

        print("\n  📊 Validity Mask:")
        print(f"     Total pixels: {total_pixels:,}")
        print(f"     Valid: {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
        print(f"     Invalid: {invalid_pixels:,} ({invalid_pixels/total_pixels:.1%})")

        if data is not None:
            nan_pixels = np.isnan(data).any(axis=-1).any(axis=0)
            nan_invalid = nan_pixels & ~valid_mask
            if nan_invalid.sum() > 0:
                print(f"\n  ℹ️  NaN/Inf pixels marked as invalid: {nan_invalid.sum():,}")

    def load_autoencoder(self) -> Optional[Dict[str, Any]]:
        """Load the pretrained autoencoder checkpoint for transfer learning."""
        autoencoder_path = self.paths["autoencoder_dir"] / "model.pth"
        if not autoencoder_path.exists():
            return None

        checkpoint = torch.load(
            autoencoder_path,
            map_location=self.config.device,
            weights_only=False
        )

        if hasattr(self.config.autoencoder, 'use_anomalies'):
            checkpoint["use_anomalies"] = self.config.autoencoder.use_anomalies

        return checkpoint

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def run(self) -> None:
        """Run the grid search."""
        self.start_time = datetime.now()

        self._print_header("🔍 GRID SEARCH")

        self._display_experiment_config()

        data_dict = self.load_data()

        ae_checkpoint = self.load_autoencoder() if self.use_transfer_learning else None
        self._display_transfer_learning_status(ae_checkpoint)

        total_combinations = len(self.config.p_values) * len(self.config.q_values)
        current = 0

        print(f"\n{'─' * 70}")
        print(f"  🚀 Starting grid search with {total_combinations} combinations")
        print(f"{'─' * 70}")

        for p in self.config.p_values:
            for q in self.config.q_values:
                current += 1

                if not self._validate_data_sufficiency(p, q, data_dict):
                    continue

                self._print_progress(current, total_combinations, p, q)

                train_ds, val_ds = self._create_datasets(p, q, data_dict)

                if len(train_ds) == 0 or len(val_ds) == 0:
                    print("  ⚠️  Dataset empty! Skipping...")
                    continue

                result = self._train_and_evaluate(p, q, train_ds, val_ds, ae_checkpoint)

                if result is not None:
                    self.results.append(result)

                del train_ds, val_ds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.save_results()

        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]
        self._print_header(f"✅ GRID SEARCH COMPLETED  ⏱  {elapsed_str}")

    # =========================================================================
    # HELPER METHODS FOR RUN
    # =========================================================================

    def _display_experiment_config(self) -> None:
        """Display the experiment configuration."""
        ds_info = self.config.get_downsample_info(self.config.region)
        ds_h, ds_w = self.config.get_downsample(self.config.region)

        print(f"  {Colors.BOLD}Region{Colors.RESET}           : {self.config.region}")
        print(f"  {Colors.BOLD}Downsampling{Colors.RESET}     : {ds_h}x{ds_w}")
        print(f"  {Colors.BOLD}Preservation{Colors.RESET}     : {ds_info['preservation_estimate']}")
        print(f"  {Colors.BOLD}Area Reduction{Colors.RESET}   : {ds_info['area_reduction']}x")
        print(f"  {Colors.BOLD}SPI scale{Colors.RESET}        : {self.spi_scale}")
        print(f"  {Colors.BOLD}Transfer Learning{Colors.RESET}: {self.use_transfer_learning}")
        print(f"  {Colors.BOLD}Optimization{Colors.RESET}     : {self.optimization_metric.upper()}")
        print(f"  {Colors.BOLD}P values{Colors.RESET}         : {self.config.p_values}")
        print(f"  {Colors.BOLD}Q values{Colors.RESET}         : {self.config.q_values}")
        print(f"  {Colors.BOLD}Suffix{Colors.RESET}           : {self.suffix}")
        print(f"{'─' * 68}")

    def _display_transfer_learning_status(self, ae_checkpoint: Optional[Dict]) -> None:
        """Display the transfer learning status."""
        if ae_checkpoint:
            print("\n  🧠 Using autoencoder for transfer learning")
            if self.load_attention:
                print("  📌 Encoder AND attention will be transferred")
            else:
                print("  📌 Only the encoder will be transferred (random attention)")
        else:
            print("\n  🔧 Training models from scratch")

    def _validate_data_sufficiency(self, p: int, q: int, data_dict: Dict) -> bool:
        """Validate whether there is enough data for the given p and q."""
        max_samples = len(data_dict["data_val"])
        min_required = p + q

        if min_required >= max_samples:
            print(f"\n  ⏭️  Skipping p={p}, q={q} (p+q={min_required} >= val_size={max_samples})")
            return False

        train_max = len(data_dict["data_train"])
        if min_required >= train_max:
            print(f"\n  ⏭️  Skipping p={p}, q={q} (p+q={min_required} >= train_size={train_max})")
            return False

        return True

    def _create_datasets(
        self,
        p: int,
        q: int,
        data_dict: Dict,
    ) -> tuple:
        """Create the training and validation datasets."""
        train_ds = ClimateDataset(
            data=data_dict["data_train"],
            spi=data_dict["spi_train"],
            months=data_dict["months_train"],
            p=p, q=q,
            valid_mask=data_dict["valid_mask"],
            temporal_decay=True,
            mode="regression",
            seed=self.config.random_seed,
            min_valid_ratio=self.config.data.min_valid_ratio,
            include_spi_channel=self.config.data.include_spi_history,
        )

        val_ds = ClimateDataset(
            data=data_dict["data_val"],
            spi=data_dict["spi_val"],
            months=data_dict["months_val"],
            p=p, q=q,
            valid_mask=data_dict["valid_mask"],
            temporal_decay=True,
            mode="regression",
            seed=self.config.random_seed,
            min_valid_ratio=self.config.data.min_valid_ratio,
            include_spi_channel=self.config.data.include_spi_history,
        )

        print(f"  📊 Training samples: {len(train_ds):>4}")
        print(f"  📊 Validation samples: {len(val_ds):>4}")

        return train_ds, val_ds

    def _train_and_evaluate(
        self,
        p: int,
        q: int,
        train_ds: ClimateDataset,
        val_ds: ClimateDataset,
        ae_checkpoint: Optional[Dict]
    ) -> Optional[Dict]:
        """Train and evaluate a model for the given (p, q) combination."""
        model = SPIPredictor(
            self.config.get_model_config("predictor")
        ).to(self.config.device)

        # ================================================================
        # TRANSFER LEARNING: ENCODER + ATTENTION
        # ================================================================
        if ae_checkpoint is not None:
            try:
                model.load_encoder_from_autoencoder(
                    ae_checkpoint,
                    load_attention=self.load_attention
                )
                print("  🧠 Encoder: transferred ✅")
                if self.load_attention:
                    print("  🧠 Attention: transferred ✅")
                else:
                    print("  🧠 Attention: kept random (ablation: load_attention=False)")
            except Exception as e:
                print(f"  ⚠️  Error transferring encoder/attention: {e}")
                print("  🔧 Training from scratch...")
                ae_checkpoint = None

        attn_suffix = "" if self.load_attention else "_noattn"
        model_dir = (
            self.paths["grid_search_pretrained"].parent / f"pretrained{attn_suffix}"
            if ae_checkpoint
            else self.paths["grid_search_scratch"]
        )
        save_path = model_dir / f"model_p{p}_q{q}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        trainer = SPIPredictorTrainer(
            model,
            train_ds,
            val_ds,
            self.config,
            save_path,
            freeze_encoder_first_epoch=bool(ae_checkpoint),
            optimization_metric=self.optimization_metric,
            load_attention=self.load_attention,
        )

        try:
            trainer.train()
        except Exception as e:
            print(f"  ❌ Training error: {e}")
            return None

        try:
            ckpt = torch.load(save_path, map_location=self.config.device, weights_only=False)
        except Exception as e:
            print(f"  ⚠️  Error loading checkpoint: {e}")
            ckpt = {}

        result = {
            "region": self.config.region,
            "p": p,
            "q": q,
            "transfer_learning": self.use_transfer_learning,
            "load_attention": self.load_attention if ae_checkpoint is not None else False,
            "best_wi": ckpt.get("best_wi", 0.0),
            "best_rmse": ckpt.get("best_rmse", float('inf')),
            "best_mae": ckpt.get("best_mae", float('inf')),
            "best_epoch": ckpt.get("best_epoch", ckpt.get("epoch", 0)),
            "train_loss": ckpt.get("train_loss", float('nan')),
        }

        del model, trainer

        return result

    # =========================================================================
    # RESULTS SAVING
    # =========================================================================

    def save_results(self) -> None:
        """Save the grid search results to disk."""
        if not self.results:
            print("  ⚠️  No results to save")
            return

        self.results_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.results)
        score_col = f"best_{self.optimization_metric}"
        ascending = self.optimization_metric in ("rmse", "mae")
        df = df.sort_values(score_col, ascending=ascending)

        excel_path = self.results_dir / f"grid_search_results_{self.suffix}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Results", index=False)

            stats_df = df.groupby(["p", "q"]).agg({
                score_col: ["mean", "std", "max" if not ascending else "min", "count"],
                "best_wi": "mean",
                "best_rmse": "mean",
                "best_mae": "mean",
            }).round(4)
            stats_df.to_excel(writer, sheet_name="Stats by p,q")

            idx_fn = df.groupby(["p", "q"])[score_col].idxmin if ascending else df.groupby(["p", "q"])[score_col].idxmax
            best_by_pq = df.loc[idx_fn()]
            best_by_pq[
                ["p", "q", score_col, "best_wi", "best_rmse", "best_mae", "best_epoch"]
            ].to_excel(writer, sheet_name="Best by p,q", index=False)

        print(f"\n  📊 Results saved to: {excel_path}")

        self.save_best_configuration()

    def save_best_configuration(self) -> None:
        """Save the best configuration and display a summary."""
        if not self.results:
            print("  ⚠️  No results to save")
            return

        df = pd.DataFrame(self.results)

        primary_col = f"best_{self.optimization_metric}"
        secondary_col = f"best_{self.secondary_metric}"
        primary_ascending = self.optimization_metric in ("rmse", "mae")
        secondary_ascending = self.secondary_metric in ("rmse", "mae")

        if primary_col in df.columns and secondary_col in df.columns:
            df_sorted = df.sort_values(
                [primary_col, secondary_col],
                ascending=[primary_ascending, secondary_ascending],
            )
        else:
            df_sorted = df.sort_values(primary_col, ascending=primary_ascending)

        best = df_sorted.iloc[0].to_dict()
        best = self._convert_numpy_types(best)

        top5_df = df_sorted.head(5)[
            ["p", "q", "transfer_learning", primary_col, secondary_col,
             "best_wi", "best_rmse", "best_mae", "best_epoch"]
        ]
        top5 = self._convert_list_types(top5_df.to_dict("records"))

        stats = {
            "total_combinations": len(self.results),
            "optimization_metric": self.optimization_metric,
            "secondary_metric": self.secondary_metric,
            f"best_{self.optimization_metric}": float(best[primary_col]),
            f"best_{self.secondary_metric}": float(best[secondary_col]) if secondary_col in best else 0.0,
            "best_wi": float(best["best_wi"]),
            "best_rmse": float(best["best_rmse"]),
            "best_mae": float(best["best_mae"]),
            "best_p": int(best["p"]),
            "best_q": int(best["q"]),
            "transfer_learning": bool(best.get("transfer_learning", self.use_transfer_learning)),
            "mean_score": float(df[primary_col].mean()),
            "std_score": float(df[primary_col].std()),
            "median_score": float(df[primary_col].median()),
        }

        config_summary = {
            "experiment_info": {
                "region": self.config.region,
                "spi_scale": self.spi_scale,
                "transfer_learning_used": self.use_transfer_learning,
                "optimization_metric": self.optimization_metric,
                "secondary_metric": self.secondary_metric,
            },
            "best_configuration": best,
            "top5_configurations": top5,
            "statistics": stats,
        }

        output_path = self.results_dir / f"best_configuration_{self.suffix}.json"
        with open(output_path, "w") as f:
            json.dump(config_summary, f, indent=2)

        self._display_best_configuration(best, primary_col, secondary_col)
        self._display_top5(top5, primary_col, secondary_col)
        self._save_summary_report(best, top5, primary_col, secondary_col)

    def _convert_numpy_types(self, obj: Dict) -> Dict:
        """Convert numpy scalar types to native Python types."""
        for key, value in obj.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                obj[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                obj[key] = float(value)
            elif isinstance(value, np.bool_):
                obj[key] = bool(value)
        return obj

    def _convert_list_types(self, records: List[Dict]) -> List[Dict]:
        """Convert numpy types in a list of records."""
        return [self._convert_numpy_types(record) for record in records]

    def _display_best_configuration(self, best: Dict, primary_col: str, secondary_col: str) -> None:
        """Display the best configuration."""
        self._print_header(f"🏆 BEST CONFIGURATION ({self.optimization_metric.upper()})")

        print(f"  {'Parameter':<20} {'Value':>15}")
        print(f"  {'─' * 38}")
        print(f"  {'p':<20} {best['p']:>15}")
        print(f"  {'q':<20} {best['q']:>15}")
        print(f"  {self.optimization_metric.upper():<20} {best[primary_col]:>15.4f}")
        print(f"  {self.secondary_metric.upper():<20} {best[secondary_col]:>15.4f}")
        print(f"  {'WI':<20} {best['best_wi']:>15.4f}")
        print(f"  {'RMSE':<20} {best['best_rmse']:>15.4f}")
        print(f"  {'MAE':<20} {best['best_mae']:>15.4f}")
        print(f"  {'Best Epoch':<20} {best['best_epoch']:>15}")
        print(f"  {'Transfer Learning':<20} {str(best['transfer_learning']):>15}")
        print(f"{'─' * 38}")

    def _display_top5(self, top5: List[Dict], primary_col: str, secondary_col: str) -> None:
        """Display the top 5 configurations."""
        self._print_section(f"📊 TOP 5 ({self.optimization_metric.upper()})")

        print(f"  {'#':<3} {'p':<4} {'q':<4} {self.optimization_metric.upper():<10} "
              f"{self.secondary_metric.upper():<10} {'WI':<10} {'RMSE':<10} "
              f"{'MAE':<10} {'Epoch':<6}")
        print(f"  {'─' * 75}")

        for i, cfg in enumerate(top5, 1):
            print(
                f"  {i:<3} {cfg['p']:<4} {cfg['q']:<4} "
                f"{cfg[primary_col]:<10.4f} {cfg[secondary_col]:<10.4f} "
                f"{cfg['best_wi']:<10.4f} {cfg['best_rmse']:<10.4f} "
                f"{cfg['best_mae']:<10.4f} {cfg['best_epoch']:<6}"
            )
        print(f"  {'─' * 75}")

    def _save_summary_report(self, best: Dict, top5: List[Dict], primary_col: str, secondary_col: str) -> None:
        """Save a plain-text summary report."""
        report_path = self.results_dir / f"grid_search_summary_{self.suffix}.txt"

        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("GRID SEARCH SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Region: {self.config.region}\n")
            f.write(f"SPI scale: {self.spi_scale}\n")
            f.write(f"Transfer Learning: {self.use_transfer_learning}\n")
            f.write(f"Optimization Metric (Primary): {self.optimization_metric.upper()}\n")
            f.write(f"Optimization Metric (Secondary): {self.secondary_metric.upper()}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("BEST CONFIGURATION:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  p: {best['p']}\n")
            f.write(f"  q: {best['q']}\n")
            f.write(f"  {self.optimization_metric.upper()}: {best[primary_col]:.4f}\n")
            f.write(f"  {self.secondary_metric.upper()}: {best[secondary_col]:.4f}\n")
            f.write(f"  WI: {best['best_wi']:.4f}\n")
            f.write(f"  RMSE: {best['best_rmse']:.4f}\n")
            f.write(f"  MAE: {best['best_mae']:.4f}\n")
            f.write(f"  Best Epoch: {best['best_epoch']}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("TOP 5 CONFIGURATIONS:\n")
            f.write("-" * 60 + "\n")
            for i, cfg in enumerate(top5, 1):
                f.write(
                    f"  {i}. p={cfg['p']}, q={cfg['q']}, "
                    f"{self.optimization_metric.upper()}={cfg[primary_col]:.4f}, "
                    f"{self.secondary_metric.upper()}={cfg[secondary_col]:.4f}, "
                    f"WI={cfg['best_wi']:.4f}, RMSE={cfg['best_rmse']:.4f}, "
                    f"MAE={cfg['best_mae']:.4f}\n"
                )

        print(f"\n  📄 Summary report saved to: {report_path}")
