"""predictor.py - Inference predictor for continuous SPI forecasting.

Derived from the binary framework's InferencePredictor. Removed entirely:
probability calibration (Platt/Isotonic), decision-threshold
optimization/fallback, and morphological post-processing of a binary mask
- none of that applies to a continuous SPI output. The model's raw output
IS the final prediction; only the domain mask is still applied (to keep
invalid pixels out of saved rasters and metrics, never to threshold the
value itself).
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json

from config import ExperimentConfig
from config.paths import get_paths
from data import load_region_timeseries, ClimateNormalizer, load_spi_cache
from models import SPIPredictor
from evaluation.metrics import compute_regression_metrics
from utils import set_reproducible_seeds
from utils.logger import Logger


class InferencePredictor:
    """Handles inference on the test period, producing continuous SPI maps."""

    def __init__(
        self,
        config: ExperimentConfig,
        base_data_path: Path,
        p: int,
        q: int,
        model_type: str = "pretrained",
        model_dir: Optional[Path] = None,
    ):
        """
        Initialize the inference predictor.

        Args:
            config: Experiment configuration.
            base_data_path: Path to the raw data directory.
            p: History length.
            q: Forecast horizon.
            model_type: 'pretrained' or 'scratch'.
            model_dir: Optional custom directory to search for the model
                checkpoint before the standard grid-search locations.
        """
        self.config = config
        self.base_data_path = base_data_path
        self.p = p
        self.q = q
        self.model_type = model_type
        self.model_dir = Path(model_dir) if model_dir else None

        self.logger = Logger()

        self.paths = get_paths(
            config.region,
            spi_scale=config.spi.scale,
            p=p,
            q=q,
            model_type=model_type,
        )

        set_reproducible_seeds(config.random_seed)

        self._load_data()
        self._load_model()

        self.logger.info("📊 InferencePredictor initialized (continuous SPI forecasting)")

    def _load_data(self):
        """Load climate data, SPI, and the normalizer."""
        self.logger.info("Loading data...")
        out = load_region_timeseries(self.base_data_path, self.config)
        self.data = out["data"]
        self.years = out["years"]
        self.months = out["months"]
        self.metadata = out["metadata"]
        self.valid_mask = out["valid_mask"]
        self._resize_valid_mask()
        self._load_spi()
        self._load_normalizer()
        self._setup_temporal_indices()

    def _resize_valid_mask(self):
        """Resize the validity mask to match the data resolution."""
        if self.valid_mask is None:
            return
        target_shape = self.data.shape[1:3]
        if self.valid_mask.shape != target_shape:
            from skimage.transform import resize
            self.valid_mask = resize(
                self.valid_mask.astype(np.float32),
                target_shape,
                order=0,
                preserve_range=True
            ).astype(bool)

    def _load_spi(self):
        """Load and resize the cached SPI series."""
        spi, _ = load_spi_cache(self.config.spi.scale, self.paths["spi_cache_dir"])
        self.spi = spi
        if self.spi is None:
            return
        target_shape = self.data.shape[1:3]
        if self.spi.shape[1:] != target_shape:
            from skimage.transform import resize
            spi_resized = np.zeros(
                (self.spi.shape[0], target_shape[0], target_shape[1]),
                dtype=self.spi.dtype
            )
            for t in range(self.spi.shape[0]):
                spi_resized[t] = resize(
                    self.spi[t],
                    target_shape,
                    order=0,
                    preserve_range=True
                )
            self.spi = spi_resized

    def _load_normalizer(self):
        """Load the climate normalizer matching this model's training.

        A pretrained/transfer-learning model's encoder was fine-tuned on
        data normalized with the autoencoder's own normalizer (see
        GridSearch.load_data) - it must be served with that exact same
        normalizer. A scratch model was trained on a fresh normalizer fit
        on the regression task's own train split, saved separately so the
        two never collide (they legitimately differ).
        """
        if self.model_type == "scratch":
            candidates = [
                self.paths["grid_search_dir"] / "normalizer_scratch.json",
                self.paths["autoencoder_dir"] / "normalizer.json",  # legacy fallback
            ]
        else:
            candidates = [
                self.paths["autoencoder_dir"] / "normalizer.json",
                self.paths["grid_search_dir"] / "normalizer_pretrained.json",
            ]

        normalizer_path = next((p for p in candidates if p.exists()), None)
        if normalizer_path is None:
            raise FileNotFoundError(f"Normalizer not found among: {candidates}")

        self.normalizer = ClimateNormalizer.load(normalizer_path)
        self.logger.success(f"Normalizer loaded ({self.model_type}): {normalizer_path}")

    def _setup_temporal_indices(self):
        """Set up temporal indices for the test period."""
        self.time_idx = np.array([y * 12 + (m - 1) for y, m in zip(self.years, self.months)])
        split = self.config.split

        test_start = split.ym_to_int(split.test[0])
        test_end = split.ym_to_int(split.test[1])
        self.test_mask = (self.time_idx >= test_start) & (self.time_idx <= test_end)
        self.test_indices = np.where(self.test_mask)[0]
        self.test_data = self.data[self.test_mask]
        self.test_months = self.months[self.test_mask]
        self.spi_test = self.spi[self.test_mask] if self.spi is not None else None

        self.logger.info(f"Test period: {len(self.test_indices)} months")

    def _find_model_path(self) -> Path:
        """Find the model checkpoint path. Only searches locations
        consistent with self.model_type (never silently falls back to the
        other type's checkpoint)."""
        model_filename = f"model_p{self.p}_q{self.q}.pth"
        type_dir = (
            self.paths["grid_search_pretrained"] if self.model_type == "pretrained"
            else self.paths["grid_search_scratch"]
        )
        candidates = []
        if self.model_dir is not None:
            candidates.append(self.model_dir / model_filename)
        candidates += [
            type_dir / model_filename,
            self.paths["grid_search_dir"] / self.model_type / model_filename,
        ]
        for path in candidates:
            if path.exists():
                self.logger.info(f"Model found: {path}")
                return path
        raise FileNotFoundError(
            f"Model not found for p={self.p}, q={self.q}, type={self.model_type} "
            f"(checked: {[str(c) for c in candidates]})"
        )

    def _load_model(self):
        """Load the trained model checkpoint.

        The model architecture is built to match what's ACTUALLY in the
        checkpoint's state_dict, not blindly from `config.get_model_config`:
        both the multiscale temporal module and the residual adapter are
        randomly initialized when the model is constructed, and if a
        checkpoint trained *before* those components existed were loaded
        into a model built with them enabled, the random extra context
        would silently corrupt predictions at inference time instead of
        failing loudly. Detecting the architecture from the checkpoint
        itself keeps old and new checkpoints both loading correctly,
        regardless of the *current* default config.
        """
        checkpoint_path = self._find_model_path()
        self.logger.info(f"Loading model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        model_config = self.config.get_model_config("predictor")
        self._reconcile_architecture_with_checkpoint(model_config, state_dict)

        self.model = SPIPredictor(model_config).to(self.config.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            self.logger.debug(f"  (state_dict missing keys, kept at init: {missing})")
        if unexpected:
            self.logger.warning(f"  ⚠️ state_dict had unexpected keys, ignored: {unexpected}")

        self.model.eval()
        self.best_wi = checkpoint.get("best_wi", float('nan'))
        self.best_rmse = checkpoint.get("best_rmse", float('nan'))
        self.best_mae = checkpoint.get("best_mae", float('nan'))
        self.logger.success(
            f"Model loaded (val WI: {self.best_wi:.4f}, val RMSE: {self.best_rmse:.4f}, "
            f"val MAE: {self.best_mae:.4f})"
        )

    def _reconcile_architecture_with_checkpoint(self, model_config: dict, state_dict: dict) -> None:
        """
        Mutate `model_config` in place so `use_attention`, `use_multiscale`
        and `use_residual_adapter` match what the checkpoint was actually
        trained with, overriding whatever the current experiment config
        says.
        """
        component_prefixes = {
            "use_attention": "attention.",
            "use_multiscale": "multiscale.",
            "use_residual_adapter": "residual_adapter.",
        }

        for config_key, prefix in component_prefixes.items():
            present_in_checkpoint = any(k.startswith(prefix) for k in state_dict)
            configured = model_config.get(config_key, True)

            if present_in_checkpoint != configured:
                self.logger.warning(
                    f"  ⚠️ Architecture mismatch for '{config_key}': config={configured}, "
                    f"checkpoint has it={present_in_checkpoint}. Using the checkpoint's "
                    "architecture (this model was likely trained before/after this "
                    "component was added)."
                )
                model_config[config_key] = present_in_checkpoint

    def _prepare_data(
        self,
        data_subset: np.ndarray,
        months_subset: np.ndarray,
        spi_subset: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Prepare data for inference (normalize + channels-first).

        Mirrors ClimateDataset._append_spi_channel: when the model was
        trained with an SPI-history input channel
        (config.data.include_spi_history), `spi_subset` must be appended
        here the same way, as the LAST channel, or the trained model would
        receive a persistence anchor from the wrong channel index.
        """
        data_norm = self.normalizer.transform(data_subset, months_subset, self.valid_mask)
        data_norm = np.nan_to_num(data_norm, nan=0.0)

        if spi_subset is not None and self.config.data.include_spi_history:
            spi_ch = np.nan_to_num(spi_subset, nan=0.0)[..., np.newaxis].astype(data_norm.dtype)
            data_norm = np.concatenate([data_norm, spi_ch], axis=-1)

        return np.transpose(data_norm, (0, 3, 1, 2))

    def _get_sample_indices(self, data_len: int) -> List[int]:
        """Get valid sample indices for a data subset."""
        min_required = self.p + self.q
        if data_len <= min_required:
            return []
        # target_idx = t + q - 1, so the last usable t is data_len - q.
        return list(range(self.p, data_len - self.q + 1))

    def predict_subset(
        self,
        data_subset: np.ndarray,
        months_subset: np.ndarray,
        spi_subset: np.ndarray,
    ) -> Dict:
        """
        Make continuous SPI predictions on a data subset.

        Args:
            data_subset: Climate data (T, H, W, C).
            months_subset: Month indices (T,).
            spi_subset: Observed SPI values (T, H, W).

        Returns:
            Dict with predicted SPI maps, observed SPI maps, metrics
            (WI/RMSE/MAE), and sample count.
        """
        data_ch = self._prepare_data(data_subset, months_subset, spi_subset)
        T, C, H, W = data_ch.shape
        indices = self._get_sample_indices(T)

        if not indices:
            return {"pred": None, "obs": None, "metrics": None, "n_samples": 0}

        all_preds = []
        all_obs = []

        mask_tensor = None
        if self.valid_mask is not None:
            mask_tensor = torch.from_numpy(self.valid_mask.astype(np.float32)).to(self.config.device)

        with torch.no_grad():
            for t in indices:
                # Matches ClimateDataset's convention: window is
                # data_ch[t-p:t] (ends at t-1), target is the qth month
                # after that window, i.e. SPI_{(t-p)+p+q-1} = SPI_{t+q-1}.
                target_idx = t + self.q - 1
                x_seq = data_ch[t - self.p:t]
                x_tensor = torch.from_numpy(x_seq).float().unsqueeze(0).to(self.config.device)
                pred, _ = self.model(x_tensor, mask=mask_tensor)
                pred_np = pred.cpu().numpy()[0, 0]
                all_preds.append(pred_np)
                all_obs.append(spi_subset[target_idx])

        pred_stack = np.stack(all_preds)
        obs_stack = np.stack(all_obs)

        metrics = compute_regression_metrics(
            pred_stack, obs_stack,
            mask=self.valid_mask if self.valid_mask is not None else None,
        )

        return {
            "pred": pred_stack,
            "obs": obs_stack,
            "metrics": metrics,
            "n_samples": len(indices),
        }

    def run_inference(self) -> Dict:
        """
        Run inference on the (held-out, previously unseen) test period.

        The test period is used only to compute the final WI/RMSE/MAE
        metrics - there is no threshold or calibration step to resolve
        beforehand, since the output is already continuous.

        Returns:
            Dict with predictions, metrics, and configuration.
        """
        self.logger.header(f"INFERENCE - p={self.p}, q={self.q}")

        if self.spi_test is None:
            self.logger.error("SPI test data not available")
            return {"success": False}

        test_data_len = len(self.test_data)
        test_samples_possible = max(0, test_data_len - self.p - self.q + 1)

        self.logger.info(f"Test period: {len(self.test_indices)} months")
        self.logger.info(f"Possible samples: {test_samples_possible} (p={self.p}, q={self.q})")

        test_result = self.predict_subset(self.test_data, self.test_months, self.spi_test)

        if test_result["pred"] is None:
            self.logger.error("No test samples available")
            return {"success": False}

        test_result["success"] = True

        self.logger.success("Inference complete")
        self.logger.info(f"  Samples: {test_result['n_samples']}")
        self.logger.info(f"  WI: {test_result['metrics']['wi']:.4f}")
        self.logger.info(f"  RMSE: {test_result['metrics']['rmse']:.4f}")
        self.logger.info(f"  MAE: {test_result['metrics']['mae']:.4f}")

        return test_result

    def save_rasters(self, result: Dict):
        """
        Save continuous SPI prediction rasters as GeoTIFF: predicted SPI,
        observed SPI, and the prediction error (Predicted - Observed).
        Invalid pixels are written with the nodata sentinel, never
        thresholded into a binary category.

        Args:
            result: Result dict from run_inference().
        """
        if result is None or result.get("pred") is None:
            self.logger.error("No results to save")
            return

        try:
            import rasterio
        except ImportError:
            self.logger.warning("rasterio not available. Saving as numpy...")
            self._save_as_numpy(result)
            return

        pred = result["pred"]
        obs = result["obs"]

        pred_dir = self.paths.get("inference_pred", self.paths["inference_dir"] / "pred")
        truth_dir = self.paths.get("inference_truth", self.paths["inference_dir"] / "truth")
        error_dir = self.paths["inference_dir"] / "error"

        pred_dir.mkdir(parents=True, exist_ok=True)
        truth_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

        profile = self._create_raster_profile(pred.shape[1:])
        valid_mask = self._get_valid_mask(pred.shape[1:])
        nodata = profile['nodata']

        saved_count = 0
        for i in range(len(pred)):
            date_info = self._get_date_info(i)
            if date_info is None:
                continue
            year, month = date_info

            pred_i = pred[i].astype(np.float32).copy()
            pred_i[~valid_mask] = nodata
            self._save_raster(pred_dir / f"pred_{year}_{month:02d}.tif", pred_i, profile)

            obs_i = obs[i].astype(np.float32).copy()
            obs_i[~valid_mask] = nodata
            self._save_raster(truth_dir / f"truth_{year}_{month:02d}.tif", obs_i, profile)

            error_i = (pred[i] - obs[i]).astype(np.float32)
            error_i[~valid_mask] = nodata
            self._save_raster(error_dir / f"error_{year}_{month:02d}.tif", error_i, profile)

            saved_count += 1

        if saved_count > 0:
            self.logger.success(f"Rasters saved to: {pred_dir.parent}")
            self.logger.info(f"  {saved_count} files generated (pred/truth/error, all continuous SPI)")
            self.logger.info(f"  nodata={nodata} for invalid pixels")
        else:
            self.logger.warning("No rasters were saved")

    def _create_raster_profile(self, shape: Tuple[int, int]) -> Dict:
        """Create a raster profile for the continuous SPI rasters."""
        H, W = shape

        profile = {
            'driver': 'GTiff',
            'height': H,
            'width': W,
            'count': 1,
            'dtype': 'float32',
            'nodata': -9999.0,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }

        if 'crs' in self.metadata:
            profile['crs'] = self.metadata['crs']
        if 'transform' in self.metadata:
            profile['transform'] = self.metadata['transform']

        return profile

    def _get_valid_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Get the validity mask resized to the given shape."""
        if self.valid_mask is not None:
            if self.valid_mask.shape != shape:
                from skimage.transform import resize
                return resize(
                    self.valid_mask.astype(np.float32),
                    shape,
                    order=0,
                    preserve_range=True
                ).astype(bool)
            return self.valid_mask
        return np.ones(shape, dtype=bool)

    def _get_date_info(self, idx: int) -> Optional[Tuple[int, int]]:
        """
        Get (year, month) for a given sample index in the test period.

        `predict_subset()` builds sample `idx` from the local index
        `t = self.p + idx` within `test_data`, with the target at
        `t + self.q - 1` (window ends at `t-1`; target is the qth month
        after that window - same convention as ClimateDataset). The global
        target index therefore has to add `p`, `idx`, and `q - 1`, or every
        saved raster ends up labeled one month later than it should.
        """
        target_local_idx = self.p + idx + self.q - 1
        if target_local_idx < len(self.test_indices):
            global_idx = self.test_indices[target_local_idx]
            if global_idx < len(self.months):
                return self.years[global_idx], self.months[global_idx]
        return None

    def _save_raster(self, path: Path, data: np.ndarray, profile: Dict):
        """Save a raster as GeoTIFF."""
        import rasterio
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1)

    def _save_as_numpy(self, result: Dict):
        """Fallback: save results as numpy files when rasterio is unavailable."""
        pred = result["pred"]
        obs = result["obs"]

        output_dir = self.paths["inference_dir"] / "numpy_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "pred.npy", pred)
        np.save(output_dir / "obs.npy", obs)
        np.save(output_dir / "error.npy", pred - obs)

        if self.valid_mask is not None:
            np.save(output_dir / "valid_mask.npy", self.valid_mask)

        meta = {
            "n_samples": len(pred),
            "p": self.p,
            "q": self.q,
            "region": self.config.region,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.logger.success(f"Results saved as numpy: {output_dir}")
