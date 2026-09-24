"""dataset.py - Dataset for autoencoder pretraining and continuous SPI regression.

Derived from the binary framework's ClimateDataset. Everything specific to
binary drought classification has been removed: there is no SPI threshold,
no binary mask, no per-sample class weights / WeightedRandomSampler, and no
extreme-drought augmentation - none of those rare-positive-class techniques
apply to a continuous regression target. Standard shuffled sampling is used
for training instead (see training.base.BaseTrainer.create_dataloaders).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List

from .preprocessing import build_temporal_valid_mask


class ClimateDataset(Dataset):
    """
    Dataset for autoencoder pretraining or continuous SPI regression.

    Temporal definition (shared by both modes):

        - Input: X_t = [I_t, I_{t+1}, ..., I_{t+p-1}] (p months)

    For mode="regression":
        - Target: Y_t = SPI_{t+p+q-1}, the continuous SPI value at each
          valid pixel, q months after the end of the input window.
        - q is therefore the forecast horizon, in months.
        - q=1 -> the month immediately after the window.
        - q=3 -> the third month after the window.
        - No threshold is ever applied to the target, during training or
          evaluation: the model learns to regress the SPI value itself.

    For mode="autoencoder":
        - Input: X_t = [I_t, I_{t+1}, ..., I_{t+p-1}] (p months)
        - Reconstruction target: the full input sequence.
    """

    def __init__(
        self,
        data: np.ndarray,
        spi: Optional[np.ndarray],
        months: np.ndarray,
        p: int,
        q: int,
        valid_mask: Optional[np.ndarray] = None,
        temporal_decay: bool = True,
        mode: str = "regression",
        seed: int = 42,
        verbose: bool = True,
        use_anomalies: bool = True,
        min_valid_ratio: float = 0.7,
        include_spi_channel: bool = False,
    ):
        """
        Initialize the ClimateDataset.

        Args:
            data: Climate data (T, H, W, C).
            spi: SPI values (T, H, W) - required for mode='regression'.
            months: Month indices (T,).
            p: History length (input timesteps).
            q: Forecast horizon (timesteps ahead) - regression only.
            valid_mask: Validity mask (H, W).
            temporal_decay: If True, apply a linear temporal decay to the input.
            mode: 'autoencoder' or 'regression'.
            seed: Random seed for reproducibility.
            verbose: If True, print dataset information.
            use_anomalies: If True and mode='autoencoder', use monthly anomalies.
            min_valid_ratio: Minimum fraction of valid_mask pixels that must
                have finite (non-NaN/Inf) data at a given timestep for that
                timestep to be considered usable. Mirrors
                ExperimentConfig.data.min_valid_ratio.
            include_spi_channel: If True and `spi` is given, append the SPI
                series as an extra input channel (see ExperimentConfig's
                DataConfig.include_spi_history). Ignored if spi is None.
                Defaults to False (unaffected legacy behavior) since this
                class is also imported directly by other projects (e.g.
                drought_datasets' compute_dataset_statistics.py) that only
                want plain dataset/sample-count statistics and must stay
                decoupled from this project's model-architecture choices;
                only this project's own training code opts in explicitly.
        """
        assert mode in ["autoencoder", "regression"], f"Invalid mode: {mode}"

        # ============================================================
        # BASIC DATA
        # ============================================================
        self.data = data.astype(np.float32)
        self.spi = spi.astype(np.float32) if spi is not None else None
        self.months = np.array(months)
        self.p = p
        self.q = q if mode == "regression" else 0
        self.mode = mode
        self.seed = seed
        self.use_anomalies = use_anomalies and mode == "autoencoder"
        self.min_valid_ratio = min_valid_ratio
        self.verbose = verbose

        # ============================================================
        # APPEND SPI HISTORY AS AN INPUT CHANNEL (last channel)
        # ============================================================
        # Without its own recent SPI values as an input feature, the model
        # has to infer the entire absolute drought index from exogenous
        # climate variables alone; SPIPredictor's persistence+delta
        # formulation (models.spi_predictor) reads this last channel's
        # final timestep as the persistence anchor. Appending it here
        # (rather than upstream in the loader/normalizer) keeps it OUT of
        # ClimateNormalizer's per-band z-scoring, since SPI is already
        # standardized by construction.
        if include_spi_channel and self.spi is not None:
            self.data = self._append_spi_channel(self.data, self.spi)

        T, H, W, C = self.data.shape

        if verbose:
            print(f"  Dataset: T={T}, H={H}, W={W}, C={C}, p={p}, q={self.q}, mode={mode}")
            if self.use_anomalies:
                print("  🔹 Using monthly anomalies for autoencoder training")
            else:
                print("  🔹 Using raw values for training")

        # ============================================================
        # VALIDITY MASK
        # ============================================================
        self.valid_mask = self._prepare_valid_mask(valid_mask, H, W)
        self.H, self.W = H, W

        # ============================================================
        # TEMPORAL INDICES
        # ============================================================
        self.indices = self._build_indices()

        # ============================================================
        # REGRESSION TARGET
        # ============================================================
        self.regression_target = None
        if mode == "regression":
            self.regression_target = self._prepare_regression_target(H, W)

        # ============================================================
        # STATISTICS
        # ============================================================
        self._compute_statistics()

        # ============================================================
        # TRAINING DATA PREPARATION
        # ============================================================
        if self.use_anomalies:
            self.data_for_training = self._compute_anomalies(self.data, self.months)
        else:
            self.data_for_training = self.data

        # Channels-first layout for PyTorch.
        self.data_ch = np.transpose(self.data_for_training, (0, 3, 1, 2))

        # Linear temporal decay applied to the input sequence.
        if temporal_decay:
            self.temporal_weights = torch.linspace(0.5, 1.0, steps=p).view(p, 1, 1, 1)
        else:
            self.temporal_weights = None

        if verbose:
            print(f"  ✅ Dataset initialized with {len(self.indices)} samples")

    # ================================================================
    # PREPARATION METHODS
    # ================================================================

    def _prepare_valid_mask(self, valid_mask: Optional[np.ndarray], H: int, W: int) -> Optional[np.ndarray]:
        """Resize the validity mask to match the data resolution, if needed."""
        if valid_mask is None:
            return None

        if valid_mask.shape != (H, W):
            from skimage.transform import resize
            mask = resize(
                valid_mask.astype(np.float32),
                (H, W),
                order=0,
                preserve_range=True
            ).astype(bool)
        else:
            mask = valid_mask

        return mask

    @staticmethod
    def _append_spi_channel(data: np.ndarray, spi: np.ndarray) -> np.ndarray:
        """Append `spi` (T, H, W) as an extra last channel of `data` (T, H,
        W, C), resizing it first if its spatial resolution doesn't already
        match (mirrors _prepare_regression_target's defensive resize)."""
        H, W = data.shape[1:3]
        if spi.shape[1:] != (H, W):
            from skimage.transform import resize
            spi_resized = np.zeros((spi.shape[0], H, W), dtype=spi.dtype)
            for t in range(spi.shape[0]):
                spi_resized[t] = resize(spi[t], (H, W), order=0, preserve_range=True)
            spi = spi_resized

        return np.concatenate([data, spi[..., np.newaxis].astype(data.dtype)], axis=-1)

    def _prepare_regression_target(self, H: int, W: int) -> np.ndarray:
        """Build the per-timestep continuous SPI target.

        Unlike a binary drought mask, there is no threshold and no
        derivation step: the target is the SPI value itself.
        """
        if self.spi is None:
            raise ValueError("SPI required for mode='regression'")

        spi = self.spi
        if spi.shape[1:] != (H, W):
            from skimage.transform import resize
            spi_resized = np.zeros((spi.shape[0], H, W), dtype=spi.dtype)
            for t in range(spi.shape[0]):
                spi_resized[t] = resize(
                    spi[t],
                    (H, W),
                    order=0,
                    preserve_range=True
                )
            spi = spi_resized
            self.spi = spi

        return spi.astype(np.float32)

    # ================================================================
    # TEMPORAL INDICES
    # ================================================================

    def _build_indices(self) -> List[int]:
        """
        Build the list of valid starting indices for sampling.

        Delegates the actual per-timestep/window validity computation to
        `preprocessing.build_temporal_valid_mask`, which is the single
        source of truth for this logic (previously duplicated here).

        Also verifies there is no temporal leakage: for every candidate
        starting index t, the input window [t, t+p-1] must end strictly
        before the target index t+p+q-1.
        """
        temporal_mask = build_temporal_valid_mask(
            data_stack=self.data,
            months=self.months,
            valid_mask=self.valid_mask,
            p=self.p,
            q=self.q,
            mode=self.mode,
            min_valid_ratio=self.min_valid_ratio,
            verbose=self.verbose,
        )
        indices = np.where(temporal_mask)[0].tolist()

        if self.mode == "regression":
            for t in indices:
                input_end = t + self.p - 1
                target_idx = t + self.p + self.q - 1
                if not (input_end < target_idx):
                    raise ValueError(
                        "Temporal leakage detected: input window ends at "
                        f"index {input_end} but target index is {target_idx} "
                        f"(t={t}, p={self.p}, q={self.q}). The target must "
                        "strictly follow the input window."
                    )

        return indices

    # ================================================================
    # ANOMALY COMPUTATION
    # ================================================================

    def _compute_anomalies(self, data: np.ndarray, months: np.ndarray) -> np.ndarray:
        """
        Compute monthly anomalies (departure from the monthly climatology).

        Args:
            data: (T, H, W, C) climate data.
            months: (T,) month indices (1-12).

        Returns:
            Anomalies (T, H, W, C).
        """
        T, H, W, C = data.shape

        climatology = np.zeros((12, H, W, C), dtype=np.float32)
        counts = np.zeros(12, dtype=np.float32)

        for t, month in enumerate(months):
            m = int(month) - 1  # 0-based index
            if self.valid_mask is not None:
                mask_3d = self.valid_mask[:, :, np.newaxis]  # (H, W, 1)
                climatology[m] += data[t] * mask_3d
            else:
                climatology[m] += data[t]
            counts[m] += 1  # occurrences of month m, NOT valid-pixel count

        counts = np.maximum(counts, 1)
        climatology = climatology / counts[:, np.newaxis, np.newaxis, np.newaxis]

        anomalies = np.zeros_like(data)
        for t, month in enumerate(months):
            m = int(month) - 1
            if self.valid_mask is not None:
                mask_3d = self.valid_mask[:, :, np.newaxis]
                anomalies[t] = (data[t] - climatology[m]) * mask_3d
            else:
                anomalies[t] = data[t] - climatology[m]

        return anomalies

    # ================================================================
    # STATISTICS
    # ================================================================

    def _compute_statistics(self) -> None:
        """Compute dataset-level statistics (target mean/std, etc.)."""
        self.total_samples = len(self.indices)
        self.target_mean = None
        self.target_std = None

        if self.mode != "regression" or self.regression_target is None:
            return

        if self.valid_mask is not None:
            values = self.regression_target[:, self.valid_mask]
        else:
            values = self.regression_target.reshape(self.regression_target.shape[0], -1)

        values = values[np.isfinite(values)]
        if values.size > 0:
            self.target_mean = float(values.mean())
            self.target_std = float(values.std())
            if self.verbose:
                print(f"  📊 SPI target stats: mean={self.target_mean:.4f}, std={self.target_std:.4f}")

    # ================================================================
    # MAIN METHODS
    # ================================================================

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a dataset item.

        Input:  X_t = [I_t, I_{t+1}, ..., I_{t+p-1}]
        Target (regression): Y_t = SPI_{t+p+q-1}
        """
        t = self.indices[idx]

        # ============================================================
        # INPUT: p consecutive timesteps
        # ============================================================
        x = torch.from_numpy(self.data_ch[t:t + self.p])

        x = torch.nan_to_num(x, nan=0.0)
        x = torch.where(torch.isinf(x), torch.zeros_like(x), x)

        if self.temporal_weights is not None:
            x = x * self.temporal_weights

        if self.mode == "regression":
            # ============================================================
            # TARGET: qth timestep after the end of the window (continuous SPI)
            # ============================================================
            target_idx = t + self.p + self.q - 1
            y_spi = self.regression_target[target_idx]

            if self.valid_mask is not None:
                mask = torch.from_numpy(self.valid_mask.astype(np.float32))
            else:
                mask = torch.ones_like(torch.from_numpy(y_spi))

            return {
                "x": x,
                "y_spi": torch.from_numpy(y_spi).float(),
                "mask": mask,
                "month": self.months[target_idx],
                "idx": idx,
            }

        # Autoencoder mode
        if self.valid_mask is not None:
            mask = torch.from_numpy(self.valid_mask.astype(np.float32))
        else:
            mask = torch.ones((self.H, self.W), dtype=torch.float32)

        return {
            "x": x,
            "mask": mask,
            "idx": idx,
        }

    # ================================================================
    # AUXILIARY METHODS
    # ================================================================

    def get_loss_mask(self) -> Optional[torch.Tensor]:
        """Return the validity mask used for loss computation."""
        if self.valid_mask is not None:
            return torch.from_numpy(self.valid_mask.astype(np.float32))
        return None

    def get_data_info(self) -> Dict[str, Any]:
        """Return a summary of the dataset configuration."""
        info = {
            "mode": self.mode,
            "p": self.p,
            "q": self.q,
            "num_samples": len(self.indices),
            "use_anomalies": self.use_anomalies,
            "has_valid_mask": self.valid_mask is not None,
        }

        if self.use_anomalies:
            info["data_type"] = "monthly_anomalies"
        else:
            info["data_type"] = "raw_values"

        if self.mode == "regression":
            info["target_mean"] = self.target_mean
            info["target_std"] = self.target_std

        return info
