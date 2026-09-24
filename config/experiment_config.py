"""experiment_config.py - Centralized project configuration.

Continuous SPI forecasting framework, derived from the binary drought-event
framework. Every binary-classification-specific configuration knob (SPI
severity threshold, class-imbalance handling, decision-threshold search,
probability calibration, extreme-drought augmentation, rolling-origin
multi-split evaluation) has been removed - see the project README for the
full migration rationale. What remains (regions, downsampling, data split,
model architecture, autoencoder pretraining, p/q grid) is unchanged from
the binary framework, since none of it is specific to the prediction task.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ============================================================================
# PER-MODULE CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Data and downsampling configuration."""

    bands: List[str] = field(default_factory=lambda: [
        "pr", "pet", "soil", "srad", "vap", "vs", "tavg"
    ])

    # Region-specific downsampling factor.
    # 3x3: fairer comparison across regions (South, Southeast, Northeast, Center-West).
    # 5x5: computational feasibility for the larger North region (lower memory footprint).
    # Keys match the raw data folder names on disk and must stay in Portuguese.
    downsample_config: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "Sul": (3, 3),
        "Sudeste": (3, 3),
        "Nordeste": (3, 3),
        "Centro-Oeste": (3, 3),
        "Norte": (5, 5),
    })

    downsample_h: int = 3      # Fallback for regions not listed above
    downsample_w: int = 3      # Fallback for regions not listed above
    min_valid_ratio: float = 0.7

    # When True, the last-p-months SPI history is appended as an extra
    # input channel (see data.dataset.ClimateDataset), and SPIPredictor
    # forecasts SPI as persistence + a learned delta rather than an
    # unconditioned absolute value (see model.predict_delta). Without an
    # autoregressive SPI signal in the input, the model has to learn the
    # entire absolute drought index from noisier exogenous climate
    # variables alone, which under MSE training collapses severe/extreme
    # SPI values toward the climatological mean (empirically confirmed:
    # predicted std was 43-64% of observed std, extreme-class magnitudes
    # underestimated by 25-75%).
    include_spi_history: bool = True


@dataclass
class SPIConfig:
    """SPI (Standardized Precipitation Index) configuration.

    Only the accumulation scale remains - there is no severity threshold
    here anymore. SPI is used directly as the continuous regression target,
    never thresholded into a binary drought/no-drought label.
    """
    scale: int = 3
    min_samples: int = 30


@dataclass
class SplitConfig:
    """Chronological train/validation/test split.

    Unchanged from the binary framework: chronological partitioning is a
    data-pipeline concern independent of the prediction target.
    """

    train_gs: Tuple[str, str] = ("1980-01", "2019-12")
    val_gs: Tuple[str, str] = ("2020-01", "2022-12")
    train_final: Tuple[str, str] = ("1980-01", "2022-12")
    test: Tuple[str, str] = ("2023-01", "2024-12")

    def ym_to_int(self, year_month: str) -> int:
        y, m = map(int, year_month.split('-'))
        return y * 12 + (m - 1)

    def get_end_indices_gs(self) -> Dict:
        return {
            "train_end": self.ym_to_int(self.train_gs[1]),
            "val_end": self.ym_to_int(self.val_gs[1]),
        }

    def get_indices(self) -> Dict:
        return {
            "trainval_end": self.ym_to_int(self.train_final[1]),
            "test_end": self.ym_to_int(self.test[1]),
        }

    def get_test_period_length(self) -> int:
        start = self.ym_to_int(self.test[0])
        end = self.ym_to_int(self.test[1])
        return end - start + 1


@dataclass
class ModelArchConfig:
    """Model architecture. Unchanged from the binary framework - the
    ConvLSTM encoder, dual temporal attention, multiscale temporal module,
    and residual adapter are all task-agnostic spatiotemporal-representation
    components."""
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    kernel_size: int = 3
    attention_dropout: float = 0.3
    use_attention: bool = True
    attention_weight: float = 0.3

    use_multiscale: bool = True
    multiscale_fast_window: int = 3
    multiscale_weight: float = 0.3

    use_residual_adapter: bool = True

    # SPI predictor only (ignored by the autoencoder): forecast SPI as
    # persistence (last observed SPI, from the appended SPI-history input
    # channel - see DataConfig.include_spi_history) plus a learned delta,
    # instead of an unconditioned absolute value. The delta head is
    # zero/near-zero initialized (see RegressionDecoder), so training
    # starts from a pure-persistence forecast and only learns corrections -
    # this preserves extreme SPI magnitudes that a from-scratch absolute
    # regression under MSE tends to smooth toward the climatological mean.
    predict_delta: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    epochs: int = 300
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3

    freeze_encoder_epochs: int = 5
    encoder_lr_factor: float = 0.3
    unfreeze_after: int = 5

    # Partial/progressive transfer learning: when enabled, the encoder is
    # unfrozen one ConvLSTM layer at a time - deepest (task-specific) first,
    # shallowest (general) last - instead of all at once.
    progressive_unfreeze: bool = False
    progressive_unfreeze_epoch_gap: int = 5
    layer_lr_decay: float = 0.7


@dataclass
class LossConfig:
    """Loss function configuration. Mean Squared Error is the only
    supported objective for continuous SPI regression.

    severity_weight (default 0.0, i.e. plain unweighted MSE): when > 0,
    each valid pixel-month's squared error is scaled by
    (1 + severity_weight * |observed SPI|), so months with more extreme
    observed SPI contribute proportionally more to the loss. Mitigates
    (does not eliminate) the residual mean-reversion attenuation that
    persists under the persistence-anchored formulation even after
    removing the dominant source of attenuation (see
    training.losses.MaskedMSELoss).
    """
    name: str = "mse"
    severity_weight: float = 0.0


@dataclass
class AutoencoderConfig:
    p: int = 12
    train: bool = True
    use_anomalies: bool = True
    reconstruct_full_sequence: bool = True
    sequence_length: int = 12
    decay_rate: float = 0.3
    skip_weight: float = 0.1
    attention_weight: float = 0.3
    diversity_weight: float = 0.01
    variable_weights: List[float] = field(default_factory=lambda: [1.0] * 7)


@dataclass
class OptimizationConfig:
    """Metrics and model selection.

    Model selection uses validation WI as the primary criterion (higher is
    better) and validation RMSE as the tie-breaking secondary criterion
    (lower is better) - see training.predictor.SPIPredictorTrainer and
    experiments.grid_search.GridSearch.
    """
    primary_metric: str = "wi"
    secondary_metric: str = "rmse"

    characterization_metrics: List[str] = field(default_factory=lambda: [
        "wi", "rmse", "mae"
    ])


# ============================================================================
# MAIN CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Region (matches the raw data folder name; keep in Portuguese)
    region: str = "Sul"

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    spi: SPIConfig = field(default_factory=SPIConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelArchConfig = field(default_factory=ModelArchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Grid search
    p_values: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    q_values: List[int] = field(default_factory=lambda: [1, 3, 6, 9, 12])

    # Flags
    use_transfer_learning: bool = True

    # Reproducibility
    random_seed: int = 42

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def num_bands(self) -> int:
        """Number of bands in the raw climate GeoTIFFs (data.loader validates
        against this) - NOT the model's input channel count, see
        model_input_dim."""
        return len(self.data.bands)

    @property
    def model_input_dim(self) -> int:
        """Number of channels SPIPredictor/ConvLSTMAutoencoder actually
        consume: the raw climate bands, plus one appended SPI-history
        channel when DataConfig.include_spi_history is set (see
        data.dataset.ClimateDataset._append_spi_channel)."""
        return self.num_bands + (1 if self.data.include_spi_history else 0)

    @property
    def device(self) -> torch.device:
        return torch.device(self.training.device)

    @property
    def optimization_metric(self) -> str:
        return self.optimization.primary_metric

    # ========================================================================
    # METHODS
    # ========================================================================

    def get_downsample(self, region: str) -> Tuple[int, int]:
        """Return the downsampling factor for a region."""
        return self.data.downsample_config.get(
            region,
            (self.data.downsample_h, self.data.downsample_w)
        )

    def get_downsample_info(self, region: str) -> Dict:
        """Return detailed downsampling information for a region."""
        ds_h, ds_w = self.get_downsample(region)

        preservation = {
            (2, 2): "~85% of events preserved",
            (3, 3): "~56% of events preserved",
            (4, 4): "~35% of events preserved",
        }.get((ds_h, ds_w), "unknown")

        return {
            "region": region,
            "downsample_h": ds_h,
            "downsample_w": ds_w,
            "preservation_estimate": preservation,
            "area_reduction": ds_h * ds_w,
        }

    def get_model_config(self, model_type: str) -> dict:
        """Return the constructor config for a given model type."""
        base = {
            "input_dim": self.model_input_dim,
            "hidden_dims": self.model.hidden_dims,
            "kernel_size": self.model.kernel_size,
            "use_attention": self.model.use_attention,
            "attention_dropout": self.model.attention_dropout,
            "attention_weight": self.model.attention_weight,
        }

        if model_type == "autoencoder":
            base["output_dim"] = self.model_input_dim
            base["reconstruct_full_sequence"] = self.autoencoder.reconstruct_full_sequence
            base["sequence_length"] = self.autoencoder.sequence_length
            base["decay_rate"] = self.autoencoder.decay_rate

        elif model_type == "predictor":
            # Single continuous SPI map per sample (the target month
            # t+p+q-1); q selects which future month is being forecast,
            # exactly as in the binary framework's direct multi-horizon
            # design (one model per q, not a multi-step decoder).
            base["output_dim"] = 1
            base["use_multiscale"] = self.model.use_multiscale
            base["multiscale_fast_window"] = self.model.multiscale_fast_window
            base["multiscale_weight"] = self.model.multiscale_weight
            base["use_residual_adapter"] = self.model.use_residual_adapter
            base["predict_delta"] = self.model.predict_delta and self.data.include_spi_history

        return base

    def get_available_metrics(self) -> List[str]:
        return ["wi", "rmse", "mae"]

    def to_dict(self) -> Dict:
        """Convert to a plain dict (for logging)."""
        return {
            "region": self.region,
            "num_bands": self.num_bands,
            "device": str(self.device),
            "random_seed": self.random_seed,
            "downsample": self.get_downsample(self.region),
            "use_transfer_learning": self.use_transfer_learning,
            "spi_scale": self.spi.scale,
        }
