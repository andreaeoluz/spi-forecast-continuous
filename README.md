# 🌧️ Continuous SPI Forecasting Framework

A framework for continuous spatial forecasting of the Standardized Precipitation Index (SPI) using spatiotemporal neural networks (ConvLSTM) with dual temporal attention and transfer learning.

This project is a derived version of `drought_forecast_binary`, which performed binary drought-event classification (SPI thresholded into drought/no-drought). Every binary-classification-specific component (SPI severity thresholds, class-imbalance handling, decision-threshold search, probability calibration, extreme-drought augmentation, morphological post-processing of binary masks) has been removed. The model now predicts the continuous SPI value itself, at every valid pixel, for a configurable forecast horizon.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Workflow](#workflow)
- [Metrics](#metrics)
- [Migration Notes](#migration-notes)

---

## 🎯 Overview

This framework predicts continuous SPI values, `SPI(x, y, t+q)`, 1 to 12 months ahead, using climate reanalysis data for Brazil's five geographic macro-regions.

### Key Features

- **🧠 ConvLSTM with Dual Temporal Attention**: Captures spatiotemporal patterns in climate data.
- **🔄 Transfer Learning**: Autoencoder pretraining for unsupervised representation learning, compared against training from scratch under an identical architecture/data/protocol.
- **📉 Masked MSE Regression**: Standard Mean Squared Error, computed only over spatially valid pixels.
- **📊 WI / RMSE / MAE Evaluation**: Willmott's Index of Agreement, Root Mean Squared Error, and Mean Absolute Error - no classification metrics.
- **🔁 Reproducibility**: Fixed seeds and deterministic operations where supported.

---

## 🏗️ Architecture

### Stage 1 - ConvLSTM Autoencoder (unsupervised pretraining)

```
Climate anomalies (B, T, C, H, W)
       |
ConvLSTM Encoder
       |
Dual Temporal Attention
       |
Latent representation (B, C_latent, H, W)
       |
Temporal Decoder
       |
Reconstruction (B, T, C_out, H, W)
```

### Stage 2 - SPI Predictor (continuous regression)

```
Input (B, T, C, H, W)
    ↓
ConvLSTM Encoder (optionally transferred from Stage 1)
    ↓
Dual Temporal Attention + Multiscale Temporal Module
    ↓
Residual Adapter
    ↓
Regression Decoder (linear output - no sigmoid/softmax)
    ↓
Continuous SPI map (B, 1, H, W)
```

The final layer is a plain linear projection: unrestricted, continuous, real-valued output.

---

## 📦 Installation

### Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Steps

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Raw Data

The raw TerraClimate rasters are expected under `DATA_BASE_PATH`, defined in `config/paths.py`, organized as one subfolder per region (e.g. `Sul/`, `Norte/`) containing files named `{region}_{year}_{month}.tif`. Edit `DATA_BASE_PATH` there if your data lives elsewhere.

---

## 📁 Project Structure

```
drought_forecast_regression/
├── config/
│   ├── experiment_config.py             # All experiment settings
│   └── paths.py                         # Path management (namespaced by SPI scale)
│
├── data/
│   ├── loader.py                        # load_region_timeseries
│   ├── preprocessing.py                 # Downsampling, validity mask, normalization
│   ├── dataset.py                       # ClimateDataset (autoencoder / regression modes)
│   └── spi.py                           # SPI computation and cache
│
├── models/
│   ├── convlstm_cell.py                 # ConvLSTMCell
│   ├── encoder.py                       # ConvLSTMEncoder
│   ├── decoder.py                       # ReconstructionDecoder, RegressionDecoder
│   ├── attention.py                     # DualTemporalAttention
│   ├── multiscale_temporal.py           # MultiscaleTemporalModule
│   ├── autoencoder.py                   # ConvLSTMAutoencoder
│   └── spi_predictor.py                 # SPIPredictor (continuous regression)
│
├── training/
│   ├── base.py                          # BaseTrainer (standard shuffled DataLoader)
│   ├── autoencoder.py                   # AutoencoderTrainer
│   ├── predictor.py                     # SPIPredictorTrainer (MSE, WI-based selection)
│   └── losses.py                        # WeightedSmoothL1Loss, MaskedMSELoss
│
├── evaluation/
│   └── metrics.py                       # WI, RMSE, MAE
│
├── experiments/
│   ├── precompute_spi.py                # SPI precomputation
│   ├── train_autoencoder.py             # Autoencoder training
│   └── grid_search.py                   # p x q x {TL, Scratch} grid search
│
├── inference/
│   ├── predictor.py                     # InferencePredictor (continuous SPI output)
│   ├── run.py                           # Inference CLI
│   └── analyze.py                       # WI/RMSE/MAE timeseries, spatial SPI maps
│
├── evaluate_autoencoder.py              # Standalone autoencoder evaluation report
├── analyze_pq_sensitivity.py            # Cross-region p/q sensitivity analysis (WI/RMSE)
├── run_all_regions.sh                   # Batch inference + analyze, all regions/p/model-type
│
├── outputs/                             # Outputs (created at runtime)
│   └── {region}/
│       ├── autoencoder/                 # Trained autoencoder + normalizer
│       ├── spi_cache/                   # Cached SPI series
│       ├── grid_search/                 # Grid search models and results
│       ├── inferences/                  # Continuous SPI rasters (pred/truth/error)
│       └── analysis/                    # Metrics and plots
│
├── requirements.txt
├── main.py
└── README.md
```

---

## ⚙️ Configuration

All settings are centralized in `config/experiment_config.py`:

```python
@dataclass
class ExperimentConfig:
    region: str = "Sul"
    data: DataConfig = field(default_factory=DataConfig)
    spi: SPIConfig = field(default_factory=SPIConfig)          # scale only - no threshold
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelArchConfig = field(default_factory=ModelArchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)       # "mse"
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)  # primary_metric="wi"
    p_values: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    q_values: List[int] = field(default_factory=lambda: [1, 3, 6, 9, 12])
    use_transfer_learning: bool = False
    random_seed: int = 42
```

### Climate Variables

The framework uses 7 TerraClimate bands (unchanged from the binary framework):

| Band | Description | Unit |
|------|-------------|------|
| `pr` | Precipitation | mm/month |
| `pet` | Potential Evapotranspiration | mm/month |
| `soil` | Soil Moisture | mm |
| `srad` | Downward Shortwave Radiation | W/m² |
| `vap` | Vapor Pressure | kPa |
| `vs` | Wind Speed | m/s |
| `tavg` | Mean Air Temperature | °C |

---

## 🚀 Usage

### 1. Precompute SPI

```bash
python main.py precompute-spi
```

### 2. Train the Autoencoder (Stage 1)

```bash
python main.py train-ae
```

### 3. Grid Search (Stage 2: p x q x {Transfer Learning, Scratch})

```bash
# Training from scratch (config.use_transfer_learning=False)
python main.py grid-search

# Transfer learning from the pretrained autoencoder
python main.py grid-search --use-transfer-learning

# Model selection by RMSE instead of WI
python main.py grid-search --optimization-metric rmse
```

### 4. Inference (continuous SPI forecast on the held-out test period)

```bash
# Automatic: uses the best configuration by validation WI
python main.py inference --region Sul

# Specific model
python main.py inference --region Sul --p 12 --q 1 --model-type pretrained

# List available trained models
python main.py inference --list-models
```

### 5. Standalone Analysis Scripts

```bash
# Detailed autoencoder reconstruction/latent-space report
python evaluate_autoencoder.py

# Inference results report (WI/RMSE/MAE timeseries, spatial SPI maps)
python -m inference.analyze --region Sul --p 12 --q 1 --model-type pretrained

# Cross-region p/q sensitivity analysis
python analyze_pq_sensitivity.py --base-dir outputs
```

### 6. Batch Inference + Analysis (all regions)

`run_all_regions.sh` runs `main.py inference` followed by `inference.analyze`
for all 5 regions x all `p` values x both model types (`pretrained`,
`scratch`), with `q=1` and the default SPI scale (SPI-3). Run it from the
project root (where `main.py` and `inference/` live):

```bash
./run_all_regions.sh
```

Failures for individual combinations are logged to `failed_runs.log` and do
not stop the run; a full log is written to `run_all_regions_<timestamp>.log`.
To cover other `q` values or SPI scales, edit the `PS`/`Q` arrays and add
`--spi-scale` to the `main.py inference` / `inference.analyze` calls inside
the script.

---

## 🔄 Workflow

```
precompute_spi.py -> SPI(x, y, t) for every timestep, continuous (no threshold)
train_autoencoder.py -> unsupervised ConvLSTM Autoencoder pretraining
grid_search.py -> for each (p, q, strategy): train SPIPredictor with masked MSE,
                   select the checkpoint with the best validation WI
inference/run.py -> continuous SPI forecast on the test period
inference/analyze.py -> WI/RMSE/MAE timeseries + observed/predicted/error SPI maps
```

---

## 📊 Metrics

### Primary Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **WI** | Willmott's Index of Agreement | [0, 1] | Higher |
| **RMSE** | Root Mean Squared Error | [0, ∞) | Lower |
| **MAE** | Mean Absolute Error | [0, ∞) | Lower |

Model selection uses maximum validation WI as the primary criterion, with minimum validation RMSE breaking ties; MAE is reported as an additional diagnostic. The test set is only ever used for the final, single evaluation of the selected checkpoint.

No classification metrics (CSI, MCC, precision, recall, F1, FAR, bias, accuracy, ROC-AUC, PR-AUC) are computed anywhere in this framework.

---

## 🔀 Migration Notes

This project was derived from `drought_forecast_binary` (binary drought-event classification) into a continuous SPI regression framework. Summary of what changed:

| Component | Binary framework | This framework |
|---|---|---|
| Target | `SPI(x,y,t+q) <= threshold` (binary mask) | `SPI(x,y,t+q)` (continuous value) |
| Output layer | Linear + sigmoid | Linear (unrestricted) |
| Loss | Focal Loss / weighted BCE | Masked MSE |
| Class imbalance | Weighted sampling, pos_weight, augmentation | None (not applicable) |
| Decision threshold | Optimized on validation (MCC/CSI) | None |
| Calibration | Platt Scaling / Isotonic Regression | None |
| Post-processing | Morphological cleanup of binary masks | None |
| Metrics | CSI, MCC, F1, precision, recall, FAR, bias | WI, RMSE, MAE |
| Encoder / Attention / Multiscale / Residual Adapter / Autoencoder | - | Unchanged (task-agnostic) |
| Chronological split, transfer learning comparison, p/q grid | - | Unchanged |
