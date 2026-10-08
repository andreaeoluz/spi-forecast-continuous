# Continuous SPI Forecasting with ConvLSTM Representation Learning

Code and results for the paper **"A Transfer Learning Framework for Continuous Drought Index Forecasting Based on ConvLSTM Representation Learning"**.

The framework forecasts the continuous SPI-3 value of every pixel at a target instant `q` months after a `p`-month input window, in Brazil's five macro-regions. A ConvLSTM autoencoder is pretrained on the climate inputs and its encoder is reused by the predictor (transfer learning, TL), which is compared with the same predictor trained from scratch.

---

## Related repositories

| Repository | Paper |
|---|---|
| [drought_forecast_binary](https://github.com/andreaeoluz/drought_forecast_binary) | Binary rare-drought forecasting with TL |
| [spi-forecast-continuous](https://github.com/andreaeoluz/spi-forecast-continuous) (this one) | Continuous SPI forecasting with TL |
| [drought-datasets](https://github.com/andreaeoluz/drought-datasets) | The two datasets used by the TL frameworks |
| [spi-forecast-ml-dl](https://github.com/andreaeoluz/spi-forecast-ml-dl) | ML vs. DL comparison for SPI forecasting |

---

## Method in brief

- **Data.** Monthly TerraClimate rasters (1980–2024): `pr`, `pet`, `soil`, `srad`, `vap`, `vs`, `tavg`, plus the SPI-3 history (8 input channels).
- **Split.** Training 1980–2019, validation 2020–2022, test 2023–2024. The SPI Gamma fit, the validity mask and the downsampling factor are derived from the training period only.
- **Model.** ConvLSTM encoder + dual temporal attention + multiscale temporal module + residual adapter + linear head. The forecast is persistence-anchored: last observed SPI + a learned correction.
- **Training.** Masked, severity-weighted MSE (weight 1 + λ·|SPI|, λ = 0.5); the checkpoint with the best validation WI is kept.
- **Grid.** p ∈ {3, 6, 9, 12} × q ∈ {1, 3, 6, 9, 12} × {TL, scratch} × 5 regions = 200 models.
- **Metrics.** Willmott's index of agreement (WI), RMSE and MAE.

---

## Repository structure

```
├── main.py                       # Entry point: precompute-spi | train-ae | grid-search | inference
├── run_full_retrain.sh           # Training: precompute-spi, train-ae, TL and scratch grid searches, 5 regions
├── run_all_regions.sh            # Test-period inference + analysis for all 200 configurations
├── config/                       # Experiment settings (experiment_config.py) and paths (paths.py)
├── data/                         # Raster loading, validity mask, SPI, datasets
├── models/                       # ConvLSTM cell, encoder, attention, multiscale module, autoencoder, SPI predictor
├── training/                     # Trainers and losses (severity-weighted MSE, reconstruction)
├── evaluation/                   # WI, RMSE, MAE
├── experiments/                  # SPI precomputation, autoencoder training, grid search
├── inference/                    # Test-period inference, GeoTIFF export, per-run analysis
├── utils/                        # Logging, GeoTIFF helpers, spatial helpers, seeds
│
├── generate_study_area_map.py    # Study-area map (shared by the binary and continuous papers)
├── analyze_spi_distribution.py   # Train/validation/test SPI statistics -> spi_split_stats.csv
├── extract_full_grid.py          # Validation grid -> grid_full.csv
├── analyze_grid_full.py          # Summaries of grid_full.csv used in the tables
├── analyze_pq_sensitivity.py     # WI heatmaps and TL vs. scratch figure
├── extract_test_period.py        # Test-period results -> test_period.csv, test_period_full.csv
├── analyze_test_period.py        # Test-period TL vs. scratch and spatial summary tables
├── generate_paper_maps.py        # Domain-mean bias and drought-class statistics of the best configurations
└── evaluate_autoencoder.py       # Autoencoder reconstruction and latent-space diagnostics
```

---

## Installation and data

```bash
git clone https://github.com/andreaeoluz/spi-forecast-continuous.git
cd spi-forecast-continuous
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.8+; a CUDA GPU is recommended for training.

The raw rasters are read from `DATA_BASE_PATH` (`config/paths.py`), one subfolder per region (`Norte`, `Nordeste`, `Centro-Oeste`, `Sudeste`, `Sul`) with files named `{region}_{year}_{month}.tif`. Edit `DATA_BASE_PATH` to point to your copy. Generated artifacts (SPI cache, checkpoints, rasters, analyses) are written to `outputs/`. `generate_study_area_map.py` also needs the IBGE shapefiles `BR_UF_2023.shp` and `Lim_america_do_sul_2021.shp`; set `IBGE_DATA_DIR` at the top of the script.

---

## Reproducing the paper

The default configuration is the one in the paper: `config/experiment_config.py` (SPI-3, persistence-anchored predictor, severity weight λ = 0.5, model selection by validation WI).

**1. Full pipeline**

```bash
./run_full_retrain.sh   # per region: precompute-spi, train-ae, grid search with TL and from scratch
./run_all_regions.sh    # test-period inference + analysis for all 200 configurations
```

`REUSE_EXISTING=1 ./run_full_retrain.sh` skips SPI and autoencoder steps whose outputs already exist.

**2. Single region / configuration**

```bash
python main.py precompute-spi --region Sul
python main.py train-ae --region Sul
python main.py grid-search --region Sul --use-transfer-learning
python main.py grid-search --region Sul --scratch
python main.py inference --region Sul --p 12 --q 1 --model-type pretrained
```

**3. Analyses, figures and tables**

```bash
python generate_study_area_map.py
python analyze_spi_distribution.py
python extract_full_grid.py && python analyze_grid_full.py
python analyze_pq_sensitivity.py
python extract_test_period.py && python analyze_test_period.py
python generate_paper_maps.py
python evaluate_autoencoder.py --region Sul
```

---

## Paper figures and tables

| Paper element | Produced by |
|---|---|
| Study-area map (`study_area_map`) | `generate_study_area_map.py` |
| Train/validation/test SPI statistics | `analyze_spi_distribution.py` |
| Validation p × q tables | `extract_full_grid.py` → `analyze_grid_full.py` |
| WI heatmaps (`heatmap_<region>_wi`) and TL vs. scratch figure (`tl_comparison_wi`) | `analyze_pq_sensitivity.py` |
| Test-period TL vs. scratch and spatial summary tables | `extract_test_period.py` → `analyze_test_period.py` |
| Domain-mean bias and predicted vs. observed drought area | `generate_paper_maps.py` |
| Autoencoder latent representation | `evaluate_autoencoder.py` |
| Best-prediction layouts (`Best_pred_<region>`) | GeoTIFFs exported by `main.py inference` (`pred_*`, `truth_*`, `error_*`, `rmse_map`), composed in ArcGIS |

---

## Included results

| File | Content |
|---|---|
| `grid_full.csv` | Validation WI/RMSE/MAE for every region × strategy × (p, q) |
| `test_period.csv` | Test-period WI/RMSE/MAE at q = 1 |
| `test_period_full.csv` | Test-period WI/RMSE/MAE for every (p, q) |
| `spi_split_stats.csv` | SPI-3 statistics per region and split (KS and trend tests) |

---

## License

MIT — see [LICENSE](LICENSE).
