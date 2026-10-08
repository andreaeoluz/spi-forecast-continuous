#!/usr/bin/env python3
"""Train vs Validation vs Test SPI-3 distributional comparison, for all
five regions. Purpose: check whether the held-out test period (2023-01 to
2024-12) is climatically representative of the historical record used for
training/validation, as context for interpreting validation-grid vs
test-period WI/RMSE differences in the continuous SPI regression results.

No severity threshold is involved (the regression target is the continuous
SPI value), so this is a plain distributional comparison: descriptive
stats per split, a two-sample KS test (train vs test, validation vs test),
and a Kendall's tau trend test on the full 1980-2024 regional-mean series.

Data: SPI-3 caches at outputs/<region>/spi_cache/spi_scale_3.pkl.
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import numpy.core
import numpy.core.numeric
import numpy.core.multiarray
sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)

REPO_ROOT = Path(__file__).parent
OUTPUTS = REPO_ROOT / "outputs"

REGIONS = [
    ("Norte", "North"),
    ("Nordeste", "Northeast"),
    ("Centro-Oeste", "Center-West"),
    ("Sudeste", "Southeast"),
    ("Sul", "South"),
]

SPLITS = {
    "Train": ("1980-01", "2019-12"),
    "Validation": ("2020-01", "2022-12"),
    "Test": ("2023-01", "2024-12"),
}


def load_region_spi(region_pt: str):
    path = OUTPUTS / region_pt / "spi_cache" / "spi_scale_3.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)
    spi = d["spi"]
    T = spi.shape[0]
    # A pixel is valid if its SPI is defined in any month. Not spi[0]: the
    # first scale-1 months are undefined for every pixel (no complete
    # accumulation window), which would leave the mask empty.
    valid_mask = np.isfinite(spi).any(axis=0)
    dates = pd.date_range("1980-01-01", periods=T, freq="MS")
    return spi, valid_mask, dates


def split_mask(dates, start, end):
    return (dates >= pd.Period(start, freq="M").start_time) & (dates <= pd.Period(end, freq="M").end_time)


def regional_mean_series(spi, valid_mask):
    flat = spi.reshape(spi.shape[0], -1)[:, valid_mask.flatten()]
    return np.nanmean(flat, axis=1)


rows = []
for region_pt, region_en in REGIONS:
    spi, valid_mask, dates = load_region_spi(region_pt)
    mean_series = regional_mean_series(spi, valid_mask)
    flat_valid = spi.reshape(spi.shape[0], -1)[:, valid_mask.flatten()]

    split_vals = {}
    for split_name, (start, end) in SPLITS.items():
        m = split_mask(dates, start, end)
        vals = flat_valid[m].flatten()
        vals = vals[np.isfinite(vals)]
        split_vals[split_name] = vals
        rows.append({
            "Region": region_en, "Split": split_name,
            "n_pixel_months": len(vals), "mean": np.mean(vals), "std": np.std(vals),
            "p05": np.percentile(vals, 5), "median": np.median(vals), "p95": np.percentile(vals, 95),
        })

    ks_tt = sps.ks_2samp(split_vals["Train"], split_vals["Test"])
    ks_vt = sps.ks_2samp(split_vals["Validation"], split_vals["Test"])
    # Undefined months (the first scale-1 ones) are skipped: kendalltau
    # returns NaN if any value is NaN.
    t_idx = np.arange(len(mean_series))
    ok = np.isfinite(mean_series)
    tau, tau_p = sps.kendalltau(t_idx[ok], mean_series[ok])

    test_mean = split_vals["Test"].mean()
    trainval_mean = np.concatenate([split_vals["Train"], split_vals["Validation"]]).mean()

    print(f"[{region_en}] Train mean={split_vals['Train'].mean():+.3f}  "
          f"Val mean={split_vals['Validation'].mean():+.3f}  Test mean={test_mean:+.3f}  "
          f"(shift vs train+val = {test_mean - trainval_mean:+.3f})")
    print(f"           KS(Train,Test): stat={ks_tt.statistic:.3f} p={ks_tt.pvalue:.4g}  "
          f"KS(Val,Test): stat={ks_vt.statistic:.3f} p={ks_vt.pvalue:.4g}  "
          f"Kendall tau (1980-2024)={tau:+.4f} p={tau_p:.4g}")

df = pd.DataFrame(rows)
df.to_csv("spi_split_stats.csv", index=False)
pd.set_option("display.width", 160)
print()
print(df.round(3).to_string(index=False))
