"""spi.py - Standardized Precipitation Index computation and cache."""

import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from scipy.stats import gamma, norm

# Some existing spi_scale_*.pkl caches on disk were written under numpy>=2.0
# (module path "numpy._core.numeric"). This environment's numpy is <2.0,
# which only has the pre-rename "numpy.core" path, so plain pickle.load
# fails with ModuleNotFoundError. This is a read-only compatibility alias
# (numpy._core is API-identical to numpy.core for array reconstruction) -
# it never touches the cache files on disk, so it cannot introduce any
# numerical drift for the other projects (e.g. drought_datasets) that load
# these same caches via this function.
import numpy.core.numeric
import numpy.core.multiarray
import numpy.core.umath
import numpy.core.numerictypes

sys.modules.setdefault("numpy._core", np.core)
for _submodule in ("numeric", "multiarray", "umath", "numerictypes"):
    # Some other dependency (observed with torch's own numpy-2.0 compat
    # shim) may have already claimed the "numpy._core" package name without
    # registering its submodules, so check each dotted name independently
    # rather than gating on the top-level package alone.
    sys.modules.setdefault(f"numpy._core.{_submodule}", getattr(np.core, _submodule))


def build_fit_mask(years: np.ndarray, months: np.ndarray, fit_end: str) -> np.ndarray:
    """Boolean (T,) mask selecting the timesteps up to and including fit_end.

    Args:
        years: Calendar year for each timestep (T,).
        months: Calendar month (1-12) for each timestep (T,).
        fit_end: Last month of the fitting period, as "YYYY-MM"
            (normally SplitConfig.train_gs[1]).
    """
    end_y, end_m = map(int, fit_end.split("-"))
    ym = np.asarray(years) * 12 + (np.asarray(months) - 1)
    return ym <= end_y * 12 + (end_m - 1)


def compute_spi(
    precipitation: np.ndarray,
    months: np.ndarray,
    scale: int = 3,
    min_samples: int = 30,
    fit_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SPI using a rolling accumulation window.

    The distribution parameters (zero probability, Gamma shape/scale, and the
    mean/std of the empirical fallback) are estimated only on the timesteps
    selected by fit_mask, then applied to every timestep. Passing the training
    period as fit_mask keeps validation/test precipitation out of the index
    definition, so the SPI used to evaluate the future never depends on the
    future.

    Args:
        precipitation: Precipitation series (T, H, W).
        months: Calendar month (1-12) for each timestep (T,).
        scale: Accumulation window length, in months.
        min_samples: Minimum number of fitting samples required to fit a
            distribution for a given calendar month / pixel.
        fit_mask: Boolean (T,) mask of the timesteps used to fit the
            distributions (see build_fit_mask). None fits on the whole series,
            which leaks validation/test statistics into the index and is kept
            only for backward compatibility.

    Returns:
        (spi, delta_spi): SPI values and month-to-month SPI change.
    """
    T, H, W = precipitation.shape

    if fit_mask is None:
        warnings.warn(
            "compute_spi called without fit_mask: SPI distributions are fitted "
            "on the whole series, including validation/test months.",
            stacklevel=2,
        )
        fit_mask = np.ones(T, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != (T,):
            raise ValueError(f"fit_mask must have shape ({T},), got {fit_mask.shape}")

    print(f"\n  Calculating rolling sum (scale={scale})...")

    acc = np.full((T, H, W), np.nan, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            series = precipitation[:, i, j]
            series_pd = pd.Series(series)
            # min_periods=scale: the first scale-1 months (and any window
            # with a missing month) have no complete accumulation and stay
            # NaN. With min_periods=1 they held 1- or 2-month partial sums,
            # scored against the 3-month distribution of their calendar
            # month, which produced a spurious region-wide "extreme drought"
            # at the start of the record (e.g. Jan/1980 mean SPI ~ -3 in Sul).
            acc_pd = series_pd.rolling(window=scale, min_periods=scale).sum()
            acc[:, i, j] = acc_pd.values.astype(np.float32)
            acc[np.isnan(precipitation[:, i, j]), i, j] = np.nan

    print("  Computing SPI by month and pixel...")

    spi = np.full_like(acc, np.nan, dtype=np.float32)
    stats_by_month = {month: {"n_pixels": 0, "n_empirical": 0, "n_gamma": 0}
                      for month in range(1, 13)}

    for month in range(1, 13):
        month_mask = (months == month)
        month_indices = np.where(month_mask)[0]
        fit_indices = np.where(month_mask & fit_mask)[0]

        if len(fit_indices) < min_samples:
            continue

        acc_month = acc[fit_indices]

        for i in range(H):
            for j in range(W):
                series = acc_month[:, i, j]
                series_clean = series[np.isfinite(series)]

                if len(series_clean) < min_samples:
                    continue

                stats_by_month[month]["n_pixels"] += 1
                mean_m = series_clean.mean()
                std_m = series_clean.std()

                use_empirical = (std_m < 1e-6)

                if not use_empirical:
                    p_zero = (series_clean == 0).mean()
                    series_pos = series_clean[series_clean > 0]

                    if len(series_pos) >= min_samples:
                        try:
                            shape, _, scale_g = gamma.fit(series_pos, floc=0)
                            use_empirical = False
                            stats_by_month[month]["n_gamma"] += 1
                        except Exception:
                            use_empirical = True
                            stats_by_month[month]["n_empirical"] += 1
                    else:
                        use_empirical = True
                        stats_by_month[month]["n_empirical"] += 1
                else:
                    stats_by_month[month]["n_empirical"] += 1

                for t_idx, t in enumerate(month_indices):
                    x = acc[t, i, j]

                    if not np.isfinite(x):
                        continue

                    if use_empirical:
                        if std_m > 1e-6:
                            spi[t, i, j] = (x - mean_m) / std_m
                        else:
                            spi[t, i, j] = 0.0
                    else:
                        if x == 0:
                            Hx = p_zero
                        else:
                            Gx = gamma.cdf(x, shape, 0, scale_g)
                            Hx = p_zero + (1 - p_zero) * Gx

                        Hx = np.clip(Hx, 1e-6, 1 - 1e-6)
                        spi[t, i, j] = norm.ppf(Hx)

    spi = np.clip(spi, -6, 6)

    delta_spi = np.full_like(spi, np.nan, dtype=np.float32)
    delta_spi[1:] = spi[1:] - spi[:-1]
    if T > 0:
        delta_spi[0] = 0.0

    return spi, delta_spi


def analyze_spi_statistics(
    spi: np.ndarray,
    valid_mask: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    scale: int = 3,
) -> Dict:
    """Compute summary statistics and drought-category breakdown for an SPI series."""
    T, H, W = spi.shape

    valid_indices = np.where(valid_mask.flatten())[0]
    n_valid_pixels = len(valid_indices)

    spi_reshaped = spi.reshape(T, -1)
    spi_masked = spi_reshaped[:, valid_indices]
    spi_clean = spi_masked[~np.isnan(spi_masked)]

    stats = {
        "scale": scale,
        "total_timesteps": T,
        "total_pixels": n_valid_pixels,
        "total_values": T * n_valid_pixels,
        "valid_values": len(spi_clean),
    }

    if len(spi_clean) == 0:
        return stats

    stats.update({
        "mean": float(np.mean(spi_clean)),
        "std": float(np.std(spi_clean)),
        "min": float(np.min(spi_clean)),
        "max": float(np.max(spi_clean)),
        "median": float(np.median(spi_clean)),
    })

    categories = {
        "Extreme Drought (≤ -2.0)": (spi_clean <= -2.0).sum(),
        "Severe Drought (-2.0 to -1.5)": ((spi_clean > -2.0) & (spi_clean <= -1.5)).sum(),
        "Moderate Drought (-1.5 to -1.0)": ((spi_clean > -1.5) & (spi_clean <= -1.0)).sum(),
        "Near Normal (-1.0 to 1.0)": ((spi_clean > -1.0) & (spi_clean < 1.0)).sum(),
        "Moderate Wet (1.0 to 1.5)": ((spi_clean >= 1.0) & (spi_clean < 1.5)).sum(),
        "Severe Wet (1.5 to 2.0)": ((spi_clean >= 1.5) & (spi_clean < 2.0)).sum(),
        "Extreme Wet (≥ 2.0)": (spi_clean >= 2.0).sum(),
    }

    stats["categories"] = {k: int(v) for k, v in categories.items()}
    stats["categories_pct"] = {k: float(100 * v / len(spi_clean)) for k, v in categories.items()}

    return stats


def save_spi_cache(
    spi: np.ndarray,
    delta_spi: np.ndarray,
    scale: int,
    months: np.ndarray,
    cache_dir: Path,
    stats: Dict = None,
    fit_period: Optional[Tuple[str, str]] = None,
) -> Path:
    """Save SPI results to a pickle cache.

    fit_period records the ("YYYY-MM", "YYYY-MM") range the distributions were
    fitted on, so a cache can be told apart from the older full-period ones.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"spi_scale_{scale}.pkl"

    save_data = {
        "spi": spi.astype(np.float32),
        "delta_spi": delta_spi.astype(np.float32),
        "scale": scale,
        "months": months,
        "fit_period": fit_period,
    }

    if stats:
        save_data["statistics"] = stats

    with open(path, "wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_spi_cache(scale: int, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a cached SPI series."""
    path = cache_dir / f"spi_scale_{scale}.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["spi"], data["delta_spi"]
