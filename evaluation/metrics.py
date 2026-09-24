"""metrics.py - Continuous SPI regression metrics.

Replaces the binary framework's classification metrics (CSI, MCC, F1,
precision/recall, FAR, bias, threshold search) with the three metrics
required for continuous spatial SPI forecasting:

    - Willmott's Index of Agreement (WI)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)

All three ignore invalid pixels and NaNs, and are numerically stable
against degenerate inputs (empty arrays, zero-variance observations).
"""

from typing import Dict, Optional
import numpy as np


def _flatten_valid(
    pred: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None,
):
    """Flatten pred/obs to 1D and keep only finite, valid-masked pairs."""
    pred_flat = np.asarray(pred, dtype=np.float64).reshape(-1)
    obs_flat = np.asarray(obs, dtype=np.float64).reshape(-1)

    valid = np.isfinite(pred_flat) & np.isfinite(obs_flat)

    if mask is not None:
        mask_flat = np.asarray(mask).reshape(-1)
        if mask_flat.size == pred_flat.size:
            valid = valid & (mask_flat > 0.5)
        elif mask_flat.size > 0 and pred_flat.size % mask_flat.size == 0:
            # A single (H, W) mask tiled over multiple timesteps/samples.
            reps = pred_flat.size // mask_flat.size
            mask_tiled = np.tile(mask_flat, reps)
            valid = valid & (mask_tiled > 0.5)

    return pred_flat[valid], obs_flat[valid]


def compute_wi(pred: np.ndarray, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Willmott's Index of Agreement.

        WI = 1 - sum((P_i - O_i)^2) / sum((|P_i - O_bar| + |O_i - O_bar|)^2)

    Range: [0, 1], where 1 is perfect agreement. Numerically stable against
    the degenerate case where the denominator is zero (which only happens
    when every observation and prediction equal the observed mean, i.e. a
    trivially perfect, constant series) by returning 1.0 there.

    Args:
        pred: Predicted SPI values (any shape).
        obs: Observed SPI values (same shape as pred).
        mask: Optional validity mask (1 = valid). Broadcastable/tileable
            against pred/obs after flattening.

    Returns:
        WI as a float, or NaN if there are no valid pixels to compare.
    """
    p, o = _flatten_valid(pred, obs, mask)
    if p.size == 0:
        return float('nan')

    o_bar = o.mean()
    numerator = np.sum((p - o) ** 2)
    denominator = np.sum((np.abs(p - o_bar) + np.abs(o - o_bar)) ** 2)

    if denominator < 1e-12:
        # Every observation (and every prediction) equals the observed
        # mean: a degenerate, perfectly-agreeing constant series.
        return 1.0

    wi = 1.0 - (numerator / denominator)
    # WI is mathematically bounded to [0, 1]; clip only to guard against
    # floating-point noise at the boundary.
    return float(np.clip(wi, 0.0, 1.0))


def compute_rmse(pred: np.ndarray, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Root Mean Squared Error, computed only over valid pixels."""
    p, o = _flatten_valid(pred, obs, mask)
    if p.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean((p - o) ** 2)))


def compute_mae(pred: np.ndarray, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean Absolute Error, computed only over valid pixels."""
    p, o = _flatten_valid(pred, obs, mask)
    if p.size == 0:
        return float('nan')
    return float(np.mean(np.abs(p - o)))


def compute_regression_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute WI, RMSE, and MAE in one pass over valid pixels.

    Args:
        pred: Predicted SPI values (any shape).
        obs: Observed SPI values (same shape as pred).
        mask: Optional validity mask.

    Returns:
        Dict with 'wi', 'rmse', 'mae', and 'n_valid'.
    """
    p, o = _flatten_valid(pred, obs, mask)

    if p.size == 0:
        return {"wi": float('nan'), "rmse": float('nan'), "mae": float('nan'), "n_valid": 0}

    o_bar = o.mean()
    sq_error = (p - o) ** 2
    numerator = np.sum(sq_error)
    denominator = np.sum((np.abs(p - o_bar) + np.abs(o - o_bar)) ** 2)

    wi = 1.0 if denominator < 1e-12 else float(np.clip(1.0 - numerator / denominator, 0.0, 1.0))
    rmse = float(np.sqrt(sq_error.mean()))
    mae = float(np.mean(np.abs(p - o)))

    return {"wi": wi, "rmse": rmse, "mae": mae, "n_valid": int(p.size)}


def is_better(current: Dict[str, float], best: Dict[str, float], primary: str = "wi") -> bool:
    """
    Model-selection comparator: higher WI is better (primary), lower RMSE
    breaks ties (secondary). MAE is diagnostic only and never drives
    selection.

    Args:
        current: Candidate metrics dict (from compute_regression_metrics).
        best: Current best metrics dict.
        primary: 'wi' (default, higher is better) or 'rmse'/'mae' (lower
            is better).

    Returns:
        True if `current` should replace `best`.
    """
    lower_is_better = primary in ("rmse", "mae")

    cur_val = current.get(primary, float('nan'))
    best_val = best.get(primary, float('nan'))

    if np.isnan(cur_val):
        return False
    if np.isnan(best_val):
        return True

    if lower_is_better:
        if cur_val < best_val - 1e-9:
            return True
        if cur_val > best_val + 1e-9:
            return False
        # Tie-break on RMSE (lower better) when primary is WI, or vice versa.
        return current.get("rmse", float('inf')) < best.get("rmse", float('inf'))

    if cur_val > best_val + 1e-9:
        return True
    if cur_val < best_val - 1e-9:
        return False
    return current.get("rmse", float('inf')) < best.get("rmse", float('inf'))
