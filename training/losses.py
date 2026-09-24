"""losses.py - Loss functions for reconstruction and continuous SPI regression.

All classification losses (Focal Loss, weighted BCE) have been removed:
they optimize a probability against a binary label, which no longer
exists. The regression objective is Mean Squared Error, computed only over
spatially valid pixels.
"""

import torch
import torch.nn as nn
from typing import Optional


class WeightedSmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss with variable weights per channel/variable.

    Used for autoencoder reconstruction with different importance weights
    for each climate variable. Unrelated to the prediction target, so kept
    unchanged from the binary framework.

    Args:
        variable_weights: Tensor of weights per variable/channel
        beta: Threshold for L1/L2 transition (default: 0.1)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        variable_weights: torch.Tensor,
        beta: float = 0.1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.register_buffer("weights", variable_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W)
            target: Targets (B, C, H, W)

        Returns:
            Weighted Smooth L1 Loss
        """
        diff = torch.abs(pred - target)

        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )

        if self.weights is not None:
            weighted_loss = loss * self.weights.view(1, -1, 1, 1)
        else:
            weighted_loss = loss

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error loss for continuous SPI regression, computed only
    over spatially valid pixels.

        loss = mean((prediction - target)^2)   over valid pixels only

    Invalid pixels (outside the domain mask) never contribute to the
    gradient. NaN/Inf targets are also excluded defensively, even though
    the target should already be finite for every valid pixel by
    construction (see data.dataset.ClimateDataset).

    Optional severity weighting (severity_weight > 0): plain MSE gives
    every valid pixel-month equal weight, so the overwhelming majority of
    near-normal months (|SPI| < 1) dominates the gradient and the
    optimal MSE predictor shrinks toward the conditional mean wherever the
    inputs cannot fully resolve the true anomaly - this is the same
    mean-reversion mechanism documented for the original absolute-value
    formulation (Section 2.3.2 of the paper), acting one level removed on
    the persistence-anchored correction instead of the raw SPI value. When
    severity_weight > 0, each valid pixel's squared error is scaled by
    (1 + severity_weight * |target|), so months with a more extreme
    observed SPI contribute proportionally more to the loss and the
    optimizer is pushed to reduce attenuation there specifically, at the
    cost of some accuracy on near-normal months. severity_weight = 0
    (the default) reproduces plain unweighted MSE exactly.
    """

    def __init__(self, reduction: str = 'mean', severity_weight: float = 0.0):
        super().__init__()
        if reduction not in ('mean', 'none'):
            raise ValueError("MaskedMSELoss only supports reduction in {'mean', 'none'}")
        if severity_weight < 0:
            raise ValueError(f"severity_weight must be >= 0, got {severity_weight}")
        self.reduction = reduction
        self.severity_weight = severity_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted SPI (B, 1, H, W) or (B, H, W).
            target: Observed SPI, same shape as pred.
            mask: Optional validity mask, broadcastable to pred's shape
                (1 = valid, 0 = invalid). If None, every finite pixel is
                treated as valid.

        Returns:
            Scalar (severity-weighted) masked MSE if reduction='mean';
            otherwise the unweighted per-element squared error with
            invalid/NaN pixels zeroed out (the caller is then responsible
            for masking, weighting, and reducing). The 'none' path is left
            unweighted regardless of severity_weight, since its consumers
            expect the raw per-pixel squared error.
        """
        finite = torch.isfinite(target)
        if mask is not None:
            valid = (mask > 0.5) & finite
        else:
            valid = finite

        target_safe = torch.where(valid, target, torch.zeros_like(target))
        sq_error = (pred - target_safe) ** 2
        sq_error = torch.where(valid, sq_error, torch.zeros_like(sq_error))

        if self.reduction == 'none':
            return sq_error

        if self.severity_weight > 0:
            weight = torch.where(
                valid, 1.0 + self.severity_weight * torch.abs(target_safe), torch.zeros_like(target_safe)
            )
            denom = weight.sum().clamp_min(1e-8)
            return (weight * sq_error).sum() / denom

        n_valid = valid.sum().clamp_min(1)
        return sq_error.sum() / n_valid


def build_loss(loss_name: str = "mse", **kwargs) -> torch.nn.Module:
    """
    Build the regression loss function.

    Args:
        loss_name: Only 'mse' is supported for the continuous SPI
            regression predictor.
        severity_weight: Optional, forwarded to MaskedMSELoss (default 0.0,
            i.e. plain unweighted MSE). See MaskedMSELoss's docstring.

    Returns:
        Loss module.
    """
    if loss_name == "mse":
        return MaskedMSELoss(
            reduction=kwargs.get("reduction", "none"),
            severity_weight=kwargs.get("severity_weight", 0.0),
        )

    raise ValueError(f"Unsupported loss: {loss_name}. Only 'mse' is supported.")
