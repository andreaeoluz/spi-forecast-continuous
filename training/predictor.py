"""predictor.py - SPI regression predictor trainer.

Derived from the binary framework's PredictorTrainer. Every classification-
specific component has been removed:

  - Focal Loss / weighted BCE -> replaced by MaskedMSELoss (training.losses).
  - pos_weight / dynamic alpha / prevalence -> gone (no class to weight).
  - Decision-threshold search (find_best_threshold) -> gone.
  - Probability calibration (Platt/Isotonic) -> gone.

Model selection uses the primary optimization metric from
config.optimization (default: validation WI, maximized; RMSE breaks ties -
see evaluation.metrics.is_better). The optimizer parameter-group /
encoder-freezing machinery for the Transfer Learning vs. Scratch comparison
is unchanged, since it only concerns weight initialization and learning
rate, not the task.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from .base import BaseTrainer
from .losses import build_loss
from utils.logger import Logger, Colors
from evaluation.metrics import compute_regression_metrics, is_better


class SPIPredictorTrainer(BaseTrainer):
    """Trainer for the ConvLSTM SPI regression predictor."""

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
        save_path: Path,
        freeze_encoder_first_epoch: bool = False,
        optimization_metric: str = None,
        load_attention: bool = True,
    ):
        super().__init__(model, train_dataset, val_dataset, config, save_path)

        self.logger = Logger()

        self.freeze_encoder_first_epoch = freeze_encoder_first_epoch
        self.freeze_encoder_epochs = getattr(config.training, 'freeze_encoder_epochs', 5)
        self.encoder_lr_factor = getattr(config.training, 'encoder_lr_factor', 0.1)
        # Whether the attention module (not just the encoder) was actually
        # transferred from the pretrained autoencoder. Only meaningful when
        # freeze_encoder_first_epoch=True; used by _create_optimizer to
        # decide whether attention gets the gentle encoder LR (it holds
        # pretrained weights) or the full decoder LR (it's random).
        self.load_attention = load_attention

        self.optimization_metric = (
            optimization_metric or config.optimization.primary_metric
        )

        self.criterion = build_loss(
            loss_name=config.loss.name, reduction="mean", severity_weight=config.loss.severity_weight
        ).to(self.device)

        self.loss_mask = None
        if hasattr(train_dataset, 'valid_mask') and train_dataset.valid_mask is not None:
            self.loss_mask = torch.from_numpy(train_dataset.valid_mask.astype(np.float32))
            self.loss_mask = self.loss_mask.to(self.device).unsqueeze(0).unsqueeze(0)

            valid_pixels = train_dataset.valid_mask.sum()
            total_pixels = train_dataset.valid_mask.size
            self.logger.info(f"📊 Loss mask: {valid_pixels:,} valid pixels / {total_pixels:,} total")
        else:
            self.logger.info("📊 No loss mask - all pixels will be used")

        self.optimizer = self._create_optimizer()

        self.logger.info("📊 SPIPredictorTrainer initialized:")
        self.logger.info(f"   Optimization metric: {self.optimization_metric.upper()}")
        self.logger.info(f"   Freeze epochs: {self.freeze_encoder_epochs if self.freeze_encoder_first_epoch else 0}")
        self.logger.info(f"   Encoder LR factor: {self.encoder_lr_factor}")

    def _create_optimizer(self):
        """
        Create the optimizer with differentiated learning rates for the
        encoder/attention and decoder parameter groups.

        The per-group learning rates are kept constant for the entire
        training run. The attention module gets the gentle encoder LR
        when it was actually transferred from the autoencoder
        (self.load_attention=True), since it then holds pretrained
        weights just like the encoder; otherwise it's randomly
        initialized and gets the full decoder LR like the rest of the
        untrained head.
        """
        if not self.freeze_encoder_first_epoch:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )

        encoder_params = []
        attention_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if name.startswith('encoder'):
                encoder_params.append(param)
            elif name.startswith('attention'):
                attention_params.append(param)
            else:
                decoder_params.append(param)

        encoder_lr = self.config.training.learning_rate * self.encoder_lr_factor
        decoder_lr = self.config.training.learning_rate

        param_groups = []

        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': encoder_lr,
                'name': 'encoder'
            })

        if attention_params:
            attention_lr = encoder_lr if self.load_attention else decoder_lr
            attention_label = 'attention (transferred)' if self.load_attention else 'attention (random)'
            param_groups.append({
                'params': attention_params,
                'lr': attention_lr,
                'name': attention_label
            })

        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': decoder_lr,
                'name': 'decoder'
            })

        self.logger.info("   Optimizer groups:")
        for group in param_groups:
            self.logger.info(f"      {group['name']}: LR={group['lr']:.2e}")

        return torch.optim.Adam(
            param_groups,
            weight_decay=self.config.training.weight_decay,
        )

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            x = batch["x"].to(self.device)
            y_spi = batch["y_spi"].to(self.device).float().unsqueeze(1)

            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)
            attn_mask = mask if mask is not None else self.loss_mask
            pred, _ = self.model(x, mask=attn_mask)

            loss = self.criterion(pred, y_spi, mask=mask if mask is not None else self.loss_mask)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)

        return total_loss / n_samples if n_samples > 0 else float('inf')

    def _collect_predictions(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """Collect predicted and observed SPI values over valid pixels only."""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y_spi = batch["y_spi"].to(self.device).float().unsqueeze(1)

                mask = batch.get("mask", None)
                attn_mask = None
                if mask is not None:
                    all_masks.append(mask.cpu().numpy().flatten())
                    attn_mask = mask.to(self.device).unsqueeze(1)
                else:
                    attn_mask = self.loss_mask

                pred, _ = self.model(x, mask=attn_mask)

                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(y_spi.cpu().numpy().flatten())

        preds = np.array(all_preds)
        targets = np.array(all_targets)

        if all_masks:
            masks = np.concatenate(all_masks)
            valid_idx = masks > 0.5
            preds = preds[valid_idx]
            targets = targets[valid_idx]

        valid = np.isfinite(preds) & np.isfinite(targets)
        preds = preds[valid]
        targets = targets[valid]

        return preds, targets

    def validate(self, val_loader) -> Dict:
        """Validate the model: compute WI, RMSE, MAE over valid pixels."""
        preds, targets = self._collect_predictions(val_loader)

        if len(preds) == 0:
            return {"metrics": {"wi": 0.0, "rmse": float('inf'), "mae": float('inf'), "n_valid": 0},
                    "wi": 0.0, "rmse": float('inf'), "mae": float('inf')}

        metrics = compute_regression_metrics(preds, targets)

        return {
            "metrics": metrics,
            "wi": metrics["wi"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        }

    def train(self):
        """Main training loop. Model selection: maximum validation WI
        (ties broken by minimum validation RMSE) - see evaluation.metrics.is_better."""
        self.logger.header("SPI PREDICTOR TRAINING")
        self.logger.info(f"Optimization metric: {self.optimization_metric.upper()}")

        if self.freeze_encoder_first_epoch:
            if self.load_attention:
                self.logger.info("🔹 Strategy: transfer learning (encoder + attention transferred)")
            else:
                self.logger.info("🔹 Strategy: transfer learning (encoder only, random attention)")
        else:
            self.logger.info("🔹 Strategy: scratch (fully random initialization)")

        train_loader, val_loader = self.create_dataloaders(shuffle_train=True)

        best_metrics = {"wi": float('-inf'), "rmse": float('inf'), "mae": float('inf')}
        patience_counter = 0
        min_epochs = 10

        if self.freeze_encoder_first_epoch:
            self.model.freeze_encoder(freeze=True)
            self.logger.info(f"🔒 Encoder frozen for {self.freeze_encoder_epochs} epochs")
            self.logger.info("   Attention is NOT frozen (random init)")

        best_epoch = 0

        for epoch in range(self.config.training.epochs):
            if self.freeze_encoder_first_epoch and epoch == self.freeze_encoder_epochs:
                self.model.freeze_encoder(freeze=False)
                self.logger.info(f"🔓 Encoder unfrozen at epoch {epoch}")
                self.logger.info(
                    f"   Keeping LRs: Encoder={self.config.training.learning_rate * self.encoder_lr_factor:.2e}, "
                    f"Decoder={self.config.training.learning_rate:.2e}"
                )

            train_loss = self.train_epoch(train_loader)
            val_results = self.validate(val_loader)
            epoch_metrics = val_results["metrics"]

            improved = is_better(epoch_metrics, best_metrics, primary=self.optimization_metric)
            if improved:
                best_metrics = epoch_metrics
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(epoch, {
                    "best_wi": epoch_metrics["wi"],
                    "best_rmse": epoch_metrics["rmse"],
                    "best_mae": epoch_metrics["mae"],
                    "train_loss": train_loss,
                })
            else:
                patience_counter += 1

            status = f"{Colors.GREEN}✓{Colors.RESET}" if improved else f"ES {patience_counter}/{self.config.training.patience}"

            self.logger.info(
                f"Epoch {epoch:3d} | Loss {train_loss:.4f} | "
                f"WI {epoch_metrics['wi']:.4f} | RMSE {epoch_metrics['rmse']:.4f} | "
                f"MAE {epoch_metrics['mae']:.4f} | {status}"
            )

            if self.freeze_encoder_first_epoch and epoch < self.freeze_encoder_epochs:
                self.logger.debug(f"   🔒 Encoder frozen (epoch {epoch + 1}/{self.freeze_encoder_epochs})")

            if epoch >= min_epochs and patience_counter >= self.config.training.patience:
                self.logger.warning(f"Early stopping at epoch {epoch}")
                break

        checkpoint = torch.load(self.save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        final_val = self.validate(val_loader)
        checkpoint["best_wi"] = final_val["wi"]
        checkpoint["best_rmse"] = final_val["rmse"]
        checkpoint["best_mae"] = final_val["mae"]
        checkpoint["best_epoch"] = best_epoch
        checkpoint["final_metrics"] = final_val["metrics"]

        torch.save(checkpoint, self.save_path)

        self.logger.success(
            f"Final - WI: {final_val['wi']:.4f} | RMSE: {final_val['rmse']:.4f} | "
            f"MAE: {final_val['mae']:.4f} | Best epoch: {best_epoch}"
        )

        return final_val["wi"]
