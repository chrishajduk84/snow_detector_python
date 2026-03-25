"""Training pipeline for the dielectric property estimator.

Handles training loop, validation, checkpointing, and metrics.
Loss is a weighted combination of:
    - MSE on eps_r (permittivity regression)
    - MSE on tan_delta (loss tangent regression)
    - BCE on presence (binary material detection)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .property_estimator import PropertyEstimator


class PropertyEstimatorTrainer:
    """Training manager for the PropertyEstimator model.

    Args:
        model: PropertyEstimator instance.
        device: Torch device ('cpu', 'cuda', etc.).
        lr: Learning rate.
        weight_eps: Loss weight for permittivity MSE.
        weight_tan: Loss weight for loss tangent MSE.
        weight_presence: Loss weight for presence BCE.
    """

    def __init__(
        self,
        model: PropertyEstimator,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_eps: float = 1.0,
        weight_tan: float = 5.0,
        weight_presence: float = 2.0,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.lr = lr

        self.weight_eps = weight_eps
        self.weight_tan = weight_tan
        self.weight_presence = weight_presence

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.history: dict[str, list[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_eps_mae': [],
            'val_tan_mae': [],
            'val_presence_acc': [],
        }

    def train(
        self,
        dataset,
        epochs: int = 100,
        batch_size: int = 32,
        val_split: float = 0.2,
        checkpoint_dir: str | Path = "models",
    ) -> dict[str, list[float]]:
        """Run the training loop.

        Args:
            dataset: PyTorch Dataset yielding (input, target) tensors.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            val_split: Fraction of data for validation.
            checkpoint_dir: Directory to save best model checkpoint.

        Returns:
            Training history dictionary.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Split dataset
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Update scheduler T_max to actual epoch count
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_metrics = self._validate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_eps_mae'].append(val_metrics['eps_mae'])
            self.history['val_tan_mae'].append(val_metrics['tan_mae'])
            self.history['val_presence_acc'].append(val_metrics['presence_acc'])

            self.scheduler.step()

            # Checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), checkpoint_dir / 'best_model.pth')

            # Progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | "
                    f"εr MAE: {val_metrics['eps_mae']:.3f} | "
                    f"tanδ MAE: {val_metrics['tan_mae']:.4f} | "
                    f"Presence: {val_metrics['presence_acc']:.1%} | "
                    f"LR: {lr:.2e}"
                )

        print(f"\nBest validation loss: {best_val_loss:.4f}")
        return self.history

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            pred: (batch, 3, L) — [eps_r, tan_delta, presence]
            target: (batch, 3, L) — same layout
        """
        eps_loss = self.mse_loss(pred[:, 0, :], target[:, 0, :])
        tan_loss = self.mse_loss(pred[:, 1, :], target[:, 1, :])
        presence_loss = self.bce_loss(pred[:, 2, :], target[:, 2, :])

        return (
            self.weight_eps * eps_loss
            + self.weight_tan * tan_loss
            + self.weight_presence * presence_loss
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self._compute_loss(pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            count += x.size(0)

        return total_loss / count if count > 0 else 0.0

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_eps_ae = 0.0
        total_tan_ae = 0.0
        total_presence_correct = 0
        total_bins = 0
        count = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            loss = self._compute_loss(pred, y)

            total_loss += loss.item() * x.size(0)
            count += x.size(0)

            # Per-bin metrics
            n_bins = pred.shape[2]
            total_eps_ae += (pred[:, 0, :] - y[:, 0, :]).abs().sum().item()
            total_tan_ae += (pred[:, 1, :] - y[:, 1, :]).abs().sum().item()
            total_presence_correct += (
                ((pred[:, 2, :] > 0.5).float() == y[:, 2, :]).sum().item()
            )
            total_bins += x.size(0) * n_bins

        return {
            'loss': total_loss / count if count > 0 else 0.0,
            'eps_mae': total_eps_ae / total_bins if total_bins > 0 else 0.0,
            'tan_mae': total_tan_ae / total_bins if total_bins > 0 else 0.0,
            'presence_acc': total_presence_correct / total_bins if total_bins > 0 else 0.0,
        }

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights from a checkpoint file."""
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
