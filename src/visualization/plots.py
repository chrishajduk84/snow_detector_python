"""Plotting functions for radar dielectric profiling.

All functions produce matplotlib figures suitable for live display
or saving to file.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_range_profile(
    complex_iq: np.ndarray,
    config=None,
    title: str = "Range Profile",
    ax_mag=None,
    ax_phase=None,
) -> plt.Figure | None:
    """Plot magnitude and phase of a complex range profile.

    Args:
        complex_iq: Complex array (num_samples,) or (num_samples, num_rx).
        config: Optional RadarConfig for physical range axis.
        title: Plot title.
        ax_mag: Optional existing axis for magnitude subplot.
        ax_phase: Optional existing axis for phase subplot.

    Returns:
        Figure if new axes were created, else None.
    """
    created_fig = False
    if ax_mag is None or ax_phase is None:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 6))
        created_fig = True

    if complex_iq.ndim == 1:
        complex_iq = complex_iq[:, np.newaxis]

    num_samples, num_rx = complex_iq.shape

    # X-axis: range bins or physical range
    if config is not None:
        x = np.arange(num_samples) * config.range_resolution_m
        xlabel = "Range (m)"
    else:
        x = np.arange(num_samples)
        xlabel = "Range Bin"

    for rx in range(num_rx):
        label = f"RX {rx}"
        ax_mag.plot(x, 20 * np.log10(np.abs(complex_iq[:, rx]) + 1e-12), label=label, alpha=0.8)
        ax_phase.plot(x, np.rad2deg(np.angle(complex_iq[:, rx])), label=label, alpha=0.8)

    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title(title)
    ax_mag.legend()
    ax_mag.grid(True, alpha=0.3)

    ax_phase.set_xlabel(xlabel)
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.legend()
    ax_phase.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_property_map(
    predictions: np.ndarray,
    ground_truth: np.ndarray | None = None,
    config=None,
    title: str = "Dielectric Property Map",
) -> plt.Figure:
    """Plot predicted (and optionally true) material properties vs range.

    Args:
        predictions: (3, num_bins) array — [eps_r, tan_delta, presence].
        ground_truth: Optional (3, num_bins) array for comparison.
        config: Optional RadarConfig for physical range axis.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    num_bins = predictions.shape[1]

    if config is not None:
        x = np.arange(num_bins) * config.range_resolution_m
        xlabel = "Range (m)"
    else:
        x = np.arange(num_bins)
        xlabel = "Range Bin"

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Permittivity
    axes[0].plot(x, predictions[0], 'b-', label='Predicted', linewidth=1.5)
    if ground_truth is not None:
        axes[0].plot(x, ground_truth[0], 'r--', label='True', linewidth=1.5)
    axes[0].set_ylabel("εr")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(predictions[0].max() * 1.2, 2.0))

    # Loss tangent
    axes[1].plot(x, predictions[1], 'b-', label='Predicted', linewidth=1.5)
    if ground_truth is not None:
        axes[1].plot(x, ground_truth[1], 'r--', label='True', linewidth=1.5)
    axes[1].set_ylabel("tan(δ)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Presence
    axes[2].fill_between(x, predictions[2], alpha=0.4, color='blue', label='Predicted')
    if ground_truth is not None:
        axes[2].step(x, ground_truth[2], 'r-', label='True', linewidth=1.5, where='mid')
    axes[2].set_ylabel("Material Present")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "Training Progress",
) -> plt.Figure:
    """Plot training and validation metrics over epochs.

    Args:
        history: Dict with keys like 'train_loss', 'val_loss', 'val_eps_mae', etc.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Loss
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], label='Train')
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], label='Validation')
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Permittivity MAE
    ax = axes[0, 1]
    if 'val_eps_mae' in history:
        ax.plot(epochs, history['val_eps_mae'], color='green')
    ax.set_ylabel("MAE")
    ax.set_title("εr Mean Absolute Error")
    ax.grid(True, alpha=0.3)

    # Loss tangent MAE
    ax = axes[1, 0]
    if 'val_tan_mae' in history:
        ax.plot(epochs, history['val_tan_mae'], color='orange')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("tan(δ) Mean Absolute Error")
    ax.grid(True, alpha=0.3)

    # Presence accuracy
    ax = axes[1, 1]
    if 'val_presence_acc' in history:
        ax.plot(epochs, [a * 100 for a in history['val_presence_acc']], color='purple')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Material Presence Accuracy")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_layer_comparison(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    config=None,
) -> plt.Figure:
    """Side-by-side comparison of predicted vs true layer properties.

    Args:
        predicted: (3, num_bins) — predicted [eps_r, tan_delta, presence].
        ground_truth: (3, num_bins) — true labels.
        config: Optional RadarConfig for range axis.

    Returns:
        Matplotlib Figure.
    """
    num_bins = predicted.shape[1]

    if config is not None:
        x = np.arange(num_bins) * config.range_resolution_m
        xlabel = "Range (m)"
    else:
        x = np.arange(num_bins)
        xlabel = "Range Bin"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Permittivity heatmap-style comparison
    ax = axes[0]
    ax.plot(x, ground_truth[0], 'r-', linewidth=2, label='True εr')
    ax.plot(x, predicted[0], 'b--', linewidth=2, label='Predicted εr')
    ax.fill_between(x, ground_truth[0], alpha=0.15, color='red')
    ax.fill_between(x, predicted[0], alpha=0.15, color='blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative Permittivity (εr)")
    ax.set_title("Permittivity Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Error profile
    ax = axes[1]
    eps_error = np.abs(predicted[0] - ground_truth[0])
    tan_error = np.abs(predicted[1] - ground_truth[1])
    ax.plot(x, eps_error, 'b-', label='|Δεr|', linewidth=1.5)
    ax.plot(x, tan_error * 100, 'r-', label='|Δtan(δ)| × 100', linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Absolute Error")
    ax.set_title("Prediction Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Layer Prediction Comparison", fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig
