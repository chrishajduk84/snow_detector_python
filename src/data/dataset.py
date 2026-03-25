"""PyTorch Dataset for dielectric profile training data.

Loads complex I/Q data from HDF5 sessions and returns
(input_channels, range_bin_labels) pairs suitable for training
the property estimator model.
"""

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .hdf5_store import load_session


class DielectricProfileDataset(Dataset):
    """Dataset that loads radar frames from HDF5 session files.

    Handles both real-measurement sessions (shared labels) and synthetic
    sessions (per-sample labels) transparently.

    Input shape:  (num_rx * 2, num_samples) — real and imaginary parts
    Target shape: (3, num_samples) — [eps_r, tan_delta, presence] per bin
    """

    def __init__(
        self,
        h5_paths: list[str | Path],
        transform: Callable | None = None,
        chirp_aggregate: str = "mean",
    ):
        """
        Args:
            h5_paths: List of HDF5 session file paths to load.
            transform: Optional transform applied to (input, target) tuple.
            chirp_aggregate: How to aggregate chirps within a frame.
                'mean' — average all chirps (default, reduces noise).
                'first' — use only the first chirp.
                'all' — each chirp becomes a separate sample.
        """
        self.transform = transform
        self.chirp_aggregate = chirp_aggregate
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []

        for path in h5_paths:
            self._load_session(Path(path))

    def _load_session(self, path: Path) -> None:
        """Load all samples from one HDF5 session file.

        Detects whether labels are shared (real session) or per-sample
        (synthetic session) and handles each format accordingly.
        """
        with h5py.File(path, 'r') as f:
            labels_grp = f['labels']
            shared_labels = 'range_bin_labels' in labels_grp

            if shared_labels:
                range_bin_labels = np.array(labels_grp['range_bin_labels'])
            else:
                range_bin_labels = None

            for key in sorted(f['samples'].keys()):
                complex_iq = np.array(f['samples'][key]['complex_iq'])

                if not shared_labels:
                    range_bin_labels = np.array(labels_grp[key]['range_bin_labels'])

                if self.chirp_aggregate == "mean":
                    aggregated = np.mean(complex_iq, axis=0)
                    self._add_sample(aggregated, range_bin_labels)
                elif self.chirp_aggregate == "first":
                    self._add_sample(complex_iq[0], range_bin_labels)
                elif self.chirp_aggregate == "all":
                    for chirp_idx in range(complex_iq.shape[0]):
                        self._add_sample(complex_iq[chirp_idx], range_bin_labels)

    def _add_sample(self, frame: np.ndarray, labels: np.ndarray) -> None:
        """Convert a (num_samples, num_rx) complex frame to channels."""
        # frame: (num_samples, num_rx) complex
        # → (num_rx * 2, num_samples) real channels
        num_samples, num_rx = frame.shape
        channels = np.zeros((num_rx * 2, num_samples), dtype=np.float32)
        for rx in range(num_rx):
            channels[2 * rx] = frame[:, rx].real
            channels[2 * rx + 1] = frame[:, rx].imag

        # labels: (num_bins, 3) → (3, num_bins)
        target = labels[:num_samples].T.astype(np.float32)

        self._samples.append((channels, target))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        channels, target = self._samples[idx]
        x = torch.from_numpy(channels)
        y = torch.from_numpy(target)

        if self.transform is not None:
            x, y = self.transform((x, y))

        return x, y


class NoiseAugmentation:
    """Add Gaussian noise to input channels for data augmentation."""

    def __init__(self, snr_db_range: tuple[float, float] = (10.0, 40.0)):
        self.snr_db_range = snr_db_range

    def __call__(self, sample: tuple[torch.Tensor, torch.Tensor]):
        x, y = sample
        snr_db = torch.empty(1).uniform_(*self.snr_db_range).item()
        signal_power = x.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(x) * noise_power.sqrt()
        return x + noise, y


class GainAugmentation:
    """Apply random gain scaling to input channels."""

    def __init__(self, gain_range_db: tuple[float, float] = (-6.0, 6.0)):
        self.gain_range_db = gain_range_db

    def __call__(self, sample: tuple[torch.Tensor, torch.Tensor]):
        x, y = sample
        gain_db = torch.empty(1).uniform_(*self.gain_range_db).item()
        gain_linear = 10 ** (gain_db / 20)
        return x * gain_linear, y
