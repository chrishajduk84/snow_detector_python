"""HDF5 storage for radar measurement sessions.

Each session file stores multiple radar frames along with ground-truth
layer stackup labels and per-range-bin property labels.

File structure:
    /metadata                (attrs: sensor_type, date, notes, config JSON)
    /samples/0000/
        complex_iq           (num_chirps, num_samples, num_rx) complex64
        timestamp            scalar float64
    /labels/
        layer_stackup        (num_layers, 3) float64 — [thickness_m, eps_r, tan_delta]
        range_bin_labels     (num_range_bins, 3) float64 — [eps_r, tan_delta, presence]
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..sensors.base import RadarConfig, RadarFrame


class MeasurementSession:
    """Write radar measurement sessions to HDF5.

    Usage:
        with MeasurementSession(path, config, layer_stackup) as session:
            session.add_frame(radar_frame)
    """

    def __init__(
        self,
        path: str | Path,
        config: RadarConfig,
        layer_stackup: np.ndarray,
        sensor_type: str = "unknown",
        notes: str = "",
    ):
        """
        Args:
            path: Output HDF5 file path.
            config: Radar configuration used for this session.
            layer_stackup: Array of shape (num_layers, 3) with columns
                           [thickness_m, epsilon_r, loss_tangent].
            sensor_type: Identifier string, e.g. 'bgt60tr13c' or 'iwr1443'.
            notes: Free-text notes about the measurement.
        """
        self._path = Path(path)
        self._config = config
        self._layer_stackup = np.asarray(layer_stackup, dtype=np.float64)
        self._sensor_type = sensor_type
        self._notes = notes
        self._file: h5py.File | None = None
        self._sample_count = 0

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, 'w')

        # Metadata
        meta = self._file.create_group('metadata')
        meta.attrs['sensor_type'] = self._sensor_type
        meta.attrs['date'] = datetime.now().isoformat()
        meta.attrs['notes'] = self._notes
        meta.attrs['config'] = json.dumps({
            'start_freq_hz': self._config.start_freq_hz,
            'end_freq_hz': self._config.end_freq_hz,
            'num_samples_per_chirp': self._config.num_samples_per_chirp,
            'num_chirps_per_frame': self._config.num_chirps_per_frame,
            'num_rx': self._config.num_rx,
            'sample_rate_hz': self._config.sample_rate_hz,
            'chirp_repetition_time_s': self._config.chirp_repetition_time_s,
            'frame_repetition_time_s': self._config.frame_repetition_time_s,
        })

        # Samples group
        self._file.create_group('samples')

        # Labels
        labels = self._file.create_group('labels')
        labels.create_dataset('layer_stackup', data=self._layer_stackup)

        # Compute per-range-bin labels from layer stackup
        range_bin_labels = compute_range_bin_labels(
            self._layer_stackup, self._config
        )
        labels.create_dataset('range_bin_labels', data=range_bin_labels)

        self._sample_count = 0

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def add_frame(self, frame: RadarFrame) -> None:
        """Add a single radar frame to the session."""
        if self._file is None:
            raise RuntimeError("Session not open")

        grp = self._file['samples'].create_group(f'{self._sample_count:04d}')
        grp.create_dataset('complex_iq', data=frame.complex_data.astype(np.complex64))
        grp.create_dataset('timestamp', data=frame.timestamp)
        self._sample_count += 1

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def compute_range_bin_labels(
    layer_stackup: np.ndarray,
    config: RadarConfig,
) -> np.ndarray:
    """Convert layer stackup to per-range-bin property labels.

    Each layer has a physical thickness which maps to an apparent range
    (electromagnetic travel distance) based on its permittivity.

    Args:
        layer_stackup: (num_layers, 3) array — [thickness_m, eps_r, tan_delta].
        config: Radar config to determine number of range bins and resolution.

    Returns:
        (num_range_bins, 3) array — [eps_r, tan_delta, presence] per bin.
    """
    c = 299_792_458.0
    num_bins = config.num_samples_per_chirp
    bin_width_m = c / (2.0 * config.bandwidth_hz)  # range per bin in free space

    labels = np.zeros((num_bins, 3), dtype=np.float64)
    # Column 0: eps_r (default 1.0 = air)
    labels[:, 0] = 1.0
    # Column 1: tan_delta (default 0.0)
    # Column 2: presence (default 0.0 = no material)

    # Walk through layers, converting physical thickness to apparent range bins
    apparent_range = 0.0  # cumulative apparent range in meters

    for i in range(layer_stackup.shape[0]):
        thickness_m = layer_stackup[i, 0]
        eps_r = layer_stackup[i, 1]
        tan_delta = layer_stackup[i, 2]

        # Apparent thickness: signal travels slower in material by sqrt(eps_r)
        apparent_thickness = thickness_m * np.sqrt(eps_r)

        start_bin = int(apparent_range / bin_width_m)
        end_bin = int((apparent_range + apparent_thickness) / bin_width_m)

        # Clamp to valid range
        start_bin = max(0, min(start_bin, num_bins - 1))
        end_bin = max(0, min(end_bin, num_bins))

        labels[start_bin:end_bin, 0] = eps_r
        labels[start_bin:end_bin, 1] = tan_delta
        labels[start_bin:end_bin, 2] = 1.0  # material present

        apparent_range += apparent_thickness

    return labels


def load_session(path: str | Path) -> dict[str, Any]:
    """Load a measurement session from HDF5.

    Returns:
        Dictionary with keys:
            'config': RadarConfig
            'layer_stackup': ndarray (num_layers, 3)
            'range_bin_labels': ndarray (num_range_bins, 3)
            'samples': list of dict with 'complex_iq' and 'timestamp'
            'metadata': dict with 'sensor_type', 'date', 'notes'
    """
    with h5py.File(path, 'r') as f:
        # Metadata
        meta = f['metadata']
        config_dict = json.loads(meta.attrs['config'])
        config = RadarConfig(**config_dict)
        metadata = {
            'sensor_type': str(meta.attrs['sensor_type']),
            'date': str(meta.attrs['date']),
            'notes': str(meta.attrs['notes']),
        }

        # Labels
        layer_stackup = np.array(f['labels/layer_stackup'])
        range_bin_labels = np.array(f['labels/range_bin_labels'])

        # Samples
        samples = []
        samples_grp = f['samples']
        for key in sorted(samples_grp.keys()):
            sample = {
                'complex_iq': np.array(samples_grp[key]['complex_iq']),
                'timestamp': float(samples_grp[key]['timestamp'][()]),
            }
            samples.append(sample)

    return {
        'config': config,
        'layer_stackup': layer_stackup,
        'range_bin_labels': range_bin_labels,
        'samples': samples,
        'metadata': metadata,
    }


def list_sessions(data_dir: str | Path) -> list[Path]:
    """List all HDF5 session files in a directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob('*.h5'))
