"""Transfer matrix simulator for synthetic radar training data.

Generates complex radar range profiles for arbitrary layered material
stackups using the 1D electromagnetic transfer matrix method.

Each layer is characterized by:
    - thickness (m)
    - relative permittivity (eps_r)
    - loss tangent (tan_delta)

The simulator produces data in the same format as real measurements,
stored in HDF5 via MeasurementSession.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from ..sensors.base import RadarConfig
from .hdf5_store import MeasurementSession, compute_range_bin_labels


C = 299_792_458.0  # speed of light in m/s
ETA_0 = 376.73  # impedance of free space in ohms


def transfer_matrix_response(
    layers: np.ndarray,
    config: RadarConfig,
) -> np.ndarray:
    """Compute complex frequency-domain radar reflection for a layered stack.

    Uses the transfer matrix method to cascade through all layers, computing
    the S11 (reflection coefficient) at each frequency point in the chirp band.

    Args:
        layers: (num_layers, 3) array — [thickness_m, eps_r, tan_delta].
        config: Radar configuration defining frequency band.

    Returns:
        Complex range profile of shape (num_samples,) after IFFT.
    """
    num_freqs = config.num_samples_per_chirp
    freqs = np.linspace(config.start_freq_hz, config.end_freq_hz, num_freqs)

    S11 = np.zeros(num_freqs, dtype=np.complex128)

    for i, f in enumerate(freqs):
        omega = 2.0 * np.pi * f

        # Start from the back (last layer → free space termination)
        # Impedance of free space behind the last layer
        eta_load = ETA_0

        # Walk backwards through layers
        for layer_idx in range(layers.shape[0] - 1, -1, -1):
            thickness = layers[layer_idx, 0]
            eps_r = layers[layer_idx, 1]
            tan_d = layers[layer_idx, 2]

            # Complex permittivity
            eps_complex = eps_r * (1.0 - 1j * tan_d)

            # Wave parameters in this layer
            k = omega * np.sqrt(eps_complex) / C
            eta = ETA_0 / np.sqrt(eps_complex)

            # Input impedance looking into this layer (terminated by eta_load)
            # Z_in = eta * (Z_L + eta * tanh(jkd)) / (eta + Z_L * tanh(jkd))
            jkd = 1j * k * thickness

            # Use tanh with overflow protection
            tanh_val = np.tanh(jkd)
            z_in = eta * (eta_load + eta * tanh_val) / (eta + eta_load * tanh_val)

            eta_load = z_in

        # Reflection coefficient at the front surface (air → first layer)
        S11[i] = (eta_load - ETA_0) / (eta_load + ETA_0)

    # IFFT to get range profile (complex)
    range_profile = np.fft.ifft(S11)

    return range_profile.astype(np.complex64)


def simulate_frame(
    layers: np.ndarray,
    config: RadarConfig,
    snr_db: float = 30.0,
) -> np.ndarray:
    """Simulate a full radar frame for a layered stackup.

    Generates the range profile and tiles it across chirps with
    added noise to simulate real acquisition.

    Args:
        layers: (num_layers, 3) — [thickness_m, eps_r, tan_delta].
        config: Radar configuration.
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        Complex I/Q array of shape (num_chirps, num_samples, num_rx).
    """
    profile = transfer_matrix_response(layers, config)

    # Create full frame by tiling across chirps and RX antennas
    # Add per-chirp, per-antenna noise and small phase variations
    frame = np.zeros(
        (config.num_chirps_per_frame, config.num_samples_per_chirp, config.num_rx),
        dtype=np.complex64,
    )

    signal_power = np.mean(np.abs(profile) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10)) if signal_power > 0 else 1e-10

    rng = np.random.default_rng()

    for chirp in range(config.num_chirps_per_frame):
        for rx in range(config.num_rx):
            # Small random phase offset per antenna (simulates hardware variation)
            phase_offset = rng.uniform(-np.pi / 16, np.pi / 16)
            noise = np.sqrt(noise_power / 2) * (
                rng.standard_normal(config.num_samples_per_chirp)
                + 1j * rng.standard_normal(config.num_samples_per_chirp)
            )
            frame[chirp, :, rx] = (profile * np.exp(1j * phase_offset) + noise).astype(np.complex64)

    return frame


def random_layer_stackup(
    rng: np.random.Generator | None = None,
    num_layers_range: tuple[int, int] = (1, 5),
    thickness_range: tuple[float, float] = (0.02, 0.15),
    eps_r_range: tuple[float, float] = (1.0, 15.0),
    tan_delta_range: tuple[float, float] = (0.0, 0.3),
) -> np.ndarray:
    """Generate a random layer stackup.

    Returns:
        (num_layers, 3) array — [thickness_m, eps_r, tan_delta].
    """
    if rng is None:
        rng = np.random.default_rng()

    num_layers = rng.integers(num_layers_range[0], num_layers_range[1] + 1)
    layers = np.zeros((num_layers, 3), dtype=np.float64)

    for i in range(num_layers):
        layers[i, 0] = rng.uniform(*thickness_range)
        layers[i, 1] = rng.uniform(*eps_r_range)
        layers[i, 2] = rng.uniform(*tan_delta_range)

    return layers


def generate_synthetic_dataset(
    output_dir: str | Path,
    config: RadarConfig,
    num_samples: int = 1000,
    frames_per_stackup: int = 10,
    snr_db_range: tuple[float, float] = (15.0, 40.0),
    seed: int | None = None,
) -> Path:
    """Generate a batch of synthetic training data and save to HDF5.

    Creates multiple sessions, each with a random layer stackup and
    several simulated frames at varying SNR levels.

    Args:
        output_dir: Directory to write HDF5 files.
        config: Radar configuration to simulate.
        num_samples: Number of unique stackups to generate.
        frames_per_stackup: Number of frames per stackup (different noise).
        snr_db_range: Range of SNR values to sample from.
        seed: Random seed for reproducibility.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_path = output_dir / f"synthetic_{timestamp}.h5"

    # We'll write all stackups into a single HDF5 file with an extended format
    import h5py

    with h5py.File(h5_path, 'w') as f:
        meta = f.create_group('metadata')
        meta.attrs['sensor_type'] = 'synthetic'
        meta.attrs['date'] = datetime.now().isoformat()
        meta.attrs['notes'] = f'{num_samples} stackups, {frames_per_stackup} frames each'
        meta.attrs['config'] = _config_to_json(config)

        samples_grp = f.create_group('samples')
        labels_grp = f.create_group('labels')

        sample_idx = 0
        for stackup_idx in range(num_samples):
            layers = random_layer_stackup(rng)
            range_bin_labels = compute_range_bin_labels(layers, config)

            for frame_idx in range(frames_per_stackup):
                snr_db = rng.uniform(*snr_db_range)
                frame_data = simulate_frame(layers, config, snr_db=snr_db)

                key = f'{sample_idx:06d}'
                grp = samples_grp.create_group(key)
                grp.create_dataset('complex_iq', data=frame_data)
                grp.create_dataset('timestamp', data=float(sample_idx))

                lbl_grp = labels_grp.create_group(key)
                lbl_grp.create_dataset('layer_stackup', data=layers)
                lbl_grp.create_dataset('range_bin_labels', data=range_bin_labels)

                sample_idx += 1

            if (stackup_idx + 1) % 100 == 0:
                print(f"  Generated {stackup_idx + 1}/{num_samples} stackups")

    print(f"Saved {sample_idx} samples to {h5_path}")
    return h5_path


def _config_to_json(config: RadarConfig) -> str:
    import json
    return json.dumps({
        'start_freq_hz': config.start_freq_hz,
        'end_freq_hz': config.end_freq_hz,
        'num_samples_per_chirp': config.num_samples_per_chirp,
        'num_chirps_per_frame': config.num_chirps_per_frame,
        'num_rx': config.num_rx,
        'sample_rate_hz': config.sample_rate_hz,
        'chirp_repetition_time_s': config.chirp_repetition_time_s,
        'frame_repetition_time_s': config.frame_repetition_time_s,
    })


class SyntheticDataset:
    """Load synthetic data from the extended HDF5 format.

    The synthetic format stores per-sample labels (since each sample
    may have a different stackup), unlike real sessions where all
    samples share the same stackup labels.
    """

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)

    def load_all(self) -> list[dict]:
        """Load all samples as a list of dicts."""
        import h5py
        samples = []
        with h5py.File(self.h5_path, 'r') as f:
            samples_grp = f['samples']
            labels_grp = f['labels']

            for key in sorted(samples_grp.keys()):
                sample = {
                    'complex_iq': np.array(samples_grp[key]['complex_iq']),
                    'timestamp': float(samples_grp[key]['timestamp'][()]),
                    'layer_stackup': np.array(labels_grp[key]['layer_stackup']),
                    'range_bin_labels': np.array(labels_grp[key]['range_bin_labels']),
                }
                samples.append(sample)

        return samples
