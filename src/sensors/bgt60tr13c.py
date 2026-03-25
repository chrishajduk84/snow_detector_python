"""Infineon BGT60TR13C radar sensor via ifxdaq/Avian SDK.

The BGT60TR13C provides real-valued IF data. We apply a Hilbert transform
to produce analytic (complex I/Q) signals for consistency with the
IWR1443 which provides native I/Q.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from .base import RadarSensor, RadarFrame, RadarConfig


class BGT60TR13CSensor(RadarSensor):
    """Infineon DEMO-BGT60TR13C sensor wrapper.

    Args:
        config_path: Path to RadarIfxFmcw JSON config file.
                     If None, uses the SDK default config.
    """

    def __init__(self, config_path: str | Path | None = None):
        self._config_path = Path(config_path) if config_path else None
        self._radar = None
        self._iterator = None
        self._config: RadarConfig | None = None
        self._start_time = 0.0

    def connect(self) -> None:
        from ifxdaq.sensor.radar_ifx import RadarIfxFmcw

        if self._config_path and self._config_path.exists():
            config_file = str(self._config_path)
        else:
            config_file = RadarIfxFmcw.create_default_config_file()

        self._radar = RadarIfxFmcw(config_file)
        self._radar.__enter__()
        self._config = self._parse_config(config_file)
        self._start_time = time.monotonic()

    def disconnect(self) -> None:
        if self._radar is not None:
            self._radar.__exit__(None, None, None)
            self._radar = None

    def get_config(self) -> RadarConfig:
        if self._config is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._config

    def get_frame(self) -> RadarFrame:
        if self._radar is None:
            raise RuntimeError("Not connected. Call connect() first.")

        frame_data = next(iter(self._radar))
        radar_frame = frame_data.get('radar')
        if radar_frame is None:
            raise RuntimeError("No radar data in frame")

        # Extract raw array: shape varies by SDK version
        if hasattr(radar_frame, 'data'):
            raw = np.array(radar_frame.data, dtype=np.float64)
        else:
            raw = np.array(radar_frame, dtype=np.float64)

        # Ensure shape is (num_chirps, num_samples, num_rx)
        config = self.get_config()
        raw = self._reshape_frame(raw, config)

        # Apply Hilbert transform per-antenna to get analytic signal
        complex_data = np.zeros_like(raw, dtype=np.complex64)
        for rx in range(raw.shape[2]):
            complex_data[:, :, rx] = hilbert(raw[:, :, rx], axis=1).astype(np.complex64)

        timestamp = time.monotonic() - self._start_time

        return RadarFrame(
            complex_data=complex_data,
            timestamp=timestamp,
            config=config,
        )

    def _reshape_frame(self, raw: np.ndarray, config: RadarConfig) -> np.ndarray:
        """Reshape raw SDK output to (num_chirps, num_samples, num_rx)."""
        expected_shape = (config.num_chirps_per_frame, config.num_samples_per_chirp, config.num_rx)

        if raw.shape == expected_shape:
            return raw

        # Common SDK output: (num_chirps, num_samples, num_rx) — already correct
        # Some versions give (num_rx, num_chirps, num_samples)
        if raw.ndim == 3 and raw.shape[0] == config.num_rx:
            return np.transpose(raw, (1, 2, 0))

        # Flat array — reshape
        if raw.ndim == 1:
            return raw.reshape(expected_shape)

        # 2D: could be (num_chirps * num_rx, num_samples) or similar
        if raw.ndim == 2:
            total_chirps = raw.shape[0]
            if total_chirps == config.num_chirps_per_frame * config.num_rx:
                reshaped = raw.reshape(config.num_rx, config.num_chirps_per_frame, config.num_samples_per_chirp)
                return np.transpose(reshaped, (1, 2, 0))

        raise ValueError(f"Cannot reshape frame with shape {raw.shape} to {expected_shape}")

    @staticmethod
    def _parse_config(config_file: str) -> RadarConfig:
        """Parse RadarIfxFmcw JSON config into RadarConfig."""
        with open(config_file, 'r') as f:
            cfg = json.load(f)

        fmcw = cfg['device_config']['fmcw_single_shape']

        return RadarConfig(
            start_freq_hz=float(fmcw['start_frequency_Hz']),
            end_freq_hz=float(fmcw['end_frequency_Hz']),
            num_samples_per_chirp=int(fmcw['num_samples_per_chirp']),
            num_chirps_per_frame=int(fmcw['num_chirps_per_frame']),
            num_rx=len(fmcw['rx_antennas']),
            sample_rate_hz=float(fmcw['sample_rate_Hz']),
            chirp_repetition_time_s=float(fmcw['chirp_repetition_time_s']),
            frame_repetition_time_s=float(fmcw['frame_repetition_time_s']),
        )
