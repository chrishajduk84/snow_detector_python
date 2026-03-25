"""Abstract base classes for radar sensor interface.

All radar sensors must implement the RadarSensor ABC, producing
RadarFrame objects with complex I/Q data in a standardized shape.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Iterator

import numpy as np


@dataclass
class RadarConfig:
    """Radar configuration parameters common to all sensors."""
    start_freq_hz: float
    end_freq_hz: float
    num_samples_per_chirp: int
    num_chirps_per_frame: int
    num_rx: int
    sample_rate_hz: float
    chirp_repetition_time_s: float
    frame_repetition_time_s: float

    @property
    def bandwidth_hz(self) -> float:
        return abs(self.end_freq_hz - self.start_freq_hz)

    @property
    def center_freq_hz(self) -> float:
        return (self.start_freq_hz + self.end_freq_hz) / 2.0

    @property
    def range_resolution_m(self) -> float:
        """Range resolution in meters (in free space)."""
        c = 299_792_458.0
        return c / (2.0 * self.bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        """Maximum unambiguous range in meters."""
        c = 299_792_458.0
        return (self.sample_rate_hz * c) / (4.0 * self.bandwidth_hz * (self.bandwidth_hz / self.chirp_repetition_time_s))

    @property
    def wavelength_m(self) -> float:
        c = 299_792_458.0
        return c / self.center_freq_hz


@dataclass
class RadarFrame:
    """Standardized radar frame output from any sensor.

    Attributes:
        complex_data: Complex I/Q data with shape (num_chirps, num_samples, num_rx).
                      dtype is complex64 or complex128.
        timestamp: Frame acquisition timestamp in seconds (monotonic).
        config: Radar configuration used to acquire this frame.
    """
    complex_data: np.ndarray  # (num_chirps, num_samples, num_rx) complex
    timestamp: float
    config: RadarConfig

    def __post_init__(self):
        if self.complex_data.ndim != 3:
            raise ValueError(
                f"complex_data must be 3D (chirps, samples, rx), got {self.complex_data.ndim}D"
            )

    @property
    def num_chirps(self) -> int:
        return self.complex_data.shape[0]

    @property
    def num_samples(self) -> int:
        return self.complex_data.shape[1]

    @property
    def num_rx(self) -> int:
        return self.complex_data.shape[2]

    def magnitude(self) -> np.ndarray:
        """Return magnitude of complex data."""
        return np.abs(self.complex_data)

    def phase(self) -> np.ndarray:
        """Return phase of complex data in radians."""
        return np.angle(self.complex_data)


class RadarSensor(ABC):
    """Abstract base class for radar sensor interfaces.

    Usage as context manager:
        with SomeSensor(config_path) as sensor:
            for frame in sensor.stream():
                process(frame)
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the radar hardware."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the radar hardware."""

    @abstractmethod
    def get_frame(self) -> RadarFrame:
        """Acquire a single radar frame.

        Returns:
            RadarFrame with complex I/Q data.

        Raises:
            RuntimeError: If frame acquisition fails.
        """

    @abstractmethod
    def get_config(self) -> RadarConfig:
        """Return the current radar configuration."""

    def stream(self) -> Iterator[RadarFrame]:
        """Yield frames continuously until disconnected."""
        while True:
            try:
                yield self.get_frame()
            except StopIteration:
                break

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
