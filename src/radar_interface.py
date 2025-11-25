"""
Radar Interface Module for DEMO-BGT60TR13C.

This module provides an interface to communicate with the Infineon
DEMO-BGT60TR13C radar development board using the Infineon Radar SDK.

Prerequisites:
    - Infineon Radar Development Kit (RDK) must be installed
    - The ifxdaq package is provided with the RDK installation
"""

import numpy as np
from typing import Optional, Callable

try:
    from ifxdaq.sensor.radar_ifx import RadarIfxAvian
except ImportError:
    RadarIfxAvian = None


class RadarInterface:
    """Interface for the DEMO-BGT60TR13C radar board.

    This class provides methods to connect to the radar board,
    configure it, and retrieve raw measurement data.

    Attributes:
        config: Dictionary containing radar configuration parameters.
        device: The connected radar device instance.
    """

    # Default configuration for DEMO-BGT60TR13C
    DEFAULT_CONFIG = {
        "sample_rate_hz": 1_000_000,
        "num_samples_per_chirp": 64,
        "num_chirps_per_frame": 16,
        "lower_frequency_hz": 60_000_000_000,
        "upper_frequency_hz": 61_500_000_000,
        "tx_power_level": 31,
        "rx_mask": 1,
        "tx_mask": 1,
        "if_gain_db": 33,
        "frame_repetition_time_s": 0.1,
    }

    def __init__(self, config: Optional[dict] = None):
        """Initialize the radar interface.

        Args:
            config: Optional configuration dictionary. If not provided,
                   DEFAULT_CONFIG will be used.
        """
        if RadarIfxAvian is None:
            raise ImportError(
                "ifxdaq package is not installed. Please install the Infineon "
                "Radar Development Kit (RDK) from: "
                "https://www.infineon.com/cms/en/product/sensor/radar-sensors/"
                "radar-sensors-for-iot/60ghz-radar/demo-bgt60tr13c/"
            )

        self.config = config if config is not None else self.DEFAULT_CONFIG.copy()
        self.device: Optional["RadarIfxAvian"] = None
        self._is_connected = False

    def connect(self) -> bool:
        """Connect to the radar board.

        Returns:
            True if connection was successful, False otherwise.
        """
        try:
            device = RadarIfxAvian(self.config)
            if device is not None:
                self.device = device
                self._is_connected = True
                return True
            else:
                self._is_connected = False
                return False
        except Exception as e:
            print(f"Failed to connect to radar: {e}")
            self.device = None
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from the radar board."""
        if self.device is not None:
            try:
                self.device.stop_acquisition()
            except Exception:
                pass
            self.device = None
        self._is_connected = False

    def is_connected(self) -> bool:
        """Check if the radar is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self._is_connected and self.device is not None

    def start_acquisition(self) -> bool:
        """Start data acquisition.

        Returns:
            True if acquisition started successfully, False otherwise.
        """
        if not self.is_connected():
            print("Radar not connected. Call connect() first.")
            return False

        try:
            self.device.start_acquisition()
            return True
        except Exception as e:
            print(f"Failed to start acquisition: {e}")
            return False

    def stop_acquisition(self) -> None:
        """Stop data acquisition."""
        if self.is_connected():
            try:
                self.device.stop_acquisition()
            except Exception as e:
                print(f"Failed to stop acquisition: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get a single frame of radar data.

        Returns:
            A numpy array containing the raw radar data for one frame,
            or None if no data is available.

        The returned array has shape:
            (num_rx_antennas, num_chirps_per_frame, num_samples_per_chirp)
        """
        if not self.is_connected():
            return None

        try:
            frame = self.device.get_next_frame()
            return np.array(frame)
        except Exception as e:
            print(f"Failed to get frame: {e}")
            return None

    def stream_frames(
        self, callback: Callable[[np.ndarray], bool], max_frames: Optional[int] = None
    ) -> None:
        """Stream frames continuously and call a callback for each frame.

        Args:
            callback: A function that takes a frame (numpy array) and returns
                     True to continue streaming, False to stop.
            max_frames: Optional maximum number of frames to stream.
                       If None, streams indefinitely until callback returns False.
        """
        if not self.is_connected():
            print("Radar not connected. Call connect() first.")
            return

        if not self.start_acquisition():
            return

        frame_count = 0
        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    continue

                frame_count += 1

                # Call the callback
                if not callback(frame):
                    break

                # Check frame limit
                if max_frames is not None and frame_count >= max_frames:
                    break

        finally:
            self.stop_acquisition()

    def get_config(self) -> dict:
        """Get the current radar configuration.

        Returns:
            A dictionary containing the current configuration parameters.
        """
        return self.config.copy()

    def set_config(self, config: dict) -> None:
        """Set new radar configuration.

        Args:
            config: Dictionary containing configuration parameters.
                   Only parameters present in the dictionary will be updated.

        Note:
            This should be called before connect(). Changing configuration
            while connected may require reconnection.
        """
        self.config.update(config)

    def __enter__(self) -> "RadarInterface":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
