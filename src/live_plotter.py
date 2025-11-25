"""
Live Plotter Module for Radar Data Visualization.

This module provides real-time plotting capabilities for radar data
using matplotlib with animation support.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Callable, Tuple, List
import threading
import queue


class LivePlotter:
    """Real-time plotter for radar data visualization.

    This class provides methods to visualize raw radar data in real-time,
    including time domain plots, range profiles, and range-Doppler maps.

    Attributes:
        fig: The matplotlib figure object.
        axes: List of axes objects for subplots.
    """

    def __init__(
        self,
        num_samples: int = 64,
        num_chirps: int = 16,
        plot_type: str = "time_domain",
        update_interval_ms: int = 50,
    ):
        """Initialize the live plotter.

        Args:
            num_samples: Number of samples per chirp.
            num_chirps: Number of chirps per frame.
            plot_type: Type of plot to display. Options:
                      - "time_domain": Raw time domain signal
                      - "range_profile": Range profile (FFT of chirp)
                      - "range_doppler": Range-Doppler map
                      - "all": All three plots
            update_interval_ms: Update interval in milliseconds.
        """
        self.num_samples = num_samples
        self.num_chirps = num_chirps
        self.plot_type = plot_type
        self.update_interval_ms = update_interval_ms

        self._data_queue: queue.Queue = queue.Queue(maxsize=10)
        self._running = False
        self._animation: Optional[FuncAnimation] = None

        self.fig: Optional[plt.Figure] = None
        self.axes: List[plt.Axes] = []
        self._lines: List = []
        self._images: List = []

    def _setup_plots(self) -> None:
        """Set up the matplotlib figure and axes."""
        plt.ion()  # Enable interactive mode

        if self.plot_type == "all":
            self.fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            self.axes = axes.flatten()[:3]  # Use first 3 subplots
            self.fig.delaxes(axes[1, 1])  # Remove unused subplot
        elif self.plot_type == "range_doppler":
            self.fig, ax = plt.subplots(figsize=(10, 8))
            self.axes = [ax]
        else:
            self.fig, ax = plt.subplots(figsize=(10, 6))
            self.axes = [ax]

        self.fig.suptitle("DEMO-BGT60TR13C Radar Live Data", fontsize=14)

        self._setup_subplot_types()
        self.fig.tight_layout()

    def _setup_subplot_types(self) -> None:
        """Configure each subplot based on plot type."""
        if self.plot_type == "time_domain" or self.plot_type == "all":
            ax_idx = 0 if self.plot_type == "time_domain" else 0
            ax = self.axes[ax_idx]
            ax.set_title("Time Domain Signal")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Amplitude")
            (line,) = ax.plot([], [], "b-", linewidth=0.8)
            self._lines.append(("time", line, ax_idx))
            ax.set_xlim(0, self.num_samples)
            ax.set_ylim(-2000, 2000)

        if self.plot_type == "range_profile" or self.plot_type == "all":
            ax_idx = 0 if self.plot_type == "range_profile" else 1
            ax = self.axes[ax_idx]
            ax.set_title("Range Profile")
            ax.set_xlabel("Range Bin")
            ax.set_ylabel("Magnitude (dB)")
            (line,) = ax.plot([], [], "r-", linewidth=0.8)
            self._lines.append(("range", line, ax_idx))
            ax.set_xlim(0, self.num_samples // 2)
            ax.set_ylim(0, 80)

        if self.plot_type == "range_doppler" or self.plot_type == "all":
            ax_idx = 0 if self.plot_type == "range_doppler" else 2
            ax = self.axes[ax_idx]
            ax.set_title("Range-Doppler Map")
            ax.set_xlabel("Doppler Bin")
            ax.set_ylabel("Range Bin")
            # Initialize with zeros
            data = np.zeros((self.num_samples // 2, self.num_chirps))
            im = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=60,
            )
            self.fig.colorbar(im, ax=ax, label="Magnitude (dB)")
            self._images.append(("rd_map", im, ax_idx))

    def _process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a radar frame to extract different views.

        Args:
            frame: Raw radar frame data with shape
                  (num_rx_antennas, num_chirps, num_samples)

        Returns:
            Tuple of (time_domain, range_profile, range_doppler_map)
        """
        # Handle different frame shapes
        if frame.ndim == 3:
            # Use first antenna
            data = frame[0]
        elif frame.ndim == 2:
            data = frame
        else:
            # Assume 1D signal, reshape
            data = frame.reshape(self.num_chirps, self.num_samples)

        # Time domain: first chirp
        time_domain = np.real(data[0])

        # Range profile: FFT of first chirp
        range_fft = np.fft.fft(data[0])
        range_profile = 20 * np.log10(np.abs(range_fft[: len(range_fft) // 2]) + 1e-6)

        # Range-Doppler map: 2D FFT
        range_fft_all = np.fft.fft(data, axis=1)
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft_all, axis=0), axes=0)
        rd_map = 20 * np.log10(
            np.abs(doppler_fft[:, : doppler_fft.shape[1] // 2].T) + 1e-6
        )

        return time_domain, range_profile, rd_map

    def _update_plot(self, frame_num: int) -> List:
        """Update function for animation.

        Args:
            frame_num: Animation frame number (unused).

        Returns:
            List of updated artists.
        """
        try:
            # Get data from queue (non-blocking)
            frame = self._data_queue.get_nowait()
        except queue.Empty:
            return self._lines + self._images

        time_domain, range_profile, rd_map = self._process_frame(frame)

        artists = []

        for plot_type, artist, ax_idx in self._lines:
            if plot_type == "time":
                artist.set_data(np.arange(len(time_domain)), time_domain)
                # Auto-scale y-axis
                max_val = max(np.abs(time_domain).max() * 1.2, 100)
                self.axes[ax_idx].set_ylim(-max_val, max_val)
            elif plot_type == "range":
                artist.set_data(np.arange(len(range_profile)), range_profile)
            artists.append(artist)

        for plot_type, artist, _ in self._images:
            if plot_type == "rd_map":
                artist.set_array(rd_map)
            artists.append(artist)

        return artists

    def push_frame(self, frame: np.ndarray) -> bool:
        """Push a new frame to the plotter.

        Args:
            frame: Raw radar frame data.

        Returns:
            True if frame was added, False if queue is full.
        """
        try:
            self._data_queue.put_nowait(frame)
            return True
        except queue.Full:
            # Drop oldest frame and add new one
            try:
                self._data_queue.get_nowait()
                self._data_queue.put_nowait(frame)
                return True
            except queue.Empty:
                return False

    def start(self, blocking: bool = True) -> None:
        """Start the live plotter.

        Args:
            blocking: If True, blocks until the window is closed.
                     If False, runs in non-blocking mode.
        """
        self._setup_plots()
        self._running = True

        self._animation = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=self.update_interval_ms,
            blit=True,
            cache_frame_data=False,
        )

        if blocking:
            plt.show(block=True)
        else:
            plt.show(block=False)

    def stop(self) -> None:
        """Stop the live plotter."""
        self._running = False
        if self._animation is not None:
            self._animation.event_source.stop()
        plt.close(self.fig)

    def is_running(self) -> bool:
        """Check if the plotter is running.

        Returns:
            True if running, False otherwise.
        """
        return self._running and plt.fignum_exists(self.fig.number if self.fig else -1)


class SimulatedRadarPlotter:
    """A plotter that uses simulated radar data for testing.

    This is useful when the actual radar hardware is not available.
    """

    def __init__(
        self,
        num_samples: int = 64,
        num_chirps: int = 16,
        plot_type: str = "all",
        update_interval_ms: int = 50,
    ):
        """Initialize the simulated radar plotter.

        Args:
            num_samples: Number of samples per chirp.
            num_chirps: Number of chirps per frame.
            plot_type: Type of plot to display.
            update_interval_ms: Update interval in milliseconds.
        """
        self.num_samples = num_samples
        self.num_chirps = num_chirps
        self.plotter = LivePlotter(
            num_samples=num_samples,
            num_chirps=num_chirps,
            plot_type=plot_type,
            update_interval_ms=update_interval_ms,
        )
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._target_distance = 5.0  # Simulated target distance (meters)
        self._target_velocity = 0.5  # Simulated target velocity (m/s)
        self._frame_count = 0

    def _generate_simulated_frame(self) -> np.ndarray:
        """Generate a simulated radar frame.

        Returns:
            Simulated radar data with shape (1, num_chirps, num_samples)
        """
        self._frame_count += 1

        # Simulate a target at a certain range with some velocity
        c = 3e8  # Speed of light
        fc = 60.5e9  # Center frequency
        bandwidth = 1.5e9  # Bandwidth
        chirp_time = 50e-6  # Chirp duration

        # Range resolution
        range_res = c / (2 * bandwidth)

        # Simulate IF signal
        frame = np.zeros((1, self.num_chirps, self.num_samples), dtype=np.complex128)

        for chirp_idx in range(self.num_chirps):
            t = np.linspace(0, chirp_time, self.num_samples)

            # Target range (slowly varying)
            target_range = self._target_distance + 0.1 * np.sin(
                2 * np.pi * 0.5 * self._frame_count / 100
            )

            # Beat frequency
            f_beat = 2 * target_range * bandwidth / (c * chirp_time)

            # Doppler shift
            f_doppler = 2 * self._target_velocity * fc / c

            # IF signal with noise
            signal = (
                500
                * np.exp(
                    1j
                    * 2
                    * np.pi
                    * (f_beat * t + f_doppler * chirp_idx * chirp_time)
                )
            )
            noise = 50 * (np.random.randn(self.num_samples) + 1j * np.random.randn(self.num_samples))
            frame[0, chirp_idx, :] = signal + noise

        return np.real(frame).astype(np.float32)

    def _data_thread(self) -> None:
        """Thread function to generate and push simulated data."""
        import time

        while self._running:
            frame = self._generate_simulated_frame()
            self.plotter.push_frame(frame)
            time.sleep(0.05)  # 20 Hz frame rate

    def start(self) -> None:
        """Start the simulated radar plotter."""
        self._running = True

        # Start data generation thread
        self._thread = threading.Thread(target=self._data_thread, daemon=True)
        self._thread.start()

        # Start plotter (blocking)
        self.plotter.start(blocking=True)

    def stop(self) -> None:
        """Stop the simulated radar plotter."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.plotter.stop()
