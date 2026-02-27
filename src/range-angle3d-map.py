# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

"""
3D Range-Azimuth-Elevation Map using BGT60TR13C

This script uses the BGT60TR13C's L-shaped antenna array (3 RX antennas)
to compute both azimuth and elevation angles, enabling true 3D object detection.

Antenna Layout:
    RX1 ---- RX2  (horizontal for azimuth)
     |
    RX3           (vertical for elevation)
"""

import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import ndimage

from ifxAvian import Avian
from helpers.fft_spectrum import *
from helpers.DigitalBeamForming2D import DigitalBeamForming2D as DBF2D
from helpers.DopplerAlgo import DopplerAlgo


def num_rx_antennas_from_config(config):
    """Count number of enabled RX antennas from config mask."""
    rx_mask = config.rx_mask
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c


def find_peaks_3d_adaptive(data, noise_floor_percentile=50, snr_threshold_db=15, 
                           min_distance=2, max_peaks=20):
    """
    Find multiple peaks in a 3D array (range x azimuth x elevation) using
    adaptive noise floor estimation.
    
    Args:
        data: 3D numpy array (range x azimuth x elevation)
        noise_floor_percentile: Percentile of data to use as noise floor
        snr_threshold_db: Minimum SNR in dB above noise floor for detection
        min_distance: Minimum distance between peaks in pixels
        max_peaks: Maximum number of peaks to return
    
    Returns:
        List of tuples: [(range_idx, az_idx, el_idx, value, snr_db), ...]
        sorted by SNR descending
    """
    # Convert to dB scale
    data_db = 10 * np.log10(np.abs(data) + 1e-10)
    
    # Calculate adaptive noise floor
    noise_floor_db = np.percentile(data_db, noise_floor_percentile)
    
    # Apply threshold
    threshold_db = noise_floor_db + snr_threshold_db
    data_thresholded = np.where(data_db >= threshold_db, np.abs(data), 0)
    
    # 3D local maximum filter
    footprint = np.ones((min_distance*2+1, min_distance*2+1, min_distance*2+1))
    local_max = (ndimage.maximum_filter(data_thresholded, footprint=footprint) == data_thresholded)
    
    # Remove border peaks
    local_max[0, :, :] = False
    local_max[-1, :, :] = False
    local_max[:, 0, :] = False
    local_max[:, -1, :] = False
    local_max[:, :, 0] = False
    local_max[:, :, -1] = False
    
    # Get peak coordinates
    peak_coords = np.argwhere(local_max & (data_thresholded > 0))
    
    if len(peak_coords) == 0:
        # Fallback to global maximum
        idx = np.unravel_index(np.abs(data).argmax(), data.shape)
        val = np.abs(data[idx])
        snr = data_db[idx] - noise_floor_db
        return [(idx[0], idx[1], idx[2], val, snr)]
    
    # Build peaks list with SNR values
    peaks = []
    for coord in peak_coords:
        val = np.abs(data[coord[0], coord[1], coord[2]])
        snr = data_db[coord[0], coord[1], coord[2]] - noise_floor_db
        peaks.append((coord[0], coord[1], coord[2], val, snr))
    
    # Sort by SNR descending
    peaks.sort(key=lambda x: x[4], reverse=True)
    
    return peaks[:max_peaks]


class LivePlot3D:
    """3D visualization for true range-azimuth-elevation radar data."""
    
    def __init__(self, max_angle_azimuth: float, max_angle_elevation: float, max_range_m: float):
        self.max_angle_az = max_angle_azimuth
        self.max_angle_el = max_angle_elevation
        self.max_range_m = max_range_m
        
        self._fig = plt.figure(figsize=(14, 10))
        self._ax = self._fig.add_subplot(111, projection='3d')
        
        self._fig.canvas.manager.set_window_title("3D Range-Azimuth-Elevation Map (BGT60TR13C)")
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True
        
        self._scatter = None
        self._projection_lines = []
        self._annotations = []
        
        # Set up axes - true 3D Cartesian coordinates
        self._ax.set_xlabel('X - Horizontal (m)', fontsize=12, fontweight='bold')
        self._ax.set_ylabel('Y - Forward (m)', fontsize=12, fontweight='bold')
        self._ax.set_zlabel('Z - Vertical (m)', fontsize=12, fontweight='bold')
        
        # Set axis limits based on max range
        self._ax.set_xlim(-max_range_m, max_range_m)
        self._ax.set_ylim(0, max_range_m)
        self._ax.set_zlim(-max_range_m/2, max_range_m/2)
        
        # Draw ground plane grid for reference
        self._draw_reference_grid(max_range_m)
        
        # Set dark background for better visibility
        self._ax.set_facecolor('#1a1a2e')
        self._fig.patch.set_facecolor('#16213e')
        self._ax.xaxis.pane.fill = False
        self._ax.yaxis.pane.fill = False
        self._ax.zaxis.pane.fill = False
        
        # Make grid lines visible
        self._ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.3)
        self._ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.3)
        self._ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.3)
        
        # White axis labels
        self._ax.tick_params(colors='white')
        self._ax.xaxis.label.set_color('white')
        self._ax.yaxis.label.set_color('white')
        self._ax.zaxis.label.set_color('white')
        
    def _draw_reference_grid(self, max_range):
        """Draw a reference grid on the ground plane (z=0)."""
        # Draw concentric range circles on z=0 plane
        for r in np.linspace(max_range/4, max_range, 4):
            theta = np.linspace(0, np.pi, 50)
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            z = np.zeros_like(x)
            self._ax.plot(x, y, z, 'c-', alpha=0.3, linewidth=0.5)
        
        # Draw radial lines
        for angle in np.linspace(-60, 60, 7):
            rad = np.radians(angle)
            x = [0, max_range * np.sin(rad)]
            y = [0, max_range * np.cos(rad)]
            z = [0, 0]
            self._ax.plot(x, y, z, 'c-', alpha=0.2, linewidth=0.5)
        
        # Draw radar position marker
        self._ax.scatter([0], [0], [0], c='cyan', marker='^', s=150, label='Radar')
        
    def draw(self, peaks_3d, title: str, max_range_m: float, num_range_bins: int,
             azimuth_angles, elevation_angles):
        """
        Draw detected peaks in true 3D Cartesian space with enhanced visibility.
        """
        if not self._is_window_open:
            return
            
        # Remove old scatter and projection lines
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None
        
        for line in self._projection_lines:
            line.remove()
        self._projection_lines = []
        
        for ann in self._annotations:
            ann.remove()
        self._annotations = []
        
        if len(peaks_3d) > 0:
            xs, ys, zs, snrs = [], [], [], []
            
            for i, (range_idx, az_idx, el_idx, value, snr_db) in enumerate(peaks_3d):
                # Convert indices to physical values
                range_m = (range_idx / num_range_bins) * max_range_m
                az_deg = azimuth_angles[az_idx]
                el_deg = elevation_angles[el_idx]
                
                # Convert spherical to Cartesian
                az_rad = np.deg2rad(az_deg)
                el_rad = np.deg2rad(el_deg)
                
                x = range_m * np.sin(az_rad) * np.cos(el_rad)
                y = range_m * np.cos(az_rad) * np.cos(el_rad)
                z = range_m * np.sin(el_rad)
                
                xs.append(x)
                ys.append(y)
                zs.append(z)
                snrs.append(snr_db)
                
                # Draw projection lines to help visualize 3D position
                # Vertical line from point to ground (z=0)
                line_v, = self._ax.plot([x, x], [y, y], [z, 0], 
                                       color='yellow', alpha=0.5, linewidth=1, linestyle='--')
                self._projection_lines.append(line_v)
                
                # Line from origin to point (range line)
                line_r, = self._ax.plot([0, x], [0, y], [0, z], 
                                       color='lime', alpha=0.4, linewidth=1.5)
                self._projection_lines.append(line_r)
                
                # Ground projection marker
                ground_marker = self._ax.scatter([x], [y], [0], 
                                                c='yellow', marker='x', s=50, alpha=0.5)
                self._projection_lines.append(ground_marker)
            
            # Size and color by SNR - make points much larger
            min_size, max_size = 200, 600
            sizes = [min_size + (max_size - min_size) * min(1, max(0, (s - 5) / 20)) for s in snrs]
            
            # Color gradient: weaker signals = yellow, stronger = red
            colors = plt.cm.hot(np.clip(np.array(snrs) / 25, 0.3, 1.0))
            
            # Plot main peaks with large, bright markers
            self._scatter = self._ax.scatter(xs, ys, zs, c=colors, s=sizes, 
                                            marker='o', alpha=0.9, edgecolors='white',
                                            linewidths=2, depthshade=False)
            
            # Add text labels for top 3 peaks
            for i in range(min(3, len(peaks_3d))):
                range_idx, az_idx, el_idx, value, snr_db = peaks_3d[i]
                range_m = (range_idx / num_range_bins) * max_range_m
                az_deg = azimuth_angles[az_idx]
                el_deg = elevation_angles[el_idx]
                
                label = f"#{i+1}\n{range_m:.1f}m"
                ann = self._ax.text(xs[i], ys[i], zs[i] + 0.15, label,
                                   color='white', fontsize=10, fontweight='bold',
                                   ha='center', va='bottom')
                self._annotations.append(ann)
        
        self._ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=20)
        
        # Rotate view slowly for dynamic feel (optional - comment out if distracting)
        # self._ax.view_init(elev=25, azim=-60 + (time.time() % 360))
        self._ax.view_init(elev=25, azim=-45)
        
        plt.draw()
        plt.pause(1e-3)
    
    def close(self, event=None):
        if not self.is_closed():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')
    
    def is_closed(self):
        return not self._is_window_open


# -------------------------------------------------
# Main logic
# -------------------------------------------------
if __name__ == '__main__':
    # 2D Beamforming parameters
    num_beams_azimuth = 32     # Number of azimuth beams
    num_beams_elevation = 32   # Number of elevation beams
    max_angle_azimuth = 60     # Max azimuth angle (+/- degrees)
    max_angle_elevation = 60   # Max elevation angle (+/- degrees)

    config = Avian.DeviceConfig(
        sample_rate_Hz = 1_000_000,       # 1MHz
        rx_mask = 7,                      # RX1, RX2, RX3 (all 3 for 2D beamforming)
        tx_mask = 1,                      # TX1
        if_gain_dB = 33,                  # gain of 33dB
        tx_power_level = 31,              # TX power level of 31
        start_frequency_Hz = 60e9,        # 60GHz 
        end_frequency_Hz = 61.5e9,        # 61.5GHz
        num_chirps_per_frame = 128,       # 128 chirps per frame
        num_samples_per_chirp = 64,       # 64 samples per chirp
        chirp_repetition_time_s = 0.0005, # 0.5ms
        frame_repetition_time_s = 1,   # 1s, frame_Rate = 1Hz
        mimo_mode = 'off'                 # MIMO not needed - using L-shaped array
    )

    with Avian.Device() as device:
        # Set configuration
        device.set_config(config)

        # Get and print metrics
        metrics = device.metrics_from_config(config)
        pprint.pprint(metrics)

        # Get maximum range
        max_range_m = metrics.max_range_m

        # Verify we have 3 RX antennas for 2D beamforming
        num_rx_antennas = num_rx_antennas_from_config(config)
        if num_rx_antennas != 3:
            raise ValueError(f"BGT60TR13C 2D beamforming requires 3 RX antennas, got {num_rx_antennas}")

        # Calculate center frequency for beamformer
        center_frequency_hz = (config.start_frequency_Hz + config.end_frequency_Hz) / 2

        # Create processing objects
        doppler = DopplerAlgo(config.num_samples_per_chirp, 
                             num_chirps_per_frame=config.num_chirps_per_frame, 
                             num_ant=num_rx_antennas)
        
        # 2D Beamformer for azimuth AND elevation
        dbf2d = DBF2D(
            num_beams_azimuth=num_beams_azimuth,
            num_beams_elevation=num_beams_elevation,
            max_angle_azimuth_deg=max_angle_azimuth,
            max_angle_elevation_deg=max_angle_elevation,
            center_frequency_hz=center_frequency_hz
        )
        
        # Get angle arrays for peak conversion
        azimuth_angles, elevation_angles = dbf2d.get_angle_axes()
        
        # 3D Plot
        plot = LivePlot3D(max_angle_azimuth, max_angle_elevation, max_range_m)

        print("\n3D Range-Azimuth-Elevation detection started...")
        print(f"Azimuth range: +/- {max_angle_azimuth}° ({num_beams_azimuth} beams)")
        print(f"Elevation range: +/- {max_angle_elevation}° ({num_beams_elevation} beams)")
        print(f"Max range: {max_range_m:.2f} m")
        print()

        while not plot.is_closed():
            # Get radar frame: (num_rx_antennas, num_samples_per_chirp, num_chirps_per_frame)
            frame = device.get_next_frame()

            # Compute range-Doppler spectrum for each antenna
            rd_spectrum = np.zeros((config.num_samples_per_chirp, 
                                   2*config.num_chirps_per_frame, 
                                   num_rx_antennas), dtype=complex)
                                   
            print(rd_spectrum.shape)

            for i_ant in range(num_rx_antennas):
                mat = frame[i_ant, :, :]
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:, :, i_ant] = dfft_dbfs

            # 2D Beamforming: produces (range, doppler, azimuth, elevation)
            beam_formed = dbf2d.run(rd_spectrum)
            
            # Get 3D energy map: (range, azimuth, elevation)
            energy_map = dbf2d.compute_range_azimuth_elevation_map(beam_formed)
            
            # Find peaks in 3D space
            detected_peaks = find_peaks_3d_adaptive(
                energy_map,
                noise_floor_percentile=50,   # Use median as noise floor
                snr_threshold_db=12,          # 12 dB SNR threshold
                min_distance=2,               # Minimum 2 pixels between peaks
                max_peaks=20                  # Detect up to 20 peaks
            )
            
            # Build title with best peak info
            if len(detected_peaks) > 0:
                best = detected_peaks[0]
                range_m = (best[0] / config.num_samples_per_chirp) * max_range_m
                az_deg = azimuth_angles[best[1]]
                el_deg = elevation_angles[best[2]]
                title = (f"{len(detected_peaks)} objects | "
                        f"Best: Az={az_deg:+.1f}° El={el_deg:+.1f}° "
                        f"R={range_m:.2f}m SNR={best[4]:.1f}dB")
            else:
                title = "No objects detected"
            
            # Plot in 3D
            plot.draw(detected_peaks, title, max_range_m, config.num_samples_per_chirp,
                     azimuth_angles, elevation_angles)

        plot.close()
