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

import pprint
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from ifxAvian import Avian
from helpers.fft_spectrum import *
from helpers.DigitalBeamForming import DigitalBeamForming as DBF
from helpers.DopplerAlgo import DopplerAlgo

def num_rx_antennas_from_config(config):
    rx_mask = config.rx_mask

    # popcount for rx_mask
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c


def find_peaks_2d_adaptive(data, noise_floor_percentile=50, snr_threshold_db=15, min_distance=3, max_peaks=10):
    """
    Find multiple peaks in a 2D array using adaptive noise floor estimation.
    
    Args:
        data: 2D numpy array (range x angle)
        noise_floor_percentile: Percentile of data to use as noise floor (default: 50 for median)
        snr_threshold_db: Minimum SNR in dB above noise floor for peak detection
        min_distance: Minimum distance between peaks in pixels
        max_peaks: Maximum number of peaks to return
    
    Returns:
        List of tuples: [(range_idx, angle_idx, value, snr_db), ...] sorted by SNR descending
    """
    # Convert to dB scale for better thresholding
    data_db = 20 * np.log10(np.abs(data) + 1e-10)
    
    # Calculate adaptive noise floor using percentile
    noise_floor_db = np.percentile(data_db, noise_floor_percentile)
    
    # Apply adaptive threshold
    threshold_db = noise_floor_db + snr_threshold_db
    data_thresholded = np.where(data_db >= threshold_db, data, 0)
    
    # Find local maxima using maximum filter
    # A point is a local maximum if it equals the maximum in its neighborhood
    footprint = np.ones((min_distance * 2 + 1, min_distance * 2 + 1))
    local_max = (ndimage.maximum_filter(data_thresholded, footprint=footprint) == data_thresholded)
    
    # Remove peaks at borders
    local_max[0, :] = False
    local_max[-1, :] = False
    local_max[:, 0] = False
    local_max[:, -1] = False
    
    # Get peak coordinates and values
    peak_coords = np.argwhere(local_max & (data_thresholded > 0))
    
    if len(peak_coords) == 0:
        # No peaks found, fall back to global maximum
        peak_range_idx, peak_angle_idx = np.unravel_index(data.argmax(), data.shape)
        peak_value = data[peak_range_idx, peak_angle_idx]
        peak_snr_db = data_db[peak_range_idx, peak_angle_idx] - noise_floor_db
        return [(peak_range_idx, peak_angle_idx, peak_value, peak_snr_db)]
    
    # Get peak values and calculate SNR
    peak_values = data_thresholded[local_max & (data_thresholded > 0)]
    peak_values_db = data_db[local_max & (data_thresholded > 0)]
    peak_snr_db = peak_values_db - noise_floor_db
    
    # Create list of peaks with their SNR
    peaks = []
    for i, (coord, value, snr) in enumerate(zip(peak_coords, peak_values, peak_snr_db)):
        peaks.append((coord[0], coord[1], value, snr))
    
    # Sort by SNR descending
    peaks.sort(key=lambda x: x[3], reverse=True)
    
    # Return up to max_peaks
    return peaks[:max_peaks]


class LivePlot:
    def __init__(self, max_angle_degrees : float, max_range_m : float):
        # max_angle_degrees: maximum supported speed
        # max_range_m:   maximum supported range
        self.h = None
        self.peak_markers = None
        self.max_angle_degrees = max_angle_degrees
        self.max_range_m = max_range_m

        self._fig, self._ax = plt.subplots(nrows=1, ncols=1)

        self._fig.canvas.manager.set_window_title("Range-Angle-Map using DBF")
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data : np.ndarray):
        # First time draw

        minmin = -60
        maxmax = 0

        self.h = self._ax.imshow(
                   data,
                   vmin=minmin, vmax=maxmax,
                   cmap='viridis',
                   extent=(-self.max_angle_degrees,
                           self.max_angle_degrees,
                           0,
                           self.max_range_m),
                   origin='lower')

        self._ax.set_xlabel("angle (degrees)")
        self._ax.set_ylabel("distance (m)")
        self._ax.set_aspect("auto")

        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar = self._fig.colorbar(self.h, cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (a.u.)")

    def _draw_next_time(self, data : np.ndarray):
        # Update data for each antenna

        self.h.set_data(data)

    def draw(self, data : np.ndarray, title : str, peaks=None):
        if self._is_window_open:
            if self.h:
                self._draw_next_time(data)
            else:
                self._draw_first_time(data)
            self._ax.set_title(title)
            
            # Remove old peak markers
            if self.peak_markers is not None:
                self.peak_markers.remove()
                self.peak_markers = None
            
            # Draw new peak markers if provided
            if peaks is not None and len(peaks) > 0:
                angles = [p[0] for p in peaks]
                ranges = [p[1] for p in peaks]
                self.peak_markers = self._ax.scatter(angles, ranges, c='red', marker='x', 
                                                     s=100, linewidths=2, zorder=5)
            self._ax.set_title(title)

            # Needed for Matplotlib ver: 3.4.0 and 3.4.1 helps with capture closing event
            plt.draw()
            plt.pause(1e-3)

    def close(self, event = None):
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
    num_beams = 127         # number of beams
    max_angle_degrees = 90 # maximum angle, angle ranges from -40 to +40 degrees

    config = Avian.DeviceConfig(
        sample_rate_Hz = 1_000_000,       # 1MHZ
        rx_mask = 7,                      # activate RX1 and RX3
        tx_mask = 1,                      # activate TX1
        if_gain_dB = 33,                  # gain of 33dB
        tx_power_level = 31,              # TX power level of 31
        start_frequency_Hz = 60e9,        # 60GHz 
        end_frequency_Hz = 61.5e9,        # 61.5GHz
        num_chirps_per_frame = 128,       # 128 chirps per frame
        num_samples_per_chirp = 64,       # 64 samples per chirp
        chirp_repetition_time_s = 0.0005, # 0.5ms
        frame_repetition_time_s = 0.15,   # 0.15s, frame_Rate = 6.667Hz
        mimo_mode = 'off'                 # MIMO disabled
    )

    with Avian.Device() as device:
        # set configuration
        device.set_config(config)

        # get metrics and print them
        metrics = device.metrics_from_config(config)
        pprint.pprint(metrics)

        # get maximum range
        max_range_m = metrics.max_range_m

        # Create frame handle
        num_rx_antennas = num_rx_antennas_from_config(config)

        # Create objects for Range-Doppler, DBF, and plotting.
        doppler = DopplerAlgo(config.num_samples_per_chirp, num_chirps_per_frame=config.num_chirps_per_frame, num_ant=num_rx_antennas)
        dbf = DBF(num_rx_antennas, num_beams = num_beams, max_angle_degrees = max_angle_degrees)
        plot = LivePlot(max_angle_degrees, max_range_m)

        # Reference max energy from first frame (set after first frame is acquired)
        reference_max_energy = None

        while not plot.is_closed():
            # frame has dimension num_rx_antennas x num_samples_per_chirp x num_chirps_per_frame
            frame = device.get_next_frame()

            rd_spectrum = np.zeros((config.num_samples_per_chirp, 2*config.num_chirps_per_frame, num_rx_antennas), dtype=complex)

            beam_range_energy = np.zeros((config.num_samples_per_chirp, num_beams))

            for i_ant in range(num_rx_antennas): # For each antenna
                # Current RX antenna (num_samples_per_chirp x num_chirps_per_frame)
                mat = frame[i_ant, :, :]

                # Compute Doppler spectrum
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:,:,i_ant] = dfft_dbfs

            # Compute Range-Angle map
            rd_beam_formed = dbf.run(rd_spectrum)
            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:,:,i_beam]
                beam_range_energy[:,i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beams)

            # Maximum energy in Range-Angle map
            max_energy = np.max(beam_range_energy)

            # Set reference max energy from first frame for consistent color scaling
            if reference_max_energy is None:
                reference_max_energy = max_energy
                print(f"Reference max energy set from first frame: {reference_max_energy:.2f}")

            # Rescale map for visualization using REFERENCE max energy (not per-frame)
            scale = 150
            beam_range_energy_display = scale*(beam_range_energy/reference_max_energy - 1)

            # Find multiple peaks using adaptive noise floor
            detected_peaks = find_peaks_2d_adaptive(
                beam_range_energy,
                noise_floor_percentile=50,  # Use median as noise floor
                snr_threshold_db=15,         # 15 dB SNR threshold
                min_distance=2,              # Minimum 2 pixels between peaks
                max_peaks=10                 # Detect up to 10 peaks
            )
            
            # Convert peak indices to physical units for display
            peak_markers = []
            peak_info_str = ""
            
            for i, (range_idx, angle_idx, value, snr_db) in enumerate(detected_peaks):
                angle_deg = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)[angle_idx]
                range_m = (range_idx / config.num_samples_per_chirp) * max_range_m
                peak_markers.append((angle_deg, range_m))
                
                if i == 0:
                    peak_info_str = f"Peak {i+1}: {angle_deg:+.1f}° @ {range_m:.2f}m (SNR: {snr_db:.1f}dB)"
                elif i < 3:  # Show first 3 peaks in title
                    peak_info_str += f" | Peak {i+1}: {angle_deg:+.1f}° @ {range_m:.2f}m"
            
            title = f"Range-Angle Map (DBF) | {len(detected_peaks)} peaks | {peak_info_str}"

            # And plot...
            plot.draw(beam_range_energy_display, title, peaks=peak_markers)

        plot.close()
