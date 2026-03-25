# ===========================================================================
# Copyright (C) 2021-2022 Infineon Technologies AG
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

import numpy as np
from scipy import signal, constants

from .fft import fft_spectrum
from ..sensors.base import RadarConfig


class RangeProfileProcessor:
    """Compute range profile (distance FFT) from raw chirp data.

    Args:
        config: Radar configuration.
        skip_bins: Number of near-range bins to skip in peak search
                   (avoids DC leakage / antenna coupling).
    """

    def __init__(self, config: RadarConfig, skip_bins: int = 8):
        self.config = config
        self.skip_bins = skip_bins
        self.num_samples = config.num_samples_per_chirp
        self.num_chirps = config.num_chirps_per_frame

        # Compute Blackman-Harris window
        if hasattr(signal, 'windows') and hasattr(signal.windows, 'blackmanharris'):
            self.range_window = signal.windows.blackmanharris(self.num_samples).reshape(1, self.num_samples)
        else:
            self.range_window = signal.blackmanharris(self.num_samples).reshape(1, self.num_samples)

        # Range bin length in meters
        fft_size = self.num_samples * 2
        self.range_bin_length = constants.c / (2 * config.bandwidth_hz * fft_size / self.num_samples)

    def compute_range_profile(self, chirp_data: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute range profile from single-antenna chirp data.

        Args:
            chirp_data: (num_chirps, num_samples) real or complex array.

        Returns:
            (peak_distance_m, range_profile) tuple.
        """
        # For complex data, use magnitude for FFT input
        if np.iscomplexobj(chirp_data):
            data = np.real(chirp_data)
        else:
            data = chirp_data

        range_fft = fft_spectrum(data, self.range_window)
        range_fft_abs = np.abs(range_fft)

        # Coherent integration across chirps
        distance_data = np.divide(range_fft_abs.sum(axis=0), self.num_chirps)

        # Peak search (skip near-range bins)
        distance_peak = np.argmax(distance_data[self.skip_bins:])
        distance_peak_m = self.range_bin_length * (distance_peak + self.skip_bins)

        return distance_peak_m, distance_data

    def compute_complex_range_profile(self, chirp_data: np.ndarray) -> np.ndarray:
        """Compute complex (magnitude + phase) range profile.

        Preserves phase information needed for dielectric property estimation.

        Args:
            chirp_data: (num_chirps, num_samples) array.

        Returns:
            Complex range profile (num_samples,) averaged across chirps.
        """
        if np.iscomplexobj(chirp_data):
            data = np.real(chirp_data)
        else:
            data = chirp_data

        range_fft = fft_spectrum(data, self.range_window)

        # Average across chirps (preserves phase)
        return np.mean(range_fft, axis=0)
