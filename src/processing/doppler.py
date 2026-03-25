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

import numpy as np
from scipy import signal

from .fft import fft_spectrum


class DopplerProcessor:
    """Compute Range-Doppler map from raw radar data.

    Args:
        num_samples: Number of ADC samples per chirp.
        num_chirps_per_frame: Number of chirps in one frame.
        num_ant: Number of RX antennas.
        mti_alpha: Moving Target Indicator filter parameter (0 to 1).
    """

    def __init__(
        self,
        num_samples: int,
        num_chirps_per_frame: int,
        num_ant: int,
        mti_alpha: float = 0.8,
    ):
        self.num_chirps_per_frame = num_chirps_per_frame

        # Compute Blackman-Harris windows
        if hasattr(signal, 'windows') and hasattr(signal.windows, 'blackmanharris'):
            self.range_window = signal.windows.blackmanharris(num_samples).reshape(1, num_samples)
            self.doppler_window = signal.windows.blackmanharris(num_chirps_per_frame).reshape(1, num_chirps_per_frame)
        else:
            self.range_window = signal.blackmanharris(num_samples).reshape(1, num_samples)
            self.doppler_window = signal.blackmanharris(num_chirps_per_frame).reshape(1, num_chirps_per_frame)

        self.mti_alpha = mti_alpha
        self.mti_history = np.zeros((num_chirps_per_frame, num_samples, num_ant))

    def compute_doppler_map(self, data: np.ndarray, i_ant: int) -> np.ndarray:
        """Compute Range-Doppler map for one antenna.

        Args:
            data: Raw chirp data (num_chirps, num_samples).
            i_ant: RX antenna index.

        Returns:
            Complex Range-Doppler map (num_samples, num_chirps * 2).
        """
        # Mean removal
        data = data - np.average(data)

        # MTI processing
        data_mti = data - self.mti_history[:, :, i_ant]
        self.mti_history[:, :, i_ant] = (
            data * self.mti_alpha + self.mti_history[:, :, i_ant] * (1 - self.mti_alpha)
        )

        # Range FFT
        fft1d = fft_spectrum(data_mti, self.range_window)

        # Transpose: distance on y-axis
        fft1d = np.transpose(fft1d)

        # Doppler windowing
        fft1d = np.multiply(fft1d, self.doppler_window)

        # Doppler FFT with zero padding
        zp2 = np.pad(fft1d, ((0, 0), (0, self.num_chirps_per_frame)), "constant")
        fft2d = np.fft.fft(zp2) / self.num_chirps_per_frame

        # Re-arrange for zero speed at centre
        return np.fft.fftshift(fft2d, (1,))
