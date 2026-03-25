# ===========================================================================
# Copyright (C) 2021 Infineon Technologies AG
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


def fft_spectrum(mat, range_window):
    """Calculate FFT spectrum from chirp data.

    Args:
        mat: Chirp data matrix (num_chirps, num_samples).
        range_window: Window function applied before FFT.

    Returns:
        Complex range FFT result (num_chirps, num_samples).
    """
    [num_chirps, num_samples] = np.shape(mat)

    # Step 1 - remove DC bias from samples
    avgs = np.average(mat, 1).reshape(num_chirps, 1)
    mat = mat - avgs

    # Step 2 - Windowing the Data
    mat = np.multiply(mat, range_window)

    # Step 3 - Zero padding (2x)
    zp1 = np.pad(mat, ((0, 0), (0, num_samples)), 'constant')

    # Step 4 - FFT for distance information
    range_fft = np.fft.fft(zp1) / num_samples

    # Keep positive spectrum only, compensate energy by doubling
    range_fft = 2 * range_fft[:, range(int(num_samples))]

    return range_fft
