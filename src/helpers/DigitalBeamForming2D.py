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


class DigitalBeamForming2D:
    """
    2D Digital Beamforming for BGT60TR13C with 3 RX antennas in L-shape arrangement.
    Computes both azimuth (horizontal) and elevation (vertical) angles.
    
    BGT60TR13C antenna layout:
        RX1 ---- RX2  (horizontal, ~lambda/2 spacing for azimuth)
         |
        RX3           (vertical, ~lambda/2 spacing for elevation)
    
    This allows measurement of both azimuth and elevation angles from a single
    radar frame without requiring MIMO.
    """
    
    def __init__(self, 
                 num_beams_azimuth: int = 32,
                 num_beams_elevation: int = 32,
                 max_angle_azimuth_deg: float = 60,
                 max_angle_elevation_deg: float = 60,
                 center_frequency_hz: float = 60.5e9):
        """
        Initialize 2D beamformer for BGT60TR13C.
        
        Args:
            num_beams_azimuth: Number of azimuth beams to form
            num_beams_elevation: Number of elevation beams to form
            max_angle_azimuth_deg: Maximum azimuth angle (+/-)
            max_angle_elevation_deg: Maximum elevation angle (+/-)
            center_frequency_hz: Center frequency for wavelength calculation
        """
        self.num_beams_az = num_beams_azimuth
        self.num_beams_el = num_beams_elevation
        self.max_angle_az = max_angle_azimuth_deg
        self.max_angle_el = max_angle_elevation_deg
        
        # Calculate wavelength from center frequency
        c = 3e8  # Speed of light (m/s)
        self.wavelength = c / center_frequency_hz
        
        # BGT60TR13C antenna spacing (approximately lambda/2 for optimal beamforming)
        self.d = self.wavelength / 2
        
        # Define antenna positions for BGT60TR13C L-shaped array
        # RX1 at origin, RX2 horizontal offset (azimuth), RX3 vertical offset (elevation)
        self.antenna_positions = np.array([
            [0, 0],           # RX1: origin (reference antenna)
            [self.d, 0],      # RX2: horizontal offset for azimuth measurement
            [0, self.d],      # RX3: vertical offset for elevation measurement
        ])
        
        # Pre-compute angle arrays
        self.azimuth_angles = np.linspace(-max_angle_azimuth_deg, max_angle_azimuth_deg, num_beams_azimuth)
        self.elevation_angles = np.linspace(-max_angle_elevation_deg, max_angle_elevation_deg, num_beams_elevation)
        
        # Pre-compute steering matrix for all angle combinations
        self.steering_matrix = self._compute_steering_matrix()
        
    def _compute_steering_matrix(self):
        """
        Compute steering vectors for all azimuth/elevation angle combinations.
        
        The steering vector represents the expected phase difference between
        antennas for a signal arriving from a specific direction.
        
        Returns:
            steering_matrix: Shape (3, num_beams_az, num_beams_el) complex array
        """
        az_rad = np.deg2rad(self.azimuth_angles)
        el_rad = np.deg2rad(self.elevation_angles)
        
        # Wave number k = 2*pi/lambda
        k = 2 * np.pi / self.wavelength
        
        # Create steering matrix
        steering = np.zeros((3, self.num_beams_az, self.num_beams_el), dtype=complex)
        
        for i_az, az in enumerate(az_rad):
            for i_el, el in enumerate(el_rad):
                # Convert spherical angles to direction cosines
                # u: x-direction component (azimuth)
                # v: y-direction component (elevation)
                u = np.sin(az) * np.cos(el)
                v = np.sin(el)
                
                # Calculate phase shift for each antenna based on position
                for i_ant, pos in enumerate(self.antenna_positions):
                    # Phase = k * (position dot direction)
                    phase = k * (pos[0] * u + pos[1] * v)
                    steering[i_ant, i_az, i_el] = np.exp(-1j * phase)
        
        return steering
    
    def run(self, data):
        """
        Apply 2D beamforming to radar data.
        
        This performs digital beamforming across all azimuth and elevation angles,
        producing a 4D output (range x doppler x azimuth x elevation).
        
        Args:
            data: Input array of shape (num_range_bins, num_doppler_bins, 3)
                  where 3 is the number of RX antennas (RX1, RX2, RX3)
        
        Returns:
            beam_formed: Shape (num_range_bins, num_doppler_bins, num_beams_az, num_beams_el)
                        Complex-valued beamformed output
        """
        num_range_bins, num_doppler_bins, num_ant = data.shape
        
        if num_ant != 3:
            raise ValueError(f"BGT60TR13C 2D beamforming requires 3 RX antennas, got {num_ant}")
        
        # Output array: 4D (range x doppler x azimuth x elevation)
        beam_formed = np.zeros((num_range_bins, num_doppler_bins, 
                               self.num_beams_az, self.num_beams_el), dtype=complex)
        
        # Vectorized beamforming for efficiency
        # For each angle combination, apply the steering vector
        for i_az in range(self.num_beams_az):
            for i_el in range(self.num_beams_el):
                # Steering vector for this direction: shape (3,)
                w = self.steering_matrix[:, i_az, i_el]
                
                # Apply steering weights to all range-doppler bins at once
                # data shape: (range, doppler, 3), w.conj() shape: (3,)
                # Result: (range, doppler)
                beam_formed[:, :, i_az, i_el] = np.tensordot(data, w.conj(), axes=([2], [0]))
        
        return beam_formed
    
    def compute_range_azimuth_elevation_map(self, beam_formed):
        """
        Collapse Doppler dimension to get 3D spatial energy map.
        
        This sums the energy across all Doppler bins to produce a
        range-azimuth-elevation map suitable for 3D target detection.
        
        Args:
            beam_formed: Shape (num_range, num_doppler, num_az, num_el)
        
        Returns:
            energy_map: Shape (num_range, num_az, num_el) - real-valued energy
        """
        # Sum squared magnitude across Doppler dimension
        return np.sum(np.abs(beam_formed)**2, axis=1)
    
    def get_angle_axes(self):
        """
        Return the azimuth and elevation angle arrays for plotting/indexing.
        
        Returns:
            azimuth_angles: 1D array of azimuth angles in degrees
            elevation_angles: 1D array of elevation angles in degrees
        """
        return self.azimuth_angles, self.elevation_angles
    
    def index_to_angles(self, az_idx, el_idx):
        """
        Convert beam indices to physical angles.
        
        Args:
            az_idx: Azimuth beam index
            el_idx: Elevation beam index
            
        Returns:
            azimuth_deg: Azimuth angle in degrees
            elevation_deg: Elevation angle in degrees
        """
        return self.azimuth_angles[az_idx], self.elevation_angles[el_idx]
