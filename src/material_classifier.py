"""
Material classification based on radar signatures.

This module provides feature extraction and classification capabilities
to identify different materials based on their radar characteristics.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RadarFeatures:
    """Radar features for material classification.
    
    Attributes:
        rcs: Radar Cross Section (amplitude in dB)
        range: Distance to target in meters
        angle: Azimuth angle in radians
        texture: Local variance in range-angle map (surface roughness indicator)
        peak_sharpness: Sharpness of the peak (1/width)
    """
    rcs: float
    range: float
    angle: float
    texture: float
    peak_sharpness: float


class MaterialClassifier:
    """Classify materials based on radar signatures.
    
    Different materials have distinct radar signatures:
    - Metal: High RCS, low texture, sharp peaks
    - Wood: Medium RCS, high texture (rough surface)
    - Plastic: Medium-low RCS, smooth
    - Snow/Water: Low RCS, high texture (scattering)
    """
    
    def __init__(self):
        """Initialize the material classifier."""
        # Material classification thresholds
        # These would ideally be learned from training data
        self.material_profiles = {
            'metal': {
                'rcs_min': -10,
                'texture_max': 2.0,
                'sharpness_min': 0.8
            },
            'rough_metal': {
                'rcs_min': -10,
                'texture_min': 2.0,
                'texture_max': 5.0
            },
            'wood': {
                'rcs_min': -20,
                'rcs_max': -10,
                'texture_min': 3.0
            },
            'plastic': {
                'rcs_min': -25,
                'rcs_max': -15,
                'texture_max': 2.5
            },
            'snow': {
                'rcs_min': -35,
                'rcs_max': -20,
                'texture_min': 2.0
            },
            'background': {
                'rcs_max': -35
            }
        }
    
    def extract_features(
        self,
        range_angle_map: np.ndarray,
        beam_idx: int,
        range_idx: int,
        angles: np.ndarray,
        max_range_m: float,
        window_size: int = 3
    ) -> RadarFeatures:
        """Extract features around a detected point.
        
        Args:
            range_angle_map: Power map (num_beams, num_range_bins)
            beam_idx: Beam index of detection
            range_idx: Range index of detection
            angles: Array of beam angles in degrees
            max_range_m: Maximum range in meters
            window_size: Size of neighborhood for texture calculation
        
        Returns:
            RadarFeatures object containing extracted features.
        """
        # RCS (amplitude)
        rcs = range_angle_map[beam_idx, range_idx]
        
        # Local texture (variance in neighborhood)
        neighborhood = self._get_neighborhood(
            range_angle_map, beam_idx, range_idx, window_size
        )
        texture = np.std(neighborhood)
        
        # Peak sharpness (inverse of width at half maximum)
        sharpness = self._compute_peak_sharpness(
            range_angle_map, beam_idx, range_idx
        )
        
        # Range and angle
        num_range_bins = range_angle_map.shape[1]
        range_resolution = max_range_m / num_range_bins
        range_val = range_idx * range_resolution
        angle_deg = angles[beam_idx]
        angle_rad = np.deg2rad(angle_deg)
        
        return RadarFeatures(
            rcs=rcs,
            range=range_val,
            angle=angle_rad,
            texture=texture,
            peak_sharpness=sharpness
        )
    
    def classify_material(self, features: RadarFeatures) -> str:
        """Classify material based on extracted features.
        
        This is a simple rule-based classifier. For production use,
        consider training a machine learning model (Random Forest, SVM, etc.)
        on labeled data.
        
        Args:
            features: Extracted radar features.
        
        Returns:
            Material classification string.
        """
        # Metal detection (high RCS)
        if features.rcs > self.material_profiles['metal']['rcs_min']:
            if features.texture < self.material_profiles['metal']['texture_max'] and \
               features.peak_sharpness > self.material_profiles['metal']['sharpness_min']:
                return "metal"
            elif features.texture >= self.material_profiles['rough_metal']['texture_min'] and \
                 features.texture < self.material_profiles['rough_metal']['texture_max']:
                return "rough_metal"
            else:
                return "metal"
        
        # Wood detection (medium RCS, high texture)
        elif features.rcs > self.material_profiles['wood']['rcs_min'] and \
             features.rcs <= self.material_profiles['wood']['rcs_max'] and \
             features.texture >= self.material_profiles['wood']['texture_min']:
            return "wood"
        
        # Plastic detection (medium-low RCS, smooth)
        elif features.rcs > self.material_profiles['plastic']['rcs_min'] and \
             features.rcs <= self.material_profiles['plastic']['rcs_max'] and \
             features.texture < self.material_profiles['plastic']['texture_max']:
            return "plastic"
        
        # Snow detection (low RCS, moderate texture)
        elif features.rcs > self.material_profiles['snow']['rcs_min'] and \
             features.rcs <= self.material_profiles['snow']['rcs_max'] and \
             features.texture >= self.material_profiles['snow']['texture_min']:
            return "snow"
        
        # Background/unknown
        else:
            return "background"
    
    def _get_neighborhood(
        self, 
        data: np.ndarray, 
        i: int, 
        j: int, 
        size: int
    ) -> np.ndarray:
        """Extract neighborhood around point.
        
        Args:
            data: 2D array
            i: Row index
            j: Column index
            size: Neighborhood size (odd number)
        
        Returns:
            Neighborhood array.
        """
        half = size // 2
        i_min = max(0, i - half)
        i_max = min(data.shape[0], i + half + 1)
        j_min = max(0, j - half)
        j_max = min(data.shape[1], j + half + 1)
        
        return data[i_min:i_max, j_min:j_max]
    
    def _compute_peak_sharpness(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        search_radius: int = 5
    ) -> float:
        """Compute sharpness of a peak.
        
        Sharpness is computed as the inverse of the width at half maximum.
        Sharp peaks (like metal reflections) have high sharpness.
        
        Args:
            data: 2D array
            i: Row index of peak
            j: Column index of peak
            search_radius: Radius to search for half-maximum
        
        Returns:
            Peak sharpness value (0-1, higher is sharper).
        """
        peak_val = data[i, j]
        half_max = peak_val - 3  # 3 dB down
        
        # Search in range direction for width
        j_min = max(0, j - search_radius)
        j_max = min(data.shape[1], j + search_radius)
        
        range_profile = data[i, j_min:j_max]
        above_half_max = range_profile > half_max
        
        if np.sum(above_half_max) == 0:
            return 0.0
        
        # Width is number of bins above half maximum
        width = np.sum(above_half_max)
        
        # Sharpness is inverse of width, normalized
        sharpness = 1.0 / (1.0 + width / 2.0)
        
        return sharpness
    
    def get_material_color(self, material: str) -> str:
        """Get color code for visualization.
        
        Args:
            material: Material classification string.
        
        Returns:
            Color string for matplotlib.
        """
        color_map = {
            'metal': 'red',
            'rough_metal': 'orange',
            'wood': 'brown',
            'plastic': 'blue',
            'snow': 'cyan',
            'background': 'gray'
        }
        return color_map.get(material, 'black')
