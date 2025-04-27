"""
COM Framework Core Model Module

This module implements the core functionality of the Continuous Oscillatory Model (COM) framework,
including methods for calculating octave positions, modulation factors, and stability metrics.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
from com_framework_constants import LZ, HQS, FIBONACCI_NORMALIZED, G, c, AU

class COMModel:
    """
    Core implementation of the COM framework for scale analysis and stability calculations.
    """
    
    def __init__(self):
        """Initialize the COM model with default parameters."""
        self.lz = LZ
        self.hqs = HQS
        self.fibonacci_pattern = FIBONACCI_NORMALIZED
        
    def octave_position(self, value, reference=1.0):
        """
        Calculate the octave position in the COM framework.
        
        Parameters:
        - value: The value to calculate octave position for
        - reference: Reference value (default: 1.0)
        
        Returns:
        - Octave position (0 to 1)
        """
        return np.log(value / reference) / np.log(self.lz) % 1
    
    def fibonacci_value(self, octave_position):
        """
        Get the Fibonacci pattern value for a given octave position.
        
        Parameters:
        - octave_position: Position in the octave (0 to 1)
        
        Returns:
        - Normalized Fibonacci pattern value
        """
        pattern_index = int(octave_position * 24) % 24
        return self.fibonacci_pattern[pattern_index]
    
    def hqs_modulation(self, octave_position):
        """
        Calculate the HQS modulation factor.
        
        Parameters:
        - octave_position: Position in the octave (0 to 1)
        
        Returns:
        - HQS modulation factor
        """
        return 1 + self.hqs * np.sin(np.pi * octave_position / self.hqs)
    
    def fibonacci_modulation(self, octave_position):
        """
        Calculate the Fibonacci pattern modulation factor.
        
        Parameters:
        - octave_position: Position in the octave (0 to 1)
        
        Returns:
        - Fibonacci modulation factor
        """
        pattern_value = self.fibonacci_value(octave_position)
        return 1 + (self.lz - 1) * pattern_value
    
    def com_correction_factor(self, octave_position):
        """
        Calculate the combined COM correction factor.
        
        Parameters:
        - octave_position: Position in the octave (0 to 1)
        
        Returns:
        - Combined correction factor
        """
        hqs_factor = self.hqs_modulation(octave_position)
        fib_factor = self.fibonacci_modulation(octave_position)
        return hqs_factor * fib_factor
    
    def scale_position(self, scale, reference_scale=1e-15):
        """
        Calculate the position of a scale in the COM framework's unified scaling structure.
        
        Parameters:
        - scale: Scale to analyze (in meters)
        - reference_scale: Reference scale (default: 1e-15 m, approximately proton scale)
        
        Returns:
        - Dictionary with scale position information
        """
        # Calculate octave number (how many LZ factors from reference)
        octave_number = np.log(scale / reference_scale) / np.log(self.lz)
        
        # Calculate octave position (fractional part)
        octave_position = octave_number % 1
        
        # Get Fibonacci pattern value
        pattern_value = self.fibonacci_value(octave_position)
        
        # Calculate HQS modulation
        hqs_mod = self.hqs_modulation(octave_position)
        
        # Calculate stability factor (higher means more stable)
        stability = hqs_mod * pattern_value
        
        return {
            "scale": scale,
            "octave_number": octave_number,
            "octave_position": octave_position,
            "pattern_value": pattern_value,
            "hqs_modulation": hqs_mod,
            "stability": stability
        }
    
    def analyze_scale_range(self, scale_min, scale_max, num_points=1000):
        """
        Analyze a range of scales using the COM framework.
        
        Parameters:
        - scale_min: Minimum scale to analyze (in meters)
        - scale_max: Maximum scale to analyze (in meters)
        - num_points: Number of points to analyze
        
        Returns:
        - Dictionary with scale analysis results
        """
        # Generate logarithmically spaced scales
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num_points)
        
        # Analyze each scale
        results = [self.scale_position(scale) for scale in scales]
        
        # Extract arrays for plotting
        octave_positions = [r["octave_position"] for r in results]
        pattern_values = [r["pattern_value"] for r in results]
        hqs_modulations = [r["hqs_modulation"] for r in results]
        stabilities = [r["stability"] for r in results]
        
        return {
            "scales": scales,
            "octave_positions": octave_positions,
            "pattern_values": pattern_values,
            "hqs_modulations": hqs_modulations,
            "stabilities": stabilities,
            "results": results
        }
