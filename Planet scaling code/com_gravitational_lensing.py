"""
COM Framework Gravitational Lensing Module

This module implements the gravitational lensing calculations using the 
Continuous Oscillatory Model (COM) framework.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
from com_framework_constants import LZ, HQS, G, c, M_SUN, AU
from com_framework_core import COMModel

class GravitationalLensingModel(COMModel):
    """
    Implementation of the COM framework for gravitational lensing calculations.
    """
    
    def __init__(self):
        """Initialize the gravitational lensing model."""
        super().__init__()
    
    def einstein_ring_radius_gr(self, mass, distance_observer, distance_source):
        """
        Calculate Einstein ring radius using General Relativity.
        
        Parameters:
        - mass: Mass of the lensing object in kg
        - distance_observer: Distance from observer to lens in meters
        - distance_source: Distance from lens to source in meters
        
        Returns:
        - Einstein ring radius in meters
        """
        # Calculate the reduced distance
        reduced_distance = (distance_observer * distance_source) / (distance_observer + distance_source)
        
        # Calculate Einstein radius
        einstein_radius = np.sqrt((4 * G * mass * reduced_distance) / c**2)
        
        return einstein_radius
    
    def einstein_ring_radius_com(self, mass, distance_observer, distance_source):
        """
        Calculate Einstein ring radius using COM framework.
        
        Parameters:
        - mass: Mass of the lensing object in kg
        - distance_observer: Distance from observer to lens in meters
        - distance_source: Distance from lens to source in meters
        
        Returns:
        - COM-modified Einstein ring radius in meters
        """
        # Calculate standard Einstein radius
        standard_radius = self.einstein_ring_radius_gr(mass, distance_observer, distance_source)
        
        # Calculate energy density at the Einstein radius
        energy_density_ratio = (G * mass) / (c**2 * standard_radius**3)
        
        # Calculate octave position in COM framework
        octave_position = self.octave_position(energy_density_ratio)
        
        # Calculate correction factor
        correction = np.sqrt(self.com_correction_factor(octave_position))
        
        # Calculate COM-modified Einstein radius
        com_radius = standard_radius * correction
        
        return com_radius
    
    def deflection_angle_gr(self, mass, impact_parameter):
        """
        Calculate light deflection angle using General Relativity.
        
        Parameters:
        - mass: Mass of the lensing object in kg
        - impact_parameter: Closest approach distance in meters
        
        Returns:
        - Deflection angle in radians
        """
        return (4 * G * mass) / (c**2 * impact_parameter)
    
    def deflection_angle_com(self, mass, impact_parameter, distance_observer=None, distance_source=None):
        """
        Calculate light deflection angle using COM framework.
        
        Parameters:
        - mass: Mass of the lensing object in kg
        - impact_parameter: Closest approach distance in meters
        - distance_observer: Distance from observer to lens in meters (optional)
        - distance_source: Distance from lens to source in meters (optional)
        
        Returns:
        - Deflection angle in radians
        """
        # Calculate standard GR deflection angle
        gr_deflection = self.deflection_angle_gr(mass, impact_parameter)
        
        # Calculate energy density ratio
        energy_density_ratio = (G * mass) / (c**2 * impact_parameter**3)
        
        # Calculate octave position in COM framework
        octave_position = self.octave_position(energy_density_ratio)
        
        # Calculate correction factor
        correction = self.com_correction_factor(octave_position)
        
        # Calculate COM-modified deflection angle
        com_deflection = gr_deflection * correction
        
        return com_deflection
    
    def magnification_gr(self, mass, impact_parameter, distance_observer, distance_source):
        """
        Calculate magnification factor using General Relativity.
        
        Parameters:
        - mass: Mass of the lensing object in kg
        - impact_parameter: Impact parameter in meters
        - distance_observer: Distance from observer to lens in meters
        - distance_source: Distance from lens to source in meters
        
        Returns:
        - Magnification factor (dimensionless)
        """
        # Calculate Einstein radius
        einstein_radius = self.einstein_ring_radius_gr(mass, distance_observer, distance_source)
        
        # Normalized impact parameter
        u = impact_parameter / einstein_radius
        
        # Magnification formula
        magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
        
        return magnification
    
    def magnification_com(self, mass, impact_parameter, distance_observer, distance_source):
        """
        Calculate magnification factor using COM framework.
        
        Parameters:
        - mass: Mass of the lensing object in kg
        - impact_parameter: Impact parameter in meters
        - distance_observer: Distance from observer to lens in meters
        - distance_source: Distance from lens to source in meters
        
        Returns:
        - Magnification factor (dimensionless)
        """
        # Calculate COM Einstein radius
        einstein_radius = self.einstein_ring_radius_com(mass, distance_observer, distance_source)
        
        # Normalized impact parameter
        u = impact_parameter / einstein_radius
        
        # Magnification formula
        magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
        
        return magnification
    
    def relativistic_correction(self, distance, velocity_fraction=None):
        """
        Calculate relativistic correction factors.
        
        Parameters:
        - distance: Distance to the object in meters
        - velocity_fraction: Velocity as fraction of c (optional)
        
        Returns:
        - Combined relativistic correction factor
        """
        # Estimate velocity as fraction of c if not provided
        if velocity_fraction is None:
            # Approximate orbital velocity using Kepler's laws
            velocity_ms = np.sqrt(G / (distance))
            velocity_fraction = velocity_ms / c
        
        # Light travel time effect (increases with distance)
        light_travel_factor = 1.0 + (distance / (30.0 * AU)) * 0.01
        
        # Relativistic Doppler effect
        doppler_factor = np.sqrt((1 + velocity_fraction) / (1 - velocity_fraction))
        
        # Gravitational time dilation
        time_dilation_factor = 1.0 / np.sqrt(1 - 2 * G / (c**2 * distance))
        
        # Combined correction factor
        correction = light_travel_factor * doppler_factor * time_dilation_factor
        
        return correction
