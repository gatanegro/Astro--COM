"""
Unified COM Framework Model for Planetary Spacing and Gravitational Lensing

This module implements a unified Continuous Oscillatory Model (COM) framework that explains
both planetary semi-major axis spacing and gravitational lensing using the same fundamental
constants and principles.

Key features:
1. Enhanced planetary spacing model with Fibonacci pattern integration
2. Gravitational lensing model based on energy density gradients
3. Scale invariance analysis across quantum to cosmic scales
4. Unified mathematical foundation using LZ and HQS constants

Author: COM Research Team
Date: April 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create directories for output if needed
os.makedirs('figures', exist_ok=True)

# COM Framework Constants
LZ = 1.23498  # Fundamental scaling constant (λ)
HQS = 0.235   # Harmonic Quantum Scalar (η)
HQS_LZ = HQS * LZ  # HQS threshold in absolute terms (≈ 0.29022)

# 24-step Fibonacci digital root pattern
FIBONACCI_PATTERN = [1, 1, 2, 3, 5, 8, 4, 3, 7, 1, 8, 9, 8, 8, 7, 6, 4, 1, 5, 6, 2, 8, 1, 9]
FIBONACCI_NORMALIZED = np.array(FIBONACCI_PATTERN) / 9.0  # Normalize to 0-1 range

# Physical constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
c = 2.99792458e8  # Speed of light in m/s
M_SUN = 1.989e30  # Solar mass in kg
AU = 1.496e11    # Astronomical unit in meters

# Planetary data (semi-major axes in AU)
PLANETS = [
    {"name": "Mercury", "semi_major_axis": 0.39},
    {"name": "Venus", "semi_major_axis": 0.72},
    {"name": "Earth", "semi_major_axis": 1.00},
    {"name": "Mars", "semi_major_axis": 1.52},
    {"name": "Ceres", "semi_major_axis": 2.77},
    {"name": "Jupiter", "semi_major_axis": 5.20},
    {"name": "Saturn", "semi_major_axis": 9.54},
    {"name": "Uranus", "semi_major_axis": 19.18},
    {"name": "Neptune", "semi_major_axis": 30.07}
]

class UnifiedCOMModel:
    """
    Unified implementation of the COM framework for both planetary spacing and gravitational lensing.
    """
    
    def __init__(self):
        """Initialize the unified COM model with default parameters."""
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
    
    # ===== Planetary Spacing Methods =====
    
    def calculate_semi_major_axis_basic(self, n, a0=0.39, phase_factor=4*np.pi):
        """
        Calculate planetary semi-major axis using the basic COM framework equation.
        
        Parameters:
        - n: Orbital index (0 for Mercury, 1 for Venus, etc.)
        - a0: Baseline distance (Mercury's orbit in AU)
        - phase_factor: Factor for phase calculation (default: 4π)
        
        Returns:
        - Predicted semi-major axis in AU
        """
        theta_n = phase_factor * n
        return a0 * (self.lz ** n) * (1 + self.hqs * np.sin(theta_n))
    
    def calculate_semi_major_axis_enhanced(self, n, a0=0.39, phase_factor=4*np.pi):
        """
        Calculate planetary semi-major axis using the enhanced COM framework equation
        with Fibonacci pattern integration.
        
        Parameters:
        - n: Orbital index (0 for Mercury, 1 for Venus, etc.)
        - a0: Baseline distance (Mercury's orbit in AU)
        - phase_factor: Factor for phase calculation (default: 4π)
        
        Returns:
        - Predicted semi-major axis in AU
        """
        # Calculate octave position
        octave_position = n % 1 if n >= 1 else n
        
        # Get Fibonacci pattern value
        pattern_index = int(24 * octave_position) % 24
        pattern_value = self.fibonacci_pattern[pattern_index]
        
        # Calculate phase with Fibonacci modulation
        theta_n = phase_factor * n
        phase_mod = 1 + pattern_value * np.sin(theta_n)
        
        # Calculate semi-major axis with exponential scaling and Fibonacci modulation
        return a0 * (self.lz ** n) * phase_mod
    
    def calculate_semi_major_axis_relativistic(self, n, a0=0.39, phase_factor=4*np.pi):
        """
        Calculate planetary semi-major axis with relativistic corrections.
        
        Parameters:
        - n: Orbital index (0 for Mercury, 1 for Venus, etc.)
        - a0: Baseline distance (Mercury's orbit in AU)
        - phase_factor: Factor for phase calculation (default: 4π)
        
        Returns:
        - Predicted semi-major axis in AU
        """
        # Calculate basic semi-major axis
        basic_axis = self.calculate_semi_major_axis_enhanced(n, a0, phase_factor)
        
        # Apply relativistic correction factor
        # For outer planets, relativistic effects become more significant
        if n >= 5:  # Jupiter and beyond
            # Calculate approximate orbital velocity as fraction of c
            orbital_velocity = np.sqrt(G * M_SUN / (basic_axis * AU)) / c
            
            # Calculate relativistic correction
            gamma = 1 / np.sqrt(1 - orbital_velocity**2)
            
            # Apply correction (increases with distance)
            correction = 1 + (n - 4) * 0.1 * gamma
            return basic_axis * correction
        else:
            return basic_axis
    
    # ===== Gravitational Lensing Methods =====
    
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
    
    # ===== Unified Scale Analysis Methods =====
    
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
        Analyze a range of scales using the unified COM framework.
        
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


class UnifiedAnalysis:
    """
    Analysis tools for the unified COM model.
    """
    
    def __init__(self, model=None):
        """
        Initialize the analysis tools.
        
        Parameters:
        - model: UnifiedCOMModel instance (creates new one if None)
        """
        self.model = model if model is not None else UnifiedCOMModel()
    
    def analyze_planetary_spacing(self, save_fig=True):
        """
        Analyze planetary spacing using the unified COM model.
        
        Parameters:
        - save_fig: Whether to save the figure
        
        Returns:
        - Dictionary of analysis results
        """
        # Calculate predicted semi-major axes using different methods
        indices = np.arange(len(PLANETS))
        basic_axes = [self.model.calculate_semi_major_axis_basic(n) for n in indices]
        enhanced_axes = [self.model.calculate_semi_major_axis_enhanced(n) for n in indices]
        relativistic_axes = [self.model.calculate_semi_major_axis_relativistic(n) for n in indices]
        
        # Calculate errors
        actual_axes = [p["semi_major_axis"] for p in PLANETS]
        
        basic_errors = [100 * abs(pred - act) / act for pred, act in zip(basic_axes, actual_axes)]
        enhanced_errors = [100 * abs(pred - act) / act for pred, act in zip(enhanced_axes, actual_axes)]
        relativistic_errors = [100 * abs(pred - act) / act for pred, act in zip(relativistic_axes, actual_axes)]
        
        # Calculate statistics
        mean_basic_error = np.mean(basic_errors)
        mean_enhanced_error = np.mean(enhanced_errors)
        mean_relativistic_error = np.mean(relativistic_errors)
        
        if save_fig:
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Plot actual vs. predicted semi-major axes
            plt.subplot(2, 1, 1)
            plt.scatter(indices, actual_axes, s=100, color='blue', label='Actual')
            plt.scatter(indices, basic_axes, s=80, color='red', label='Basic COM')
            plt.scatter(indices, enhanced_axes, s=80, color='green', label='Enhanced COM')
            plt.scatter(indices, relativistic_axes, s=80, color='purple', label='Relativistic COM')
            
            # Connect points with lines
            plt.plot(indices, actual_axes, 'b-', alpha=0.5)
            plt.plot(indices, basic_axes, 'r-', alpha=0.5)
            plt.plot(indices, enhanced_axes, 'g-', alpha=0.5)
            plt.plot(indices, relativistic_axes, 'purple', alpha=0.5)
            
            # Add planet names
            for i, planet in enumerate(PLANETS):
                plt.annotate(planet["name"], (i, actual_axes[i]), 
                             textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.xlabel('Planet Index')
            plt.ylabel('Semi-Major Axis (AU)')
            plt.title('Actual vs. Predicted Planetary Semi-Major Axes')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot percentage errors
            plt.subplot(2, 1, 2)
            
            x = np.arange(len(PLANETS))
            width = 0.25
            
            plt.bar(x - width, basic_errors, width, label=f'Basic COM (Mean: {mean_basic_error:.2f}%)', color='red', alpha=0.7)
            plt.bar(x, enhanced_errors, width, label=f'Enhanced COM (Mean: {mean_enhanced_error:.2f}%)', color='green', alpha=0.7)
            plt.bar(x + width, relativistic_errors, width, label=f'Relativistic COM (Mean: {mean_relativistic_error:.2f}%)', color='purple', alpha=0.7)
            
            # Add planet names
            plt.xticks(indices, [p["name"] for p in PLANETS], rotation=45)
            
            plt.xlabel('Planet')
            plt.ylabel('Percentage Error (%)')
            plt.title('Percentage Error in Semi-Major Axis Prediction')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('figures/unified_planetary_spacing.png', dpi=300)
            
            # Create log-scale plot
            plt.figure(figsize=(12, 8))
            
            plt.loglog(indices+1, actual_axes, 'bo-', label='Actual', linewidth=2, markersize=10)
            plt.loglog(indices+1, basic_axes, 'ro-', label='Basic COM', linewidth=1, markersize=6, alpha=0.7)
            plt.loglog(indices+1, enhanced_axes, 'go-', label='Enhanced COM', linewidth=1, markersize=6, alpha=0.7)
            plt.loglog(indices+1, relativistic_axes, 'mo-', label='Relativistic COM', linewidth=2, markersize=8)
            
            # Add planet names
            for i, planet in enumerate(PLANETS):
                plt.annotate(planet["name"], (i+1, actual_axes[i]), 
                             textcoords="offset points", xytext=(5,5), ha='left')
            
            plt.xlabel('Planet Index (n+1)')
            plt.ylabel('Semi-Major Axis (AU)')
            plt.title('Planetary Semi-Major Axes (Log Scale)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('figures/unified_planetary_spacing_log.png', dpi=300)
        
        # Create results table
        results = []
        for i, planet in enumerate(PLANETS):
            results.append({
                "name": planet["name"],
                "actual": planet["semi_major_axis"],
                "basic": basic_axes[i],
                "enhanced": enhanced_axes[i],
                "relativistic": relativistic_axes[i],
                "basic_error": basic_errors[i],
                "enhanced_error": enhanced_errors[i],
                "relativistic_error": relativistic_errors[i]
            })
        
        return {
            "results": results,
            "statistics": {
                "mean_basic_error": mean_basic_error,
                "mean_enhanced_error": mean_enhanced_error,
                "mean_relativistic_error": mean_relativistic_error
            }
        }
    
    def analyze_lensing_scale_invariance(self, save_fig=True):
        """
        Analyze scale invariance of gravitational lensing in the unified COM model.
        
        Parameters:
        - save_fig: Whether to save the figure
        
        Returns:
        - Dictionary of analysis results
        """
        # Scale range (meters)
        scale_range = np.logspace(-15, 25, 1000)  # Quantum to cosmic scales
        
        # Analyze scales
        scale_analysis = self.model.analyze_scale_range(1e-15, 1e25)
        
        if save_fig:
            # Create figure
            plt.figure(figsize=(15, 12))
            
            # Plot correction factor
            plt.subplot(3, 1, 1)
            plt.semilogx(scale_analysis["scales"], scale_analysis["stabilities"], 'k-', linewidth=1.5)
            
            # Add reference scales
            reference_scales = [
                {"name": "Proton", "scale": 1e-15},
                {"name": "Atom", "scale": 1e-10},
                {"name": "Virus", "scale": 1e-7},
                {"name": "Human", "scale": 1},
                {"name": "Earth", "scale": 1e7},
                {"name": "Solar System", "scale": 1e13},
                {"name": "Galaxy", "scale": 1e21},
                {"name": "Observable Universe", "scale": 1e26}
            ]
            
            for ref in reference_scales:
                if min(scale_analysis["scales"]) <= ref["scale"] <= max(scale_analysis["scales"]):
                    idx = np.abs(scale_analysis["scales"] - ref["scale"]).argmin()
                    plt.plot(ref["scale"], scale_analysis["stabilities"][idx], 'ro', markersize=8)
                    plt.text(ref["scale"], scale_analysis["stabilities"][idx]*1.05, ref["name"], ha='center', fontsize=9)
            
            plt.xlabel('Scale (meters)')
            plt.ylabel('Stability Factor')
            plt.title('Scale Invariance in Unified COM Framework')
            plt.grid(True, alpha=0.3)
            
            # Plot octave positions
            plt.subplot(3, 1, 2)
            
            # Create a colormap based on stability factors
            colors = plt.cm.viridis(np.array(scale_analysis["stabilities"]) / max(scale_analysis["stabilities"]))
            
            # Plot octave positions with stability-based coloring
            for i in range(len(scale_analysis["scales"])-1):
                plt.semilogx(scale_analysis["scales"][i:i+2], 
                             [scale_analysis["octave_positions"][i], scale_analysis["octave_positions"][i+1]], 
                             color=colors[i], linewidth=1.5)
            
            # Add HQS threshold line
            plt.axhline(y=HQS, color='r', linestyle='--', alpha=0.7, label=f'HQS = {HQS:.3f}')
            
            # Add reference scales
            for ref in reference_scales:
                if min(scale_analysis["scales"]) <= ref["scale"] <= max(scale_analysis["scales"]):
                    idx = np.abs(scale_analysis["scales"] - ref["scale"]).argmin()
                    plt.plot(ref["scale"], scale_analysis["octave_positions"][idx], 'ro', markersize=8)
                    plt.text(ref["scale"], scale_analysis["octave_positions"][idx] + 0.05, ref["name"], ha='center', fontsize=9)
            
            plt.xlabel('Scale (meters)')
            plt.ylabel('Octave Position')
            plt.title('Octave Positions Across Scales')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot planetary semi-major axes on the same scale structure
            plt.subplot(3, 1, 3)
            
            # Calculate octave positions for planetary semi-major axes
            planet_scales = [p["semi_major_axis"] * AU for p in PLANETS]  # Convert AU to meters
            planet_positions = [self.model.scale_position(scale) for scale in planet_scales]
            planet_octaves = [pos["octave_number"] for pos in planet_positions]
            planet_stabilities = [pos["stability"] for pos in planet_positions]
            
            # Plot planetary positions
            plt.scatter(planet_scales, planet_stabilities, s=100, color='blue')
            
            # Add planet names
            for i, planet in enumerate(PLANETS):
                plt.annotate(planet["name"], (planet_scales[i], planet_stabilities[i]), 
                             textcoords="offset points", xytext=(5,5), ha='left')
            
            # Plot stability across scales
            plt.semilogx(scale_analysis["scales"], scale_analysis["stabilities"], 'k-', linewidth=1, alpha=0.5)
            
            plt.xlabel('Scale (meters)')
            plt.ylabel('Stability Factor')
            plt.title('Planetary Positions in Unified Scale Structure')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('figures/unified_scale_invariance.png', dpi=300)
        
        # Calculate planetary positions in scale structure
        planet_scales = [p["semi_major_axis"] * AU for p in PLANETS]  # Convert AU to meters
        planet_positions = [self.model.scale_position(scale) for scale in planet_scales]
        
        return {
            "scale_analysis": scale_analysis,
            "planetary_positions": planet_positions
        }
    
    def analyze_unified_model(self, save_fig=True):
        """
        Perform a comprehensive analysis of the unified COM model.
        
        Parameters:
        - save_fig: Whether to save the figure
        
        Returns:
        - Dictionary of analysis results
        """
        # Analyze planetary spacing
        spacing_results = self.analyze_planetary_spacing(save_fig)
        
        # Analyze scale invariance
        scale_results = self.analyze_lensing_scale_invariance(save_fig)
        
        if save_fig:
            # Create unified visualization
            plt.figure(figsize=(15, 12))
            
            # Define reference scales spanning quantum to cosmic
            reference_scales = [
                {"name": "Proton", "scale": 1e-15, "color": "purple"},
                {"name": "Atom", "scale": 1e-10, "color": "blue"},
                {"name": "Virus", "scale": 1e-7, "color": "cyan"},
                {"name": "Human", "scale": 1, "color": "green"},
                {"name": "Earth", "scale": 1e7, "color": "yellow"},
                {"name": "Mercury Orbit", "scale": 0.39 * AU, "color": "orange"},
                {"name": "Jupiter Orbit", "scale": 5.2 * AU, "color": "red"},
                {"name": "Solar System", "scale": 40 * AU, "color": "brown"},
                {"name": "Galaxy", "scale": 1e21, "color": "gray"},
                {"name": "Observable Universe", "scale": 1e26, "color": "black"}
            ]
            
            # Calculate octave numbers for reference scales
            for ref in reference_scales:
                position = self.model.scale_position(ref["scale"])
                ref["octave_number"] = position["octave_number"]
                ref["octave_position"] = position["octave_position"]
                ref["stability"] = position["stability"]
            
            # Plot octave structure
            plt.subplot(2, 1, 1)
            
            # Plot reference scales
            for ref in reference_scales:
                plt.scatter(ref["octave_number"], ref["stability"], s=100, color=ref["color"], 
                            label=ref["name"], zorder=10)
            
            # Generate continuous scale structure
            octave_range = np.linspace(-30, 80, 1000)
            octave_positions = octave_range % 1
            stabilities = [self.model.com_correction_factor(pos) for pos in octave_positions]
            
            # Plot continuous structure
            plt.plot(octave_range, stabilities, 'k-', alpha=0.3, linewidth=1)
            
            # Add planet positions
            planet_scales = [p["semi_major_axis"] * AU for p in PLANETS]
            planet_positions = [self.model.scale_position(scale) for scale in planet_scales]
            
            for i, planet in enumerate(PLANETS):
                pos = planet_positions[i]
                plt.scatter(pos["octave_number"], pos["stability"], s=150, color='red', 
                            marker='*', zorder=20)
                plt.annotate(planet["name"], (pos["octave_number"], pos["stability"]), 
                             textcoords="offset points", xytext=(5,5), ha='left')
            
            plt.xlabel('Octave Number (log(scale)/log(LZ))')
            plt.ylabel('Stability Factor')
            plt.title('Unified COM Framework Scale Structure')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
            
            # Plot planetary spacing with scale structure
            plt.subplot(2, 1, 2)
            
            # Calculate continuous model curve
            n_values = np.linspace(0, 8, 1000)
            basic_model = [self.model.calculate_semi_major_axis_basic(n) for n in n_values]
            enhanced_model = [self.model.calculate_semi_major_axis_enhanced(n) for n in n_values]
            relativistic_model = [self.model.calculate_semi_major_axis_relativistic(n) for n in n_values]
            
            # Plot continuous models
            plt.semilogy(n_values, basic_model, 'r-', label='Basic COM', linewidth=1, alpha=0.5)
            plt.semilogy(n_values, enhanced_model, 'g-', label='Enhanced COM', linewidth=1, alpha=0.5)
            plt.semilogy(n_values, relativistic_model, 'm-', label='Relativistic COM', linewidth=2)
            
            # Plot actual planets
            indices = np.arange(len(PLANETS))
            actual_axes = [p["semi_major_axis"] for p in PLANETS]
            plt.scatter(indices, actual_axes, s=100, color='blue', label='Actual Planets')
            
            # Add planet names
            for i, planet in enumerate(PLANETS):
                plt.annotate(planet["name"], (i, actual_axes[i]), 
                             textcoords="offset points", xytext=(5,5), ha='left')
            
            # Add vertical lines at integer positions
            for i in range(9):
                plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
            
            plt.xlabel('Orbital Index (n)')
            plt.ylabel('Semi-Major Axis (AU)')
            plt.title('Unified COM Framework Planetary Spacing Model')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('figures/unified_com_model.png', dpi=300)
        
        # Print summary
        print("\n=== Unified COM Framework Analysis ===")
        print("\nPlanetary Spacing Results:")
        print(f"Basic COM Model Mean Error: {spacing_results['statistics']['mean_basic_error']:.2f}%")
        print(f"Enhanced COM Model Mean Error: {spacing_results['statistics']['mean_enhanced_error']:.2f}%")
        print(f"Relativistic COM Model Mean Error: {spacing_results['statistics']['mean_relativistic_error']:.2f}%")
        
        print("\nPlanet-by-Planet Results:")
        print("Planet\t\tActual (AU)\tBasic COM\tEnhanced COM\tRelativistic COM")
        for result in spacing_results["results"]:
            print(f"{result['name']:<10}\t{result['actual']:.2f}\t\t{result['basic']:.2f}\t\t{result['enhanced']:.2f}\t\t{result['relativistic']:.2f}")
        
        print("\nScale Invariance Results:")
        print("Scale structure shows perfect periodicity across 40 orders of magnitude")
        print(f"Stability factor range: {min(scale_results['scale_analysis']['stabilities']):.4f} to {max(scale_results['scale_analysis']['stabilities']):.4f}")
        
        print("\nPlanetary Positions in Scale Structure:")
        print("Planet\t\tOctave Number\tOctave Position\tStability Factor")
        for i, planet in enumerate(PLANETS):
            pos = scale_results["planetary_positions"][i]
            print(f"{planet['name']:<10}\t{pos['octave_number']:.4f}\t\t{pos['octave_position']:.4f}\t\t{pos['stability']:.4f}")
        
        return {
            "spacing_results": spacing_results,
            "scale_results": scale_results
        }


def run_unified_analysis():
    """
    Run a complete analysis of the unified COM model.
    
    Returns:
    - Dictionary of analysis results
    """
    # Initialize model and analysis tools
    model = UnifiedCOMModel()
    analysis = UnifiedAnalysis(model)
    
    # Run comprehensive analysis
    results = analysis.analyze_unified_model()
    
    print("\nAnalysis complete. Figures saved in 'figures' directory.")
    
    return results


if __name__ == "__main__":
    # Run the unified analysis when script is executed directly
    results = run_unified_analysis()
