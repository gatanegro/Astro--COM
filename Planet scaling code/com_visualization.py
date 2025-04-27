"""
COM Framework Visualization Module

This module implements visualization tools for the Continuous Oscillatory Model (COM) framework,
generating plots for planetary spacing, gravitational lensing, and scale invariance analysis.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

from com_framework_constants import PLANETS, REFERENCE_SCALES, LZ, HQS, AU, G, c, M_SUN
from com_framework_core import COMModel
from com_planetary_spacing import PlanetarySpacingModel
from com_gravitational_lensing import GravitationalLensingModel

# Create directories for output if needed
os.makedirs('figures', exist_ok=True)

class COMVisualization:
    """
    Visualization tools for the COM framework.
    """
    
    def __init__(self, planetary_model=None, lensing_model=None):
        """
        Initialize the visualization tools.
        
        Parameters:
        - planetary_model: PlanetarySpacingModel instance (creates new one if None)
        - lensing_model: GravitationalLensingModel instance (creates new one if None)
        """
        self.planetary_model = planetary_model if planetary_model is not None else PlanetarySpacingModel()
        self.lensing_model = lensing_model if lensing_model is not None else GravitationalLensingModel()
        self.core_model = COMModel()
    
    def plot_planetary_spacing(self, save_path='figures/planetary_spacing.png'):
        """
        Create visualization of planetary spacing predictions.
        
        Parameters:
        - save_path: Path to save the figure
        
        Returns:
        - Figure object
        """
        # Calculate predicted semi-major axes using different methods
        indices = np.arange(len(PLANETS))
        basic_axes = [self.planetary_model.calculate_semi_major_axis_basic(n) for n in indices]
        enhanced_axes = [self.planetary_model.calculate_semi_major_axis_enhanced(n) for n in indices]
        relativistic_axes = [self.planetary_model.calculate_semi_major_axis_relativistic(n) for n in indices]
        
        # Calculate errors
        actual_axes = [p["semi_major_axis"] for p in PLANETS]
        
        basic_errors = [100 * abs(pred - act) / act for pred, act in zip(basic_axes, actual_axes)]
        enhanced_errors = [100 * abs(pred - act) / act for pred, act in zip(enhanced_axes, actual_axes)]
        relativistic_errors = [100 * abs(pred - act) / act for pred, act in zip(relativistic_axes, actual_axes)]
        
        # Calculate statistics
        mean_basic_error = np.mean(basic_errors)
        mean_enhanced_error = np.mean(enhanced_errors)
        mean_relativistic_error = np.mean(relativistic_errors)
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
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
        plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_planetary_spacing_log(self, save_path='figures/planetary_spacing_log.png'):
        """
        Create log-scale visualization of planetary spacing predictions.
        
        Parameters:
        - save_path: Path to save the figure
        
        Returns:
        - Figure object
        """
        # Calculate predicted semi-major axes using different methods
        indices = np.arange(len(PLANETS))
        basic_axes = [self.planetary_model.calculate_semi_major_axis_basic(n) for n in indices]
        enhanced_axes = [self.planetary_model.calculate_semi_major_axis_enhanced(n) for n in indices]
        relativistic_axes = [self.planetary_model.calculate_semi_major_axis_relativistic(n) for n in indices]
        
        # Get actual axes
        actual_axes = [p["semi_major_axis"] for p in PLANETS]
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
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
        plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_planetary_spacing_continuous(self, save_path='figures/planetary_spacing_continuous.png'):
        """
        Create visualization of continuous planetary spacing model.
        
        Parameters:
        - save_path: Path to save the figure
        
        Returns:
        - Figure object
        """
        # Generate continuous model curve
        n_values = np.linspace(0, 8, 1000)
        basic_model = [self.planetary_model.calculate_semi_major_axis_basic(n) for n in n_values]
        enhanced_model = [self.planetary_model.calculate_semi_major_axis_enhanced(n) for n in n_values]
        relativistic_model = [self.planetary_model.calculate_semi_major_axis_relativistic(n) for n in n_values]
        
        # Get actual axes
        indices = np.arange(len(PLANETS))
        actual_axes = [p["semi_major_axis"] for p in PLANETS]
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Plot continuous models
        plt.semilogy(n_values, basic_model, 'r-', label='Basic COM', linewidth=1, alpha=0.5)
        plt.semilogy(n_values, enhanced_model, 'g-', label='Enhanced COM', linewidth=1, alpha=0.5)
        plt.semilogy(n_values, relativistic_model, 'm-', label='Relativistic COM', linewidth=2)
        
        # Plot actual planets
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
        plt.title('COM Framework Planetary Spacing Model')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_asteroid_belt_analysis(self, save_path='figures/asteroid_belt_analysis.png'):
        """
        Create visualization of asteroid belt analysis.
        
        Parameters:
        - save_path: Path to save the figure
        
        Returns:
        - Figure object
        """
        # Generate continuous model curve
        n_values = np.linspace(0, 8, 1000)
        continuous_model = [self.planetary_model.calculate_semi_major_axis_basic(n) for n in n_values]
        
        # Get asteroid belt analysis
        asteroid_belt_results = self.planetary_model.analyze_asteroid_belt()
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Plot continuous model
        plt.semilogy(n_values, continuous_model, 'r-', label='COM Model', linewidth=2)
        
        # Plot actual planets
        indices = np.arange(len(PLANETS))
        actual_axes = [p["semi_major_axis"] for p in PLANETS]
        plt.scatter(indices, actual_axes, s=100, color='blue', label='Actual Planets')
        
        # Highlight asteroid belt region
        plt.axvspan(3.5, 4.5, color='gray', alpha=0.2, label='Asteroid Belt Region')
        
        # Mark extrema in asteroid belt region
        extrema_n = asteroid_belt_results["extrema"]["n_values"]
        extrema_values = asteroid_belt_results["extrema"]["semi_major_axes"]
        plt.scatter(extrema_n, extrema_values, s=80, color='green', marker='x', label='Stability Extrema')
        
        # Add planet names
        for i, planet in enumerate(PLANETS):
            plt.annotate(planet["name"], (i, actual_axes[i]), 
                         textcoords="offset points", xytext=(5,5), ha='left')
        
        # Add vertical lines at integer positions
        for i in range(9):
            plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
        
        plt.xlabel('Orbital Index (n)')
        plt.ylabel('Semi-Major Axis (AU)')
        plt.title('Asteroid Belt Analysis in COM Framework Model')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_scale_invariance(self, save_path='figures/scale_invariance.png'):
        """
        Create visualization of scale invariance in the COM framework.
        
        Parameters:
        - save_path: Path to save the figure
        
        Returns:
        - Figure object
        """
        # Analyze scales
        scale_analysis = self.core_model.analyze_scale_range(1e-15, 1e25)
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # Plot correction factor
        plt.subplot(3, 1, 1)
        plt.semilogx(scale_analysis["scales"], scale_analysis["stabilities"], 'k-', linewidth=1.5)
        
        # Add reference scales
        for ref in REFERENCE_SCALES:
            if min(scale_analysis["scales"]) <= ref["scale"] <= max(scale_analysis["scales"]):
                idx = np.abs(scale_analysis["scales"] - ref["scale"]).argmin()
                plt.plot(ref["scale"], scale_analysis["stabilities"][idx], 'ro', markersize=8)
                plt.text(ref["scale"], scale_analysis["stabilities"][idx]*1.05, ref["name"], ha='center', fontsize=9)
        
        plt.xlabel('Scale (meters)')
        plt.ylabel('Stability Factor')
        plt.title('Scale Invariance in COM Framework')
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
        for ref in REFERENCE_SCALES:
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
        planet_positions = [self.core_model.scale_position(scale) for scale in planet_scales]
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
        plt.title('Planetary Positions in Scale Structure')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_unified_model(self, save_path='figures/unified_model.png'):
        """
        Create unified visualization of the COM framework.
        
        Parameters:
        - save_path: Path to save the figure
        
        Returns:
        - Figure object
        """
        # Analyze scales
        scale_analysis = self.core_model.analyze_scale_range(1e-15, 1e25)
        
        # Calculate octave numbers for reference scales
        reference_scales_with_positions = []
        for ref in REFERENCE_SCALES:
            position = self.core_model.scale_position(ref["scale"])
            ref_copy = ref.copy()
            ref_copy["octave_number"] = position["octave_number"]
            ref_copy["octave_position"] = position["octave_position"]
            ref_copy["stability"] = position["stability"]
            reference_scales_with_positions.append(ref_copy)
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # Plot octave structure
        plt.subplot(2, 1, 1)
        
        # Plot reference scales
        for ref in reference_scales_with_positions:
            plt.scatter(ref["octave_number"], ref["stability"], s=100, color=ref["color"], 
                        label=ref["name"], zorder=10)
        
        # Generate continuous scale structure
        octave_range = np.linspace(-30, 80, 1000)
        octave_positions = octave_range % 1
        stabilities = [self.core_model.com_correction_factor(pos) for pos in octave_positions]
        
        # Plot continuous structure
        plt.plot(octave_range, stabilities, 'k-', alpha=0.3, linewidth=1)
        
        # Add planet positions
        planet_scales = [p["semi_major_axis"] * AU for p in PLANETS]
        planet_positions = [self.core_model.scale_position(scale) for scale in planet_scales]
        
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
        basic_model = [self.planetary_model.calculate_semi_major_axis_basic(n) for n in n_values]
        enhanced_model = [self.planetary_model.calculate_semi_major_axis_enhanced(n) for n in n_values]
        relativistic_model = [self.planetary_model.calculate_semi_major_axis_relativistic(n) for n in n_values]
        
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
        plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_lensing_mass_scaling(self, save_path='figures/lensing_mass_scaling.png'):
        """
        Create visualization of lensing predictions scaling with mass.
        
        Parameters:
        - save_path: Path to save the figure
        
        Returns:
        - Figure object
        """
        # Mass range (solar masses)
        mass_range = np.logspace(0, 15, 1000)  # 1 to 10^15 solar masses
        
        # Fixed parameters
        distance_observer = 1.0e22  # 10 kpc (typical galactic distance)
        distance_source = 1.0e23    # 100 kpc (typical intergalactic distance)
        
        # Calculate Einstein radii
        gr_radii = np.array([self.lensing_model.einstein_ring_radius_gr(m * M_SUN, distance_observer, distance_source) 
                             for m in mass_range])
        com_radii = np.array([self.lensing_model.einstein_ring_radius_com(m * M_SUN, distance_observer, distance_source) 
                              for m in mass_range])
        
        # Calculate ratio
        ratio = com_radii / gr_radii
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Plot Einstein radii
        plt.subplot(2, 1, 1)
        plt.loglog(mass_range, gr_radii, 'b-', label='General Relativity', linewidth=2)
        plt.loglog(mass_range, com_radii, 'r-', label='COM Framework', linewidth=2)
        
        # Add reference points for known objects
        lensing_objects = [
            {"name": "Sun", "mass": 1.989e30, "radius": 6.957e8, "distance": 1.496e11},  # 1 AU
            {"name": "Jupiter", "mass": 1.898e27, "radius": 6.9911e7, "distance": 7.785e11},  # 5.2 AU
            {"name": "Sagittarius A*", "mass": 4.154e6 * 1.989e30, "radius": 1.2e10, "distance": 2.55e20},  # 8.178 kpc
            {"name": "M87 Black Hole", "mass": 6.5e9 * 1.989e30, "radius": 2.0e13, "distance": 5.23e23},  # 16.8 Mpc
            {"name": "Typical Galaxy Cluster", "mass": 1e15 * 1.989e30, "radius": 3.086e22, "distance": 3.086e24}  # 1 Gpc
        ]
        
        for obj in lensing_objects:
            mass_solar = obj["mass"] / M_SUN
            if 1 <= mass_solar <= 1e15:
                gr_radius = self.lensing_model.einstein_ring_radius_gr(obj["mass"], distance_observer, distance_source)
                com_radius = self.lensing_model.einstein_ring_radius_com(obj["mass"], distance_observer, distance_source)
                plt.plot(mass_solar, gr_radius, 'bo', markersize=8)
                plt.plot(mass_solar, com_radius, 'ro', markersize=8)
                plt.text(mass_solar, gr_radius*1.2, obj["name"], ha='center', fontsize=9)
        
        plt.xlabel('Mass (Solar Masses)')
        plt.ylabel('Einstein Ring Radius (m)')
        plt.title('Einstein Ring Radius vs. Mass')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot ratio
        plt.subplot(2, 1, 2)
        plt.semilogx(mass_range, ratio, 'k-', linewidth=2)
        
        # Add horizontal line at ratio=1
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        
        # Add vertical lines at pattern transitions
        for i in range(1, 25):
            transition_mass = 10**(i/8 * 15 / 24)
            plt.axvline(x=transition_mass, color='g', linestyle='-', alpha=0.2)
        
        plt.xlabel('Mass (Solar Masses)')
        plt.ylabel('COM/GR Ratio')
        plt.title('Ratio of COM to GR Einstein Ring Radius')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        
        return fig
    
    def plot_lensing_observation_comparison(self, save_path='figures/lensing_observation_comparison.png'):
        """
        Create visualization comparing lensing predictions with observations.
        
        Parameters:
        - save_path: Path to save the figure
        
        Returns:
        - Figure object
        """
        # Observational data (approximate values from literature)
        observations = [
            {"name": "Solar Deflection (Eddington)", "mass": M_SUN, "impact": 6.957e8, 
             "observed": 1.75, "uncertainty": 0.2},  # arcseconds
            {"name": "Solar Deflection (Modern)", "mass": M_SUN, "impact": 6.957e8, 
             "observed": 1.66, "uncertainty": 0.18},  # arcseconds
            {"name": "Quasar Twin QSO 0957+561", "mass": 1.6e14 * M_SUN, "impact": 1e20, 
             "observed": 1.5, "uncertainty": 0.3},  # arcseconds
            {"name": "Abell 1689 Cluster", "mass": 1.3e15 * M_SUN, "impact": 3.086e22, 
             "observed": 35.0, "uncertainty": 5.0},  # arcseconds
            {"name": "SDSS J1004+4112", "mass": 1e14 * M_SUN, "impact": 2e22, 
             "observed": 14.7, "uncertainty": 0.8}  # arcseconds
        ]
        
        # Calculate predictions
        for obs in observations:
            # Calculate GR prediction
            gr_deflection = self.lensing_model.deflection_angle_gr(obs["mass"], obs["impact"])
            obs["gr_prediction"] = gr_deflection * 206265  # convert to arcseconds
            
            # Calculate COM prediction
            com_deflection = self.lensing_model.deflection_angle_com(obs["mass"], obs["impact"])
            obs["com_prediction"] = com_deflection * 206265  # convert to arcseconds
            
            # Calculate deviations
            obs["gr_deviation"] = (obs["gr_prediction"] - obs["observed"]) / obs["uncertainty"]
            obs["com_deviation"] = (obs["com_prediction"] - obs["observed"]) / obs["uncertainty"]
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Plot predictions vs observations
        plt.subplot(2, 1, 1)
        
        x = np.arange(len(observations))
        width = 0.25
        
        plt.bar(x - width, [o["observed"] for o in observations], width, label='Observed', color='green')
        plt.bar(x, [o["gr_prediction"] for o in observations], width, label='GR Prediction', color='blue')
        plt.bar(x + width, [o["com_prediction"] for o in observations], width, label='COM Prediction', color='red')
        
        # Add error bars for observations
        plt.errorbar(x - width, [o["observed"] for o in observations], 
                     yerr=[o["uncertainty"] for o in observations], fmt='o', color='black')
        
        plt.xlabel('Lensing System')
        plt.ylabel('Deflection Angle (arcsec)')
        plt.title('Observed vs. Predicted Deflection Angles')
        plt.xticks(x, [o["name"] for o in observations], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot deviations
        plt.subplot(2, 1, 2)
        
        plt.bar(x - width/2, [o["gr_deviation"] for o in observations], width, label='GR Deviation', color='blue')
        plt.bar(x + width/2, [o["com_deviation"] for o in observations], width, label='COM Deviation', color='red')
        
        # Add horizontal lines at deviation = ±1
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.7)
        
        plt.xlabel('Lensing System')
        plt.ylabel('Deviation (σ)')
        plt.title('Deviation from Observations (in units of observational uncertainty)')
        plt.xticks(x, [o["name"] for o in observations], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        
        return fig
    
    def generate_all_plots(self):
        """
        Generate all visualizations for the COM framework.
        
        Returns:
        - Dictionary of figure objects
        """
        figures = {}
        
        # Planetary spacing plots
        figures["planetary_spacing"] = self.plot_planetary_spacing()
        figures["planetary_spacing_log"] = self.plot_planetary_spacing_log()
        figures["planetary_spacing_continuous"] = self.plot_planetary_spacing_continuous()
        figures["asteroid_belt_analysis"] = self.plot_asteroid_belt_analysis()
        
        # Scale invariance plots
        figures["scale_invariance"] = self.plot_scale_invariance()
        
        # Unified model plot
        figures["unified_model"] = self.plot_unified_model()
        
        # Lensing plots
        figures["lensing_mass_scaling"] = self.plot_lensing_mass_scaling()
        figures["lensing_observation_comparison"] = self.plot_lensing_observation_comparison()
        
        print("All plots generated and saved to 'figures' directory.")
        
        return figures


if __name__ == "__main__":
    # Create visualization tools
    visualizer = COMVisualization()
    
    # Generate all plots
    figures = visualizer.generate_all_plots()
