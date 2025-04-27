"""
COM Framework Gravitational Lensing Model

This module implements the Continuous Oscillatory Model (COM) framework for gravitational lensing,
reinterpreting lensing as an energy density gradient effect rather than spacetime curvature.

The model incorporates the COM framework's fundamental constants (LZ=1.23498 and HQS=0.235)
and the 24-step Fibonacci pattern to predict lensing effects across different scales.

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
LZ = 1.23498  # Fundamental scaling constant
HQS = 0.235   # Harmonic Quantum Scalar (23.5% of LZ)
HQS_LZ = HQS * LZ  # HQS threshold in absolute terms (≈ 0.29022)

# 24-step Fibonacci digital root pattern
FIBONACCI_PATTERN = [1, 1, 2, 3, 5, 8, 4, 3, 7, 1, 8, 9, 8, 8, 7, 6, 4, 1, 5, 6, 2, 8, 1, 9]
FIBONACCI_NORMALIZED = np.array(FIBONACCI_PATTERN) / 9.0  # Normalize to 0-1 range

# Physical constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
c = 2.99792458e8  # Speed of light in m/s
M_SUN = 1.989e30  # Solar mass in kg
AU = 1.496e11    # Astronomical unit in meters


class COMLensingModel:
    """
    Implementation of the COM framework for gravitational lensing calculations.
    """
    
    def __init__(self):
        """Initialize the COM lensing model with default parameters."""
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


class LensingAnalysis:
    """
    Analysis tools for comparing COM and GR lensing predictions with observations.
    """
    
    def __init__(self, model=None):
        """
        Initialize the analysis tools.
        
        Parameters:
        - model: COMLensingModel instance (creates new one if None)
        """
        self.model = model if model is not None else COMLensingModel()
        
        # Astronomical objects for lensing validation
        self.lensing_objects = [
            {"name": "Sun", "mass": 1.989e30, "radius": 6.957e8, "distance": 1.496e11},  # 1 AU
            {"name": "Jupiter", "mass": 1.898e27, "radius": 6.9911e7, "distance": 7.785e11},  # 5.2 AU
            {"name": "Sagittarius A*", "mass": 4.154e6 * 1.989e30, "radius": 1.2e10, "distance": 2.55e20},  # 8.178 kpc
            {"name": "M87 Black Hole", "mass": 6.5e9 * 1.989e30, "radius": 2.0e13, "distance": 5.23e23},  # 16.8 Mpc
            {"name": "Typical Galaxy Cluster", "mass": 1e15 * 1.989e30, "radius": 3.086e22, "distance": 3.086e24}  # 1 Gpc
        ]
        
        # Observational data (approximate values from literature)
        self.observations = [
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
    
    def analyze_mass_scaling(self, mass_range=None, save_fig=True):
        """
        Analyze how lensing predictions scale with mass.
        
        Parameters:
        - mass_range: Array of masses in solar masses (creates default if None)
        - save_fig: Whether to save the figure
        
        Returns:
        - Dictionary of results
        """
        # Mass range (solar masses)
        if mass_range is None:
            mass_range = np.logspace(0, 15, 1000)  # 1 to 10^15 solar masses
        
        # Fixed parameters
        distance_observer = 1.0e22  # 10 kpc (typical galactic distance)
        distance_source = 1.0e23    # 100 kpc (typical intergalactic distance)
        
        # Calculate Einstein radii
        gr_radii = np.array([self.model.einstein_ring_radius_gr(m * M_SUN, distance_observer, distance_source) 
                             for m in mass_range])
        com_radii = np.array([self.model.einstein_ring_radius_com(m * M_SUN, distance_observer, distance_source) 
                              for m in mass_range])
        
        # Calculate ratio
        ratio = com_radii / gr_radii
        
        if save_fig:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot Einstein radii
            plt.subplot(2, 1, 1)
            plt.loglog(mass_range, gr_radii, 'b-', label='General Relativity', linewidth=2)
            plt.loglog(mass_range, com_radii, 'r-', label='COM Framework', linewidth=2)
            
            # Add reference points for known objects
            for obj in self.lensing_objects:
                mass_solar = obj["mass"] / M_SUN
                if 1 <= mass_solar <= 1e15:
                    gr_radius = self.model.einstein_ring_radius_gr(obj["mass"], distance_observer, distance_source)
                    com_radius = self.model.einstein_ring_radius_com(obj["mass"], distance_observer, distance_source)
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
            plt.savefig('figures/lensing_mass_scaling.png', dpi=300)
            plt.close()
        
        return {
            "mass_range": mass_range,
            "gr_radii": gr_radii,
            "com_radii": com_radii,
            "ratio": ratio
        }
    
    def analyze_deflection_profiles(self, save_fig=True):
        """
        Analyze deflection angles vs. impact parameter for different objects.
        
        Parameters:
        - save_fig: Whether to save the figure
        
        Returns:
        - Dictionary of results
        """
        # Impact parameter range (as fraction of object radius)
        impact_range = np.logspace(0, 3, 1000)  # 1 to 1000 times the object radius
        
        # Results for different objects
        results = {}
        
        if save_fig:
            # Create figure
            plt.figure(figsize=(15, 10))
        
        # For each lensing object
        for i, obj in enumerate(self.lensing_objects):
            # Calculate deflection angles for different impact parameters
            impact_parameters = impact_range * obj["radius"]
            
            gr_deflections = np.array([self.model.deflection_angle_gr(obj["mass"], b) for b in impact_parameters])
            com_deflections = np.array([self.model.deflection_angle_com(obj["mass"], b) for b in impact_parameters])
            
            # Convert to arcseconds
            gr_deflections_arcsec = gr_deflections * 206265  # radians to arcseconds
            com_deflections_arcsec = com_deflections * 206265
            
            # Store results
            results[obj["name"]] = {
                "impact_parameters": impact_parameters,
                "gr_deflections": gr_deflections_arcsec,
                "com_deflections": com_deflections_arcsec
            }
            
            if save_fig:
                # Plot deflection angles
                plt.subplot(len(self.lensing_objects), 2, 2*i+1)
                plt.loglog(impact_range, gr_deflections_arcsec, 'b-', label='General Relativity', linewidth=2)
                plt.loglog(impact_range, com_deflections_arcsec, 'r-', label='COM Framework', linewidth=2)
                
                plt.xlabel('Impact Parameter (object radii)')
                plt.ylabel('Deflection Angle (arcsec)')
                plt.title(f'Light Deflection by {obj["name"]}')
                plt.grid(True, alpha=0.3)
                if i == 0:
                    plt.legend()
                
                # Plot ratio
                plt.subplot(len(self.lensing_objects), 2, 2*i+2)
                ratio = com_deflections_arcsec / gr_deflections_arcsec
                plt.semilogx(impact_range, ratio, 'k-', linewidth=2)
                
                # Add horizontal line at ratio=1
                plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
                
                # Add pattern structure
                pattern_positions = []
                for j in range(24):
                    if j % 8 == 0:  # Highlight every 8th position
                        pattern_positions.append(j/24)
                        
                for pos in pattern_positions:
                    # Find approximate impact parameter for this pattern position
                    octave_transitions = np.exp(np.log(LZ) * np.arange(10))  # Several octaves
                    for oct in octave_transitions:
                        # Calculate impact parameter that would give this octave position
                        b_approx = obj["radius"] * oct
                        if obj["radius"] <= b_approx <= 1000 * obj["radius"]:
                            plt.axvline(x=b_approx/obj["radius"], color='g', linestyle='-', alpha=0.2)
                
                plt.xlabel('Impact Parameter (object radii)')
                plt.ylabel('COM/GR Ratio')
                plt.title(f'Ratio of COM to GR Deflection for {obj["name"]}')
                plt.grid(True, alpha=0.3)
        
        if save_fig:
            plt.tight_layout()
            plt.savefig('figures/lensing_deflection_profiles.png', dpi=300)
            plt.close()
        
        return results
    
    def analyze_scale_invariance(self, save_fig=True):
        """
        Analyze scale invariance properties of the COM framework.
        
        Parameters:
        - save_fig: Whether to save the figure
        
        Returns:
        - Dictionary of results
        """
        # Scale range (meters)
        scale_range = np.logspace(-15, 25, 1000)  # Quantum to cosmic scales
        
        # Calculate COM correction factor across scales
        correction_factors = []
        octave_positions = []
        
        for scale in scale_range:
            # Calculate octave position
            octave_position = self.model.octave_position(scale, 1e-15)
            octave_positions.append(octave_position)
            
            # Calculate correction factor
            correction = self.model.com_correction_factor(octave_position)
            correction_factors.append(correction)
        
        if save_fig:
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Plot correction factor
            plt.subplot(2, 1, 1)
            plt.semilogx(scale_range, correction_factors, 'k-', linewidth=1.5)
            
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
                if min(scale_range) <= ref["scale"] <= max(scale_range):
                    idx = np.abs(scale_range - ref["scale"]).argmin()
                    plt.plot(ref["scale"], correction_factors[idx], 'ro', markersize=8)
                    plt.text(ref["scale"], correction_factors[idx]*1.05, ref["name"], ha='center', fontsize=9)
            
            plt.xlabel('Scale (meters)')
            plt.ylabel('COM Correction Factor')
            plt.title('Scale Invariance in COM Framework')
            plt.grid(True, alpha=0.3)
            
            # Plot pattern structure
            plt.subplot(2, 1, 2)
            
            # Create a colormap based on correction factors
            colors = plt.cm.viridis(np.array(correction_factors) / max(correction_factors))
            
            # Plot octave positions with correction-based coloring
            for i in range(len(scale_range)-1):
                plt.semilogx(scale_range[i:i+2], [octave_positions[i], octave_positions[i+1]], 
                             color=colors[i], linewidth=1.5)
            
            # Add HQS threshold line
            plt.axhline(y=HQS, color='r', linestyle='--', alpha=0.7, label=f'HQS = {HQS:.3f}')
            
            # Add reference scales
            for ref in reference_scales:
                if min(scale_range) <= ref["scale"] <= max(scale_range):
                    octave_pos = self.model.octave_position(ref["scale"], 1e-15)
                    plt.plot(ref["scale"], octave_pos, 'ro', markersize=8)
                    plt.text(ref["scale"], octave_pos + 0.05, ref["name"], ha='center', fontsize=9)
            
            plt.xlabel('Scale (meters)')
            plt.ylabel('Octave Position')
            plt.title('Octave Positions Across Scales')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('figures/lensing_scale_invariance.png', dpi=300)
            plt.close()
        
        return {
            "scale_range": scale_range,
            "correction_factors": correction_factors,
            "octave_positions": octave_positions
        }
    
    def compare_with_observations(self, save_fig=True):
        """
        Compare model predictions with observational data.
        
        Parameters:
        - save_fig: Whether to save the figure
        
        Returns:
        - List of observation results with predictions
        """
        # Calculate predictions
        for obs in self.observations:
            # Calculate GR prediction
            gr_deflection = self.model.deflection_angle_gr(obs["mass"], obs["impact"])
            obs["gr_prediction"] = gr_deflection * 206265  # convert to arcseconds
            
            # Calculate COM prediction
            com_deflection = self.model.deflection_angle_com(obs["mass"], obs["impact"])
            obs["com_prediction"] = com_deflection * 206265  # convert to arcseconds
            
            # Calculate deviations
            obs["gr_deviation"] = (obs["gr_prediction"] - obs["observed"]) / obs["uncertainty"]
            obs["com_deviation"] = (obs["com_prediction"] - obs["observed"]) / obs["uncertainty"]
        
        if save_fig:
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Plot predictions vs observations
            plt.subplot(2, 1, 1)
            
            x = np.arange(len(self.observations))
            width = 0.25
            
            plt.bar(x - width, [o["observed"] for o in self.observations], width, label='Observed', color='green')
            plt.bar(x, [o["gr_prediction"] for o in self.observations], width, label='GR Prediction', color='blue')
            plt.bar(x + width, [o["com_prediction"] for o in self.observations], width, label='COM Prediction', color='red')
            
            # Add error bars for observations
            plt.errorbar(x - width, [o["observed"] for o in self.observations], 
                         yerr=[o["uncertainty"] for o in self.observations], fmt='o', color='black')
            
            plt.xlabel('Lensing System')
            plt.ylabel('Deflection Angle (arcsec)')
            plt.title('Observed vs. Predicted Deflection Angles')
            plt.xticks(x, [o["name"] for o in self.observations], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot deviations
            plt.subplot(2, 1, 2)
            
            plt.bar(x - width/2, [o["gr_deviation"] for o in self.observations], width, label='GR Deviation', color='blue')
            plt.bar(x + width/2, [o["com_deviation"] for o in self.observations], width, label='COM Deviation', color='red')
            
            # Add horizontal lines at deviation = ±1
            plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
            plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.7)
            
            plt.xlabel('Lensing System')
            plt.ylabel('Deviation (σ)')
            plt.title('Deviation from Observations (in units of observational uncertainty)')
            plt.xticks(x, [o["name"] for o in self.observations], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('figures/lensing_observation_comparison.png', dpi=300)
            plt.close()
        
        return self.observations


def run_full_analysis():
    """
    Run a complete analysis of the COM lensing model and compare with observations.
    
    Returns:
    - Dictionary of analysis results
    """
    # Initialize model and analysis tools
    model = COMLensingModel()
    analysis = LensingAnalysis(model)
    
    # Run analyses
    mass_scaling_results = analysis.analyze_mass_scaling()
    deflection_results = analysis.analyze_deflection_profiles()
    scale_invariance_results = analysis.analyze_scale_invariance()
    observation_comparison = analysis.compare_with_observations()
    
    # Print summary
    print("\n=== COM Lensing Model Analysis ===")
    print("\nMass Scaling Results:")
    print(f"Einstein radius ratio range: {min(mass_scaling_results['ratio']):.4f} to {max(mass_scaling_results['ratio']):.4f}")

    print("\nDeflection Profile Results:")
    for obj_name, results in deflection_results.items():
        ratio = results["com_deflections"] / results["gr_deflections"]
        print(f"{obj_name}: Deflection ratio range: {min(ratio):.4f} to {max(ratio):.4f}")

    print("\nScale Invariance Results:")
    print(f"Correction factor range: {min(scale_invariance_results['correction_factors']):.4f} to {max(scale_invariance_results['correction_factors']):.4f}")

    print("\nObservation Comparison:")
    print("System\t\tObserved\tGR Pred\tCOM Pred\tGR Dev\tCOM Dev")
    for obs in observation_comparison:
        print(f"{obs['name'][:15]}...\t{obs['observed']:.2f}±{obs['uncertainty']:.2f}\t{obs['gr_prediction']:.2f}\t{obs['com_prediction']:.2f}\t{obs['gr_deviation']:.2f}σ\t{obs['com_deviation']:.2f}σ")

    print("\nAnalysis complete. Figures saved in 'figures' directory.")
    
    return {
        "mass_scaling": mass_scaling_results,
        "deflection_profiles": deflection_results,
        "scale_invariance": scale_invariance_results,
        "observation_comparison": observation_comparison
    }


if __name__ == "__main__":
    # Run the full analysis when script is executed directly
    results = run_full_analysis()
