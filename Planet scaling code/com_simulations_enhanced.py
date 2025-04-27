import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Create directories for figures
os.makedirs('figures', exist_ok=True)

# COM Framework Constants
LZ = 1.23498  # Fundamental scaling constant
HQS = 0.235  # Harmonic Quantum Scalar (23.5% of LZ)
HQS_LZ = HQS * LZ  # HQS threshold in absolute terms (≈ 0.29022)

# Reference values
r0_mercury = 0.39  # Mercury's orbital radius in AU
G = 1.327e20  # Gravitational constant * Sun mass in m³/s²
c = 2.998e8  # Speed of light in m/s
AU = 1.496e11  # Astronomical unit in meters

# 24-step Fibonacci digital root pattern
FIBONACCI_PATTERN = [1, 1, 2, 3, 5, 8, 4, 3, 7, 1, 8, 9, 8, 8, 7, 6, 4, 1, 5, 6, 2, 8, 1, 9]

# Planet data
planets = [
    {"name": "Mercury", "actual_distance": 0.39, "index": 0, "color": "gray"},
    {"name": "Venus", "actual_distance": 0.72, "index": 1, "color": "orange"},
    {"name": "Earth", "actual_distance": 1.00, "index": 2, "color": "blue"},
    {"name": "Mars", "actual_distance": 1.52, "index": 3, "color": "red"},
    {"name": "Ceres", "actual_distance": 2.77, "index": 4, "color": "brown"},
    {"name": "Jupiter", "actual_distance": 5.20, "index": 5, "color": "tan"},
    {"name": "Saturn", "actual_distance": 9.54, "index": 6, "color": "gold"},
    {"name": "Uranus", "actual_distance": 19.20, "index": 7, "color": "lightblue"},
    {"name": "Neptune", "actual_distance": 30.10, "index": 8, "color": "darkblue"}
]

# Traditional Bode's Law
def bodes_law(n):
    return 0.4 + 0.3 * 2**n

# Basic COM-based orbital spacing model
def basic_com_orbital_model(n):
    # Special case for Mercury (n=0)
    if n == 0:
        return r0_mercury
    
    # Basic formula with exponential scaling
    return r0_mercury * LZ**(n * (1 + HQS * (n-1)/10))

# Enhanced COM model incorporating Jupiter's gravitational influence
def enhanced_com_orbital_model(n):
    # Special case for Mercury (n=0)
    if n == 0:
        return r0_mercury
    
    # Get Fibonacci pattern value for this position
    pattern_value = FIBONACCI_PATTERN[n % 24] / 9.0  # Normalize to 0-1 range
    
    # Calculate resonance factor based on Jupiter's influence
    # This factor increases for positions beyond the asteroid belt
    jupiter_resonance = 1.0
    if n > 4:  # Beyond asteroid belt
        jupiter_resonance = 1.0 + 0.15 * (n - 4)  # Increasing influence with distance
    
    # Enhanced formula with Fibonacci pattern and Jupiter resonance
    return r0_mercury * (LZ * jupiter_resonance)**(n * (1 + pattern_value * HQS/3))

# Relativistic correction for observer bias
def relativistic_correction(distance, velocity_fraction=None):
    """
    Apply relativistic correction to account for observer bias
    
    Parameters:
    - distance: Distance in AU
    - velocity_fraction: Orbital velocity as fraction of c (if None, estimated from distance)
    
    Returns:
    - Corrected distance
    """
    # Estimate velocity as fraction of c if not provided
    if velocity_fraction is None:
        # Approximate orbital velocity using Kepler's laws
        velocity_ms = np.sqrt(G / (distance * AU))
        velocity_fraction = velocity_ms / c
    
    # Light travel time effect (increases with distance)
    light_travel_factor = 1.0 + (distance / 30.0) * 0.01  # 1% effect at Neptune's distance
    
    # Relativistic Doppler effect
    doppler_factor = np.sqrt((1 + velocity_fraction) / (1 - velocity_fraction))
    
    # Gravitational time dilation
    time_dilation_factor = 1.0 / np.sqrt(1 - 2 * G / (c**2 * distance * AU))
    
    # Combined correction factor
    correction = light_travel_factor * doppler_factor * time_dilation_factor
    
    return distance * correction

# Octave position function
def octave_position(r):
    return np.log(r/r0_mercury) / np.log(LZ) % 1

# Stability function incorporating Jupiter's influence
def stability(r):
    # Basic stability from octave position
    basic_stability = np.sin(np.pi * octave_position(r) / HQS)**2
    
    # Jupiter's influence creates a stability minimum in the asteroid belt region
    jupiter_influence = 1.0
    asteroid_belt_effect = 0.0
    
    # Reduce stability in asteroid belt region (2.2-3.3 AU)
    if isinstance(r, np.ndarray):
        asteroid_belt_mask = (r >= 2.2) & (r <= 3.3)
        jupiter_influence = np.ones_like(r)
        asteroid_belt_effect = np.zeros_like(r)
        
        # Calculate influence based on distance from Jupiter
        distance_from_jupiter = np.abs(r - 5.2)
        jupiter_influence[asteroid_belt_mask] = 0.5 + 0.5 * (distance_from_jupiter[asteroid_belt_mask] / 3.0)
        
        # Calculate resonance effects
        for resonance in [2.06, 2.5, 2.82, 3.27]:  # Known Kirkwood gaps
            resonance_effect = 0.8 * np.exp(-(r - resonance)**2 / 0.01)
            asteroid_belt_effect = np.maximum(asteroid_belt_effect, resonance_effect)
    else:
        if 2.2 <= r <= 3.3:
            distance_from_jupiter = abs(r - 5.2)
            jupiter_influence = 0.5 + 0.5 * (distance_from_jupiter / 3.0)
            
            # Calculate resonance effects
            for resonance in [2.06, 2.5, 2.82, 3.27]:  # Known Kirkwood gaps
                resonance_effect = 0.8 * np.exp(-(r - resonance)**2 / 0.01)
                asteroid_belt_effect = max(asteroid_belt_effect, resonance_effect)
    
    # Combined stability with Jupiter's influence
    return basic_stability * jupiter_influence - asteroid_belt_effect

# Phase transition boundary function
def is_boundary(r, epsilon=0.1):
    return abs(np.sin(np.pi * np.log(r/r0_mercury) / np.log(LZ) / HQS)) < epsilon

# Simulation 1: Enhanced Planetary Spacing Analysis
def simulate_enhanced_planetary_spacing():
    # Calculate predictions
    indices = np.array([p["index"] for p in planets])
    actual_distances = np.array([p["actual_distance"] for p in planets])
    bode_predictions = np.array([bodes_law(n) for n in indices])
    basic_com_predictions = np.array([basic_com_orbital_model(n) for n in indices])
    enhanced_com_predictions = np.array([enhanced_com_orbital_model(n) for n in indices])
    
    # Apply relativistic corrections to enhanced model for outer planets
    relativistic_com_predictions = enhanced_com_predictions.copy()
    for i, planet in enumerate(planets):
        if planet["index"] > 4:  # Apply only to outer planets
            relativistic_com_predictions[i] = relativistic_correction(enhanced_com_predictions[i])
    
    # Calculate errors
    bode_errors = (bode_predictions - actual_distances) / actual_distances * 100
    basic_com_errors = (basic_com_predictions - actual_distances) / actual_distances * 100
    enhanced_com_errors = (enhanced_com_predictions - actual_distances) / actual_distances * 100
    relativistic_com_errors = (relativistic_com_predictions - actual_distances) / actual_distances * 100
    
    # Create figure
    plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Plot 1: Orbital distances
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(indices, actual_distances, 'ko-', label='Actual Distances', linewidth=2)
    ax1.plot(indices, bode_predictions, 'b--', label='Bode\'s Law', linewidth=1.5)
    ax1.plot(indices, basic_com_predictions, 'g-.', label='Basic COM Model', linewidth=1.5)
    ax1.plot(indices, enhanced_com_predictions, 'r-', label='Enhanced COM Model', linewidth=1.5)
    ax1.plot(indices, relativistic_com_predictions, 'm:', label='Relativistic COM Model', linewidth=2)
    
    # Add planet markers
    for i, planet in enumerate(planets):
        ax1.plot(planet["index"], planet["actual_distance"], 'o', 
                 color=planet["color"], markersize=10)
        ax1.text(planet["index"], planet["actual_distance"]*1.1, 
                 planet["name"], ha='center', fontsize=9)
    
    # Add asteroid belt region
    ax1.axvspan(3.5, 4.5, alpha=0.2, color='brown', label='Asteroid Belt')
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Planet Index')
    ax1.set_ylabel('Orbital Distance (AU)')
    ax1.set_title('Planetary Orbital Distances: Actual vs. Predicted')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Prediction errors
    ax2 = plt.subplot(gs[1, 0])
    bar_width = 0.2
    x = np.arange(len(planets))
    ax2.bar(x - 1.5*bar_width, bode_errors, bar_width, label='Bode\'s Law Error')
    ax2.bar(x - 0.5*bar_width, basic_com_errors, bar_width, label='Basic COM Error')
    ax2.bar(x + 0.5*bar_width, enhanced_com_errors, bar_width, label='Enhanced COM Error')
    ax2.bar(x + 1.5*bar_width, relativistic_com_errors, bar_width, label='Relativistic COM Error')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p["name"] for p in planets], rotation=45)
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Prediction Errors')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Octave positions with Jupiter influence
    ax3 = plt.subplot(gs[1, 1])
    r_values = np.logspace(np.log10(0.2), np.log10(40), 1000)
    stability_values = stability(r_values)
    octave_positions = octave_position(r_values)
    
    # Create a colormap based on stability
    colors = plt.cm.viridis(stability_values)
    
    # Plot octave positions with stability-based coloring
    for i in range(len(r_values)-1):
        ax3.plot(r_values[i:i+2], octave_positions[i:i+2], color=colors[i], linewidth=1.5)
    
    # Add planet markers
    for planet in planets:
        oct_pos = octave_position(planet["actual_distance"])
        ax3.plot(planet["actual_distance"], oct_pos, 'o', 
                 color=planet["color"], markersize=8)
        ax3.text(planet["actual_distance"]*1.05, oct_pos, 
                 planet["name"], fontsize=8, va='center')
    
    # Add HQS threshold line
    ax3.axhline(y=HQS, color='r', linestyle='--', alpha=0.7, label=f'HQS = {HQS:.3f}')
    
    # Add asteroid belt region
    ax3.axvspan(2.2, 3.3, alpha=0.2, color='brown')
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Orbital Distance (AU)')
    ax3.set_ylabel('Octave Position')
    ax3.set_title('Octave Positions of Planets with Jupiter Influence')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Relativistic correction factors
    ax4 = plt.subplot(gs[2, 0])
    
    # Distance range
    distances = np.linspace(0.1, 40, 1000)
    
    # Calculate correction factors
    light_travel_corrections = 1.0 + (distances / 30.0) * 0.01
    
    # Approximate velocity as fraction of c
    velocities = np.sqrt(G / (distances * AU)) / c
    doppler_corrections = np.sqrt((1 + velocities) / (1 - velocities))
    
    # Gravitational time dilation
    time_dilation_corrections = 1.0 / np.sqrt(1 - 2 * G / (c**2 * distances * AU))
    
    # Combined correction
    combined_corrections = light_travel_corrections * doppler_corrections * time_dilation_corrections
    
    # Plot correction factors
    ax4.plot(distances, light_travel_corrections, 'b-', label='Light Travel Time', linewidth=1.5)
    ax4.plot(distances, doppler_corrections, 'g-', label='Doppler Effect', linewidth=1.5)
    ax4.plot(distances, time_dilation_corrections, 'r-', label='Time Dilation', linewidth=1.5)
    ax4.plot(distances, combined_corrections, 'k-', label='Combined Effect', linewidth=2)
    
    # Add planet markers
    for planet in planets:
        dist = planet["actual_distance"]
        correction = 1.0
        if planet["index"] > 4:  # Only show for outer planets
            correction = combined_corrections[np.abs(distances - dist).argmin()]
            ax4.plot(dist, correction, 'o', color=planet["color"], markersize=8)
            ax4.text(dist, correction*1.01, planet["name"], fontsize=8, ha='center')
    
    ax4.set_xscale('log')
    ax4.set_xlabel('Orbital Distance (AU)')
    ax4.set_ylabel('Relativistic Correction Factor')
    ax4.set_title('Observer Relativity Bias Correction Factors')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Jupiter's gravitational influence
    ax5 = plt.subplot(gs[2, 1])
    
    # Calculate Jupiter's gravitational influence
    jupiter_distance = 5.2  # AU
    distances = np.linspace(0.1, 40, 1000)
    
    # Simplified model of Jupiter's gravitational influence
    # Based on resonance zones and distance
    jupiter_influence = np.ones_like(distances)
    
    # Reduce stability in asteroid belt region (2.2-3.3 AU)
    asteroid_belt_mask = (distances >= 2.2) & (distances <= 3.3)
    distance_from_jupiter = np.abs(distances - jupiter_distance)
    jupiter_influence[asteroid_belt_mask] = 0.5 + 0.5 * (distance_from_jupiter[asteroid_belt_mask] / 3.0)
    
    # Add resonance effects
    resonance_effect = np.zeros_like(distances)
    for resonance in [2.06, 2.5, 2.82, 3.27]:  # Known Kirkwood gaps
        effect = 0.8 * np.exp(-(distances - resonance)**2 / 0.01)
        resonance_effect = np.maximum(resonance_effect, effect)
    
    # Plot Jupiter's influence
    ax5.plot(distances, jupiter_influence, 'b-', label='Jupiter\'s Gravitational Influence', linewidth=1.5)
    ax5.plot(distances, resonance_effect, 'r-', label='Resonance Effects', linewidth=1.5)
    ax5.plot(distances, jupiter_influence - resonance_effect, 'k-', label='Combined Effect', linewidth=2)
    
    # Add asteroid belt region
    ax5.axvspan(2.2, 3.3, alpha=0.2, color='brown', label='Asteroid Belt')
    
    # Add planet markers
    for planet in planets:
        if 0.5 <= planet["actual_distance"] <= 20:  # Only show relevant planets
            dist = planet["actual_distance"]
            influence = jupiter_influence[np.abs(distances - dist).argmin()]
            resonance = resonance_effect[np.abs(distances - dist).argmin()]
            ax5.plot(dist, influence - resonance, 'o', color=planet["color"], markersize=8)
            ax5.text(dist, influence - resonance + 0.05, planet["name"], fontsize=8, ha='center')
    
    ax5.set_xscale('log')
    ax5.set_xlabel('Orbital Distance (AU)')
    ax5.set_ylabel('Stability Factor')
    ax5.set_title('Jupiter\'s Gravitational Influence on Orbital Stability')
    ax5.set_ylim(0, 1.2)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    plt.tight_layout()
    plt.savefig('figures/enhanced_planetary_spacing_analysis.png', dpi=300)
    
    # Create a second figure for the enhanced orbital stability model
    plt.figure(figsize=(12, 8))
    
    # Generate continuous model predictions
    r_range = np.logspace(np.log10(0.2), np.log10(40), 1000)
    stability_continuous = stability(r_range)
    
    # Plot continuous stability function
    plt.plot(r_range, stability_continuous, 'k-', alpha=0.7, linewidth=1.5, label='Orbital Stability')
    
    # Highlight boundary regions
    boundary_mask = np.array([is_boundary(r) for r in r_range])
    plt.fill_between(r_range, 0, 1, where=boundary_mask, 
                     color='gray', alpha=0.3, label='Phase Transition Boundaries')
    
    # Add planet positions
    for planet in planets:
        stab = stability(planet["actual_distance"])
        plt.plot(planet["actual_distance"], stab, 'o', 
                 color=planet["color"], markersize=10, label=planet["name"])
        plt.text(planet["actual_distance"], stab + 0.05, 
                 planet["name"], ha='center', fontsize=9)
    
    # Add asteroid belt region with Kirkwood gaps
    plt.axvspan(2.2, 3.3, alpha=0.2, color='brown', label='Asteroid Belt')
    
    # Add Kirkwood gaps
    for gap in [2.06, 2.5, 2.82, 3.27]:
        plt.axvline(x=gap, color='brown', linestyle='--', alpha=0.5)
        plt.text(gap, 0.05, f'{gap:.2f}', rotation=90, ha='center', fontsize=8)
    
    # Add Jupiter's position
    plt.axvline(x=5.2, color='tan', linestyle='-', alpha=0.5, label='Jupiter')
    plt.text(5.2, 0.05, 'Jupiter', rotation=90, ha='center', fontsize=8)
    
    plt.xscale('log')
    plt.xlabel('Orbital Distance (AU)')
    plt.ylabel('Orbital Stability')
    plt.title('Enha
(Content truncated due to size limit. Use line ranges to read in chunks)