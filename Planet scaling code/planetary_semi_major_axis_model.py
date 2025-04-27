"""
COM Framework Planetary Semi-Major Axis Model

This module implements the Continuous Oscillatory Model (COM) framework for planetary
semi-major axis prediction using the equation:
    an = a0 · λn · (1 + η · sin(θn))

where:
    - λ = 1.23498 (LZ scaling)
    - η = 0.235 (HQS modulation)
    - θn = 4nπ (phase term)
    - a0 is Mercury's semi-major axis (0.39 AU)

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

def calculate_semi_major_axis(n, a0=0.39, lz=LZ, hqs=HQS, phase_factor=4*np.pi):
    """
    Calculate planetary semi-major axis using the COM framework equation.
    
    Parameters:
    - n: Orbital index (0 for Mercury, 1 for Venus, etc.)
    - a0: Baseline distance (Mercury's orbit in AU)
    - lz: LZ scaling factor (default: 1.23498)
    - hqs: HQS modulation factor (default: 0.235)
    - phase_factor: Factor for phase calculation (default: 4π)
    
    Returns:
    - Predicted semi-major axis in AU
    """
    theta_n = phase_factor * n
    return a0 * (lz ** n) * (1 + hqs * np.sin(theta_n))

def analyze_model():
    """
    Analyze the COM framework planetary semi-major axis model.
    
    Returns:
    - Dictionary of analysis results
    """
    # Calculate predicted semi-major axes
    indices = np.arange(len(PLANETS))
    predicted_axes = [calculate_semi_major_axis(n) for n in indices]
    
    # Calculate errors
    actual_axes = [p["semi_major_axis"] for p in PLANETS]
    absolute_errors = [abs(pred - act) for pred, act in zip(predicted_axes, actual_axes)]
    percentage_errors = [100 * abs(pred - act) / act for pred, act in zip(predicted_axes, actual_axes)]
    
    # Calculate statistics
    mean_absolute_error = np.mean(absolute_errors)
    mean_percentage_error = np.mean(percentage_errors)
    max_percentage_error = max(percentage_errors)
    
    # Create results table
    results = []
    for i, planet in enumerate(PLANETS):
        results.append({
            "name": planet["name"],
            "actual": planet["semi_major_axis"],
            "predicted": predicted_axes[i],
            "absolute_error": absolute_errors[i],
            "percentage_error": percentage_errors[i]
        })
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot actual vs. predicted semi-major axes
    plt.subplot(2, 1, 1)
    plt.scatter(indices, actual_axes, s=100, color='blue', label='Actual')
    plt.scatter(indices, predicted_axes, s=100, color='red', label='Predicted')
    
    # Connect points with lines
    plt.plot(indices, actual_axes, 'b-', alpha=0.5)
    plt.plot(indices, predicted_axes, 'r-', alpha=0.5)
    
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
    plt.bar(indices, percentage_errors, color='green', alpha=0.7)
    
    # Add horizontal line at mean error
    plt.axhline(y=mean_percentage_error, color='red', linestyle='--', 
                label=f'Mean Error: {mean_percentage_error:.2f}%')
    
    # Add planet names
    plt.xticks(indices, [p["name"] for p in PLANETS], rotation=45)
    
    plt.xlabel('Planet')
    plt.ylabel('Percentage Error (%)')
    plt.title('Percentage Error in Semi-Major Axis Prediction')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('figures/planetary_semi_major_axis_model.png', dpi=300)
    
    # Create log-scale plot
    plt.figure(figsize=(12, 8))
    
    plt.loglog(indices+1, actual_axes, 'bo-', label='Actual', linewidth=2, markersize=10)
    plt.loglog(indices+1, predicted_axes, 'ro-', label='Predicted', linewidth=2, markersize=10)
    
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
    plt.savefig('figures/planetary_semi_major_axis_log.png', dpi=300)
    
    # Create continuous model plot
    plt.figure(figsize=(12, 8))
    
    # Generate continuous model curve
    n_values = np.linspace(0, 8, 1000)
    continuous_model = [calculate_semi_major_axis(n) for n in n_values]
    
    # Plot continuous model and actual data points
    plt.semilogy(n_values, continuous_model, 'r-', label='COM Model', linewidth=2)
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
    plt.savefig('figures/planetary_semi_major_axis_continuous.png', dpi=300)
    
    # Print results
    print("\n=== COM Framework Planetary Semi-Major Axis Model Analysis ===")
    print("\nModel Parameters:")
    print(f"a0 = {0.39} AU (Mercury's orbit)")
    print(f"λ = {LZ} (LZ scaling)")
    print(f"η = {HQS} (HQS modulation)")
    print(f"θn = {4}nπ (phase term)")
    
    print("\nResults:")
    print(f"Mean Absolute Error: {mean_absolute_error:.2f} AU")
    print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
    print(f"Maximum Percentage Error: {max_percentage_error:.2f}%")
    
    print("\nPlanet-by-Planet Results:")
    print("Planet\t\tActual (AU)\tPredicted (AU)\tError (%)")
    for result in results:
        print(f"{result['name']:<10}\t{result['actual']:.2f}\t\t{result['predicted']:.2f}\t\t{result['percentage_error']:.2f}%")
    
    return {
        "parameters": {
            "a0": 0.39,
            "lz": LZ,
            "hqs": HQS,
            "phase_factor": 4*np.pi
        },
        "results": results,
        "statistics": {
            "mean_absolute_error": mean_absolute_error,
            "mean_percentage_error": mean_percentage_error,
            "max_percentage_error": max_percentage_error
        }
    }

def parameter_sensitivity_analysis():
    """
    Analyze the sensitivity of the model to different parameter values.
    
    Returns:
    - Dictionary of sensitivity analysis results
    """
    # Parameter ranges to test
    a0_values = np.linspace(0.3, 0.5, 5)
    lz_values = np.linspace(1.2, 1.3, 5)
    hqs_values = np.linspace(0.2, 0.3, 5)
    phase_factor_values = np.linspace(2*np.pi, 6*np.pi, 5)
    
    # Initialize results
    sensitivity_results = {
        "a0": [],
        "lz": [],
        "hqs": [],
        "phase_factor": []
    }
    
    # Test a0 sensitivity
    for a0 in a0_values:
        predicted_axes = [calculate_semi_major_axis(n, a0=a0) for n in range(len(PLANETS))]
        actual_axes = [p["semi_major_axis"] for p in PLANETS]
        percentage_errors = [100 * abs(pred - act) / act for pred, act in zip(predicted_axes, actual_axes)]
        mean_error = np.mean(percentage_errors)
        sensitivity_results["a0"].append({"value": a0, "mean_error": mean_error})
    
    # Test lz sensitivity
    for lz in lz_values:
        predicted_axes = [calculate_semi_major_axis(n, lz=lz) for n in range(len(PLANETS))]
        actual_axes = [p["semi_major_axis"] for p in PLANETS]
        percentage_errors = [100 * abs(pred - act) / act for pred, act in zip(predicted_axes, actual_axes)]
        mean_error = np.mean(percentage_errors)
        sensitivity_results["lz"].append({"value": lz, "mean_error": mean_error})
    
    # Test hqs sensitivity
    for hqs in hqs_values:
        predicted_axes = [calculate_semi_major_axis(n, hqs=hqs) for n in range(len(PLANETS))]
        actual_axes = [p["semi_major_axis"] for p in PLANETS]
        percentage_errors = [100 * abs(pred - act) / act for pred, act in zip(predicted_axes, actual_axes)]
        mean_error = np.mean(percentage_errors)
        sensitivity_results["hqs"].append({"value": hqs, "mean_error": mean_error})
    
    # Test phase_factor sensitivity
    for phase_factor in phase_factor_values:
        predicted_axes = [calculate_semi_major_axis(n, phase_factor=phase_factor) for n in range(len(PLANETS))]
        actual_axes = [p["semi_major_axis"] for p in PLANETS]
        percentage_errors = [100 * abs(pred - act) / act for pred, act in zip(predicted_axes, actual_axes)]
        mean_error = np.mean(percentage_errors)
        sensitivity_results["phase_factor"].append({"value": phase_factor, "mean_error": mean_error})
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot sensitivity to a0
    plt.subplot(2, 2, 1)
    plt.plot([r["value"] for r in sensitivity_results["a0"]], 
             [r["mean_error"] for r in sensitivity_results["a0"]], 
             'bo-', linewidth=2)
    plt.xlabel('a0 (AU)')
    plt.ylabel('Mean Percentage Error (%)')
    plt.title('Sensitivity to Baseline Distance (a0)')
    plt.grid(True, alpha=0.3)
    
    # Plot sensitivity to lz
    plt.subplot(2, 2, 2)
    plt.plot([r["value"] for r in sensitivity_results["lz"]], 
             [r["mean_error"] for r in sensitivity_results["lz"]], 
             'ro-', linewidth=2)
    plt.xlabel('λ (LZ Scaling)')
    plt.ylabel('Mean Percentage Error (%)')
    plt.title('Sensitivity to LZ Scaling (λ)')
    plt.grid(True, alpha=0.3)
    
    # Plot sensitivity to hqs
    plt.subplot(2, 2, 3)
    plt.plot([r["value"] for r in sensitivity_results["hqs"]], 
             [r["mean_error"] for r in sensitivity_results["hqs"]], 
             'go-', linewidth=2)
    plt.xlabel('η (HQS Modulation)')
    plt.ylabel('Mean Percentage Error (%)')
    plt.title('Sensitivity to HQS Modulation (η)')
    plt.grid(True, alpha=0.3)
    
    # Plot sensitivity to phase_factor
    plt.subplot(2, 2, 4)
    plt.plot([r["value"]/(np.pi) for r in sensitivity_results["phase_factor"]], 
             [r["mean_error"] for r in sensitivity_results["phase_factor"]], 
             'mo-', linewidth=2)
    plt.xlabel('Phase Factor (multiples of π)')
    plt.ylabel('Mean Percentage Error (%)')
    plt.title('Sensitivity to Phase Factor')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/planetary_semi_major_axis_sensitivity.png', dpi=300)
    
    # Find optimal parameters
    optimal_a0 = min(sensitivity_results["a0"], key=lambda x: x["mean_error"])
    optimal_lz = min(sensitivity_results["lz"], key=lambda x: x["mean_error"])
    optimal_hqs = min(sensitivity_results["hqs"], key=lambda x: x["mean_error"])
    optimal_phase_factor = min(sensitivity_results["phase_factor"], key=lambda x: x["mean_error"])
    
    print("\n=== Parameter Sensitivity Analysis ===")
    print(f"Optimal a0: {optimal_a0['value']:.4f} AU (Error: {optimal_a0['mean_error']:.2f}%)")
    print(f"Optimal λ: {optimal_lz['value']:.4f} (Error: {optimal_lz['mean_error']:.2f}%)")
    print(f"Optimal η: {optimal_hqs['value']:.4f} (Error: {optimal_hqs['mean_error']:.2f}%)")
    print(f"Optimal Phase Factor: {optimal_phase_factor['value']/np.pi:.2f}π (Error: {optimal_phase_factor['mean_error']:.2f}%)")
    
    return {
        "sensitivity_results": sensitivity_results,
        "optimal_parameters": {
            "a0": optimal_a0,
            "lz": optimal_lz,
            "hqs": optimal_hqs,
            "phase_factor": optimal_phase_factor
        }
    }

def analyze_asteroid_belt():
    """
    Analyze the asteroid belt region using the COM framework model.
    
    Returns:
    - Dictionary of asteroid belt analysis results
    """
    # Generate continuous model curve
    n_values = np.linspace(0, 8, 1000)
    continuous_model = [calculate_semi_major_axis(n) for n in n_values]
    
    # Find where the model predicts the asteroid belt should be
    asteroid_belt_actual = 2.77  # Ceres semi-major axis as reference
    
    # Calculate model values around asteroid belt region
    asteroid_region_n = np.linspace(3, 5, 1000)
    asteroid_region_model = [calculate_semi_major_axis(n) for n in asteroid_region_n]
    
    # Find local minima and maxima in the asteroid belt region
    asteroid_region_diff = np.diff(asteroid_region_model)
    sign_changes = np.where(np.diff(np.signbit(asteroid_region_diff)))[0]
    
    extrema_n = [asteroid_region_n[i+1] for i in sign_changes]
    extrema_values = [calculate_semi_major_axis(n) for n in extrema_n]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot continuous model
    plt.semilogy(n_values, continuous_model, 'r-', label='COM Model', linewidth=2)
    
    # Plot actual planets
    indices = np.arange(len(PLANETS))
    actual_axes = [p["semi_major_axis"] for p in PLANETS]
    plt.scatter(indices, actual_axes, s=100, color='blue', label='Actual Planets')
    
    # Highlight asteroid belt region
    plt.axvspan(3.5, 4.5, color='gray', alpha=0.2, label='Asteroid Belt Region')
    
    # Mark extrema in asteroid belt region
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
    plt.savefig('figures/asteroid_belt_analysis.png', dpi=300)
    
    # Calculate Kirkwood gaps positions
    kirkwood_gaps = [2.06, 2.5, 2.82, 3.27]  # Known Kirkwood gaps in AU
    
    # Find corresponding n values
    kirkwood_n_values = []
    for gap in kirkwood_gaps:
        # Find n value that gives closest semi-major axis to the gap
        n_range = np.linspace(2, 5, 1000)
        predicted_values = [calculate_semi_major_axis(n) for n in n_range]
        differences = [abs(pred - gap) for pred in predicted_values]
        best_index = np.argmin(differences)
        kirkwood_n_values.append(n_range[best_index])
    
    print("\n=== Asteroid Belt Analysis ===")
    print("Extrema in Asteroid Belt Region:")
    for i, n in enumerate(extrema_n):
        extrema_type = "Minimum" if i % 2 == 0 else "Maximum"
        print(f"{extrema_type} at n = {n:.2f}, semi-major axis = {extrema_values[i]:.2f} AU")
    
    print("\nKirkwood Gaps Analysis:")
    print("Gap (AU)\tApproximate n value")
    for i, gap in enumerate(kirkwo
(Content truncated due to size limit. Use line ranges to read in chunks)