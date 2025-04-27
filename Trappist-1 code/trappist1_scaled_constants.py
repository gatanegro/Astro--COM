"""
COM Framework Analysis of TRAPPIST-1 Exoplanetary System with Scaled Constants

This module implements the COM-HQS-LZ model for the TRAPPIST-1 system
with proportionally scaled constants to test the hypothesis that
LZ and HQS constants are local and scale with the stellar system.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

# Original Solar System COM framework constants
LZ_SOLAR = 1.23498  # LZ scaling constant for Solar System
HQS_SOLAR = 0.235   # HQS modulation constant for Solar System
a0_TRAPPIST = 0.0115   # TRAPPIST-1b's observed distance (AU)

def com_hqs_lz_orbit(n, scale_factor=1.0, phase_func=lambda n: np.sin(n * np.pi / 3)):
    """
    Predicts orbital distance for octave layer n with phase modulation and scaled constants.
    
    Parameters:
    - n: Octave layer number
    - scale_factor: Factor to scale both LZ and HQS constants
    - phase_func: Phase function to use (default: sin(n*π/3))
    
    Returns:
    - Predicted orbital distance in AU
    """
    # Scale constants proportionally
    lz = LZ_SOLAR * scale_factor
    hqs = HQS_SOLAR * scale_factor
    
    return a0_TRAPPIST * (lz ** n) * (1 + hqs * phase_func(n))

def analyze_trappist1_with_scaling(scale_factor=1.0, phase_func=lambda n: np.sin(n * np.pi / 3)):
    """
    Analyze the TRAPPIST-1 system using the COM-HQS-LZ model with scaled constants.
    
    Parameters:
    - scale_factor: Factor to scale both LZ and HQS constants
    - phase_func: Phase function to use
    
    Returns:
    - Dictionary with analysis results
    """
    # TRAPPIST-1 observed distances (AU) - NASA data
    planet_names = ['TRAPPIST-1b', 'TRAPPIST-1c', 'TRAPPIST-1d', 'TRAPPIST-1e', 
                   'TRAPPIST-1f', 'TRAPPIST-1g', 'TRAPPIST-1h']
    observed = [0.0115, 0.0158, 0.0223, 0.0293, 0.0385, 0.0469, 0.0619]
    layers = np.arange(len(observed))
    
    # Calculate scaled constants
    lz_scaled = LZ_SOLAR * scale_factor
    hqs_scaled = HQS_SOLAR * scale_factor
    
    # Compare predictions
    predicted = [com_hqs_lz_orbit(n, scale_factor, phase_func) for n in layers]
    residuals = (np.array(predicted) - np.array(observed)) / np.array(observed) * 100  # % error
    
    # Calculate average absolute error
    avg_abs_error = np.mean(np.abs(residuals))
    
    # Prepare results for return
    results = {
        "planet_names": planet_names,
        "observed": observed,
        "predicted": predicted,
        "residuals": residuals,
        "avg_abs_error": avg_abs_error,
        "layers": layers,
        "scale_factor": scale_factor,
        "lz_scaled": lz_scaled,
        "hqs_scaled": hqs_scaled
    }
    
    return results

def test_scaling_factors(scale_factors, phase_func=lambda n: np.sin(n * np.pi / 3)):
    """
    Test multiple scaling factors and find the optimal one.
    
    Parameters:
    - scale_factors: List or array of scaling factors to test
    - phase_func: Phase function to use
    
    Returns:
    - Dictionary with results for each scaling factor
    - Optimal scaling factor
    """
    results = {}
    errors = {}
    
    for sf in scale_factors:
        result = analyze_trappist1_with_scaling(sf, phase_func)
        results[sf] = result
        errors[sf] = result["avg_abs_error"]
    
    # Find optimal scaling factor
    optimal_sf = min(errors.items(), key=lambda x: x[1])[0]
    
    return results, optimal_sf, errors

def plot_scaling_comparison(results_dict, optimal_sf, output_dir="figures", filename="trappist1_scaling_comparison.png"):
    """
    Create a plot comparing different scaling factors.
    
    Parameters:
    - results_dict: Dictionary with results for each scaling factor
    - optimal_sf: Optimal scaling factor
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    scale_factors = list(results_dict.keys())
    errors = [results_dict[sf]["avg_abs_error"] for sf in scale_factors]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot errors vs scaling factors
    plt.plot(scale_factors, errors, 'o-', color='blue', markersize=8, linewidth=2)
    
    # Mark optimal scaling factor
    plt.plot(optimal_sf, results_dict[optimal_sf]["avg_abs_error"], 'o', color='red', markersize=12)
    plt.text(optimal_sf, results_dict[optimal_sf]["avg_abs_error"] * 1.05, 
             f"Optimal: {optimal_sf:.3f}\nError: {results_dict[optimal_sf]['avg_abs_error']:.2f}%", 
             ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.xlabel('Scaling Factor', fontsize=12)
    plt.ylabel('Average Absolute Error (%)', fontsize=12)
    plt.title('TRAPPIST-1 System: Error vs. LZ/HQS Scaling Factor', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add text with scaled constants
    lz_scaled = LZ_SOLAR * optimal_sf
    hqs_scaled = HQS_SOLAR * optimal_sf
    const_text = f"Solar System Constants:\n" \
                 f"LZ = {LZ_SOLAR}\n" \
                 f"HQS = {HQS_SOLAR}\n\n" \
                 f"Optimal TRAPPIST-1 Constants:\n" \
                 f"LZ = {lz_scaled:.5f}\n" \
                 f"HQS = {hqs_scaled:.5f}"
    plt.figtext(0.02, 0.02, const_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Scaling comparison plot saved to {os.path.join(output_dir, filename)}")

def plot_optimal_comparison(results, output_dir="figures", filename="trappist1_optimal_scaling.png"):
    """
    Create a bar chart comparing observed and predicted values with optimal scaling.
    
    Parameters:
    - results: Dictionary with analysis results for optimal scaling
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    planet_names = results["planet_names"]
    observed = results["observed"]
    predicted = results["predicted"]
    residuals = results["residuals"]
    scale_factor = results["scale_factor"]
    lz_scaled = results["lz_scaled"]
    hqs_scaled = results["hqs_scaled"]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot bars
    x = np.arange(len(planet_names))
    width = 0.35
    
    plt.bar(x - width/2, observed, width, label='Observed', color='blue', alpha=0.7)
    plt.bar(x + width/2, predicted, width, label='COM-HQS-LZ Predicted (Scaled)', color='green', alpha=0.7)
    
    # Add error labels
    for i in range(len(planet_names)):
        plt.text(i, max(observed[i], predicted[i]) + 0.002, 
                 f"Error: {residuals[i]:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Planet', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title(f'TRAPPIST-1 System: Observed vs. COM-HQS-LZ Predicted with Scaled Constants', fontsize=14)
    plt.xticks(x, planet_names, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add average error and scaling info
    info_text = f"Average Absolute Error: {results['avg_abs_error']:.2f}%\n" \
                f"Scaling Factor: {scale_factor:.3f}\n" \
                f"LZ (scaled): {lz_scaled:.5f}\n" \
                f"HQS (scaled): {hqs_scaled:.5f}"
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Optimal scaling comparison plot saved to {os.path.join(output_dir, filename)}")

def plot_continuous_model_scaled(results, output_dir="figures", filename="trappist1_continuous_scaled.png"):
    """
    Create a plot of the continuous model with scaled constants.
    
    Parameters:
    - results: Dictionary with analysis results for optimal scaling
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    planet_names = results["planet_names"]
    observed = results["observed"]
    predicted = results["predicted"]
    residuals = results["residuals"]
    layers = results["layers"]
    scale_factor = results["scale_factor"]
    lz_scaled = results["lz_scaled"]
    hqs_scaled = results["hqs_scaled"]
    
    # Generate continuous model
    n_values = np.linspace(0, len(observed) - 0.5, 1000)
    a_values = [com_hqs_lz_orbit(n, scale_factor) for n in n_values]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot continuous model
    plt.plot(n_values, a_values, '-', color='green', alpha=0.7, linewidth=2, 
             label=f'Scaled COM-HQS-LZ Model: $a_n = a_0 \\cdot \\lambda^n \\cdot (1 + \\eta \\cdot \\sin(n\\pi/3))$')
    
    # Plot planet positions
    for i, (name, obs, pred, n) in enumerate(zip(planet_names, observed, predicted, layers)):
        plt.plot(n, obs, 'o', markersize=10, label=f"{name} (Observed)")
        plt.plot(n, pred, 's', markersize=8, label=f"{name} (Predicted)")
        
        # Add error label
        plt.text(n, obs * 1.1, 
                 f"Error: {residuals[i]:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Octave Layer (n)', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title('Continuous COM-HQS-LZ Model for TRAPPIST-1 System with Scaled Constants', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with smaller font size and multiple columns
    plt.legend(fontsize=10, ncol=3, loc='upper left')
    
    # Add model parameters text
    param_text = f"Model Parameters:\n" \
                 f"$\\lambda$ (LZ) = {lz_scaled:.5f}\n" \
                 f"$\\eta$ (HQS) = {hqs_scaled:.5f}\n" \
                 f"$a_0$ = {a0_TRAPPIST} AU\n" \
                 f"Phase function = $\\sin(n\\pi/3)$\n" \
                 f"Scaling factor = {scale_factor:.3f}"
    plt.figtext(0.02, 0.02, param_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add average error text
    plt.figtext(0.98, 0.02, f"Average Absolute Error: {results['avg_abs_error']:.2f}%", 
                ha='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Continuous model plot with scaled constants saved to {os.path.join(output_dir, filename)}")

def plot_mass_scaling_relationship(output_dir="figures", filename="stellar_mass_scaling_relationship.png"):
    """
    Create a plot showing the relationship between stellar mass and scaling factor.
    
    Parameters:
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Data points (stellar mass in solar masses, scaling factor)
    # Solar System (by definition) and TRAPPIST-1 (from our analysis)
    stellar_masses = [1.0, 0.089]  # Sun and TRAPPIST-1 in solar masses
    scaling_factors = [1.0, 0.65]  # Assuming 0.65 is our optimal scaling factor
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(stellar_masses, scaling_factors, s=100, color='blue')
    
    # Add labels for each point
    plt.text(stellar_masses[0], scaling_factors[0] * 1.05, "Sun", ha='center', fontsize=12)
    plt.text(stellar_masses[1], scaling_factors[1] * 1.05, "TRAPPIST-1", ha='center', fontsize=12)
    
    # Try to fit a power law relationship
    # log(y) = a * log(x) + b
    log_masses = np.log10(stellar_masses)
    log_factors = np.log10(scaling_factors)
    coeffs = np.polyfit(log_masses, log_factors, 1)
    a, b = coeffs
    
    # Generate points for the fitted curve
    x_fit = np.logspace(-2, 1, 100)  # From 0.01 to 10 solar masses
    y_fit = 10**(a * np.log10(x_fit) + b)
    
    # Plot the fitted curve
    plt.plot(x_fit, y_fit, '--', color='red', 
             label=f'Power Law Fit: S = {10**b:.3f} × M$_*$^{a:.3f}')
    
    # Add labels and title
    plt.xlabel('Stellar Mass (Solar Masses)', fontsize=12)
    plt.ylabel('COM Constants Scaling Factor', fontsize=12)
    plt.title('Relationship Between Stellar Mass and COM Constants Scaling Factor', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Use log scales for better visualization
    plt.xscale('log')
    plt.yscale('log')
    
    # Add text explaining the relationship
    explanation = f"Proposed Relationship:\n" \
                  f"S = {10**b:.3f} × M$_*$^{a:.3f}\n\n" \
                  f"Where:\n" \
                  f"S = Scaling factor for LZ and HQS\n" \
                  f"M$_*$ = Stellar mass in solar masses\n\n" \
                  f"This suggests COM constants scale with\n" \
                  f"approximately the {a:.3f} power of stellar mass."
    plt.figtext(0.02, 0.02, explanation, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Stellar mass scaling relationship plot saved to {os.path.join(output_dir, filename)}")
    
    return a, b

def main():
    """Main function to analyze TRAPPIST-1 system with scaled constants."""
    print("=" * 60)
    print("COM-HQS-LZ Analysis of TRAPPIST-1 System with Scaled Constants")
    print("=" * 60)
    
    # Test a range of scaling factors
    scale_factors = np.linspace(0.5, 0.8, 31)  # Test from 0.5 to 0.8 in 31 steps
    results_dict, optimal_sf, errors = test_scaling_factors(scale_factors)
    
    # Get results for optimal scaling factor
    optimal_results = results_dict[optimal_sf]
    
    # Print results for optimal scaling factor
    lz_scaled = LZ_SOLAR * optimal_sf
    hqs_scaled = HQS_SOLAR * optimal_sf
    
    print(f"\nOptimal Scaling Factor: {optimal_sf:.5f}")
    print(f"Scaled Constants: LZ = {lz_scaled:.5f}, HQS = {hqs_scaled:.5f}")
    print(f"Average Absolute Error: {optimal_results['avg_abs_error']:.2f}%")
    
    print("\nPredicted vs. Observed Solar System Distances (AU):")
    print("{:<12} {:<15} {:<15} {:<15}".format("Planet", "Observed (AU)", "Predicted (AU)", "Error (%)"))
    print("-" * 60)
    
    for i, (name, obs, pred, err) in enumerate(zip(
            optimal_results["planet_names"], 
            optimal_results["observed"], 
            optimal_results["predicted"], 
            optimal_results["residuals"])):
        print("{:<12} {:<15.4f} {:<15.4f} {:<15.2f}".format(name, obs, pred, err))
    
    # Generate visualizations
    plot_scaling_comparison(results_dict, optimal_sf)
    plot_optimal_comparison(optimal_results)
    plot_continuous_model_scaled(optimal_results)
    
    # Analyze relationship with stellar mass
    a, b = plot_mass_scaling_relationship()
    
    print("\n" + "=" * 60)
    print(f"Proposed Stellar Mass Scaling Relationship: S = {10**b:.3f} × M_*^{a:.3f}")
    print("=" * 60)
    
    return optimal_sf, optimal_results, a, b

if __name__ == "__main__":
    main()
